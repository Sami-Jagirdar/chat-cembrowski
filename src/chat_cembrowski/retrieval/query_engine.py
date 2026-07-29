from __future__ import annotations

from dataclasses import dataclass, field

import voyageai
from openai import OpenAI
from qdrant_client import QdrantClient

from chat_cembrowski.data.vectordb import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)

from . import authors, nih
from .nih import NIHResult
from .prompts import (
    CLASSIFIER_PROMPT,
    CONDENSE_PROMPT,
    META_ANSWER,
    NIH_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
)

# A single prior turn: {"role": "user" | "assistant", "content": str}. Already
# the OpenAI message shape, so it drops straight into a `messages` list.
ChatMessage = dict[str, str]

CHAT_MODEL = "gpt-4.1"
CLASSIFIER_MODEL = "gpt-4.1-mini"

# Minimum top-hit Qdrant cosine score to trust the Cembrowski corpus for a
# question classified as Cembrowski-specific. Below this, retrieval is too
# weak to be reliable, so the question is routed to NIH instead.
#
# Measured against the live collection (voyage-multimodal-3.5, page-based
# chunks). On-topic questions score 0.32-0.65; general medical questions that
# belong on the NIH path top out at 0.24. Short on-topic questions sit at the
# bottom of that range -- "tell me about gem 4000" scores 0.33 -- so the
# previous value of 0.4 cut through the middle of the on-topic band and sent
# terse but valid corpus questions to NIH. 0.30 sits in the empty gap between
# the two populations. Re-measure before changing the embedding model or the
# chunking strategy, since both shift the score distribution.
SCORE_THRESHOLD = 0.30


@dataclass
class RetrievedChunk:
    score: float
    source_type: str        # "paper" or "document"
    title: str
    text: str
    chunk_index: int
    # Paper-specific
    paper_id: str | None = None
    publication: str | None = None
    year: int | None = None
    page_label: str | None = None
    authors: list[str] | None = None
    page: int | None = None
    # Set by scripts/link_posters.py — ties a poster chunk to its page on the
    # website. Absent on chunks that predate the backfill, so always optional.
    site_path: str | None = None    # e.g. "/presentation/gem-4000-cartridge-instability"
    poster_id: str | None = None    # e.g. "pos-gem-4000-cartridge-instability"
    # Document-specific
    file_type: str | None = None
    # Image-specific
    caption: str | None = None
    image_type: str | None = None


@dataclass
class SourceRef:
    """
    One entry in a numbered source list, aligned by position with the
    `SOURCE {i}` blocks handed to the model. `index` is 1-based and is what the
    model cites as `[index]`.

    `kind` is "poster" (Cembrowski research with a page on the site),
    "document" (an internal note/code file, no public page), or "nih"
    (MedlinePlus/PubMed). `url` is a site-relative path for posters, an absolute
    URL for NIH, and None for documents.
    """
    index: int
    kind: str
    title: str
    authors: list[str] = field(default_factory=list)
    url: str | None = None
    page: int | None = None
    publication: str | None = None
    year: int | None = None

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "kind": self.kind,
            "title": self.title,
            "authors": self.authors,
            "url": self.url,
            "page": self.page,
            "publication": self.publication,
            "year": self.year,
        }


@dataclass
class QueryResult:
    """Answer plus the ordered sources it was grounded in and the route taken."""
    answer: str
    route: str              # "cembrowski" or "nih"
    sources: list[SourceRef] = field(default_factory=list)


class QueryEngine:
    def __init__(
        self,
        qdrant_client: QdrantClient,
        openai_client: OpenAI,
        voyage_client: voyageai.Client, # type: ignore
        top_k: int = 10,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        self.qdrant = qdrant_client
        self.openai = openai_client
        self.voyage = voyage_client
        self.top_k = top_k
        self.collection_name = collection_name

    def query(
        self, question: str, history: list[ChatMessage] | None = None
    ) -> str:
        """End-to-end RAG query pipeline. See `query_structured` for details."""
        return self.query_structured(question, history).answer

    def query_with_route(
        self, question: str, history: list[ChatMessage] | None = None
    ) -> tuple[str, str]:
        """
        End-to-end RAG query pipeline, returning `(answer, route)`.

        Thin wrapper over `query_structured` for callers that want the route
        label but not the source list. See `query_structured` for the steps.
        """
        result = self.query_structured(question, history)
        return result.answer, result.route

    def query_structured(
        self, question: str, history: list[ChatMessage] | None = None
    ) -> QueryResult:
        """
        End-to-end RAG query pipeline with source routing and structured sources.

        Steps:
        0. If there's prior conversation, condense the follow-up into a
           standalone question (cheap LLM call) — used for classification,
           embedding, and search so context-dependent follow-ups (e.g. "what
           about in women?") still retrieve correctly.
        1. Check the (standalone) question against every author name known to
           the corpus (fuzzy match, no LLM call). A confident hit answers
           from that person's works across the corpus (metadata filter, not
           vector search — author names are never embedded) and skips
           classification entirely.
        2. Otherwise, classify the question as "cembrowski", "general", or
           "meta" (cheap LLM call).
        2. If "meta" (a question about the site/assistant itself, not a
           health/research topic): answer with the static META_ANSWER —
           no retrieval at all, Cembrowski or NIH.
        3. If "cembrowski": embed + search Qdrant. If the top hit clears
           SCORE_THRESHOLD, answer from the Cembrowski corpus.
        4. Otherwise (classified "general", or Cembrowski retrieval was too
           weak to trust): answer from NIH (MedlinePlus + PubMed).

        The raw `question` (plus `history`) is what's sent to the model for
        answer generation, so phrasing and tone stay natural; only retrieval
        and routing operate on the condensed standalone version.

        Returns a QueryResult whose `sources` are 1-based and aligned by
        position with the `SOURCE {i}` blocks the model was shown, so a `[i]`
        citation in the answer maps straight to `sources[i - 1]`.
        """
        history = history or []
        search_question = (
            self._condense_question(question, history) if history else question
        )

        matched_author = self._match_author(search_question)
        if matched_author:
            answer, sources = self._answer_author(question, matched_author, history)
            return QueryResult(answer=answer, route="author", sources=sources)

        label = self._classify(search_question)

        if label == "meta":
            return QueryResult(answer=META_ANSWER, route="meta", sources=[])

        if label != "general":
            query_embedding = self._embed_query(search_question)
            retrieved_chunks = self._search(query_embedding)

            strong_match = (
                bool(retrieved_chunks)
                and retrieved_chunks[0].score >= SCORE_THRESHOLD
            )
            if strong_match:
                answer, sources = self._answer_cembrowski(
                    question, retrieved_chunks, history
                )
                return QueryResult(answer=answer, route="cembrowski", sources=sources)

        answer, sources = self._answer_nih(question, history, search_question)
        return QueryResult(answer=answer, route="nih", sources=sources)

    def _condense_question(
        self, question: str, history: list[ChatMessage]
    ) -> str:
        """
        Rewrite a follow-up question as a standalone question using the prior
        conversation (cheap LLM call). Used only to drive classification,
        embedding, and search — the original `question` is still what gets
        answered.

        Falls back to the raw question on any API failure or empty response,
        so a condensation hiccup degrades to today's stateless behavior rather
        than breaking the request.
        """
        try:
            response = self.openai.chat.completions.create(
                model=CLASSIFIER_MODEL,
                temperature=0,
                max_tokens=256,
                messages=[
                    {"role": "system", "content": CONDENSE_PROMPT},
                    *history, # type: ignore
                    {
                        "role": "user",
                        "content": f"Follow-up question: {question}\n\nStandalone question:",
                    },
                ],
            )
            standalone = (response.choices[0].message.content or "").strip()
        except Exception:
            return question

        return standalone or question

    def _classify(self, question: str) -> str:
        """
        Classify a question as "cembrowski", "general", or "meta" via a
        cheap LLM call.

        Defaults to "cembrowski" on any API failure or an unrecognized
        label — the retrieval-score check in `query_structured` still
        catches weak/off-topic matches and routes them to NIH, so failing
        open here doesn't bypass that fallback. (Never defaults to "meta":
        that route skips retrieval entirely, so a wrong "meta" guess would
        leave the answer stuck with a canned response.)
        """
        try:
            response = self.openai.chat.completions.create(
                model=CLASSIFIER_MODEL,
                temperature=0,
                max_tokens=5,
                messages=[
                    {"role": "system", "content": CLASSIFIER_PROMPT},
                    {"role": "user", "content": question},
                ],
            )
            label = (response.choices[0].message.content or "").strip().lower()
        except Exception:
            return "cembrowski"

        if "meta" in label:
            return "meta"
        if "general" in label:
            return "general"
        return "cembrowski"

    def _answer_cembrowski(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        history: list[ChatMessage] | None = None,
    ) -> tuple[str, list[SourceRef]]:
        """
        Build a grounded prompt from Cembrowski corpus chunks and generate an
        answer, returning it alongside the numbered sources it was shown.

        Chunks split into `citable` (resolves to a real URL — a linked
        poster) and `background` (documents, and posters not yet linked via
        scripts/link_posters.py). Only `citable` chunks become numbered
        SOURCE blocks and SourceRefs, so the reader-facing sources list is
        never handed something unclickable. `background` chunks still inform
        the answer, just without a citation number — see prompts.SYSTEM_PROMPT
        rule 8.
        """
        citable_chunks = [c for c in chunks if c.site_path]
        background_chunks = [c for c in chunks if not c.site_path]

        context = self._build_context(citable_chunks)
        background = self._build_background_context(background_chunks)
        sources = self._cembrowski_sources(citable_chunks)

        user_content = f"""
Question:
{question}

Context:
{context}
"""
        if background:
            user_content += f"""
Additional background (not citable — do not cite these with a bracket number, use only to inform your answer):
{background}
"""

        response = self.openai.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0.1,
            max_tokens=1500,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                *(history or []), # type: ignore
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
        )

        return response.choices[0].message.content or "", sources

    def _build_background_context(self, chunks: list[RetrievedChunk]) -> str:
        """
        Unnumbered context from non-citable chunks (documents, unlinked
        posters) — informs the answer but is never assigned a SOURCE number,
        so it can never be cited.
        """
        if not chunks:
            return ""

        sections = []
        for chunk in chunks:
            header_lines = [f"Title: {chunk.title}"]
            if chunk.file_type:
                header_lines.append(f"Type: {chunk.file_type}")
            if chunk.publication:
                header_lines.append(f"Publication: {chunk.publication}")
            if chunk.year:
                header_lines.append(f"Year: {chunk.year}")
            header = "\n".join(header_lines)
            body = chunk.caption or chunk.text
            sections.append(f"{header}\n\nContent:\n{body}")

        return "\n\n----\n\n".join(sections)

    def _match_author(self, question: str) -> str | None:
        """Fuzzy-match `question` against every author name known to the corpus."""
        try:
            known = authors.get_known_authors(self.qdrant, self.collection_name)
            return authors.match_author(question, known)
        except Exception:
            return None

    def _answer_author(
        self,
        question: str,
        author_name: str,
        history: list[ChatMessage] | None = None,
    ) -> tuple[str, list[SourceRef]]:
        """
        Answer a "who is X" question from every distinct work crediting
        `author_name` across the corpus (a payload filter, not a vector
        search — exhaustive rather than whatever lands in the top-k).

        Collapses to one representative chunk per distinct work (lowest
        chunk_index, i.e. first page) before handing off to
        `_answer_cembrowski`, which applies the same citable/background split
        as any other Cembrowski-routed answer.
        """
        points = authors.fetch_chunks_by_author(
            self.qdrant, self.collection_name, author_name
        )

        best_by_work: dict[str, RetrievedChunk] = {}
        for point in points:
            chunk = self._point_to_chunk(point)
            key = chunk.paper_id or chunk.title
            current = best_by_work.get(key)
            if current is None or chunk.chunk_index < current.chunk_index:
                best_by_work[key] = chunk

        return self._answer_cembrowski(question, list(best_by_work.values()), history)

    def _cembrowski_sources(
        self, chunks: list[RetrievedChunk]
    ) -> list[SourceRef]:
        """
        One SourceRef per chunk, numbered from 1 in the same order as
        `_build_context` writes the `SOURCE {i}` blocks — so `[i]` in the answer
        maps to `sources[i - 1]`. Documents carry no `site_path`, so their `url`
        is None and they render as an unlinked, title-only citation.
        """
        sources: list[SourceRef] = []
        for i, chunk in enumerate(chunks, start=1):
            kind = "document" if chunk.source_type == "document" else "poster"
            sources.append(
                SourceRef(
                    index=i,
                    kind=kind,
                    title=chunk.title,
                    authors=chunk.authors or [],
                    url=chunk.site_path,
                    page=chunk.page,
                    publication=chunk.publication,
                    year=chunk.year,
                )
            )
        return sources

    def _search_nih(self, question: str) -> list[NIHResult]:
        return nih.search_nih(question)

    def _build_nih_context(self, results: list[NIHResult]) -> str:
        """Build retrieval context for the LLM from NIH (MedlinePlus/PubMed) results."""
        sections = []

        for i, result in enumerate(results, start=1):
            header_lines = [f"Source: {result.source}", f"Title: {result.title}"]
            if result.journal:
                header_lines.append(f"Journal: {result.journal}")
            if result.year:
                header_lines.append(f"Year: {result.year}")
            header_lines.append(f"URL: {result.url}")
            header = "\n".join(header_lines)
            body = result.summary or "(no summary available)"

            sections.append(f"SOURCE {i}\n\n{header}\n\nContent:\n{body}")

        return "\n\n====================\n\n".join(sections)

    def _answer_nih(
        self,
        question: str,
        history: list[ChatMessage] | None = None,
        search_question: str | None = None,
    ) -> tuple[str, list[SourceRef]]:
        """
        Answer a general medical question from NIH (MedlinePlus/PubMed) search
        results, returning the answer alongside the numbered sources shown.
        """
        results = self._search_nih(search_question or question)

        if not results:
            return (
                "I couldn't find reliable NIH information to answer this "
                "question. Please try rephrasing, or consult a healthcare "
                "professional.\n\n"
                "This is general information, not medical advice.",
                [],
            )

        context = self._build_nih_context(results)
        sources = [
            SourceRef(
                index=i,
                kind="nih",
                title=result.title,
                url=result.url,
                publication=result.journal or result.source,
                year=int(result.year) if result.year and result.year.isdigit() else None,
            )
            for i, result in enumerate(results, start=1)
        ]

        response = self.openai.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0.1,
            max_tokens=1500,
            messages=[
                {
                    "role": "system",
                    "content": NIH_SYSTEM_PROMPT,
                },
                *(history or []), # type: ignore
                {
                    "role": "user",
                    "content": f"""
Question:
{question}

Context:
{context}
""",
                },
            ],
        )

        return response.choices[0].message.content or "", sources

    def _embed_query(self, question: str) -> list[float]:
        result = self.voyage.multimodal_embed(
            inputs=[[question]],
            model=EMBEDDING_MODEL,
            input_type="query",
        )
        return result.embeddings[0]

    def _search(self, query_embedding: list[float]) -> list[RetrievedChunk]:
        results = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=self.top_k,
            with_payload=True,
        ).points

        return [self._point_to_chunk(point, score=point.score) for point in results]

    def _point_to_chunk(self, point, score: float = 0.0) -> RetrievedChunk:
        """Convert one Qdrant point into a RetrievedChunk, routed by payload shape."""
        payload = point.payload or {}
        chunk_category = payload.get("chunk_category", "text")
        source_type = payload.get("source_type", "paper")

        if chunk_category == "image":
            return RetrievedChunk(
                score=score,
                source_type="image",
                title=payload.get("title", "Unknown Title"),
                text=payload.get("text", ""),
                chunk_index=payload.get("chunk_index", -1),
                paper_id=payload.get("paper_id"),
                publication=payload.get("publication"),
                year=payload.get("year"),
                page_label=payload.get("page_label"),
                authors=payload.get("authors"),
                page=payload.get("page"),
                site_path=payload.get("site_path"),
                poster_id=payload.get("poster_id"),
                caption=payload.get("caption"),
                image_type=payload.get("image_type"),
            )
        elif source_type == "document":
            return RetrievedChunk(
                score=score,
                source_type="document",
                title=payload.get("title", "Unknown Document"),
                text=payload.get("text", ""),
                chunk_index=payload.get("chunk_index", -1),
                file_type=payload.get("file_type"),
            )
        else:
            return RetrievedChunk(
                score=score,
                source_type="paper",
                title=payload.get("title", "Unknown Title"),
                text=payload.get("text", ""),
                chunk_index=payload.get("chunk_index", -1),
                paper_id=payload.get("paper_id"),
                publication=payload.get("publication"),
                year=payload.get("year"),
                page_label=payload.get("page_label"),
                authors=payload.get("authors"),
                page=payload.get("page"),
                site_path=payload.get("site_path"),
                poster_id=payload.get("poster_id"),
            )

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        """Build retrieval context for the LLM, rendering papers and documents differently."""
        sections = []

        for i, chunk in enumerate(chunks, start=1):
            if chunk.source_type == "document":
                header_lines = [f"Title: {chunk.title}"]
                if chunk.file_type:
                    header_lines.append(f"Type: {chunk.file_type}")
                header = "\n".join(header_lines)
                body = chunk.text

            elif chunk.source_type == "image":
                header_lines = [f"Title: {chunk.title}"]
                if chunk.publication:
                    header_lines.append(f"Publication: {chunk.publication}")
                if chunk.year:
                    header_lines.append(f"Year: {chunk.year}")
                if chunk.page_label:
                    header_lines.append(f"Pages: {chunk.page_label}")
                if chunk.image_type:
                    header_lines.append(f"Image Type: {chunk.image_type}")
                header = "\n".join(header_lines)
                body = chunk.caption or chunk.text

            else:
                header_lines = [f"Title: {chunk.title}"]
                if chunk.publication:
                    header_lines.append(f"Publication: {chunk.publication}")
                if chunk.year:
                    header_lines.append(f"Year: {chunk.year}")
                if chunk.page_label:
                    header_lines.append(f"Pages: {chunk.page_label}")
                header = "\n".join(header_lines)
                body = chunk.text

            sections.append(
                f"SOURCE {i}\n\n{header}\n\nContent:\n{body}"
            )

        return "\n\n====================\n\n".join(sections)