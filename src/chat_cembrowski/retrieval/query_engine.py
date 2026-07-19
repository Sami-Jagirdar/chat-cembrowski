from __future__ import annotations

from dataclasses import dataclass

import voyageai
from openai import OpenAI
from qdrant_client import QdrantClient

from chat_cembrowski.data.vectordb import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)

from . import nih
from .nih import NIHResult
from .prompts import CLASSIFIER_PROMPT, NIH_SYSTEM_PROMPT, SYSTEM_PROMPT


CHAT_MODEL = "gpt-4.1"
CLASSIFIER_MODEL = "gpt-4.1-mini"

# Minimum top-hit Qdrant cosine score to trust the Cembrowski corpus for a
# question classified as Cembrowski-specific. Below this, retrieval is too
# weak to be reliable, so the question is routed to NIH instead.
SCORE_THRESHOLD = 0.4


@dataclass
class RetrievedChunk:
    score: float
    source_type: str        # "paper" or "document"
    title: str
    text: str
    chunk_index: int
    # Paper-specific
    publication: str | None = None
    year: int | None = None
    page_label: str | None = None
    # Document-specific
    file_type: str | None = None
    # Image-specific
    caption: str | None = None
    image_type: str | None = None


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

    def query(self, question: str) -> str:
        """End-to-end RAG query pipeline. See `query_with_route` for details."""
        answer, _route = self.query_with_route(question)
        return answer

    def query_with_route(self, question: str) -> tuple[str, str]:
        """
        End-to-end RAG query pipeline with source routing.

        Steps:
        1. Classify the question as "cembrowski" or "general" (cheap LLM call).
        2. If "cembrowski": embed + search Qdrant. If the top hit clears
           SCORE_THRESHOLD, answer from the Cembrowski corpus.
        3. Otherwise (classified "general", or Cembrowski retrieval was too
           weak to trust): answer from NIH (MedlinePlus + PubMed).

        Returns:
            (answer, route) where route is "cembrowski" or "nih".
        """
        label = self._classify(question)

        if label != "general":
            query_embedding = self._embed_query(question)
            retrieved_chunks = self._search(query_embedding)

            strong_match = (
                bool(retrieved_chunks)
                and retrieved_chunks[0].score >= SCORE_THRESHOLD
            )
            if strong_match:
                return self._answer_cembrowski(question, retrieved_chunks), "cembrowski"

        return self._answer_nih(question), "nih"

    def _classify(self, question: str) -> str:
        """
        Classify a question as "cembrowski" or "general" via a cheap LLM call.

        Defaults to "cembrowski" on any API failure — the retrieval-score
        check in `query_with_route` still catches weak/off-topic matches and
        routes them to NIH, so failing open here doesn't bypass the fallback.
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

        return "general" if "general" in label else "cembrowski"

    def _answer_cembrowski(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> str:
        """Build a grounded prompt from Cembrowski corpus chunks and generate an answer."""
        context = self._build_context(chunks)

        response = self.openai.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0.1,
            max_tokens=1500,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
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

        return response.choices[0].message.content or ""

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

    def _answer_nih(self, question: str) -> str:
        """Answer a general medical question from NIH (MedlinePlus/PubMed) search results."""
        results = self._search_nih(question)

        if not results:
            return (
                "I couldn't find reliable NIH information to answer this "
                "question. Please try rephrasing, or consult a healthcare "
                "professional.\n\n"
                "This is general information, not medical advice."
            )

        context = self._build_nih_context(results)

        response = self.openai.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0.1,
            max_tokens=1500,
            messages=[
                {
                    "role": "system",
                    "content": NIH_SYSTEM_PROMPT,
                },
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

        return response.choices[0].message.content or ""

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

        chunks = []

        for point in results:
            payload = point.payload or {}
            chunk_category = payload.get("chunk_category", "text")
            source_type = payload.get("source_type", "paper")

            if chunk_category == "image":
                chunks.append(
                    RetrievedChunk(
                        score=point.score,
                        source_type="image",
                        title=payload.get("title", "Unknown Title"),
                        text=payload.get("text", ""),
                        chunk_index=payload.get("chunk_index", -1),
                        publication=payload.get("publication"),
                        year=payload.get("year"),
                        page_label=payload.get("page_label"),
                        caption=payload.get("caption"),
                        image_type=payload.get("image_type"),
                    )
                )
            elif source_type == "document":
                chunks.append(
                    RetrievedChunk(
                        score=point.score,
                        source_type="document",
                        title=payload.get("title", "Unknown Document"),
                        text=payload.get("text", ""),
                        chunk_index=payload.get("chunk_index", -1),
                        file_type=payload.get("file_type"),
                    )
                )
            else:
                chunks.append(
                    RetrievedChunk(
                        score=point.score,
                        source_type="paper",
                        title=payload.get("title", "Unknown Title"),
                        text=payload.get("text", ""),
                        chunk_index=payload.get("chunk_index", -1),
                        publication=payload.get("publication"),
                        year=payload.get("year"),
                        page_label=payload.get("page_label"),
                    )
                )

        return chunks

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