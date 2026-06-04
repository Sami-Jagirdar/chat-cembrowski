from __future__ import annotations

from dataclasses import dataclass

import voyageai
from openai import OpenAI
from qdrant_client import QdrantClient

from chat_cembrowski.data.vectordb import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)

from .prompts import SYSTEM_PROMPT


CHAT_MODEL = "gpt-4.1-mini"


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


class QueryEngine:
    def __init__(
        self,
        qdrant_client: QdrantClient,
        openai_client: OpenAI,
        voyage_client: voyageai.Client, # type: ignore
        top_k: int = 10,
    ) -> None:
        self.qdrant = qdrant_client
        self.openai = openai_client
        self.voyage = voyage_client
        self.top_k = top_k

    def query(self, question: str) -> str:
        """
        End-to-end RAG query pipeline.

        Steps:
        1. Embed query
        2. Retrieve top-k chunks from Qdrant
        3. Build grounded prompt
        4. Generate answer
        """
        query_embedding = self._embed_query(question)

        retrieved_chunks = self._search(query_embedding)

        context = self._build_context(retrieved_chunks)

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

    def _embed_query(self, question: str) -> list[float]:
        result = self.voyage.multimodal_embed(
            inputs=[[question]],
            model=EMBEDDING_MODEL,
            input_type="query",
        )
        return result.embeddings[0]

    def _search(self, query_embedding: list[float]) -> list[RetrievedChunk]:
        results = self.qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=self.top_k,
            with_payload=True,
        ).points

        chunks = []

        for point in results:
            payload = point.payload or {}
            source_type = payload.get("source_type", "paper")

            if source_type == "document":
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
            else:
                header_lines = [f"Title: {chunk.title}"]
                if chunk.publication:
                    header_lines.append(f"Publication: {chunk.publication}")
                if chunk.year:
                    header_lines.append(f"Year: {chunk.year}")
                if chunk.page_label:
                    header_lines.append(f"Pages: {chunk.page_label}")
                header = "\n".join(header_lines)

            sections.append(
                f"SOURCE {i}\n\n{header}\n\nContent:\n{chunk.text}"
            )

        return "\n\n====================\n\n".join(sections)