from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict

from openai import OpenAI
from qdrant_client import QdrantClient

from chat_cembrowski.data.vectordb import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS,
)

from .prompts import SYSTEM_PROMPT


CHAT_MODEL = "gpt-4.1-mini"


@dataclass
class RetrievedChunk:
    score: float
    title: str
    publication: str
    year: int | None
    page_label: str
    text: str
    chunk_index: int


class QueryEngine:
    def __init__(
        self,
        qdrant_client: QdrantClient,
        openai_client: OpenAI,
        top_k: int = 10,
    ) -> None:
        self.qdrant = qdrant_client
        self.openai = openai_client
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

        return response.choices[0].message.content

    def _embed_query(self, question: str) -> list[float]:
        response = self.openai.embeddings.create(
            model=EMBEDDING_MODEL,
            input=question,
            dimensions=EMBEDDING_DIMENSIONS,
        )

        return response.data[0].embedding

    def _search(self, query_embedding: list[float]) -> list[RetrievedChunk]:
        results = self.qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=self.top_k,
            with_payload=True,
        ).points

        chunks = []

        for point in results:
            payload = point.payload

            chunks.append(
                RetrievedChunk(
                    score=point.score,
                    title=payload["title"],
                    publication=payload["publication"],
                    year=payload.get("year"),
                    page_label=payload["page_label"],
                    text=payload["text"],
                    chunk_index=payload["chunk_index"],
                )
            )

        return chunks

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        """
        Build retrieval context for the LLM.

        Deduplicates repeated chunk references while preserving order.
        """
        sections = []

        for i, chunk in enumerate(chunks, start=1):
            sections.append(
                f"""
SOURCE {i}

Title: {chunk.title}
Publication: {chunk.publication}
Year: {chunk.year}
Pages: {chunk.page_label}

Content:
{chunk.text}
""".strip()
            )

        return "\n\n====================\n\n".join(sections)