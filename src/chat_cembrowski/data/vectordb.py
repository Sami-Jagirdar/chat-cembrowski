"""
Qdrant client factory, collection management, and batch upsert for Chunk objects.
"""

import logging
import os
from itertools import islice
from typing import Iterator
from pathlib import Path
from dotenv import load_dotenv


from qdrant_client import QdrantClient
from openai import OpenAI
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from .chunker import Chunk

logger = logging.getLogger(__name__)
load_dotenv()


COLLECTION_NAME = "cembrowski_papers_v3"

# text-embedding-3-large supports Matryoshka truncation via the `dimensions`
# parameter.  1024 dims retains most retrieval quality (~94 % of full 3072)
# while using only 1/3 the storage.  Switch VECTOR_DIM + EMBEDDING_DIMENSIONS
# together if you ever want to experiment with other sizes.
EMBEDDING_MODEL      = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 1024           # truncated via OpenAI's native param
VECTOR_DIM           = EMBEDDING_DIMENSIONS

UPSERT_BATCH_SIZE = 64                # points per upsert call

PROJECT_ROOT = Path(__file__).resolve().parents[3]
QDRANT_LOCAL_PATH = PROJECT_ROOT / "data" / "vectors"


def get_qdrant_client() -> QdrantClient:
    """
    Return a QdrantClient configured from environment variables.

    Local dev  → set nothing (uses embedded local DB at ROOT/data/vectors)
    Production → set QDRANT_CLUSTER_ENDPOINT and QDRANT_API_KEY in your
                 environment / .env
    """
    qdrant_url = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    qdrant_local_path = os.getenv("QDRANT_LOCAL_PATH", str(QDRANT_LOCAL_PATH))

    if qdrant_url:
        logger.info(f"Connecting to Qdrant Cloud at {qdrant_url}")
        return QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    logger.info(f"Using local Qdrant at '{qdrant_local_path}'")
    return QdrantClient(path=qdrant_local_path)


def get_openai_client() -> OpenAI:
    """
    Return an OpenAI client configured from environment variables.
    Requires OPENAI_API_KEY in your environment / .env.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not set in environment.")
    return OpenAI(api_key=openai_api_key)


def ensure_collection(client: QdrantClient, recreate: bool = False) -> None:
    """
    Create the collection if it doesn't exist.

    Args:
        client:    Active QdrantClient.
        recreate:  If True, drop and recreate (useful for re-indexing from
                   scratch).  Note: if you change VECTOR_DIM you must recreate.
    """
    exists = client.collection_exists(COLLECTION_NAME)

    if exists and recreate:
        logger.warning(f"Dropping existing collection '{COLLECTION_NAME}'.")
        client.delete_collection(COLLECTION_NAME)
        exists = False

    if not exists:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        logger.info(
            f"Created collection '{COLLECTION_NAME}' "
            f"(model={EMBEDDING_MODEL}, dim={VECTOR_DIM})."
        )
    else:
        logger.info(
            f"Collection '{COLLECTION_NAME}' already exists — skipping creation."
        )


def _batched(iterable, n: int) -> Iterator[list]:
    """Yield successive n-sized batches from an iterable."""
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch


def _embed_texts(openai_client: OpenAI, texts: list[str]) -> list[list[float]]:
    """
    Call the OpenAI embeddings endpoint for a batch of texts.

    Uses the `dimensions` parameter to truncate text-embedding-3-large output
    to EMBEDDING_DIMENSIONS, keeping storage lean without a separate PCA step.
    OpenAI's Matryoshka training ensures truncated vectors still rank well.
    """
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
        dimensions=EMBEDDING_DIMENSIONS,
    )
    # Response items are guaranteed to be in the same order as the input.
    return [item.embedding for item in response.data]


def _chunks_to_points(
    chunks: list[Chunk],
    embeddings: list[list[float]],
) -> list[PointStruct]:
    """Zip chunks with their embeddings into Qdrant PointStructs."""
    assert len(chunks) == len(embeddings), (
        f"Chunk count ({len(chunks)}) != embedding count ({len(embeddings)})"
    )
    return [
        PointStruct(id=chunk.id, vector=embedding, payload=chunk.payload)
        for chunk, embedding in zip(chunks, embeddings)
    ]


def embed_and_upsert(
    client: QdrantClient,
    openai_client: OpenAI,
    chunks: list[Chunk],
) -> int:
    """
    Embed a list of Chunk objects via OpenAI and upsert them into Qdrant in
    batches.

    Args:
        client:        Active QdrantClient (from get_qdrant_client()).
        openai_client: Active OpenAI client (from get_openai_client()).
        chunks:        Output of chunk_paper() — one or more papers' worth.

    Returns:
        Total number of points upserted.
    """
    if not chunks:
        logger.warning("embed_and_upsert called with empty chunk list.")
        return 0

    total_upserted = 0

    for batch in _batched(chunks, UPSERT_BATCH_SIZE):
        texts = [c.text for c in batch]

        try:
            embeddings = _embed_texts(openai_client, texts)
        except Exception as e:
            logger.error(f"OpenAI embedding failed for batch: {e}")
            continue

        points = _chunks_to_points(batch, embeddings)

        try:
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            total_upserted += len(points)
            logger.debug(f"Upserted {len(points)} points.")
        except Exception as e:
            logger.error(f"Qdrant upsert failed: {e}")

    logger.info(f"Total upserted: {total_upserted} points.")
    return total_upserted


if __name__ == "__main__":
    from .chunker import chunk_paper
    from .serialization import load_papers_from_json, save_paper

    logging.basicConfig(level=logging.INFO)

    client = get_qdrant_client()
    openai_client = get_openai_client()

    try:
        ensure_collection(client, recreate=False)
        for paper in load_papers_from_json():
            if paper.processed:
                logger.info(
                    f"Paper '{paper.title}' (ID: {paper.id}) already "
                    "processed — skipping.\n"
                )
                continue

            chunks = chunk_paper(paper)

            if embed_and_upsert(client, openai_client, chunks):
                logger.info(
                    f"Successfully embedded and upserted chunks for paper "
                    f"'{paper.title}' (ID: {paper.id}).\n"
                )
                paper.processed = True
                save_paper(paper)
    finally:
        client.close()