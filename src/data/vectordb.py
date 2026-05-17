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
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Document,
)

from .chunker import Chunk

logger = logging.getLogger(__name__)
load_dotenv()


COLLECTION_NAME = "cembrowski_papers_test"
VECTOR_DIM = 384            # embedding model's output dimension
                            # 384 for all-MiniLM-L6-v2
UPSERT_BATCH_SIZE = 64      # points per upsert call; tune to your memory / API limits

PROJECT_ROOT = Path(__file__).resolve().parents[2]
QDRANT_LOCAL_PATH = PROJECT_ROOT / "data" / "vectors"


def get_qdrant_client() -> QdrantClient:
    """
    Return a QdrantClient configured from environment variables.

    Local dev  → set nothing (uses embedded local DB at ROOT/data/vectors)
    Production → set QDRANT_URL and QDRANT_API_KEY in your environment / .env
    """
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    qdrant_local_path = os.getenv("QDRANT_LOCAL_PATH", QDRANT_LOCAL_PATH)

    if qdrant_url:
        logger.info(f"Connecting to Qdrant Cloud at {qdrant_url}")
        return QdrantClient(url=qdrant_url, api_key=qdrant_api_key, cloud_inference=True)

    logger.info(f"Using local Qdrant at '{qdrant_local_path}'")
    return QdrantClient(path=qdrant_local_path)


def ensure_collection(client: QdrantClient, recreate: bool = False) -> None:
    """
    Create the collection if it doesn't exist.

    Args:
        client:    Active QdrantClient.
        recreate:  If True, drop and recreate (useful for re-indexing from scratch).
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
        logger.info(f"Created collection '{COLLECTION_NAME}' (dim={VECTOR_DIM}).")
    else:
        logger.info(f"Collection '{COLLECTION_NAME}' already exists — skipping creation.")


def _batched(iterable, n: int) -> Iterator[list]:
    """Yield successive n-sized batches from an iterable."""
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch


def _chunks_to_points(chunks: list[Chunk], texts: list[str]) -> list[PointStruct]:
    """Zip chunks with their embeddings into Qdrant PointStructs."""
    assert len(chunks) == len(texts), (
        f"Chunk count ({len(chunks)}) != text count ({len(texts)})"
    )
    return [
        PointStruct(id=chunk.id, vector=Document(text=text, model="sentence-transformers/all-MiniLM-L6-v2"), payload=chunk.payload)
        for chunk, text in zip(chunks, texts)
    ]

def embed_and_upsert(
    client: QdrantClient,
    chunks: list[Chunk],
) -> int:
    """
    Embed a list of Chunk objects and upsert them into Qdrant in batches.

    Args:
        client:    Active QdrantClient (from get_qdrant_client()).
        chunks:    Output of chunk_paper() — one or more papers' worth of chunks.

    Returns:
        Total number of points upserted.
    """
    if not chunks:
        logger.warning("embed_and_upsert called with empty chunk list.")
        return 0

    total_upserted = 0

    for batch in _batched(chunks, UPSERT_BATCH_SIZE):
        texts = [c.text for c in batch]

        points = _chunks_to_points(batch, texts)

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
    ensure_collection(client, recreate=False)

    for paper in load_papers_from_json():
        if paper.processed:
            logger.info(f"Paper '{paper.title}' (ID: {paper.id}) already processed — skipping.\n")
            continue

        chunks = chunk_paper(paper)

        if (embed_and_upsert(client, chunks)):
            logger.info(f"Successfully embedded and upserted chunks for paper '{paper.title}' (ID: {paper.id}).\n")
            paper.processed = True
            save_paper(paper)
            

