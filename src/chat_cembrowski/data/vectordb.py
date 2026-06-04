#type: ignore
"""
Qdrant client factory, collection management, and batch upsert for Chunk objects.
Uses Voyage AI multimodal embeddings (voyage-multimodal-3.5) for both text and image chunks.
"""

import logging
import os
from itertools import islice
from typing import Iterator
from pathlib import Path
from dotenv import load_dotenv

import PIL.Image
import voyageai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .chunker import Chunk

logger = logging.getLogger(__name__)
load_dotenv()


COLLECTION_NAME = "jenna_rimkus_papers"

EMBEDDING_MODEL = "voyage-multimodal-3.5"
VECTOR_DIM = 1024

TEXT_BATCH_SIZE = 64
IMAGE_BATCH_SIZE = 16    # images are token-heavier; keep batches smaller

PROJECT_ROOT = Path(__file__).resolve().parents[3]
QDRANT_LOCAL_PATH = PROJECT_ROOT / "data" / "vectors"
IMAGES_DIR = PROJECT_ROOT / "data" / "images"


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


def get_voyage_client() -> voyageai.Client:
    """Return a Voyage AI client. Reads VOYAGE_API_KEY from environment / .env."""
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        raise ValueError("VOYAGE_API_KEY not set in environment.")
    return voyageai.Client(api_key=api_key)


def ensure_collection(client: QdrantClient, recreate: bool = False) -> None:
    """
    Create the collection if it doesn't exist.

    Args:
        client:   Active QdrantClient.
        recreate: If True, drop and recreate (required when changing VECTOR_DIM
                  or switching embedding models).
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
        logger.info(f"Collection '{COLLECTION_NAME}' already exists — skipping creation.")


def _batched(iterable, n: int) -> Iterator[list]:
    """Yield successive n-sized batches from an iterable."""
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch


def _embed_text_batch(vo: voyageai.Client, chunks: list[Chunk]) -> list[list[float]]:
    """Embed a batch of text chunks via the Voyage multimodal endpoint."""
    result = vo.multimodal_embed(
        inputs=[[c.text] for c in chunks],
        model=EMBEDDING_MODEL,
        input_type="document",
    )
    return result.embeddings


def _embed_image_batch(
    vo: voyageai.Client,
    chunks: list[Chunk],
    images_dir: Path,
) -> tuple[list[Chunk], list[list[float]]]:
    """
    Embed a batch of image chunks via the Voyage multimodal endpoint.

    Each input is a [text, PIL.Image] pair where text is the caption + paper
    metadata assembled by the chunker.  Chunks whose image file is missing or
    unreadable are skipped; the returned lists are always parallel.
    """
    valid_chunks: list[Chunk] = []
    inputs = []

    for chunk in chunks:
        image_path = images_dir / chunk.payload["source_file"]
        if not image_path.exists():
            logger.warning(f"Image file not found, skipping: {image_path.name}")
            continue
        try:
            img = PIL.Image.open(image_path)
        except Exception as e:
            logger.warning(f"Failed to open image {image_path.name}: {e}")
            continue

        # Always include text if present; Voyage handles text-only or mixed inputs.
        inputs.append([chunk.text, img] if chunk.text else [img])
        valid_chunks.append(chunk)

    if not inputs:
        return [], []

    result = vo.multimodal_embed(
        inputs=inputs,
        model=EMBEDDING_MODEL,
        input_type="document",
    )
    return valid_chunks, result.embeddings


def _chunks_to_points(
    chunks: list[Chunk],
    embeddings: list[list[float]],
) -> list[PointStruct]:
    assert len(chunks) == len(embeddings), (
        f"Chunk count ({len(chunks)}) != embedding count ({len(embeddings)})"
    )
    return [
        PointStruct(id=chunk.id, vector=embedding, payload=chunk.payload)
        for chunk, embedding in zip(chunks, embeddings)
    ]


def embed_and_upsert(
    client: QdrantClient,
    vo: voyageai.Client,
    chunks: list[Chunk],
    images_dir: Path = IMAGES_DIR,
    collection_name: str = COLLECTION_NAME,
) -> int:
    """
    Embed and upsert a mixed list of text and image Chunk objects into Qdrant.

    Routes by chunk_category: text chunks are embedded as plain text; image
    chunks are embedded as (text, image) pairs using the Voyage multimodal model.

    Args:
        client:     Active QdrantClient.
        vo:         Active Voyage AI client.
        chunks:     Output of chunk_paper() + chunk_paper_images() for one or
                    more papers.
        images_dir: Directory containing extracted image files.
        collection_name: Qdrant collection to upsert into.

    Returns:
        Total number of points successfully upserted.
    """
    if not chunks:
        logger.warning("embed_and_upsert called with empty chunk list.")
        return 0

    text_chunks  = [c for c in chunks if c.payload.get("chunk_category") == "text"]
    image_chunks = [c for c in chunks if c.payload.get("chunk_category") == "image"]

    total_upserted = 0

    for batch in _batched(text_chunks, TEXT_BATCH_SIZE):
        try:
            embeddings = _embed_text_batch(vo, batch)
        except Exception as e:
            logger.error(f"Voyage text embedding failed: {e}")
            continue
        points = _chunks_to_points(batch, embeddings)
        try:
            client.upsert(collection_name=collection_name, points=points)
            total_upserted += len(points)
            logger.debug(f"Upserted {len(points)} text points.")
        except Exception as e:
            logger.error(f"Qdrant upsert failed (text batch): {e}")

    for batch in _batched(image_chunks, IMAGE_BATCH_SIZE):
        try:
            valid_chunks, embeddings = _embed_image_batch(vo, batch, images_dir)
        except Exception as e:
            logger.error(f"Voyage image embedding failed: {e}")
            continue
        if not valid_chunks:
            continue
        points = _chunks_to_points(valid_chunks, embeddings)
        try:
            client.upsert(collection_name=collection_name, points=points)
            total_upserted += len(points)
            logger.debug(f"Upserted {len(points)} image points.")
        except Exception as e:
            logger.error(f"Qdrant upsert failed (image batch): {e}")

    logger.info(f"Total upserted: {total_upserted} points.")
    return total_upserted


if __name__ == "__main__":
    from .chunker import chunk_paper, chunk_paper_images
    from .serialization import load_papers_from_json, save_paper

    logging.basicConfig(level=logging.INFO)

    client = get_qdrant_client()
    vo = get_voyage_client()
    collection_name = "jenna_rimkus_papers"  # change if you want to use a different collection

    try:
        ensure_collection(client, recreate=False)
        for paper in load_papers_from_json():
            if paper.processed:
                logger.info(
                    f"Paper '{paper.title}' (ID: {paper.id}) already "
                    "processed — skipping.\n"
                )
                continue

            chunks = chunk_paper(paper) + chunk_paper_images(paper)

            if embed_and_upsert(client, vo, chunks, collection_name=collection_name) > 0:
                logger.info(
                    f"Embedded and upserted chunks for paper "
                    f"'{paper.title}' (ID: {paper.id}).\n"
                )
                paper.processed = True
                save_paper(paper)
    finally:
        client.close()
