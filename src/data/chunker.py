"""
This module chunks text (extracted from articles) for embedding and RAG.
Currently implements recursive chunking using LangChain's markdown-aware splitter.
Each chunk will also carry paper metadata (title, authors, etc.) in its payload dict, ready for Qdrant upsert.
"""

import logging
import uuid
from pathlib import Path


from models import Paper, Chunk
from parser import preprocess_markdown
from serialization import load_paper


from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1024 # characters per chunk
CHUNK_OVERLAP = 128

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data/papers"
JSON_DIR = PROJECT_ROOT / "data/json"

def _build_payload(paper: Paper, chunk_index: int, chunk_text: str) -> dict:
    """
    Assemble Qdrant payload metadata for a given chunk.
    Args:
        paper: Paper object containing metadata
        chunk_index: Index of the current chunk (0-based)
        chunk_text: Text content of the current chunk
    Returns:
        Dictionary payload with paper metadata and chunk info
    """
    return {
        "paper_id": paper.id,
        "title": paper.title,
        "authors": paper.authors,
        "year": paper.year,
        "publication": paper.publication,
        "authors": paper.authors,
        "chunk_index": chunk_index,
        "text": chunk_text,
    }


def chunk_paper(paper: Paper) -> list[Chunk]:
    """
    Splits paper's cleaned text into overlapping chunks with metadata payload.

    Args:
        paper: Paper object containing text and metadata

    Returns:
        List of Chunk objects with text and metadata payload
    """

    if not paper.text or not paper.text.strip():
        logger.warning(f"Paper {paper.id} has empty text. Skipping chunking.")
        return []
    
    cleaned_text = preprocess_markdown(paper.text)

    text_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    try:
        raw_chunks = text_splitter.split_text(cleaned_text)
    except Exception as e:
        logger.error(f"Error splitting text for paper {paper.id}: {e}")
        return []
    
    chunks: list[Chunk] = []
    for i, chunk_text in enumerate(raw_chunks):
        chunk_text = chunk_text.strip()
        if not chunk_text:
            continue

        chunks.append(
            Chunk(
                id=str(uuid.uuid4()),
                text=chunk_text,
                payload=_build_payload(paper, i, chunk_text),
            )
        )

    logger.info(f"Paper {paper.id} split into {len(chunks)} chunks.")
    return chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage: load papers from JSON, chunk them, and print chunk info
    paper = load_paper(JSON_DIR / "DaZlHqgIxjgJ.json")
    try:
        chunks = chunk_paper(paper)
        logger.info(f"Paper '{paper.title}' produced {len(chunks)} chunks.")
    except Exception as e:
        logger.error(f"Failed to chunk paper '{paper.title}': {e}")
    

