"""
This module chunks text (extracted from articles) for embedding and RAG.
Currently implements recursive chunking using LangChain's markdown-aware splitter.
Each chunk will also carry paper metadata (title, authors, etc.) in its payload dict, ready for Qdrant upsert.
"""

import logging
import uuid
from pathlib import Path


from .models import Paper, Chunk, ImageRecord
from .parser import preprocess_markdown, parse_pdf_for_pages
from .serialization import load_paper, load_image_records_for_paper


from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1024 # characters per chunk
CHUNK_OVERLAP = 128

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data/papers"
JSON_DIR = PROJECT_ROOT / "data/json"
IMAGE_JSON_DIR = PROJECT_ROOT / "data/image_json"

def _build_payload(paper: Paper, chunk_index: int, chunk_text: str, page_start: int, page_end: int) -> dict:
    """
    Assemble Qdrant payload metadata for a given chunk.
    Args:
        paper: Paper object containing metadata
        chunk_index: Index of the current chunk (0-based)
        chunk_text: Text content of the current chunk
        page_start: Starting page number of the chunk
        page_end: Ending page number of the chunk
    Returns:
        Dictionary payload with paper metadata and chunk info
    """
    return {
        "chunk_category": "text",
        "paper_id": paper.id,
        "title": paper.title,
        "authors": paper.authors,
        "year": paper.year,
        "publication": paper.publication,
        "chunk_index": chunk_index,
        "page_start": page_start,
        "page_end": page_end,
        "page_label": (
            f"p. {page_start}"
            if page_start == page_end
            else f"pp. {page_start}–{page_end}"
        ),
        "text": chunk_text,
    }

# How many characters to overlap between adjacent pages to preserve
# cross-page sentence continuity. Mirrors your chunk overlap strategy.
PAGE_STITCH_OVERLAP = CHUNK_OVERLAP

def chunk_paper(paper: Paper) -> list[Chunk]:
    """
    Splits a paper into overlapping chunks with page-number metadata.

    Strategy:
      1. Parse the PDF into per-page text blocks.
      2. Stitch adjacent pages with a character overlap so sentences that
         span a page break are not severed.
      3. Split the stitched text with RecursiveCharacterTextSplitter.
      4. Map each resulting chunk back to the source page(s) it came from
         by scanning a character-offset index built from the stitched text.

    Args:
        paper: Paper object containing source_file path and metadata.

    Returns:
        List of Chunk objects; each payload includes page_start / page_end.
    """
    pages = parse_pdf_for_pages(DATA_DIR / paper.source_file)
    if not pages:
        logger.warning(f"Paper {paper.id} yielded no page text. Skipping.")
        return []

    # journal_page = pdf_page + page_offset
    # Anchors the first non-empty PDF page to first_page_number, handling blank leading pages.
    page_offset = (
        paper.first_page_number - pages[0]["page"]
        if paper.first_page_number is not None
        else 0
    )

    # ------------------------------------------------------------------
    # 1. Build a single stitched string and record the char offset at
    #    which each page begins.  The overlap is taken from the *tail* of
    #    the previous page so the splitter can see both sides of a break.
    # ------------------------------------------------------------------
    stitched_parts: list[str] = []
    # page_offsets[i] = (char_start, char_end, page_number) in stitched text
    page_offsets: list[tuple[int, int, int]] = []
    cursor = 0

    for i, page in enumerate(pages):
        page_text = preprocess_markdown(page["text"])

        if i > 0 and PAGE_STITCH_OVERLAP > 0:
            # Prepend the tail of the previous page's contribution so the
            # splitter can bridge the boundary.  We do NOT advance cursor
            # for this overlap region — it is intentionally double-counted
            # so the page attribution below stays correct.
            prev_tail = stitched_parts[-1][-PAGE_STITCH_OVERLAP:]
            page_text = prev_tail + page_text

        start = cursor
        end = cursor + len(page_text)
        page_offsets.append((start, end, page["page"]))
        stitched_parts.append(page_text)
        cursor = end

    stitched_text = "".join(stitched_parts)

    # ------------------------------------------------------------------
    # 2. Split
    # ------------------------------------------------------------------
    text_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )

    try:
        docs = text_splitter.create_documents([stitched_text])
    except Exception as e:
        logger.error(f"Error splitting text for paper {paper.id}: {e}")
        return []

    # ------------------------------------------------------------------
    # 3. Map each chunk to page(s)
    # ------------------------------------------------------------------
    def page_for_offset(char_offset: int) -> int:
        """Return the 1-based page number that owns char_offset."""
        for start, end, page_num in page_offsets:
            if start <= char_offset < end:
                return page_num
        # Fallback: last page
        return page_offsets[-1][2]

    chunks: list[Chunk] = []
    for i, doc in enumerate(docs):
        chunk_text = doc.page_content.strip()
        if not chunk_text:
            continue

        pos = doc.metadata.get("start_index", 0)
        chunk_start_page = page_for_offset(pos)
        chunk_end_page = page_for_offset(pos + len(chunk_text) - 1)

        chunks.append(
            Chunk(
                id=str(uuid.uuid4()),
                text=_build_text_embed_text(paper, chunk_text),
                payload=_build_payload(
                    paper, i, chunk_text,
                    chunk_start_page + page_offset,
                    chunk_end_page + page_offset,
                ),
            )
        )

    logger.info(f"Paper {paper.id} split into {len(chunks)} chunks.")
    return chunks


def chunk_paper_text_only(paper: Paper) -> list[Chunk]:
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
                text=_build_text_embed_text(paper, chunk_text),
                payload=_build_payload(paper, i, chunk_text, 1, 1),
            )
        )

    logger.info(f"Paper {paper.id} split into {len(chunks)} chunks.")
    return chunks



def _build_text_embed_text(paper: Paper, chunk_text: str) -> str:
    """Prepend paper metadata to chunk text so the embedding carries provenance context."""
    parts: list[str] = []
    if paper.title:
        parts.append(f"Title: {paper.title}")
    if paper.year:
        parts.append(f"Year: {paper.year}")
    if paper.publication:
        parts.append(f"Publication: {paper.publication}")
    parts.append(chunk_text)
    return "\n".join(parts)


def _build_image_embed_text(record: ImageRecord) -> str:
    """
    Compose the text input sent to Voyage alongside the image.
    Caption anchors semantics; paper metadata improves retrieval precision.
    """
    parts: list[str] = []
    if record.caption:
        parts.append(record.caption)
    if record.title:
        parts.append(f"Title: {record.title}")
    if record.year:
        parts.append(f"Year: {record.year}")
    if record.publication:
        parts.append(f"Publication: {record.publication}")
    return "\n".join(parts)


def _build_image_payload(record: ImageRecord, chunk_index: int) -> dict:
    return {
        "chunk_category": "image",
        "paper_id": record.paper_id,
        "title": record.title,
        "authors": record.authors,
        "year": record.year,
        "publication": record.publication,
        "chunk_index": chunk_index,
        "page": record.page,
        "page_label": f"p. {record.page}",
        "source_file": record.source_file,
        "bbox": list(record.bbox),
        "caption": record.caption,
        "image_type": record.image_type,
        "text": _build_image_embed_text(record),
    }


def chunk_paper_images(
    paper: Paper,
    image_json_dir: Path = IMAGE_JSON_DIR,
) -> list[Chunk]:
    """
    Convert all extracted ImageRecords for a paper into Chunk objects.

    Chunk.text is the text input Voyage will receive alongside the image
    (caption + title + year + publication).  Chunk.id matches ImageRecord.id
    so vectordb can locate the image file without an extra lookup.
    """
    records = load_image_records_for_paper(paper.id, image_json_dir)
    if not records:
        logger.info(f"No image records found for paper {paper.id}.")
        return []

    chunks: list[Chunk] = []
    for i, record in enumerate(records):
        chunks.append(
            Chunk(
                id=record.id,
                text=_build_image_embed_text(record),
                payload=_build_image_payload(record, i),
            )
        )

    logger.info(f"Paper {paper.id} produced {len(chunks)} image chunks.")
    return chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage: load papers from JSON, chunk them, and print chunk info
    paper = load_paper(JSON_DIR / "DaZlHqgIxjgJ.json")
    if not paper:
        logger.error("Failed to load paper for chunking.")
    else:
        try:
            chunks = chunk_paper(paper)
            logger.info(f"Paper '{paper.title}' produced {len(chunks)} chunks.")
        except Exception as e:
            logger.error(f"Failed to chunk paper '{paper.title}': {e}")
        

