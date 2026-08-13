"""
This module provides functions to parse PDF files, extract text as markdown, clean it for RAG chunking, and store it in corresponding JSON files. It uses the pymupdf4llm library for PDF parsing and includes error handling and logging for robustness.
"""

import re
from typing import Any, Dict, List
import pymupdf4llm
import fitz
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data/papers"
JSON_DIR = PROJECT_ROOT / "data/json"

DEFAULT_PAGE_RENDER_ZOOM = 2.0  # ~144 DPI (72 * zoom); higher = sharper text, more tokens

# Voyage multimodal embeddings reject images over 16 million pixels. Some
# source PDFs are physically large (e.g. conference posters), so a fixed
# zoom can blow past that on a per-page basis. We clamp per-page instead of
# lowering the global default, so normal-sized pages still render at the
# full requested zoom.
MAX_PAGE_IMAGE_PIXELS = 16_000_000
PAGE_IMAGE_PIXEL_SAFETY_MARGIN = 0.95  # stay safely under the cap after integer pixel rounding

def _to_markdown_forcing_invisible_text(pdf_path: Path, **kwargs):
    """pymupdf4llm.to_markdown(), but temporarily forced onto the classic
    (non-layout) engine with ignore_alpha=True so invisible text is accepted
    unconditionally.

    Used as a fallback for PDFs where a page mixes an invisible OCR text
    layer (common for ABBYY-scanned sources: render mode 3, "ignore-text")
    with a small amount of normal visible text, e.g. running headers. Both
    pymupdf4llm's default layout engine and the classic engine's
    page_is_ocr() heuristic only accept invisible text when a page is
    *purely* invisible text, so on a mixed page the OCR content — the
    actual body of the document — is silently dropped, even though
    has_text_layer() sees the layer and skips the scanned-PDF OCR fallback.
    Forcing the classic engine with ignore_alpha=True bypasses that
    heuristic and recovers the text.
    """
    pymupdf4llm.use_layout(False)
    try:
        return pymupdf4llm.to_markdown(str(pdf_path), ignore_alpha=True, **kwargs)
    finally:
        pymupdf4llm.use_layout(True)


def _parse_pdf_for_text(pdf_path: Path) -> str | List[Dict[str, Any]]:
    """
     Parse a single PDF file and extract text as markdown.

    Args:
        pdf_path: Path to PDF file

    Returns:
        pdf text or empty string if parsing fails
    """
    try:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            return ""

        text = pymupdf4llm.to_markdown(str(pdf_path))
        if not text:
            logger.warning(
                f"No text extracted from {pdf_path} via the default pymupdf4llm "
                f"path; retrying with invisible text forced on."
            )
            text = _to_markdown_forcing_invisible_text(pdf_path)
        if not text:
            logger.warning(f"No text extracted from {pdf_path}")
            return ""
        return text
    except Exception as e:
        logger.error(f"Error parsing {pdf_path}: {e}")
        return ""


def _page_chunks_to_pages(page_chunks) -> list[dict]:
    pages = []
    for page_data in page_chunks:
        if not isinstance(page_data, dict):
            continue
        text = page_data.get("text", "")
        page_number = page_data.get("metadata", {}).get("page_number", 0)
        if text.strip():
            pages.append({"text": text, "page": page_number})
    return pages


def parse_pdf_for_pages(pdf_path: Path) -> list[dict]:
    """
    Parse a single PDF and extract text per page using pymupdf4llm.

    Args:
        pdf_path: Path to PDF file

    Returns:
        List of page dicts with keys: 'text' (str), 'page' (1-based int).
        Returns empty list if parsing fails.
    """
    try:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            return []

        page_chunks = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
        pages = _page_chunks_to_pages(page_chunks) if page_chunks else []

        if not pages:
            logger.warning(
                f"No text extracted from {pdf_path} via the default pymupdf4llm "
                f"path; retrying with invisible text forced on (handles mixed "
                f"invisible/visible OCR text layers)."
            )
            page_chunks = _to_markdown_forcing_invisible_text(pdf_path, page_chunks=True)
            pages = _page_chunks_to_pages(page_chunks) if page_chunks else []

        if not pages:
            logger.warning(f"No text extracted from {pdf_path}")

        return pages

    except Exception as e:
        logger.error(f"Error parsing {pdf_path}: {e}")
        return []


def render_pdf_pages_as_images(
    pdf_path: Path,
    out_dir: Path,
    paper_id: str,
    zoom: float = DEFAULT_PAGE_RENDER_ZOOM,
) -> list[dict]:
    """
    Render every page of a PDF to a standalone PNG image.

    Used by the page-based multimodal chunking strategy, where each page's
    full visual layout (text, figures, tables) is embedded as a single image
    alongside the page's markdown text.

    Args:
        pdf_path: Path to the source PDF.
        out_dir: Directory to write rendered PNGs into (created if missing).
        paper_id: Paper ID, used to build a unique, stable filename per page.
        zoom: Render scale factor applied to both axes (72 DPI * zoom).
              2.0 (~144 DPI) balances text legibility against embedding cost.

    Returns:
        List of dicts with keys: 'page' (1-based int), 'image_file' (str
        filename, relative to out_dir). Empty list if rendering fails.
    """
    try:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            return []

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        doc = fitz.open(str(pdf_path))
        max_pixels = MAX_PAGE_IMAGE_PIXELS * PAGE_IMAGE_PIXEL_SAFETY_MARGIN

        pages: list[dict] = []
        for page in doc:
            page_number = page.number + 1  # 1-based
            filename = f"{paper_id}_p{page_number}.png"

            # Clamp zoom per-page so oversized physical pages (e.g. posters)
            # still fit under Voyage's pixel budget.
            rect = page.rect
            page_area = rect.width * rect.height
            effective_zoom = zoom
            if page_area > 0:
                requested_pixels = page_area * (zoom ** 2)
                if requested_pixels > max_pixels:
                    effective_zoom = (max_pixels / page_area) ** 0.5
                    logger.warning(
                        f"{pdf_path.name} page {page_number} is oversized for "
                        f"zoom={zoom} ({requested_pixels:,.0f} px); "
                        f"clamping to zoom={effective_zoom:.3f}."
                    )

            pix = page.get_pixmap(matrix=fitz.Matrix(effective_zoom, effective_zoom))
            pix.save(str(out_dir / filename))
            pages.append({"page": page_number, "image_file": filename})

        logger.info(f"Rendered {len(pages)} page images for {pdf_path.name}")
        return pages

    except Exception as e:
        logger.error(f"Error rendering pages for {pdf_path}: {e}")
        return []


def preprocess_markdown(text: str) -> str:
    """
    Clean markdown extracted from PDFs for RAG chunking.

    Args:   
        text: Raw markdown text extracted from PDF
    
    Returns:
        Cleaned markdown text
    """

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove standalone page numbers 
    text = re.sub(
        r"(?m)^\s*(?:page\s+)?[-–—]?\s*\d+\s*[-–—]?\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Fix hyphenated line breaks from PDFs (e.g. "hyphen-\nated" -> "hyphenated")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Merge accidental single newlines inside paragraphs
    text = re.sub(
        r"(?<!\n)\n(?!\n|#|\* |- |\d+\.|```|>)",
        " ",
        text,
    )

    # Remove repeated spaces/tabs
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Trim trailing whitespace
    text = re.sub(r"[ \t]+\n", "\n", text)

    return str(text.strip())

def store_text_from_pdf(json_path: Path, pdf_path: Path):
    """
    Extract text from a PDF file and store in the corresponding JSON file.

    Args:
        json_path: Path to JSON file
        pdf_path: Path to PDF file
    """
    text = _parse_pdf_for_text(pdf_path)
    if not text:
        logger.warning(f"No text extracted from {pdf_path}, skipping JSON update.")
        return

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # storing raw text since preprocessing step might be modified later
        data["text"] = text

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        logger.info(f"Updated JSON with text from {pdf_path.name}")
    except Exception as e:
        logger.error(f"Failed to update JSON for {pdf_path}: {e}")


def populate_all_json_with_text(json_dir: Path = JSON_DIR, data_dir: Path = DATA_DIR):
    """
    Populate all JSON files with text extracted from their corresponding PDFs.

    Args:
        json_dir: Directory containing JSON files
        data_dir: Directory containing PDF files
    """
    if not json_dir.exists():
        logger.error(f"JSON directory not found: {json_dir}")
        return
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return

    json_files = json_dir.glob("*.json")
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            source_file = data.get("source_file")
            if not source_file:
                logger.warning(f"No source_file in {json_file}, skipping.")
                continue
            
            pdf_path = data_dir / source_file
            if not pdf_path.exists():
                logger.warning(f"PDF file {pdf_path} not found for {json_file}, skipping.")
                continue
            
            store_text_from_pdf(json_file, pdf_path)
        except Exception as e:
            logger.error(f"Failed to process {json_file}: {e}")
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    populate_all_json_with_text()