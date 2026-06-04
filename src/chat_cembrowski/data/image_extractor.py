#type: ignore
"""
Extracts images from paper PDFs, finds captions via spatial search + regex,
and writes per-image metadata JSON files ready for multimodal embedding.
"""

import json
import logging
import re
import uuid
from pathlib import Path
from typing import Optional

import fitz

from .models import ImageRecord, Paper
from .serialization import load_papers_from_json

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PAPERS_DIR = PROJECT_ROOT / "data" / "papers"
IMAGES_DIR = PROJECT_ROOT / "data" / "images"
IMAGE_JSON_DIR = PROJECT_ROOT / "data" / "image_json"

MIN_IMAGE_DIM = 50       # minimum width or height in PDF points
CAPTION_PROXIMITY = 80.0  # max vertical distance (points) to search for caption

# Matches labels like: "Figure 1.", "Fig. 2A:", "Table 3.", "Chart 1:", "Image."
CAPTION_RE = re.compile(
    r"\b(fig(?:ure)?|table|chart|image|plate)\.?\s*(?:[A-Z]?\d+[a-zA-Z]?)?\s*[.:]",
    re.IGNORECASE,
)


def _block_text(block: dict) -> str:
    return " ".join(
        span["text"]
        for line in block.get("lines", [])
        for span in line.get("spans", [])
    ).strip()


def _find_caption(
    page: fitz.Page, img_rect: fitz.Rect
) -> tuple[Optional[str], Optional[str]]:
    """
    Search for a caption near img_rect using bounding-box proximity.
    Searches below the image first (figures), then above (tables).
    Returns (caption_text, image_type) or (None, None).
    """
    text_blocks = [b for b in page.get_text("dict")["blocks"] if b.get("type") == 0]

    def h_overlap(bbox) -> bool:
        return bbox[0] < img_rect.x1 and bbox[2] > img_rect.x0

    below = sorted(
        [
            b for b in text_blocks
            if b["bbox"][1] >= img_rect.y1 - 5
            and b["bbox"][1] <= img_rect.y1 + CAPTION_PROXIMITY
            and h_overlap(b["bbox"])
        ],
        key=lambda b: b["bbox"][1],
    )
    above = sorted(
        [
            b for b in text_blocks
            if b["bbox"][3] <= img_rect.y0 + 5
            and b["bbox"][3] >= img_rect.y0 - CAPTION_PROXIMITY
            and h_overlap(b["bbox"])
        ],
        key=lambda b: -b["bbox"][3],
    )

    for block in (below or above):
        text = _block_text(block)
        m = CAPTION_RE.search(text)
        if m:
            label = m.group(1).lower()
            image_type = "figure" if label.startswith("fig") else label
            return text, image_type

    return None, None


def extract_images_from_paper(
    paper: Paper,
    papers_dir: Path = PAPERS_DIR,
    images_dir: Path = IMAGES_DIR,
    image_json_dir: Path = IMAGE_JSON_DIR,
) -> list[ImageRecord]:
    """
    Extract all non-trivial images from a paper's PDF.
    Saves image files to images_dir and metadata JSONs to image_json_dir.
    Page numbers are converted to journal page numbers via paper.first_page_number.
    """
    pdf_path = papers_dir / paper.source_file
    if not pdf_path.exists():
        logger.warning(f"PDF not found: {pdf_path.name}")
        return []

    images_dir.mkdir(parents=True, exist_ok=True)
    image_json_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))

    # Mirror the chunker's offset logic: anchor first non-empty PDF page to first_page_number
    first_content_pdf_page = next(
        (i + 1 for i, p in enumerate(doc) if p.get_text().strip()),
        1,
    )
    page_offset = (
        paper.first_page_number - first_content_pdf_page
        if paper.first_page_number is not None
        else 0
    )

    records: list[ImageRecord] = []
    seen_xrefs: set[int] = set()

    for page_num, page in enumerate(doc):
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            rects = page.get_image_rects(xref)
            if not rects:
                continue
            img_rect = rects[0]

            if img_rect.width < MIN_IMAGE_DIM or img_rect.height < MIN_IMAGE_DIM:
                continue

            try:
                base_image = doc.extract_image(xref)
            except Exception as e:
                logger.warning(f"Could not extract image xref={xref} from {pdf_path.name}: {e}")
                continue

            image_bytes = base_image["image"]
            ext = base_image["ext"]
            image_id = str(uuid.uuid7())
            filename = f"{image_id}.{ext}"

            (images_dir / filename).write_bytes(image_bytes)

            caption, image_type = _find_caption(page, img_rect)
            journal_page = (page_num + 1) + page_offset

            record = ImageRecord(
                id=image_id,
                paper_id=paper.id,
                source_file=filename,
                page=journal_page,
                bbox=(img_rect.x0, img_rect.y0, img_rect.x1, img_rect.y1),
                caption=caption,
                image_type=image_type,
                title=paper.title,
                authors=paper.authors,
                year=paper.year,
                publication=paper.publication,
            )

            with open(image_json_dir / f"{image_id}.json", "w", encoding="utf-8") as f:
                json.dump(record.to_dict(), f, indent=2, ensure_ascii=False)

            records.append(record)
            logger.info(
                f"Extracted {filename} from {pdf_path.name} "
                f"(page {journal_page}, type={image_type!r}, caption={caption!r})"
            )

    logger.info(f"Extracted {len(records)} images from {paper.source_file}")
    return records


def extract_all_images(
    papers_dir: Path = PAPERS_DIR,
    json_dir: Optional[Path] = None,
    images_dir: Path = IMAGES_DIR,
    image_json_dir: Path = IMAGE_JSON_DIR,
) -> list[ImageRecord]:
    """
    Run image extraction for all papers in json_dir.
    Skips non-PDF source files.
    """
    if json_dir is None:
        json_dir = PROJECT_ROOT / "data" / "json"

    all_records: list[ImageRecord] = []
    for paper in load_papers_from_json(json_dir):
        if not paper.source_file.lower().endswith(".pdf"):
            logger.info(f"Skipping non-PDF: {paper.source_file}")
            continue
        all_records.extend(
            extract_images_from_paper(paper, papers_dir, images_dir, image_json_dir)
        )

    logger.info(f"Total images extracted: {len(all_records)}")
    return all_records


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    records = extract_all_images()
    print(f"Extracted {len(records)} images across all papers.")
