import re
import pymupdf4llm
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data/papers"
JSON_DIR = PROJECT_ROOT / "data/json"

def _parse_pdf_for_text(pdf_path: Path) -> str:
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
            logger.warning(f"No text extracted from {pdf_path}")
            return ""
        return text
    except Exception as e:
        logger.error(f"Error parsing {pdf_path}: {e}")
        return ""

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

    return text.strip()

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