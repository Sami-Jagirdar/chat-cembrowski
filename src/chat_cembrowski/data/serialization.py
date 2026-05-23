import json
import logging
from pathlib import Path
from typing import Iterator, Optional

from .models import Paper

logger = logging.getLogger(__name__)

def save_paper(paper: Paper, output_dir: Optional[str | Path] = None) -> Optional[Path]:
    """
    Save a single Paper object to a JSON file.

    Args:
        paper: Paper object to save
        output_dir: Directory to save the JSON file (default: data/json)

    Returns:
        Path to the saved JSON file or None if saving fails
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[3] / "data" / "json"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    filename = paper.id + ".json"
    filepath = output_dir / filename

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(paper.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Saved: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save {filepath}: {e}")
        return None

def save_papers_to_json(papers: list[Paper], output_dir: Optional[str | Path] = None) -> Path:
    """
    Save Paper objects to JSON files.

    Args:
        papers: List of Paper objects to save
        output_dir: Directory to save JSON files (default: data/json)

    Returns:
        Path to the output directory
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[3] / "data" / "json"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    for paper in papers:
        filename = paper.id + ".json"
        filepath = output_dir / filename

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(paper.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"Saved: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save {filepath}: {e}")

    logger.info(f"Saved {len(papers)} papers to {output_dir}")
    return output_dir

def load_paper(json_file: str | Path) -> Optional[Paper]:
    """
    Load a single Paper object from a JSON file.

    Args:
        json_file: Path to the JSON file
    Returns:
        Paper object or None if loading fails
    """
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        paper = Paper(
            source_file=data["source_file"],
            id=data["id"],
            title=data.get("title"),
            authors=data.get("authors"),
            year=data.get("year"),
            publication=data.get("publication"),
            processed=data.get("processed", False),
            text=data.get("text", ""),
        )
        logger.info(f"Loaded: {json_file.name}")
        return paper
    except Exception as e:
        logger.error(f"Failed to load {json_file}: {e}")
        return None


def load_papers_from_json(json_dir: Optional[str | Path] = None) -> Iterator[Paper]:
    """
    Load Paper objects from JSON files.

    Args:
        json_dir: Directory containing JSON files (default: data/json)

    Returns:
        List of Paper objects
    """
    if json_dir is None:
        json_dir = Path(__file__).resolve().parents[3] / "data" / "json"
    else:
        json_dir = Path(json_dir)

    if not json_dir.exists():
        logger.warning(f"JSON directory not found: {json_dir}")
        return []

    json_files = json_dir.glob("*.json")

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            paper = Paper(
                source_file=data["source_file"],
                text=data["text"],
                id=data["id"],
                title=data.get("title"),
                authors=data.get("authors"),
                year=data.get("year"),
                publication=data.get("publication"),
                processed=data.get("processed", False),
            )
            logger.info(f"Loaded: {json_file.name}")
            yield paper
        except Exception as e:
            logger.error(f"Failed to load {json_file}: {e}")
