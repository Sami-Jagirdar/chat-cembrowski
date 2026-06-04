from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Paper:
    """Represents a research paper."""
    id: str
    source_file: str
    title: Optional[str] = None
    authors: Optional[list[str]] = None
    year: Optional[int] = None
    publication: Optional[str] = None
    first_page_number: Optional[int] = None
    processed: bool = False
    text: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source_file": self.source_file,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "publication": self.publication,
            "first_page_number": self.first_page_number,
            "processed": self.processed,
            "text": self.text,
        }
    
@dataclass
class Chunk:
    """Represents a chunk of text from a paper, along with metadata for RAG."""
    id: str
    text: str
    payload: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "payload": self.payload,
    }

@dataclass
class ImageRecord:
    """Represents an image extracted from a paper, with spatial and caption metadata."""
    id: str
    paper_id: str
    source_file: str                          # filename in data/images/
    page: int                                 # journal page number (offset applied)
    bbox: tuple[float, float, float, float]   # (x0, y0, x1, y1) in PDF points
    caption: Optional[str] = None
    image_type: Optional[str] = None          # "figure", "table", "chart", "image"
    title: Optional[str] = None
    authors: Optional[list[str]] = None
    year: Optional[int] = None
    publication: Optional[str] = None
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "paper_id": self.paper_id,
            "source_file": self.source_file,
            "page": self.page,
            "bbox": list(self.bbox),
            "caption": self.caption,
            "image_type": self.image_type,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "publication": self.publication,
        }
