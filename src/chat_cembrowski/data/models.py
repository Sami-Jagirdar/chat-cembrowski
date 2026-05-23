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
