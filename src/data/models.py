from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Paper:
    """Represents a research paper."""
    id: str
    source_file: str
    text: str
    title: Optional[str] = None
    authors: Optional[list[str]] = None
    year: Optional[int] = None
    publication: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source_file": self.source_file,
            "text": self.text,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "publication": self.publication,
        }
