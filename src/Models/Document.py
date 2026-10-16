"""Parsed scientific document model."""

from dataclasses import asdict, dataclass, field

from src.Models.Reference import Reference
from src.Models.Section import DocumentSection


@dataclass
class ExtractedDocument:
    """Structured content extracted from a scientific PDF."""

    paper_id: str
    sections: list[DocumentSection] = field(default_factory=list)
    references: list[Reference] = field(default_factory=list)

    extraction_method: str = "grobid"
    tei_path: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)