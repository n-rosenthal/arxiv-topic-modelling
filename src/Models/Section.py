"""Scientific document section model."""

from dataclasses import asdict, dataclass, field

from src.Models.Citation import Citation


@dataclass
class DocumentSection:
    """A section or subsection extracted from a scientific paper."""

    section_id: str
    paper_id: str
    title: str
    level: int
    order: int
    raw_text: str

    processed_text: str = ""
    tokens: list[str] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)

    parent_section_id: str | None = None
    extraction_method: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        result = asdict(self)
        return result