"""In-text citation model."""

from dataclasses import asdict, dataclass, field


@dataclass
class Citation:
    """An occurrence of a citation within a document section."""

    marker: str
    target_reference_ids: list[str] = field(default_factory=list)

    # Offsets are relative to the section's raw_text.
    start: int | None = None
    end: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)