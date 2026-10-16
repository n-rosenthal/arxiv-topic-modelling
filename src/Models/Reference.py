"""Bibliographic reference model."""

from dataclasses import asdict, dataclass


@dataclass
class Reference:
    """A bibliographic reference extracted from a scientific paper."""

    reference_id: str
    raw_text: str

    title: str | None = None
    authors: list[str] | None = None
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None

    def __post_init__(self) -> None:
        if self.authors is None:
            self.authors = []

    def to_dict(self) -> dict:
        return asdict(self)