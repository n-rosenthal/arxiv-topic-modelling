"""
src/Models/Paper.py

Data models for arXiv papers and their processing lifecycle.

The models distinguish source metadata, processing state, and
generated research artifacts.

Python version: >= 3.10
"""
from __future__ import annotations

__author__ = "n-rosenthal"
__version__ = "1.0.0"
__date__ = "2026-10-15"


from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any, Optional
import json

from src.Models.Processing import (
    ProcessingStatus, 
    ProcessingStage,  
    StageState,      
    ProcessingInfo,   
    SourceInfo,       
    TopicAssignment   
)

def utc_now() -> datetime:
    """Return the current timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def normalize_datetime(value: Optional[datetime]) -> Optional[str]:
    """Serialize a datetime as an ISO 8601 string."""
    if value is None:
        return None

    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)

    return value.isoformat()


@dataclass
class Paper:
    """
    Represent an arXiv paper throughout the research pipeline.

    Source metadata is retained independently from processing state
    and generated artifacts.
    """

    # Identity and source metadata
    arxiv_id: str
    title: str
    authors: list[str]
    submission_date: Optional[str]
    abstract: str

    comments: Optional[str] = None
    primary_subject: Optional[str] = None
    subjects: Optional[str] = None
    doi: Optional[str] = None
    file_path: Optional[str] = None

    # Normalized and derived metadata
    submission_date_iso: Optional[str] = None
    subject_categories: list[str] = field(default_factory=list)

    # Provenance and lifecycle
    source: SourceInfo = field(default_factory=SourceInfo)
    processing: ProcessingInfo = field(default_factory=ProcessingInfo)

    # Topic assignments belong to experiments, not to the paper's
    # immutable source metadata.
    topic_assignments: list[TopicAssignment] = field(default_factory=list)

    # Record-level timestamps
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        """Validate essential metadata and normalize simple fields."""
        self.arxiv_id = self.arxiv_id.strip()
        self.title = self.title.strip()
        self.abstract = self.abstract.strip()
        self.authors = [
            author.strip()
            for author in self.authors
            if author.strip()
        ]

        if not self.arxiv_id:
            raise ValueError("arxiv_id cannot be empty.")

        if not self.title:
            raise ValueError("title cannot be empty.")

    @property
    def identifier(self) -> str:
        """Return the canonical paper identifier."""
        return self.arxiv_id

#    @property
#    def pdf_downloaded(self) -> bool:
#        """Whether the PDF download stage completed successfully."""
#        return self.processing.downloaded

#    @property
#    def text_processed(self) -> bool:
#        """Whether text preprocessing completed successfully."""
#        return self.processing.processed

    def update_timestamp(self) -> None:
        """Update the record's modification timestamp."""
        self.updated_at = utc_now()

    def calculate_source_hash(self) -> str:
        """
        Compute a stable hash of the source metadata.

        Useful for detecting changes to a paper's source record.
        """
        payload = {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "authors": self.authors,
            "submission_date": self.submission_date,
            "comments": self.comments,
            "primary_subject": self.primary_subject,
            "subjects": self.subjects,
            "doi": self.doi,
            "abstract": self.abstract,
            "file_path": self.file_path,
        }

        encoded = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")

        return sha256(encoded).hexdigest()

    @classmethod
    def from_source_record(
        cls,
        record: dict[str, Any],
        *,
        dataset_id: str = "nick007x/arxiv-papers",
        dataset_split: Optional[str] = None,
        source_record_index: Optional[int] = None,
        source_revision: Optional[str] = None,
    ) -> Paper:
        """
        Construct a Paper from a source dataset record.

        Extra source fields are ignored by this model. Preserve the
        complete original record separately if lossless archival
        is required.
        """
        arxiv_id = record.get("arxiv_id")
        title = record.get("title")
        authors = record.get("authors")
        abstract = record.get("abstract")

        if not arxiv_id:
            raise ValueError("Source record is missing arxiv_id.")

        if not title:
            raise ValueError(
                f"Source record {arxiv_id!r} is missing title."
            )

        if not isinstance(authors, list):
            raise ValueError(
                f"Source record {arxiv_id!r} must have authors "
                "as a list of strings."
            )

        if not isinstance(abstract, str):
            raise ValueError(
                f"Source record {arxiv_id!r} must have an abstract "
                "string."
            )

        paper = cls(
            arxiv_id=str(arxiv_id),
            title=str(title),
            authors=[str(author) for author in authors],
            submission_date=record.get("submission_date"),
            comments=record.get("comments"),
            primary_subject=record.get("primary_subject"),
            subjects=record.get("subjects"),
            doi=record.get("doi"),
            abstract=abstract,
            file_path=record.get("file_path"),
            source=SourceInfo(
                dataset_id=dataset_id,
                dataset_split=dataset_split,
                source_file_path=record.get("file_path"),
                source_record_index=source_record_index,
                source_revision=source_revision,
            ),
        )

        paper.source.source_record_hash = paper.calculate_source_hash()

        return paper

    def to_dict(self) -> dict[str, Any]:
        """Serialize the paper into JSON-compatible data."""
        def convert(value: Any) -> Any:
            if isinstance(value, Enum):
                return value.value

            if isinstance(value, datetime):
                return normalize_datetime(value)

            if isinstance(value, Path):
                return str(value)

            if isinstance(value, dict):
                return {
                    convert(key): convert(item)
                    for key, item in value.items()
                }

            if isinstance(value, list):
                return [convert(item) for item in value]

            if hasattr(value, "__dataclass_fields__"):
                return convert(asdict(value))

            return value

        return convert(asdict(self))

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        """Serialize the paper as JSON."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
        )
