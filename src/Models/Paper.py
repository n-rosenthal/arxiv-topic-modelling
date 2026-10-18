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

from src.Models.Processing import ProcessingStatus

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

class ProcessingStage(str, Enum):
    """Stages in the paper-processing pipeline."""

    METADATA = "metadata"
    PDF_DOWNLOAD = "pdf_download"
    PDF_VALIDATION = "pdf_validation"
    TEXT_EXTRACTION = "text_extraction"
    PREPROCESSING = "preprocessing"
    EMBEDDING = "embedding"
    GRAPH_CONSTRUCTION = "graph_construction"
    TOPIC_MODELING = "topic_modeling"


@dataclass
class StageState:
    """
    Track the execution state of one processing stage.

    Attributes
    ----------
    status:
        Current stage status.
    started_at:
        Timestamp of the most recent execution attempt.
    completed_at:
        Timestamp of the most recent successful completion.
    updated_at:
        Timestamp of the most recent state transition.
    attempts:
        Number of execution attempts.
    error:
        Error message from the most recent failure.
    """

    status: ProcessingStatus = ProcessingStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    updated_at: datetime = field(default_factory=utc_now)
    attempts: int = 0
    error: Optional[str] = None

    def start(self) -> None:
        """Mark the stage as running."""
        self.status = ProcessingStatus.RUNNING
        self.started_at = utc_now()
        self.updated_at = self.started_at
        self.attempts += 1
        self.error = None

    def complete(self) -> None:
        """Mark the stage as successfully completed."""
        now = utc_now()
        self.status = ProcessingStatus.COMPLETED
        self.completed_at = now
        self.updated_at = now
        self.error = None

    def fail(self, error: str) -> None:
        """Mark the stage as failed."""
        now = utc_now()
        self.status = ProcessingStatus.FAILED
        self.updated_at = now
        self.error = error

    def skip(self, reason: Optional[str] = None) -> None:
        """Mark the stage as intentionally skipped."""
        now = utc_now()
        self.status = ProcessingStatus.SKIPPED
        self.updated_at = now
        self.error = reason

    def reset(self) -> None:
        """Reset the stage so it can be executed again."""
        now = utc_now()
        self.status = ProcessingStatus.PENDING
        self.started_at = None
        self.completed_at = None
        self.updated_at = now
        self.error = None

    @property
    def is_complete(self) -> bool:
        """Whether the stage completed successfully."""
        return self.status == ProcessingStatus.COMPLETED


@dataclass
class SourceInfo:
    """
    Record the provenance of a paper's metadata.

    The source path is a reference to the original dataset, not
    necessarily a path on the local filesystem.
    """

    dataset_id: str = "nick007x/arxiv-papers"
    dataset_split: Optional[str] = None
    source_file_path: Optional[str] = None
    source_record_index: Optional[int] = None
    source_revision: Optional[str] = None
    imported_at: datetime = field(default_factory=utc_now)
    source_record_hash: Optional[str] = None


@dataclass
class DownloadInfo:
    """Record information about a downloaded PDF."""

    local_path: Optional[str] = None
    file_size_bytes: Optional[int] = None
    sha256: Optional[str] = None
    downloaded_at: Optional[datetime] = None
    content_type: Optional[str] = None
    source_url: Optional[str] = None
    verified: bool = False


@dataclass
class TextArtifact:
    """Metadata about text extracted from or processed from a paper."""

    local_path: str | None = None
    sha256: str | None = None

    character_count: int = 0
    token_count: int | None = None
    page_count: int | None = None

    language: str | None = None
    extraction_method: str | None = None

    created_at: datetime | None = None


@dataclass
class EmbeddingInfo:
    """Describe a generated document embedding."""

    model_name: Optional[str] = None
    model_revision: Optional[str] = None
    dimensions: Optional[int] = None
    artifact_path: Optional[str] = None
    sha256: Optional[str] = None
    created_at: Optional[datetime] = None


@dataclass
class GraphInfo:
    """Describe a paper's representation in a research graph."""

    graph_id: Optional[str] = None
    node_id: Optional[str] = None
    graph_version: Optional[str] = None
    edge_count: Optional[int] = None
    created_at: Optional[datetime] = None


@dataclass
class TopicAssignment:
    """Record a paper's assignment in a particular topic-model run."""

    experiment_id: str
    model_name: str
    topic_id: Optional[str] = None
    confidence: Optional[float] = None
    assigned_at: datetime = field(default_factory=utc_now)


@dataclass
class ProcessingInfo:
    """Track processing state and generated artifacts."""

    stages: dict[ProcessingStage, StageState] = field(
        default_factory=lambda: {
            stage: StageState()
            for stage in ProcessingStage
        }
    )

    download: DownloadInfo = field(default_factory=DownloadInfo)
    extracted_text: TextArtifact = field(default_factory=TextArtifact)
    processed_text: TextArtifact = field(default_factory=TextArtifact)
    embedding: EmbeddingInfo = field(default_factory=EmbeddingInfo)
    graph: GraphInfo = field(default_factory=GraphInfo)

    def state(self, stage: ProcessingStage) -> StageState:
        """Return the state associated with a stage."""
        return self.stages[stage]

    def is_complete(self, stage: ProcessingStage) -> bool:
        """Check whether a stage completed successfully."""
        return self.state(stage).is_complete

    @property
    def downloaded(self) -> bool:
        """Whether PDF downloading completed successfully."""
        return self.is_complete(ProcessingStage.PDF_DOWNLOAD)

    @property
    def processed(self) -> bool:
        """
        Whether text preprocessing completed successfully.

        This does not imply that embedding, graph construction,
        or topic modelling has completed.
        """
        return self.is_complete(ProcessingStage.PREPROCESSING)


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

    @property
    def pdf_downloaded(self) -> bool:
        """Whether the PDF download stage completed successfully."""
        return self.processing.downloaded

    @property
    def text_processed(self) -> bool:
        """Whether text preprocessing completed successfully."""
        return self.processing.processed

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
