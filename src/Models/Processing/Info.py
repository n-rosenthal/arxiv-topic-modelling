""""
src/Models/Processing/Info.py
Definitions for experiments and the pipeline stage information and payloads/temporary objects.
"""

__author__  = "n-rosenthal"
__version__ = "1.0.1"
__date__    = "2026-10-18"

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from src.Models.Processing import ProcessingStatus, ProcessingStage, StageState

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

#    @property
#    def downloaded(self) -> bool:
#        """Whether PDF downloading completed successfully."""
#        return self.is_complete(ProcessingStage.PDF_DOWNLOAD)

#    @property
#    def processed(self) -> bool:
#        """
#        Whether text preprocessing completed successfully.
#
#        This does not imply that embedding, graph construction,
#        or topic modelling has completed.
#        """
#        return self.is_complete(ProcessingStage.PREPROCESSING)
