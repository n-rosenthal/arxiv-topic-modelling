"""Data models for arxiv-topic-modelling."""

from src.Models.Processing import (
    DownloadInfo,
    EmbeddingInfo,
    GraphInfo,
    ProcessingInfo,
    ProcessingStage,
    ProcessingStatus,
    SourceInfo,
    StageState,
    TextArtifact,
    TopicAssignment,
)

from src.Models.Paper    import Paper
from src.Models.Document import ExtractedDocument


__all__ = [
    "DownloadInfo",
    "EmbeddingInfo",
    "GraphInfo",
    "ProcessingInfo",
    "ProcessingStage",
    "ProcessingStatus",
    "SourceInfo",
    "StageState",
    "TextArtifact",
    "TopicAssignment",

    "Paper",
    "ExtractedDocument",
]
