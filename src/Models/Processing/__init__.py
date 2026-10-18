"""
src/Models/Processing/__init__.py

Defines the `Models.Processing` package for use in the project.
"""

__author__  = "n-rosenthal"
__version__ = "1.0.1"
__date__    = "2026-10-18"

# Status of a processing action
from src.Models.Processing.Status import ProcessingStatus

# Stage in the processing pipeline
from src.Models.Processing.Stage  import ProcessingStage

# State for a stage in the processing pipeline
from src.Models.Processing.StageState import StageState

# Attributes and temporary objects over pipeline stages
from src.Models.Processing.Info import (
    SourceInfo,
    DownloadInfo,

    TextArtifact,

    ProcessingInfo,
    
    EmbeddingInfo,
    GraphInfo,
    TopicAssignment,
)


__all__ = [
    "ProcessingStatus",
    "ProcessingStage",
    "StageState",
    
    "SourceInfo",
    "DownloadInfo",
    "TextArtifact",
    "ProcessingInfo",
    "EmbeddingInfo",
    "GraphInfo",
    "TopicAssignment",
]
