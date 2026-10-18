"""
src/Models/Processing/Status.py

Defines the `ProcessingStatus` enum class for registering individual statuses in the scientific paper procesisng pipeline.
"""
from __future__ import annotations

__author__  = "n-rosenthal"
__version__ = "1.0.1"
__date__    = "2026-10-18"

from enum import Enum

class ProcessingStatus(str, Enum):
    """Status of an individual processing stage."""

    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"
    SKIPPED   = "skipped"
