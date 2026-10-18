"""
src/Models/Processing/StageState.py
State for the current stage of the pipeline being runned.
"""
from __future__ import annotations

__author__  = "n-rosenthal"
__version__ = "1.0.1"
__date__    = "2026-10-18"

from datetime import datetime, timezone
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from src.Models.Processing import ProcessingStage, ProcessingStatus

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
