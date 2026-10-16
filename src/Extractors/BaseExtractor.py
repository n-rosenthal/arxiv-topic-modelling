from abc import ABC, abstractmethod
from pathlib import Path

from src.Models.Document import ExtractedDocument


class BaseExtractor(ABC):
    """Interface implemented by scientific document extractors."""

    @abstractmethod
    def extract(
        self,
        pdf_path: str | Path,
        paper_id: str,
    ) -> ExtractedDocument:
        """Extract a structured document from a PDF."""
        raise NotImplementedError