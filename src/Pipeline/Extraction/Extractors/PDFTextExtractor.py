"""
PDF text extraction for arXiv papers.

Extracts text from PDF files using PyMuPDF and records the resulting
artifact and processing state in the Paper data model.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile

from datetime import datetime, timezone
from pathlib import Path

import pymupdf

from src.Models import Paper, ProcessingStage


__author__  = "n-rosenthal"
__version__ = "1.0.1"
__date__    = "2026-10-18"


logger = logging.getLogger(__name__)


class PDFExtractionError(Exception):
    """Raised when a PDF cannot be validated or extracted."""


class PDFTextExtractor:
    """Extract and persist text from research-paper PDFs."""

    def __init__(
        self,
        output_dir: str | Path = "data/raw/extracted_text",
        *,
        overwrite: bool = False,
        min_characters: int = 50,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.overwrite = overwrite
        self.min_characters = min_characters

        if min_characters < 0:
            raise ValueError("min_characters must be non-negative")

    @staticmethod
    def _sha256(path: Path) -> str:
        """Calculate a file's SHA-256 checksum without loading it all at once."""
        digest = hashlib.sha256()

        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)

        return digest.hexdigest()

    @staticmethod
    def _safe_filename(identifier: str) -> str:
        """Convert an arXiv identifier into a safe filename."""
        filename = re.sub(r"[^A-Za-z0-9._-]+", "_", identifier)
        filename = filename.strip("._")

        if not filename:
            raise ValueError("Paper has an invalid arXiv identifier")

        return filename

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    def _validate_pdf(self, pdf_path: Path) -> None:
        """Check that the input exists, is a file, and is a readable PDF."""
        if not pdf_path.exists():
            raise PDFExtractionError(f"PDF does not exist: {pdf_path}")

        if not pdf_path.is_file():
            raise PDFExtractionError(f"PDF path is not a file: {pdf_path}")

        if pdf_path.stat().st_size == 0:
            raise PDFExtractionError(f"PDF is empty: {pdf_path}")

        try:
            with pymupdf.open(pdf_path) as document:
                if not document.is_pdf:
                    raise PDFExtractionError(
                        f"Input is not a PDF: {pdf_path}"
                    )

                if document.page_count == 0:
                    raise PDFExtractionError(
                        f"PDF contains no pages: {pdf_path}"
                    )

        except PDFExtractionError:
            raise
        except Exception as exc:
            raise PDFExtractionError(
                f"Could not open PDF {pdf_path}: {exc}"
            ) from exc

    def _extract_pages(self, pdf_path: Path) -> list[str]:
        """Extract text from each page, preserving page boundaries."""
        try:
            with pymupdf.open(pdf_path) as document:
                return [
                    page.get_text("text", sort=True)
                    for page in document
                ]
        except Exception as exc:
            raise PDFExtractionError(
                f"Text extraction failed for {pdf_path}: {exc}"
            ) from exc

    def _write_atomically(self, output_path: Path, text: str) -> None:
        """Write to a temporary file and atomically replace the destination."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=output_path.parent,
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                temporary_file.write(text)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())

            os.replace(temporary_path, output_path)

        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def extract(
        self,
        paper: Paper,
        pdf_path: str | Path | None = None,
    ) -> Path:
        """
        Extract a paper's text and update its processing metadata.

        Args:
            paper: Paper instance to update.
            pdf_path: Optional explicit PDF path. When omitted, the path
                recorded in paper.processing.download.local_path is used.

        Returns:
            Path to the extracted UTF-8 text file.

        Raises:
            PDFExtractionError: If validation or extraction fails.
            FileExistsError: If the output exists and overwrite is disabled.
        """
        if pdf_path is None:
            recorded_path = paper.processing.download.local_path

            if not recorded_path:
                raise PDFExtractionError(
                    f"No downloaded PDF path is recorded for "
                    f"{paper.arxiv_id}"
                )

            source_path = Path(recorded_path)
        else:
            source_path = Path(pdf_path)

        output_path = (
            self.output_dir
            / f"{self._safe_filename(paper.arxiv_id)}.txt"
        )

        if output_path.exists() and not self.overwrite:
            raise FileExistsError(
                f"Extraction already exists: {output_path}"
            )

        #  begins extraction process
        paper.stage = ProcessingStage.PDF_TO_TEXT
        
        try:
            self._validate_pdf(source_path)

            # finishes extraction process
            paper.stage = ProcessingStage.TEXT_AVAILABLE
            
            pages = self._extract_pages(source_path)

            # Form-feed separators preserve the boundaries between pages.
            extracted_text = "\n\f\n".join(pages).strip()

            if len(extracted_text) < self.min_characters:
                raise PDFExtractionError(
                    f"Extracted only {len(extracted_text)} characters "
                    f"from {paper.arxiv_id}; the PDF may be scanned or "
                    f"contain little extractable text."
                )

            self._write_atomically(output_path, extracted_text)

            artifact = paper.processing.extracted_text
            artifact.local_path = str(output_path)
            artifact.sha256 = self._sha256(output_path)
            artifact.character_count = len(extracted_text)
            artifact.page_count = len(pages)
            artifact.extraction_method = (
                f"PyMuPDF {pymupdf.VersionBind}; "
                "page.get_text('text', sort=True)"
            )
            artifact.created_at = self._utc_now()

            # Record the input PDF's checksum if it has not already
            # been recorded by the downloader.
            if not paper.processing.download.sha256:
                paper.processing.download.sha256 = self._sha256(
                    source_path
                )

            paper.update_timestamp()

            logger.info(
                "Extracted %s: %d pages, %d characters -> %s",
                paper.arxiv_id,
                len(pages),
                len(extracted_text),
                output_path,
            )

            return output_path

        except Exception as exc:
            # Preserve the exception for the caller, but record the failure
            # so the pipeline can report or retry it later.
            if isinstance(exc, FileExistsError):
                # An existing artifact is not necessarily an extraction
                # failure; it may simply be a request to avoid overwriting.
                raise

            # the extraction process failed
            paper.stage = ProcessingStage.INVALID_TEXT

            try:
                paper.update_timestamp()
            except Exception:
                logger.exception(
                    "Could not update timestamp for %s",
                    paper.arxiv_id,
                )

            if isinstance(exc, PDFExtractionError):
                raise

            raise PDFExtractionError(
                f"Could not extract {paper.arxiv_id}: {exc}"
            ) from exc
