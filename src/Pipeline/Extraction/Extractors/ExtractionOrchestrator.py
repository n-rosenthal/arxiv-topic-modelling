import logging
from pathlib import Path

from src.Pipeline.Extraction.Extractors.GROBIDExtractor import GROBIDExtractor
from src.Pipeline.Extraction.Extractors.PyMuPDFExtractor import PyMuPDFExtractor
from src.Models import ExtractedDocument


logger = logging.getLogger(__name__)


class ExtractionOrchestrator:
    """Try the primary extractor and fall back on failure."""

    def __init__(
        self,
        grobid_url: str = "http://localhost:8070",
        extracted_dir: str | Path = "data/extracted",
    ) -> None:
        self.grobid = GROBIDExtractor(
            base_url=grobid_url,
            output_dir=extracted_dir,
        )

        self.fallback = PyMuPDFExtractor(
            output_dir=extracted_dir,
        )

    def extract(
        self,
        pdf_path: str | Path,
        paper_id: str,
    ) -> ExtractedDocument:
        errors: list[str] = []

        try:
            document = self.grobid.extract(
                pdf_path=pdf_path,
                paper_id=paper_id,
            )

            if document.sections and any(
                section.raw_text.strip()
                for section in document.sections
            ):
                return document

            errors.append(
                "GROBID returned no non-empty sections."
            )

        except Exception as exc:
            errors.append(
                f"GROBID failed: {type(exc).__name__}: {exc}"
            )

            logger.exception(
                "GROBID extraction failed for paper %s.",
                paper_id,
            )

        try:
            document = self.fallback.extract(
                pdf_path=pdf_path,
                paper_id=paper_id,
            )

            document.warnings.extend(errors)

            return document

        except Exception as fallback_exc:
            logger.exception(
                "Fallback extraction failed for paper %s.",
                paper_id,
            )

            raise RuntimeError(
                f"All extraction methods failed for {paper_id}. "
                f"Primary errors: {errors}. "
                f"Fallback error: "
                f"{type(fallback_exc).__name__}: "
                f"{fallback_exc}"
            ) from fallback_exc

    def close(self) -> None:
        self.grobid.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
