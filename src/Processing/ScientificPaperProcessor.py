"""End-to-end section extraction and preprocessing."""

from __future__ import annotations

import json
import logging
import re

from pathlib import Path

from src.Extractors.GROBIDExtractor import GROBIDExtractor
from src.Models.Document import ExtractedDocument
from src.Processing.NLPPreprocessor import NLPPreprocessor
from src.Processing.ScientificContentCleaner import (
    ScientificContentCleaner,
)

logger = logging.getLogger(__name__)


class ScientificPaperProcessor:
    """Extract sections and prepare them for topic modelling."""

    def __init__(
        self,
        grobid_url: str = "http://localhost:8070",
        extracted_dir: str | Path = "data/extracted",
        sections_dir: str | Path = "data/sections",
        spacy_model: str | None = None,
    ) -> None:
        self.extracted_dir = Path(extracted_dir)
        self.sections_dir = Path(sections_dir)

        self.extracted_dir.mkdir(parents=True, exist_ok=True)
        self.sections_dir.mkdir(parents=True, exist_ok=True)

        self.extractor = GROBIDExtractor(
            base_url=grobid_url,
            output_dir=self.extracted_dir,
        )

        self.cleaner = ScientificContentCleaner()
        self.preprocessor = NLPPreprocessor(
            spacy_model=spacy_model,
        )

    @staticmethod
    def _safe_identifier(paper_id: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]", "_", paper_id)

    def process(
        self,
        pdf_path: str | Path,
        paper_id: str,
    ) -> ExtractedDocument:
        """Extract, clean, tokenize, and persist one scientific paper."""
        pdf_path = Path(pdf_path)

        document = self.extractor.extract(
            pdf_path=pdf_path,
            paper_id=paper_id,
        )

        for section in document.sections:
            section.processed_text = self.cleaner.clean(section)
            section.tokens = self.preprocessor.preprocess(
                section.processed_text
            )

            if not section.processed_text:
                section.warnings.append(
                    "Section became empty after preprocessing."
                )

            if not section.tokens:
                section.warnings.append(
                    "No tokens remain after NLP preprocessing."
                )

        output_path = (
            self.sections_dir
            / f"{self._safe_identifier(paper_id)}.json"
        )

        temporary_path = output_path.with_suffix(".json.tmp")

        temporary_path.write_text(
            json.dumps(
                document.to_dict(),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        temporary_path.replace(output_path)

        logger.info(
            "Processed %s: %d sections, %d references; saved to %s",
            paper_id,
            len(document.sections),
            len(document.references),
            output_path,
        )

        return document

    def close(self) -> None:
        self.extractor.close()

    def __enter__(self) -> "ScientificPaperProcessor":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()