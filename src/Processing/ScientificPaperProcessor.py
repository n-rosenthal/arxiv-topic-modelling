"""End-to-end section extraction and preprocessing.

The work is split in three steps that can be used separately:

    extract(pdf_path, paper_id)      PDF -> GROBID -> TEI on disk -> sections
    preprocess(document)             clean each section and tokenize it
    save(document)                   write the sections JSON

``process()`` runs all three. The TEI stored by ``extract`` is the durable
raw artifact: ``extract_from_tei()`` rebuilds the very same document from
it, so changing the cleaner or the NLP settings only needs
``extract_from_tei() -> preprocess() -> save()``, with no PDF and no GROBID.
"""

from __future__ import annotations

import json
import logging
import re

from pathlib import Path

from src.Pipeline.Extraction.Extractors.GROBIDExtractor import GROBIDExtractor
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
        cleaner: ScientificContentCleaner | None = None,
        preprocessor: NLPPreprocessor | None = None,
    ) -> None:
        self.extracted_dir = Path(extracted_dir)
        self.sections_dir = Path(sections_dir)

        self.extracted_dir.mkdir(parents=True, exist_ok=True)
        self.sections_dir.mkdir(parents=True, exist_ok=True)

        self.extractor = GROBIDExtractor(
            base_url=grobid_url,
            output_dir=self.extracted_dir,
        )

        # Both can be injected, e.g. ScientificContentCleaner(remove_math=False).
        self.cleaner = (
            cleaner if cleaner is not None else ScientificContentCleaner()
        )
        self.preprocessor = (
            preprocessor
            if preprocessor is not None
            else NLPPreprocessor(spacy_model=spacy_model)
        )

    @staticmethod
    def _safe_identifier(paper_id: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]", "_", paper_id)

    def sections_path(self, paper_id: str) -> Path:
        """Caminho do JSON de seções de um paper (existe depois de `save`)."""
        return self.sections_dir / f"{self._safe_identifier(paper_id)}.json"

    def tei_path(self, paper_id: str) -> Path:
        """Caminho do TEI XML de um paper (existe depois de `extract`)."""
        return self.extractor.tei_path(paper_id)

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def extract(
        self,
        pdf_path: str | Path,
        paper_id: str,
    ) -> ExtractedDocument:
        """PDF -> GROBID -> TEI stored in ``extracted_dir`` -> sections."""
        return self.extractor.extract(
            pdf_path=Path(pdf_path),
            paper_id=paper_id,
        )

    def extract_from_tei(
        self,
        paper_id: str,
        tei_path: str | Path | None = None,
    ) -> ExtractedDocument:
        """Rebuild the document from a stored TEI. No PDF, no GROBID."""
        return self.extractor.extract_from_tei(paper_id, tei_path)

    def preprocess(
        self,
        document: ExtractedDocument,
    ) -> ExtractedDocument:
        """Clean and tokenize every section, in place. Performs no I/O."""
        for section in document.sections:
            section.processed_text = self.cleaner.clean(section)
            section.tokens = self.preprocessor.preprocess(
                section.processed_text
            )

            if not section.processed_text:
                self._warn(
                    section,
                    "Section became empty after preprocessing.",
                )

            if not section.tokens:
                self._warn(
                    section,
                    "No tokens remain after NLP preprocessing.",
                )

        return document

    @staticmethod
    def _warn(section, message: str) -> None:
        # Preprocessing may run more than once on the same object.
        if message not in section.warnings:
            section.warnings.append(message)

    def save(self, document: ExtractedDocument) -> Path:
        """Write the sections JSON atomically and return its path."""
        output_path = self.sections_path(document.paper_id)

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
            "Saved %s: %d sections, %d references; %s",
            document.paper_id,
            len(document.sections),
            len(document.references),
            output_path,
        )

        return output_path

    def process(
        self,
        pdf_path: str | Path,
        paper_id: str,
    ) -> ExtractedDocument:
        """Extract, clean, tokenize, and persist one scientific paper."""
        document = self.extract(pdf_path, paper_id)
        self.preprocess(document)
        self.save(document)

        return document

    def close(self) -> None:
        self.extractor.close()

    def __enter__(self) -> "ScientificPaperProcessor":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
