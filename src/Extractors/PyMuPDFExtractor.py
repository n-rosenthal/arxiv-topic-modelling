import re
from pathlib import Path

import pymupdf

from src.Extractors.BaseExtractor import BaseExtractor
from src.Models.Document import ExtractedDocument
from src.Models.Section import DocumentSection


class PyMuPDFExtractor(BaseExtractor):
    """Fallback extractor based on PyMuPDF.

    Extracts page text and heuristically identifies section headings.
    Does not claim to provide reliable bibliographic references or
    citation-to-reference links.
    """

    HEADING_PATTERNS = [
        re.compile(
            r"^(?:\d+(?:\.\d+)*\.?\s+)?"
            r"(?:introduction|background|related work|"
            r"methodology|methods|method|"
            r"experiments|experimental results|results|"
            r"discussion|conclusion|conclusions|"
            r"future work|acknowledgments|"
            r"acknowledgements|references|"
            r"appendix(?:\s+[A-Z0-9]+)?)$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\d+(?:\.\d+)*\.?\s+[A-Z][^\n]{1,100}$"
        ),
    ]

    def __init__(
        self,
        output_dir: str | Path = "data/extracted",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    @classmethod
    def _is_heading(cls, text: str) -> bool:
        text = " ".join(text.split())

        if not text or len(text) > 120:
            return False

        if text.endswith((".", ",", ";")):
            return False

        return any(
            pattern.fullmatch(text)
            for pattern in cls.HEADING_PATTERNS
        )

    @staticmethod
    def _clean_page_text(text: str) -> str:
        text = text.replace("\x00", "")
        text = text.replace("\u00ad", "")
        return text.strip()

    def extract(
        self,
        pdf_path: str | Path,
        paper_id: str,
    ) -> ExtractedDocument:
        pdf_path = Path(pdf_path)

        if not pdf_path.is_file():
            raise FileNotFoundError(
                f"PDF does not exist: {pdf_path}"
            )

        sections: list[DocumentSection] = []
        current_title = "Main text"
        current_parts: list[str] = []
        section_counter = 0

        def flush_section() -> None:
            nonlocal section_counter
            nonlocal current_parts

            raw_text = "\n".join(
                part for part in current_parts if part
            ).strip()

            if not raw_text:
                current_parts = []
                return

            section_counter += 1

            sections.append(
                DocumentSection(
                    section_id=(
                        f"{paper_id}-s{section_counter:04d}"
                    ),
                    paper_id=paper_id,
                    title=current_title,
                    level=1,
                    order=section_counter,
                    raw_text=raw_text,
                    citations=[],
                    parent_section_id=None,
                    extraction_method="pymupdf",
                    warnings=[
                        "Section boundaries were inferred "
                        "heuristically from PDF text."
                    ],
                )
            )

            current_parts = []

        with pymupdf.open(pdf_path) as document:
            if document.is_encrypted:
                raise ValueError(
                    f"Encrypted PDF cannot be processed: {pdf_path}"
                )

            for page_number, page in enumerate(document, start=1):
                page_text = self._clean_page_text(
                    page.get_text("text", sort=True)
                )

                if not page_text:
                    continue

                for line in page_text.splitlines():
                    line = line.strip()

                    if not line:
                        continue

                    if self._is_heading(line):
                        flush_section()
                        current_title = line
                    else:
                        current_parts.append(line)

        flush_section()

        if not sections:
            raise ValueError(
                f"No extractable text found in {pdf_path}. "
                "The PDF may be scanned or image-only."
            )

        return ExtractedDocument(
            paper_id=paper_id,
            sections=sections,
            references=[],
            extraction_method="pymupdf",
            warnings=[
                "Fallback extraction used PyMuPDF.",
                "Section headings were inferred heuristically.",
                "Bibliographic references and citation links "
                "were not extracted.",
                "Scanned PDFs may require OCR.",
            ],
        )