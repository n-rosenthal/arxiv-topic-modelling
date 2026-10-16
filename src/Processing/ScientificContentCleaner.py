"""Clean extracted scientific prose for topic modelling."""

from __future__ import annotations

import re
import unicodedata

from src.Models.Section import DocumentSection


class ScientificContentCleaner:
    """Remove structural noise while retaining useful scientific prose."""

    # Numeric citation fallback for citations not marked up by GROBID.
    NUMERIC_CITATION_PATTERN = re.compile(
        r"\[\s*\d+(?:\s*[-,;]\s*\d+)*\s*\]"
    )

    # Inline and display TeX math.
    TEX_MATH_PATTERNS = [
        re.compile(r"\$\$.*?\$\$", re.DOTALL),
        re.compile(r"(?<!\$)\$(?!\$).*?(?<!\$)\$(?!\$)", re.DOTALL),
        re.compile(r"\\\[.*?\\\]", re.DOTALL),
        re.compile(r"\\\(.*?\\\)", re.DOTALL),
        re.compile(
            r"\\begin\{(?:equation\*?|align\*?|gather\*?|"
            r"multline\*?|displaymath)\}"
            r".*?"
            r"\\end\{(?:equation\*?|align\*?|gather\*?|"
            r"multline\*?|displaymath)\}",
            re.DOTALL,
        ),
    ]

    # Captions frequently survive in flattened PDF text.
    CAPTION_PATTERN = re.compile(
        r"^\s*(?:fig(?:ure)?|table)\s*\.?\s*\d+[A-Za-z]?"
        r"\s*[:.)-].*$",
        re.IGNORECASE,
    )

    PAGE_NUMBER_PATTERN = re.compile(
        r"^\s*(?:page\s+)?\d+\s*$",
        re.IGNORECASE,
    )

    def __init__(
        self,
        remove_numeric_citations: bool = True,
        remove_math: bool = True,
        remove_captions: bool = True,
        remove_page_numbers: bool = True,
    ) -> None:
        self.remove_numeric_citations = remove_numeric_citations
        self.remove_math = remove_math
        self.remove_captions = remove_captions
        self.remove_page_numbers = remove_page_numbers

    @staticmethod
    def _remove_spans(
        text: str,
        spans: list[tuple[int, int]],
    ) -> str:
        """Remove text spans in descending order to preserve offsets."""
        valid_spans = sorted(
            (
                (start, end)
                for start, end in spans
                if 0 <= start < end <= len(text)
            ),
            reverse=True,
        )

        last_start = len(text) + 1

        for start, end in valid_spans:
            # Ignore overlapping spans.
            if end > last_start:
                continue

            text = text[:start] + " " + text[end:]
            last_start = start

        return text

    @staticmethod
    def _normalize_unicode(text: str) -> str:
        # NFKC normalizes compatibility characters such as common ligatures.
        text = unicodedata.normalize("NFKC", text)

        # Normalize whitespace-like characters.
        text = text.replace("\u00a0", " ")
        text = text.replace("\u200b", "")
        text = text.replace("\xad", "")

        return text

    @staticmethod
    def _repair_line_hyphenation(text: str) -> str:
        # Join a word broken by a hyphen immediately before a line break.
        # The hyphen is removed only when it occurs at the line boundary.
        return re.sub(
            r"(?<=[^\W\d_])-\s*\n\s*(?=[^\W\d_])",
            "",
            text,
            flags=re.UNICODE,
        )

    def clean(self, section: DocumentSection) -> str:
        """Produce cleaned prose from a section's original extracted text."""
        text = section.raw_text

        # Remove known citation occurrences using offsets from GROBID.
        spans = [
            (citation.start, citation.end)
            for citation in section.citations
            if citation.start is not None
            and citation.end is not None
        ]

        text = self._remove_spans(text, spans)

        # Normalize before applying the remaining cleaning rules.
        text = self._normalize_unicode(text)
        text = self._repair_line_hyphenation(text)

        if self.remove_numeric_citations:
            text = self.NUMERIC_CITATION_PATTERN.sub(" ", text)

        if self.remove_math:
            for pattern in self.TEX_MATH_PATTERNS:
                text = pattern.sub(" ", text)

        lines: list[str] = []

        for line in text.splitlines():
            stripped = line.strip()

            if not stripped:
                lines.append("")
                continue

            if self.remove_captions and self.CAPTION_PATTERN.match(stripped):
                continue

            if (
                self.remove_page_numbers
                and self.PAGE_NUMBER_PATTERN.match(stripped)
            ):
                continue

            lines.append(stripped)

        text = "\n".join(lines)

        # Avoid merging sentences across removed blocks.
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"([.!?])\s*\n(?=[A-Z])", r"\1 ", text)
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)

        return text.strip()