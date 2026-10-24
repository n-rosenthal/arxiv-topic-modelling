"""Extract structured scientific-document content using GROBID."""

from __future__ import annotations

import logging
import re
import time

from pathlib import Path
from xml.etree import ElementTree as ET

import requests

from src.Models.Citation import Citation
from src.Models.Document import ExtractedDocument
from src.Models.Reference import Reference
from src.Models.Section import DocumentSection

logger = logging.getLogger(__name__)


class GROBIDExtractionError(RuntimeError):
    """Raised when GROBID fails to parse a scientific document."""


class GROBIDExtractor:
    """Extract sections, citations, and references from scientific PDFs."""

    XML_ID = "{http://www.w3.org/XML/1998/namespace}id"

    EXCLUDED_TAGS = {
        "figure",
        "table",
        "formula",
        "graphic",
        "media",
        "fw",  # Running headers and footers, when identified by TEI.
    }

    def __init__(
        self,
        base_url: str = "http://localhost:8070",
        output_dir: str | Path = "data/extracted",
        timeout: tuple[float, float] = (10.0, 300.0),
        verify_tls: bool = True,
        max_retries: int = 3,
        retry_wait: float = 5.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.output_dir = Path(output_dir)
        self.timeout = timeout
        self.verify_tls = verify_tls

        # GROBID answers 503 when all its worker threads are busy. That is
        # a transient condition, so the request is repeated before giving up.
        self.max_retries = max(1, max_retries)
        self.retry_wait = retry_wait

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "arxiv-topic-modelling/0.1"}
        )

    @staticmethod
    def _local_name(tag: str) -> str:
        """Return an XML tag without its namespace."""
        return tag.rsplit("}", 1)[-1]

    @classmethod
    def _children(
        cls,
        element: ET.Element,
        name: str,
    ) -> list[ET.Element]:
        return [
            child
            for child in element
            if cls._local_name(child.tag) == name
        ]

    @classmethod
    def _first_child(
        cls,
        element: ET.Element,
        name: str,
    ) -> ET.Element | None:
        for child in element:
            if cls._local_name(child.tag) == name:
                return child
        return None

    @classmethod
    def _descendants(
        cls,
        element: ET.Element,
        name: str,
    ) -> list[ET.Element]:
        return [
            node
            for node in element.iter()
            if cls._local_name(node.tag) == name
        ]

    @classmethod
    def _text(cls, element: ET.Element | None) -> str:
        if element is None:
            return ""

        return re.sub(
            r"\s+",
            " ",
            "".join(element.itertext()),
        ).strip()

    @classmethod
    def _spaced_text(cls, element: ET.Element) -> str:
        """Text of an element with its parts separated by a space."""
        return re.sub(
            r"\s+",
            " ",
            " ".join(element.itertext()),
        ).strip()

    def _retry_delay(
        self,
        response: requests.Response,
        attempt: int,
    ) -> float:
        """Seconds to wait after a 503; the server's Retry-After wins."""
        header = response.headers.get("Retry-After")

        if header:
            try:
                return max(0.0, float(header))
            except ValueError:
                pass

        return self.retry_wait * attempt

    def _request_tei(self, pdf_path: Path) -> str:
        """Submit a PDF to GROBID and return the resulting TEI XML."""
        endpoint = (
            f"{self.base_url}/api/processFulltextDocument"
        )

        try:
            for attempt in range(1, self.max_retries + 1):
                # The file is reopened on every attempt: the previous POST
                # already consumed its read pointer.
                with pdf_path.open("rb") as pdf_file:
                    response = self.session.post(
                        endpoint,
                        files={
                            "input": (
                                pdf_path.name,
                                pdf_file,
                                "application/pdf",
                            )
                        },
                        data={
                            "consolidateHeader": "0",
                            "consolidateCitations": "0",
                        },
                        timeout=self.timeout,
                        verify=self.verify_tls,
                    )

                if response.status_code != 503 or attempt == self.max_retries:
                    break

                wait = self._retry_delay(response, attempt)

                logger.info(
                    "GROBID busy (503) for %s; retry %d/%d in %.1fs",
                    pdf_path.name,
                    attempt,
                    self.max_retries,
                    wait,
                )

                time.sleep(wait)

            response.raise_for_status()

        except requests.RequestException as exc:
            raise GROBIDExtractionError(
                f"GROBID request failed for {pdf_path}: {exc}"
            ) from exc

        xml_text = response.text.strip()

        if not xml_text:
            raise GROBIDExtractionError(
                f"GROBID returned an empty response for {pdf_path}."
            )

        try:
            ET.fromstring(xml_text)
        except ET.ParseError as exc:
            raise GROBIDExtractionError(
                f"GROBID returned invalid XML for {pdf_path}."
            ) from exc

        return xml_text

    def _extract_references(
        self,
        root: ET.Element,
    ) -> list[Reference]:
        """Extract bibliography entries from TEI biblStruct elements."""
        references: list[Reference] = []

        for element in self._descendants(root, "biblStruct"):
            reference_id = (
                element.attrib.get(self.XML_ID)
                or element.attrib.get("id")
            )

            if not reference_id:
                # Stable fallback for malformed or incomplete TEI.
                reference_id = f"ref-{len(references) + 1}"

            raw_text = self._text(element)

            title = None
            titles = self._descendants(element, "title")

            if titles:
                # Prefer an article or chapter title over a journal title.
                preferred = next(
                    (
                        candidate
                        for candidate in titles
                        if candidate.attrib.get("level") in {"a", "m"}
                    ),
                    titles[0],
                )
                title = self._text(preferred) or None

            authors: list[str] = []

            for author in self._descendants(element, "author"):
                person = self._first_child(author, "persName")

                # GROBID writes <forename>..</forename><surname>..</surname>
                # with no whitespace between them, so the parts must be
                # joined explicitly or the name comes out as "TKipf".
                name = self._spaced_text(
                    author if person is None else person
                )

                if name and name not in authors:
                    authors.append(name)

            year = None

            for date in self._descendants(element, "date"):
                candidate = date.attrib.get("when") or self._text(date)
                match = re.search(r"\b(?:19|20)\d{2}\b", candidate)

                if match:
                    year = int(match.group())
                    break

            doi = None
            arxiv_id = None

            for identifier in self._descendants(element, "idno"):
                id_type = identifier.attrib.get("type", "").lower()
                value = self._text(identifier)

                if not value:
                    continue

                if id_type == "doi":
                    doi = value.removeprefix(
                        "https://doi.org/"
                    ).removeprefix("http://doi.org/")

                elif id_type in {"arxiv", "arxiv-id"}:
                    arxiv_id = value.removeprefix("arXiv:")

            references.append(
                Reference(
                    reference_id=reference_id,
                    raw_text=raw_text,
                    title=title,
                    authors=authors,
                    year=year,
                    doi=doi,
                    arxiv_id=arxiv_id,
                )
            )

        return references

    @classmethod
    def _render_element(
        cls,
        element: ET.Element,
        citations: list[Citation],
        current_offset: int,
    ) -> tuple[str, list[Citation]]:
        """Render an element's prose while recording citation occurrences.

        Figure/table/formula subtrees are excluded. Citation offsets refer
        to the returned text and are later adjusted to section coordinates.
        """
        parts: list[str] = []
        local_citations: list[Citation] = []
        length = 0

        def append(value: str) -> None:
            nonlocal length
            parts.append(value)
            length += len(value)

        def visit(node: ET.Element) -> None:
            if node.text:
                append(node.text)

            for child in node:
                tag = cls._local_name(child.tag)

                if tag in cls.EXCLUDED_TAGS:
                    if child.tail:
                        append(child.tail)
                    continue

                if tag == "ref" and child.attrib.get("type") == "bibr":
                    marker = cls._text(child)

                    if marker:
                        start = length
                        append(marker)

                        targets = [
                            target.lstrip("#")
                            for target in child.attrib.get(
                                "target", ""
                            ).split()
                            if target.startswith("#")
                        ]

                        local_citations.append(
                            Citation(
                                marker=marker,
                                target_reference_ids=targets,
                                start=current_offset + start,
                                end=current_offset + length,
                            )
                        )
                else:
                    visit(child)

                if child.tail:
                    append(child.tail)

        visit(element)

        text = "".join(parts)
        citations.extend(local_citations)

        return text, local_citations

    @classmethod
    def _direct_paragraphs(
        cls,
        element: ET.Element,
    ) -> list[ET.Element]:
        """Return paragraphs belonging directly to a section.

        Paragraphs nested inside a child <div> belong to that subsection,
        so they must not be duplicated in the parent section.
        """
        paragraphs: list[ET.Element] = []

        def visit_container(container: ET.Element) -> None:
            for child in container:
                tag = cls._local_name(child.tag)

                if tag == "div":
                    continue

                if tag == "p":
                    paragraphs.append(child)
                    continue

                if tag in {"listBibl", "back", "figure", "table"}:
                    continue

                visit_container(child)

        visit_container(element)
        return paragraphs
    
    def _extract_sections(
        self,
        root: ET.Element,
        paper_id: str,
    ) -> list[DocumentSection]:
        """Extract section hierarchy and section-level citations.

        Citation offsets refer to character positions in each section's
        raw_text. Offsets are adjusted when paragraph whitespace is trimmed.
        """
        text_element = next(
            (
                node
                for node in root.iter()
                if self._local_name(node.tag) == "text"
            ),
            None,
        )

        if text_element is None:
            raise GROBIDExtractionError(
                "TEI document does not contain a <text> element."
            )

        body = next(
            (
                node
                for node in text_element
                if self._local_name(node.tag) == "body"
            ),
            None,
        )

        if body is None:
            raise GROBIDExtractionError(
                "TEI document does not contain a <body> element."
            )

        sections: list[DocumentSection] = []
        section_counter = 0

        def render_paragraphs(
            paragraphs: list[ET.Element],
        ) -> tuple[str, list[Citation]]:
            """Render paragraphs and maintain citation offsets.

            Paragraphs are joined with a single newline. Citation offsets
            are relative to the resulting joined text.
            """
            text_parts: list[str] = []
            citations: list[Citation] = []
            offset = 0

            for paragraph in paragraphs:
                rendered_text, paragraph_citations = self._render_element(
                    paragraph,
                    citations=[],
                    current_offset=offset,
                )

                # Determine the trimmed prefix before stripping it.
                leading = len(rendered_text) - len(
                    rendered_text.lstrip()
                )
                paragraph_text = rendered_text.strip()

                if not paragraph_text:
                    continue

                # _render_element() calculated offsets against the
                # untrimmed rendered paragraph. Move them to the trimmed
                # paragraph's coordinate system.
                for citation in paragraph_citations:
                    if citation.start is not None:
                        citation.start -= leading

                    if citation.end is not None:
                        citation.end -= leading

                text_parts.append(paragraph_text)
                citations.extend(paragraph_citations)

                # The next paragraph starts after the joining newline.
                offset += len(paragraph_text) + 1

            return "\n".join(text_parts), citations

        def process_div(
            div: ET.Element,
            level: int,
            parent_id: str | None,
        ) -> None:
            nonlocal section_counter

            head = self._first_child(div, "head")
            title = self._text(head) or "Untitled section"

            section_counter += 1
            section_id = f"{paper_id}-s{section_counter:04d}"

            paragraphs = self._direct_paragraphs(div)
            raw_text, citations = render_paragraphs(paragraphs)

            section = DocumentSection(
                section_id=section_id,
                paper_id=paper_id,
                title=title,
                level=level,
                order=section_counter,
                raw_text=raw_text,
                citations=citations,
                parent_section_id=parent_id,
                extraction_method="grobid",
            )

            if not raw_text:
                section.warnings.append(
                    "Section contains no extractable prose."
                )

            # Keep empty sections as well: nested sections can refer to
            # them through parent_section_id.
            sections.append(section)

            for child in div:
                if self._local_name(child.tag) == "div":
                    process_div(
                        child,
                        level=level + 1,
                        parent_id=section_id,
                    )

        top_level_divs = [
            child
            for child in body
            if self._local_name(child.tag) == "div"
        ]

        for div in top_level_divs:
            process_div(
                div,
                level=1,
                parent_id=None,
            )

        # Some documents contain body paragraphs outside any <div>.
        body_paragraphs = [
            child
            for child in body
            if self._local_name(child.tag) == "p"
        ]

        if body_paragraphs:
            raw_text, citations = render_paragraphs(body_paragraphs)

            if raw_text:
                section_counter += 1
                section_id = f"{paper_id}-s{section_counter:04d}"

                # These paragraphs occur outside the formal section
                # hierarchy, so place their section before the others.
                sections.insert(
                    0,
                    DocumentSection(
                        section_id=section_id,
                        paper_id=paper_id,
                        title="Main text",
                        level=1,
                        order=1,
                        raw_text=raw_text,
                        citations=citations,
                        parent_section_id=None,
                        extraction_method="grobid",
                        warnings=[
                            "Some body paragraphs were outside TEI sections."
                        ],
                    ),
                )

        # Normalize order after inserting the optional fallback section.
        for index, section in enumerate(sections, start=1):
            section.order = index

        if not any(section.raw_text.strip() for section in sections):
            raise GROBIDExtractionError(
                "No non-empty body sections could be extracted."
            )

        return sections

    @staticmethod
    def _safe_identifier(paper_id: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]", "_", paper_id)

    def tei_path(self, paper_id: str) -> Path:
        """Where the TEI XML of a paper is (or will be) stored."""
        return (
            self.output_dir
            / f"{self._safe_identifier(paper_id)}.tei.xml"
        )

    def _save_tei(self, xml_text: str, paper_id: str) -> Path:
        tei_path = self.tei_path(paper_id)
        temporary_path = tei_path.with_suffix(".tei.xml.tmp")

        temporary_path.write_text(
            xml_text,
            encoding="utf-8",
        )
        temporary_path.replace(tei_path)

        return tei_path

    def extract(
        self,
        pdf_path: str | Path,
        paper_id: str,
    ) -> ExtractedDocument:
        """Parse a PDF and return its sections, citations, and references.

        The TEI returned by GROBID is stored in ``output_dir`` before it is
        parsed, so it survives a parsing failure and can be parsed again
        later with :meth:`extract_from_tei`, without the PDF or GROBID.
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.is_file():
            raise FileNotFoundError(
                f"PDF does not exist: {pdf_path}"
            )

        xml_text = self._request_tei(pdf_path)
        tei_path = self._save_tei(xml_text, paper_id)

        return self.parse_tei(xml_text, paper_id, tei_path)

    def extract_from_tei(
        self,
        paper_id: str,
        tei_path: str | Path | None = None,
    ) -> ExtractedDocument:
        """Rebuild a document from a TEI file stored by :meth:`extract`.

        Needs neither the PDF nor a running GROBID server. ``tei_path``
        defaults to the location :meth:`tei_path` gives for ``paper_id``.
        """
        path = Path(tei_path) if tei_path else self.tei_path(paper_id)

        if not path.is_file():
            raise FileNotFoundError(f"TEI does not exist: {path}")

        return self.parse_tei(
            path.read_text(encoding="utf-8"),
            paper_id,
            path,
        )

    def parse_tei(
        self,
        xml_text: str,
        paper_id: str,
        tei_path: str | Path | None = None,
    ) -> ExtractedDocument:
        """Turn GROBID TEI XML into sections, citations, and references."""
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            raise GROBIDExtractionError(
                f"Could not parse TEI for {paper_id}."
            ) from exc

        references = self._extract_references(root)
        sections = self._extract_sections(root, paper_id)

        known_reference_ids = {
            reference.reference_id
            for reference in references
        }

        warnings: list[str] = []

        for section in sections:
            for citation in section.citations:
                unresolved = [
                    target
                    for target in citation.target_reference_ids
                    if target not in known_reference_ids
                ]

                if unresolved:
                    section.warnings.append(
                        "Citation targets missing from bibliography: "
                        + ", ".join(unresolved)
                    )

        document = ExtractedDocument(
            paper_id=paper_id,
            sections=sections,
            references=references,
            extraction_method="grobid",
            tei_path=str(tei_path) if tei_path else None,
            warnings=warnings,
        )

        logger.info(
            "Extracted %d sections and %d references from %s",
            len(sections),
            len(references),
            paper_id,
        )

        return document

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self.session.close()

    def __enter__(self) -> "GROBIDExtractor":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
