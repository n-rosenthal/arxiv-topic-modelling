"""src.Pipeline.Extraction.Tei
Leitura do TEI XML produzido pelo GROBID.

O TEI completo é guardado em disco como fonte de verdade (contém
seções, referências e, se pedido, sentenças e coordenadas). Este módulo
só extrai dele o necessário para gerar o texto simples usado em PLN:
título, resumo e as seções do corpo, em ordem.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from dataclasses import dataclass, field


TEI_NS = "http://www.tei-c.org/ns/1.0"
_NS = {"tei": TEI_NS}
_DIV = f"{{{TEI_NS}}}div"
_P = f"{{{TEI_NS}}}p"

_WHITESPACE = re.compile(r"\s+")


def _clean(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def _element_text(element: ET.Element) -> str:
    """
    Texto de um elemento. Quando o GROBID marcou sentenças (`<s>`),
    elas são unidas por espaço — o TEI não garante um espaço entre
    `</s>` e `<s>`.
    """
    sentences = element.findall("tei:s", _NS)

    if sentences:
        return _clean(" ".join("".join(s.itertext()) for s in sentences))

    return _clean("".join(element.itertext()))


@dataclass
class Section:
    heading: str
    paragraphs: list[str]
    number: str = ""   # numeração original do artigo, ex.: "3.1"


@dataclass
class ParsedTei:
    title: str = ""
    abstract: str = ""
    sections: list[Section] = field(default_factory=list)
    reference_count: int = 0

    def to_text(self, include_headings: bool = True) -> str:
        """
        Texto simples: título, resumo e corpo, separados por linha em
        branco. Com `include_headings`, resumo e seções ganham um
        cabeçalho no estilo Markdown (`## 3.1 Método`).
        """
        parts: list[str] = []

        if self.title:
            parts.append(f"# {self.title}" if include_headings else self.title)

        if self.abstract:
            if include_headings:
                parts.append("## Abstract")
            parts.append(self.abstract)

        for section in self.sections:
            if include_headings and section.heading:
                label = f"{section.number} {section.heading}".strip()
                parts.append(f"## {label}")
            parts.extend(section.paragraphs)

        return "\n\n".join(parts)


def parse_tei(xml: str | bytes) -> ParsedTei:
    """
    Extrai título, resumo, seções e contagem de referências de um TEI
    do GROBID.

    Raises:
        ValueError: o XML é inválido.
    """
    data = xml.encode("utf-8") if isinstance(xml, str) else xml

    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise ValueError(f"invalid TEI XML: {exc}") from exc

    parsed = ParsedTei()

    title = root.find("tei:teiHeader/tei:fileDesc/tei:titleStmt/tei:title", _NS)
    if title is not None:
        parsed.title = _clean("".join(title.itertext()))

    abstract = root.find("tei:teiHeader/tei:profileDesc/tei:abstract", _NS)
    if abstract is not None:
        paragraphs = [_element_text(p) for p in abstract.iter(_P)]
        parsed.abstract = "\n\n".join(p for p in paragraphs if p)

        if not parsed.abstract:
            parsed.abstract = _clean("".join(abstract.itertext()))

    body = root.find("tei:text/tei:body", _NS)
    if body is not None:
        # `iter` percorre também divs aninhadas; cada div contribui só
        # com o próprio cabeçalho e parágrafos diretos, sem duplicar.
        for div in body.iter(_DIV):
            head = div.find("tei:head", _NS)
            heading = _element_text(head) if head is not None else ""
            number = (head.get("n") or "").strip() if head is not None else ""

            paragraphs = [_element_text(p) for p in div.findall("tei:p", _NS)]
            paragraphs = [p for p in paragraphs if p]

            if heading or paragraphs:
                parsed.sections.append(Section(heading, paragraphs, number))

    parsed.reference_count = len(
        root.findall("tei:text/tei:back//tei:listBibl/tei:biblStruct", _NS)
    )

    return parsed