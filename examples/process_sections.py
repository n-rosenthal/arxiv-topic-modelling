"""Extract and preprocess sections from a paper in the metadata JSONL."""

from __future__ import annotations

import argparse
import json
import logging

from pathlib import Path

from src.Processing.ScientificPaperProcessor import (
    ScientificPaperProcessor,
)


def read_record(path: Path, index: int) -> dict:
    """Read a JSONL record by zero-based line index."""
    if index < 0:
        raise ValueError("index must be non-negative.")

    with path.open(encoding="utf-8") as file:
        for current_index, line in enumerate(file):
            if current_index == index:
                if not line.strip():
                    raise ValueError(
                        f"Record at index {index} is empty."
                    )
                return json.loads(line)

    raise IndexError(
        f"No record at index {index} in {path}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract scientific sections, citations, and references, "
            "then preprocess the text for topic modelling."
        )
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/raw/nick007x-arxiv-papers.jsonl"),
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=Path("data/pdfs"),
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Zero-based JSONL record index.",
    )
    parser.add_argument(
        "--grobid-url",
        default="http://localhost:8070",
    )
    parser.add_argument(
        "--spacy-model",
        default=None,
        help="Optional installed spaCy model, e.g. en_core_web_sm.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s | %(name)s | %(message)s",
    )

    record = read_record(args.metadata, args.index)

    arxiv_id = record.get("arxiv_id")

    if not arxiv_id:
        raise ValueError(
            f"Metadata record {args.index} has no arxiv_id."
        )

    # Match the filename convention used by PDFDownloader.
    pdf_filename = arxiv_id.replace("/", "_") + ".pdf"
    pdf_path = args.pdf_dir / pdf_filename

    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}\n"
            "Download the paper first using your existing PDFDownloader."
        )

    with ScientificPaperProcessor(
        grobid_url=args.grobid_url,
        spacy_model=args.spacy_model,
    ) as processor:
        document = processor.process(
            pdf_path=pdf_path,
            paper_id=arxiv_id,
        )

    print(f"Paper: {arxiv_id}")
    print(f"Sections: {len(document.sections)}")
    print(f"References: {len(document.references)}")
    print(f"TEI XML: {document.tei_path}")

    for section in document.sections:
        print(
            f"\n{'  ' * (section.level - 1)}"
            f"{section.order}. {section.title}"
        )
        print(f"   Characters: {len(section.raw_text)}")
        print(f"   Tokens: {len(section.tokens)}")
        print(f"   Citations: {len(section.citations)}")

    output_path = (
        Path("data/sections")
        / f"{arxiv_id.replace('/', '_')}.json"
    )
    print(f"\nStructured output: {output_path}")


if __name__ == "__main__":
    main()