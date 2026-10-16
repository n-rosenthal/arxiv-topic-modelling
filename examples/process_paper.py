"""Download a paper's PDF and extract its text."""

from __future__ import annotations

import argparse
import json
import logging

from pathlib import Path

from src.Pipeline import PaperPipeline
from src.Models import Paper


def read_record(path: Path, index: int) -> dict:
    """Read one JSONL record by zero-based record index."""
    if index < 0:
        raise ValueError("index must be non-negative")

    with path.open("r", encoding="utf-8") as file:
        for current_index, line in enumerate(file):
            if current_index == index:
                return json.loads(line)

    raise IndexError(
        f"Record {index} not found in {path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and extract an arXiv paper."
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/raw/nick007x-arxiv-papers.jsonl"),
        help="Path to the metadata JSONL file.",
    )

    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Zero-based record index in the JSONL file.",
    )

    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=Path("data/pdfs"),
    )

    parser.add_argument(
        "--text-dir",
        type=Path,
        default=Path("data/texts"),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload the PDF and overwrite extracted text.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    record = read_record(args.metadata, args.index)

    paper = Paper.from_source_record(
        record,
        dataset_id="nick007x/arxiv-papers",
        dataset_split="train",
        source_record_index=args.index,
    )

    pipeline = PaperPipeline(
        pdf_dir=args.pdf_dir,
        text_dir=args.text_dir,
        overwrite_pdfs=args.overwrite,
        overwrite_text=args.overwrite,
    )

    pipeline.process(paper)

    print(f"arXiv ID:       {paper.arxiv_id}")
    print(f"Title:          {paper.title}")
    print(f"PDF:            {paper.processing.download.local_path}")
    print(f"PDF size:       {paper.processing.download.file_size_bytes} bytes")
    print(f"Extracted text: {paper.processing.extracted_text.local_path}")
    print(f"Pages:          {paper.processing.extracted_text.page_count}")
    print(f"Characters:     {paper.processing.extracted_text.character_count}")
    print(f"PDF SHA-256:    {paper.processing.download.sha256}")
    print(f"Text SHA-256:   {paper.processing.extracted_text.sha256}")


if __name__ == "__main__":
    main()