"""Test PDF text extraction on a single paper."""

import argparse
from pathlib import Path

from src.Extractors import PDFTextExtractor
from src.Models import Paper


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract text from a paper's PDF."
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        required=True,
        help="Path to the downloaded PDF.",
    )
    parser.add_argument(
        "--arxiv-id",
        required=True,
        help="arXiv identifier, e.g. 2301.00001.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/texts"),
    )
    args = parser.parse_args()

    # Construct a minimal Paper object for this extraction test.
    paper = Paper(
        arxiv_id=args.arxiv_id,
        title="PDF extraction test",
        authors=[],
        submission_date="18 Feb 2009",
        abstract="Test document.",
        file_path=str(args.pdf),
    )

    paper.processing.download.local_path = str(args.pdf)

    extractor = PDFTextExtractor(
        output_dir=args.output_dir,
        overwrite=False,
    )

    output_path = extractor.extract(paper)

    print(f"Paper:      {paper.arxiv_id}")
    print(f"Output:     {output_path}")
    print(f"Characters: {paper.processing.extracted_text.character_count}")
    print(f"Pages:      {paper.processing.extracted_text.page_count}")
    print(f"SHA-256:    {paper.processing.extracted_text.sha256}")
    print(f"Processed:  {paper.text_processed}")


if __name__ == "__main__":
    main()