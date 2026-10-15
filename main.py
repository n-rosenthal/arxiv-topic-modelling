"""
main.py

Command-line entry point for the arxiv-topic-modelling project.

Examples
--------
Download 100 papers:
    python main.py

Download 1,000 papers:
    python main.py --limit 1000

Download 500 papers starting at index 1,000:
    python main.py --start 1000 --limit 500

Download the remainder of the dataset split:
    python main.py --all

Specify a different dataset:
    python main.py --dataset CShorten/ML-ArXiv-Papers

Specify an output file:
    python main.py --output data/raw/papers.jsonl
"""

import argparse
import logging
import sys
from pathlib import Path

from src.Extractors.Downloader import ArxivDownloader


__author__ = "n-rosenthal"
__version__ = "1.0.0"
__date__ = "2026-10-15"


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "arxiv-papers.jsonl"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="arxiv-topic-modelling",
        description=(
            "Download arXiv papers from a Hugging Face dataset."
        ),
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=ArxivDownloader.DEFAULT_DATASET_ID,
        help="Hugging Face dataset repository ID.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default=ArxivDownloader.DEFAULT_SPLIT,
        help="Dataset split to download (default: train).",
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Zero-based index of the first record (default: 0).",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of records to download (default: 100).",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all remaining records in the selected split.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Destination JSONL file (default: {DEFAULT_OUTPUT}).",
    )

    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory for the Hugging Face dataset cache.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output file.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )

    args = parser.parse_args()

    if args.start < 0:
        parser.error("--start must be non-negative.")

    if args.limit < 0:
        parser.error("--limit must be non-negative.")

    if args.all and args.limit != 100:
        parser.error(
            "--all cannot be combined with a custom --limit."
        )

    return args


def main() -> int:
    """Run the downloader."""
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    logger = logging.getLogger(__name__)

    limit = None if args.all else args.limit

    downloader = ArxivDownloader(
        dataset_id=args.dataset,
        split=args.split,
        cache_dir=args.cache_dir,
    )

    try:
        count = downloader.download(
            output_path=args.output,
            start=args.start,
            limit=limit,
            overwrite=args.overwrite,
        )

    except KeyboardInterrupt:
        logger.warning("Download interrupted by the user.")
        return 130

    except Exception:
        logger.exception("Download failed.")
        return 1

    logger.info("Download completed successfully.")
    logger.info("Records written: %d", count)
    logger.info("Output file: %s", args.output.resolve())

    return 0


if __name__ == "__main__":
    sys.exit(main())