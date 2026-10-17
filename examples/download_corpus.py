#!/usr/bin/env python3
"""Download and validate PDFs listed in the corpus manifest.

Usage:
    python -m examples.download_corpus --limit 5
    python -m examples.download_corpus
    python -m examples.download_corpus --retry-failures
    python -m examples.download_corpus --force
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = PROJECT_ROOT / "data" / "corpus" / "manifest.jsonl"
PDF_DIR = PROJECT_ROOT / "data" / "pdfs"
REPORT_PATH = PROJECT_ROOT / "data" / "corpus" / "download_report.jsonl"

USER_AGENT = "arxiv-topic-modelling/0.1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_filename(arxiv_id: str) -> str:
    """Convert an arXiv identifier into a safe filename."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", arxiv_id)


def pdf_path_for(arxiv_id: str) -> Path:
    return PDF_DIR / f"{safe_filename(arxiv_id)}.pdf"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, reporting malformed records with line numbers."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object in {path}, line {line_number}"
                )

            records.append(record)

    return records


def write_jsonl_atomic(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    """Replace a JSONL report atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    os.replace(temporary_path, path)


def validate_pdf(path: Path) -> tuple[bool, str, int]:
    """Validate the PDF signature and open the document with PyMuPDF.

    Returns:
        (is_valid, message, page_count)
    """
    if not path.is_file():
        return False, "File does not exist", 0

    if path.stat().st_size < 8:
        return False, "File is too small to be a valid PDF", 0

    try:
        with path.open("rb") as f:
            signature = f.read(1024)

        if b"%PDF-" not in signature:
            return False, "PDF signature not found near file beginning", 0

    except OSError as exc:
        return False, f"Could not read file: {exc}", 0

    try:
        import pymupdf

        with pymupdf.open(path) as document:
            page_count = document.page_count

            if page_count < 1:
                return False, "PDF contains no pages", 0

            return True, "PDF validated successfully", page_count

    except ImportError:
        raise RuntimeError(
            "PyMuPDF is required to validate downloaded PDFs. "
            "Install it with: pip install PyMuPDF"
        )

    except Exception as exc:
        return False, f"PyMuPDF could not open the PDF: {exc}", 0


def resolve_pdf_url(record: dict[str, Any]) -> str:
    """Return the PDF URL from metadata, or construct an arXiv URL."""
    url = record.get("pdf_url")

    if isinstance(url, str) and url.strip():
        return url.strip()

    arxiv_id = record.get("versioned_arxiv_id") or record.get("arxiv_id")

    if not isinstance(arxiv_id, str) or not arxiv_id.strip():
        raise ValueError("Record has neither a PDF URL nor a usable arXiv ID")

    encoded_id = urllib.parse.quote(arxiv_id.strip(), safe="/.")
    return f"https://arxiv.org/pdf/{encoded_id}"


def download_pdf(
    url: str,
    destination: Path,
    *,
    timeout: float,
    retries: int,
) -> tuple[int, int]:
    """Download a PDF to a temporary file and atomically install it.

    Returns:
        (number_of_attempts, bytes_downloaded)
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = destination.with_suffix(".pdf.part")
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "application/pdf,application/octet-stream;q=0.9",
                },
            )

            with urllib.request.urlopen(
                request,
                timeout=timeout,
            ) as response:
                status = getattr(response, "status", 200)

                if status != 200:
                    raise OSError(f"Unexpected HTTP status: {status}")

                with temporary_path.open("wb") as output:
                    while True:
                        chunk = response.read(1024 * 128)

                        if not chunk:
                            break

                        output.write(chunk)

            # Check the downloaded content before replacing the destination.
            valid, message, _ = validate_pdf(temporary_path)

            if not valid:
                raise ValueError(f"Downloaded file is not a valid PDF: {message}")

            bytes_downloaded = temporary_path.stat().st_size
            os.replace(temporary_path, destination)

            return attempt, bytes_downloaded

        except Exception as exc:
            last_error = exc

            temporary_path.unlink(missing_ok=True)

            retryable = (
                not isinstance(exc, urllib.error.HTTPError)
                or exc.code == 429
                or 500 <= exc.code < 600
            )

            if not retryable or attempt == retries:
                break

            wait_seconds = min(2 ** (attempt - 1) * 5, 60)

            print(
                f"    Attempt {attempt}/{retries} failed: {exc}. "
                f"Retrying in {wait_seconds}s..."
            )

            time.sleep(wait_seconds)

    raise RuntimeError(
        f"Download failed after {attempt} attempt(s): {last_error}"
    ) from last_error


def load_previous_report() -> dict[str, dict[str, Any]]:
    """Load the latest report record for each paper."""
    if not REPORT_PATH.exists():
        return {}

    previous: dict[str, dict[str, Any]] = {}

    for record in read_jsonl(REPORT_PATH):
        arxiv_id = record.get("arxiv_id")

        if isinstance(arxiv_id, str):
            previous[arxiv_id] = record

    return previous


def download_record(
    record: dict[str, Any],
    *,
    timeout: float,
    retries: int,
    force: bool,
) -> dict[str, Any]:
    """Download one paper and produce a structured outcome record."""
    arxiv_id = record.get("arxiv_id")

    if not isinstance(arxiv_id, str) or not arxiv_id.strip():
        return {
            "arxiv_id": None,
            "title": record.get("title"),
            "status": "failed",
            "error": "Missing or invalid arXiv ID",
            "timestamp": utc_now(),
        }

    arxiv_id = arxiv_id.strip()
    destination = pdf_path_for(arxiv_id)

    base_report = {
        "arxiv_id": arxiv_id,
        "title": record.get("title"),
        "pdf_path": str(destination.relative_to(PROJECT_ROOT)),
        "timestamp": utc_now(),
    }

    try:
        url = resolve_pdf_url(record)
        base_report["pdf_url"] = url

        if destination.exists() and not force:
            valid, message, page_count = validate_pdf(destination)

            if valid:
                return {
                    **base_report,
                    "status": "skipped_existing",
                    "message": message,
                    "page_count": page_count,
                    "bytes": destination.stat().st_size,
                    "attempts": 0,
                }

            print(f"    Existing PDF is invalid: {message}")

        attempts, bytes_downloaded = download_pdf(
            url,
            destination,
            timeout=timeout,
            retries=retries,
        )

        valid, message, page_count = validate_pdf(destination)

        if not valid:
            raise RuntimeError(f"Post-download validation failed: {message}")

        return {
            **base_report,
            "status": "downloaded",
            "message": message,
            "page_count": page_count,
            "bytes": bytes_downloaded,
            "attempts": attempts,
            "error": None,
        }

    except Exception as exc:
        return {
            **base_report,
            "status": "failed",
            "error": str(exc),
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and validate PDFs in the corpus manifest."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N manifest records.",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=3.0,
        help="Seconds between papers (default: 3).",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Network timeout in seconds (default: 60).",
    )

    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Maximum download attempts per paper (default: 3).",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload even if a valid local PDF exists.",
    )

    parser.add_argument(
        "--retry-failures",
        action="store_true",
        help="Process only papers not already downloaded successfully.",
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    if args.delay < 0:
        parser.error("--delay cannot be negative")

    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    if args.retries < 1:
        parser.error("--retries must be at least 1")

    manifest = read_jsonl(MANIFEST_PATH)
    previous = load_previous_report()

    if args.retry_failures:
        manifest = [
            record
            for record in manifest
            if previous.get(record.get("arxiv_id"), {}).get("status")
            not in {"downloaded", "skipped_existing"}
        ]

    if args.limit is not None:
        manifest = manifest[:args.limit]

    if not manifest:
        print("No papers to process.")
        return

    PDF_DIR.mkdir(parents=True, exist_ok=True)

    # Store outcomes by ID so each update replaces the previous result.
    report = dict(previous)

    total = len(manifest)

    print(f"Papers to process: {total}")
    print(f"PDF directory: {PDF_DIR}")
    print(f"Report: {REPORT_PATH}\n")

    for index, record in enumerate(manifest, start=1):
        arxiv_id = record.get("arxiv_id", "<missing ID>")
        title = record.get("title", "<untitled>")

        print(f"[{index}/{total}] {arxiv_id} — {title}")

        outcome = download_record(
            record,
            timeout=args.timeout,
            retries=args.retries,
            force=args.force,
        )

        if outcome["status"] == "downloaded":
            print(
                f"    OK: {outcome['page_count']} pages, "
                f"{outcome['bytes']:,} bytes"
            )
        elif outcome["status"] == "skipped_existing":
            print(f"    SKIP: existing PDF is valid ({outcome['page_count']} pages)")
        else:
            print(f"    FAILED: {outcome.get('error', 'Unknown error')}")

        if outcome.get("arxiv_id"):
            report[outcome["arxiv_id"]] = outcome

        # Persist after each paper, so an interruption loses at most
        # the current paper's report update.
        write_jsonl_atomic(REPORT_PATH, list(report.values()))

        if index < total:
            time.sleep(args.delay)

    outcomes = [
        report.get(record.get("arxiv_id"), {})
        for record in manifest
    ]

    downloaded = sum(r.get("status") == "downloaded" for r in outcomes)
    skipped = sum(r.get("status") == "skipped_existing" for r in outcomes)
    failed = sum(r.get("status") == "failed" for r in outcomes)

    print("\nDownload summary")
    print("----------------")
    print(f"Downloaded:       {downloaded}")
    print(f"Already available: {skipped}")
    print(f"Failed:           {failed}")
    print(f"Report:           {REPORT_PATH}")


if __name__ == "__main__":
    main()