#!/usr/bin/env python3
"""Build a small arXiv corpus for topic-modelling experiments."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import arxiv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "corpus"

CATEGORY = "cs.LG"
MAX_RESULTS = 100

# Restrict the pilot to a recent, fixed interval.
# arXiv query dates use YYYYMMDD.
START_DATE = "20250101"
END_DATE = "20260101"


def build_query() -> str:
    return (
        f"cat:{CATEGORY} "
        f"AND submittedDate:[{START_DATE}0000 TO {END_DATE}2359]"
    )


def normalize_paper(result: arxiv.Result) -> dict:
    """Convert an arXiv API result into a JSON-serializable record."""
    arxiv_id = result.get_short_id().split("v")[0]

    categories = sorted(result.categories)

    return {
        "arxiv_id": arxiv_id,
        "versioned_arxiv_id": result.get_short_id(),
        "title": " ".join(result.title.split()),
        "authors": [author.name for author in result.authors],
        "abstract": " ".join(result.summary.split()),
        "primary_subject": result.primary_category,
        "subjects": categories,
        "submission_date": result.published.isoformat(),
        "updated_date": result.updated.isoformat(),
        "doi": result.doi,
        "comment": result.comment,
        "pdf_url": result.pdf_url,
        "abstract_url": result.entry_id,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_path = OUTPUT_DIR / "manifest.jsonl"
    metadata_path = OUTPUT_DIR / "metadata.json"

    query = build_query()

    print(f"Category: {CATEGORY}")
    print(f"Date interval: {START_DATE}–{END_DATE}")
    print(f"Maximum papers: {MAX_RESULTS}")
    print(f"Query: {query}")
    print("Contacting arXiv API...")

    client = arxiv.Client(
        page_size=100,
        delay_seconds=3.0,
        num_retries=3,
    )

    search = arxiv.Search(
        query=query,
        max_results=MAX_RESULTS,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    records: list[dict] = []
    seen_ids: set[str] = set()

    for result in client.results(search):
        record = normalize_paper(result)

        if record["arxiv_id"] in seen_ids:
            continue

        seen_ids.add(record["arxiv_id"])
        records.append(record)

        print(
            f"[{len(records):03d}/{MAX_RESULTS}] "
            f"{record['arxiv_id']} — {record['title']}"
        )

        # Be polite to the API. The client also enforces its request delay.
        time.sleep(0.1)

    # Write the manifest atomically so an interrupted run doesn't leave
    # a partially written manifest at the final path.
    temporary_path = manifest_path.with_suffix(".jsonl.tmp")

    with temporary_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    temporary_path.replace(manifest_path)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "arXiv API",
        "query": query,
        "category": CATEGORY,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "requested_limit": MAX_RESULTS,
        "record_count": len(records),
        "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
    }

    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nSaved {len(records)} papers.")
    print(f"Manifest: {manifest_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()