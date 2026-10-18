#!/usr/bin/env python3
"""Collect arXiv metadata for a category x time-interval experiment design.

Outputs:
  - experiment_design.csv: all cells in the Cartesian product (56 rows)
  - experiment_counts.csv: per-cell counts and metadata quality indicators
  - manifest.jsonl: deduplicated metadata records for downstream pipeline stages
  - cells/*.jsonl: resumable cache, one file per experiment cell

This script collects metadata only. It does not download PDFs or run GROBID.
Requires the existing `arxiv` Python package.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import arxiv


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "experiments.json"
LOG = logging.getLogger("arxiv_experiments")


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    expected = len(config["time_intervals"]) * len(config["categories"])
    if len(config.get("experiments", [])) != expected:
        raise ValueError(
            f"Config must contain {expected} product cells; "
            f"found {len(config.get('experiments', []))}."
        )
    return config


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
    return rows


def normalize_result(result: arxiv.Result) -> dict:
    short_id = result.get_short_id()
    arxiv_id = short_id.rsplit("v", 1)[0] if re.search(r"v\d+$", short_id) else short_id
    published = result.published.isoformat() if result.published else None
    updated = result.updated.isoformat() if result.updated else None
    abstract = " ".join((result.summary or "").split())
    return {
        "arxiv_id": arxiv_id,
        "versioned_arxiv_id": short_id,
        "title": " ".join((result.title or "").split()),
        "authors": [author.name for author in result.authors],
        "abstract": abstract,
        "primary_subject": result.primary_category,
        "subjects": sorted(set(result.categories or [])),
        "submission_date": published,
        "updated_date": updated,
        "doi": result.doi,
        "comment": result.comment,
        "pdf_url": result.pdf_url,
        "abstract_url": result.entry_id,
        "abstract_word_count": len(abstract.split()),
    }


def query_for(category: str, start_year: int, end_year: int) -> str:
    start = f"{start_year}01010000"
    end = f"{end_year}12312359"
    return f"cat:{category} AND submittedDate:[{start} TO {end}]"


def cell_cache_path(cache_dir: Path, experiment_id: str) -> Path:
    return cache_dir / f"{safe_name(experiment_id)}.jsonl"


def collect_cell(
    client: arxiv.Client,
    experiment: dict,
    cache_dir: Path,
    limit_per_cell: int | None,
    refresh: bool,
) -> tuple[list[dict], str, str | None]:
    cache_path = cell_cache_path(cache_dir, experiment["experiment_id"])
    meta_path = cache_path.with_suffix(".meta.json")

    if cache_path.exists() and not refresh:
        cache_meta = {}
        if meta_path.exists():
            cache_meta = json.loads(meta_path.read_text(encoding="utf-8"))

        cached_is_limited = bool(cache_meta.get("limited", False))
        # A pilot cache must never masquerade as a complete collection.
        if limit_per_cell is not None or not cached_is_limited:
            rows = read_jsonl(cache_path)
            LOG.info("%s: cache hit (%d records)", experiment["experiment_id"], len(rows))
            return rows, ("limited_cached" if cached_is_limited else "cached"), None
        LOG.info("%s: replacing limited pilot cache with full collection", experiment["experiment_id"])

    query = query_for(
        experiment["category"], experiment["start_year"], experiment["end_year"]
    )
    LOG.info("%s: query=%s", experiment["experiment_id"], query)
    search = arxiv.Search(
        query=query,
        max_results=limit_per_cell,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    records: list[dict] = []
    seen: set[str] = set()
    try:
        for result in client.results(search):
            record = normalize_result(result)
            if record["arxiv_id"] in seen:
                continue
            seen.add(record["arxiv_id"])
            # The API query uses category membership; retain both membership
            # and primary-category status for later analysis.
            record["matched_category"] = experiment["category"]
            record["matched_interval_id"] = experiment["interval_id"]
            record["experiment_id"] = experiment["experiment_id"]
            records.append(record)
            if limit_per_cell is not None and len(records) >= limit_per_cell:
                break
        atomic_jsonl(cache_path, records)
        cache_meta = {
            "experiment_id": experiment["experiment_id"],
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "limited": limit_per_cell is not None and len(records) >= limit_per_cell,
            "limit_per_cell": limit_per_cell,
            "record_count": len(records),
            "query": query,
        }
        meta_tmp = meta_path.with_name(meta_path.name + ".tmp")
        meta_tmp.write_text(
            json.dumps(cache_meta, indent=2, ensure_ascii=False) + "\\n",
            encoding="utf-8",
        )
        meta_tmp.replace(meta_path)
        return records, ("limited" if limit_per_cell is not None else "collected"), None
    except Exception as exc:
        LOG.exception("Collection failed for %s", experiment["experiment_id"])
        return records, "failed", f"{type(exc).__name__}: {exc}"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def build_design_rows(config: dict) -> list[dict]:
    intervals = {item["id"]: item for item in config["time_intervals"]}
    categories = {item["id"]: item for item in config["categories"]}
    rows = []
    for exp in config["experiments"]:
        interval = intervals[exp["interval_id"]]
        category = categories[exp["category"]]
        rows.append({
            "experiment_id": exp["experiment_id"],
            "category": category["id"],
            "category_area": category["area"],
            "category_label": category["label"],
            "interval_id": interval["id"],
            "start_year": interval["start_year"],
            "end_year": interval["end_year"],
        })
    return rows


def summarize_cell(experiment: dict, records: list[dict], status: str, error: str | None) -> dict:
    category = experiment["category"]
    primary_count = sum(row.get("primary_subject") == category for row in records)
    secondary_count = sum(
        category in row.get("subjects", []) and row.get("primary_subject") != category
        for row in records
    )
    abstract_counts = [row.get("abstract_word_count", 0) for row in records]
    years = [
        int(row["submission_date"][:4])
        for row in records
        if row.get("submission_date")
    ]
    return {
        "experiment_id": experiment["experiment_id"],
        "category": category,
        "interval_id": experiment["interval_id"],
        "start_year": experiment["start_year"],
        "end_year": experiment["end_year"],
        "query_matches": len(records),
        "primary_category_count": primary_count,
        "secondary_category_count": secondary_count,
        "abstract_available_count": sum(bool(row.get("abstract")) for row in records),
        "pdf_url_available_count": sum(bool(row.get("pdf_url")) for row in records),
        "mean_abstract_words": round(mean(abstract_counts), 2) if abstract_counts else 0,
        "min_submission_year_observed": min(years) if years else "",
        "max_submission_year_observed": max(years) if years else "",
        "collection_status": status,
        "error": error or "",
    }


def merge_manifest(all_cell_records: list[dict]) -> list[dict]:
    """Deduplicate papers globally while preserving all experiment memberships."""
    merged: dict[str, dict] = {}
    memberships: dict[str, list[dict]] = defaultdict(list)

    for record in all_cell_records:
        paper_id = record["arxiv_id"]
        membership = {
            "experiment_id": record["experiment_id"],
            "category": record["matched_category"],
            "interval_id": record["matched_interval_id"],
        }
        memberships[paper_id].append(membership)
        if paper_id not in merged:
            base = dict(record)
            base.pop("matched_category", None)
            base.pop("matched_interval_id", None)
            base.pop("experiment_id", None)
            merged[paper_id] = base

    output = []
    for paper_id, record in merged.items():
        record["experiment_memberships"] = memberships[paper_id]
        record["corpus_index"] = len(output)
        output.append(record)
    output.sort(key=lambda row: (row.get("submission_date") or "", row["arxiv_id"]))
    for index, row in enumerate(output):
        row["corpus_index"] = index
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory from config (default: configured directory).",
    )
    parser.add_argument(
        "--limit-per-cell",
        type=int,
        default=None,
        help="Optional pilot limit per category/interval cell; results will be incomplete.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        help="Optional experiment IDs to run; by default run all cells.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore per-cell caches and query the API again.",
    )
    parser.add_argument(
        "--design-only",
        action="store_true",
        help="Write the 56-cell design CSV without contacting arXiv.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    config = load_config(args.config.resolve())
    configured_output = config["outputs"]["directory"]
    output_dir = args.output_dir or (ROOT / configured_output)
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    design_rows = build_design_rows(config)
    write_csv(
        output_dir / "experiment_design.csv",
        design_rows,
        ["experiment_id", "category", "category_area", "category_label",
         "interval_id", "start_year", "end_year"],
    )
    LOG.info("Design written: %s (%d cells)", output_dir / "experiment_design.csv", len(design_rows))
    if args.design_only:
        return 0

    selected = config["experiments"]
    if args.only:
        wanted = set(args.only)
        unknown = wanted - {exp["experiment_id"] for exp in selected}
        if unknown:
            parser.error(f"Unknown experiment IDs: {', '.join(sorted(unknown))}")
        selected = [exp for exp in selected if exp["experiment_id"] in wanted]

    if args.limit_per_cell is not None and args.limit_per_cell < 1:
        parser.error("--limit-per-cell must be positive.")

    cache_dir = output_dir / "cells"
    cache_dir.mkdir(parents=True, exist_ok=True)
    client = arxiv.Client(
        page_size=config["collection"]["arxiv_page_size"],
        delay_seconds=config["collection"]["delay_seconds"],
        num_retries=config["collection"]["num_retries"],
    )

    summaries_by_id = {}
    all_records = []
    failed = 0
    selected_ids = {exp["experiment_id"] for exp in selected}

    for number, experiment in enumerate(selected, 1):
        LOG.info("[%d/%d] %s", number, len(selected), experiment["experiment_id"])
        records, status, error = collect_cell(
            client, experiment, cache_dir, args.limit_per_cell, args.refresh
        )
        if status == "failed":
            failed += 1
        all_records.extend(records)
        summaries_by_id[experiment["experiment_id"]] = summarize_cell(
            experiment, records, status, error
        )

    # Keep one CSV row for every factorial cell, even when --only is used.
    # Reuse cached records for unselected cells, otherwise mark them not_run.
    for experiment in config["experiments"]:
        experiment_id = experiment["experiment_id"]
        if experiment_id in selected_ids:
            continue
        cache_path = cell_cache_path(cache_dir, experiment_id)
        if cache_path.exists():
            cached = read_jsonl(cache_path)
            all_records.extend(cached)
            meta_path = cache_path.with_suffix(".meta.json")
            cache_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
            cached_status = "limited_cached" if cache_meta.get("limited", False) else "cached"
            summaries_by_id[experiment_id] = summarize_cell(
                experiment, cached, cached_status, None
            )
        else:
            summaries_by_id[experiment_id] = summarize_cell(
                experiment, [], "not_run", None
            )

    summaries = list(summaries_by_id.values())
    summaries.sort(key=lambda row: (row["start_year"], row["category"]))
    write_csv(
        output_dir / "experiment_counts.csv",
        summaries,
        ["experiment_id", "category", "interval_id", "start_year", "end_year",
         "query_matches", "primary_category_count", "secondary_category_count",
         "abstract_available_count", "pdf_url_available_count",
         "mean_abstract_words", "min_submission_year_observed",
         "max_submission_year_observed", "collection_status", "error"],
    )

    manifest = merge_manifest(all_records)
    atomic_jsonl(output_dir / "manifest.jsonl", manifest)

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config.resolve()),
        "experiment_cells_in_this_run": len(selected),
        "total_design_cells": len(config["experiments"]),
        "unique_papers_in_manifest": len(manifest),
        "failed_cells_in_this_run": failed,
        "limit_per_cell": args.limit_per_cell,
        "note": (
            "Counts are complete only for cells with collection_status=collected "
            "or cached from a complete run. limited/limited_cached cells are pilot samples; "
            "not_run cells have not been queried."
        ),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    LOG.info("Counts CSV: %s", output_dir / "experiment_counts.csv")
    LOG.info("Deduplicated manifest: %s (%d unique papers)", output_dir / "manifest.jsonl", len(manifest))
    if failed:
        LOG.warning("%d cells failed. Re-run to retry; successful cell caches are preserved.", failed)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
