 #!/usr/bin/env python3
"""Constrói um corpus de artigos do arXiv para modelagem de tópicos."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import arxiv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "corpus"

DEFAULT_CATEGORY = "cs.LG"
DEFAULT_LIMIT = 100
DEFAULT_START_DATE = "20250101"
DEFAULT_END_DATE = "20251231"


def parse_args() -> argparse.Namespace:
    """Interpreta os argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description=(
            "Constrói um corpus do arXiv filtrando por categoria principal "
            "e intervalo de datas."
        )
    )

    parser.add_argument(
        "--category",
        default=DEFAULT_CATEGORY,
        help=f"Categoria principal desejada (padrão: {DEFAULT_CATEGORY}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Quantidade de artigos (padrão: {DEFAULT_LIMIT}).",
    )
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
        help="Data inicial inclusiva, no formato YYYYMMDD.",
    )
    parser.add_argument(
        "--end-date",
        default=DEFAULT_END_DATE,
        help="Data final inclusiva, no formato YYYYMMDD.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Diretório de saída do manifesto e dos metadados.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Permite substituir um manifesto e metadados existentes.",
    )

    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit deve ser maior que zero.")

    for option, value in (
        ("--start-date", args.start_date),
        ("--end-date", args.end_date),
    ):
        try:
            datetime.strptime(value, "%Y%m%d")
        except ValueError:
            parser.error(f"{option} deve estar no formato YYYYMMDD.")

    if args.start_date > args.end_date:
        parser.error("--start-date não pode ser posterior a --end-date.")

    return args


def build_query(category: str, start_date: str, end_date: str) -> str:
    """Monta uma consulta com intervalo de submissão inclusivo."""
    return (
        f"cat:{category} "
        f"AND submittedDate:[{start_date}0000 TO {end_date}2359]"
    )


def normalize_paper(result: arxiv.Result) -> dict:
    """Converte um resultado da API em um registro serializável."""
    arxiv_id = result.get_short_id().split("v")[0]

    return {
        "arxiv_id": arxiv_id,
        "versioned_arxiv_id": result.get_short_id(),
        "title": " ".join(result.title.split()),
        "authors": [author.name for author in result.authors],
        "abstract": " ".join(result.summary.split()),
        "primary_subject": result.primary_category,
        "subjects": sorted(result.categories),
        "submission_date": result.published.isoformat(),
        "updated_date": result.updated.isoformat(),
        "doi": result.doi,
        "comment": result.comment,
        "pdf_url": result.pdf_url,
        "abstract_url": result.entry_id,
    }


def write_json_atomic(path: Path, data: dict) -> None:
    """Grava JSON usando arquivo temporário e substituição atômica."""
    temporary_path = path.with_name(path.name + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
 
    temporary_path.replace(path)


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.jsonl"
    metadata_path = output_dir / "metadata.json"

    existing_files = [
        path for path in (manifest_path, metadata_path) if path.exists()
    ]

    if existing_files and not args.force:
        files = ", ".join(str(path) for path in existing_files)
        raise SystemExit(
            "Os seguintes arquivos já existem:\n"
            f"  {files}\n"
            "Faça um backup e execute novamente com --force "
            "se desejar substituí-los."
        )

    query = build_query(
        args.category,
        args.start_date,
        args.end_date,
    )

    print(f"Categoria principal desejada: {args.category}")
    print(f"Intervalo: {args.start_date}–{args.end_date}")
    print(f"Quantidade desejada: {args.limit}")
    print(f"Consulta: {query}")
    print("Consultando a API do arXiv...")

    client = arxiv.Client(
        page_size=100,
        delay_seconds=3.0,
        num_retries=3,
    )

    search = arxiv.Search(
        query=query,
        max_results=None,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    records: list[dict] = []
    seen_ids: set[str] = set()
    examined = 0

    for result in client.results(search):
        examined += 1
        record = normalize_paper(result)

        # A consulta por categoria pode retornar artigos cuja categoria
        # principal não corresponde à categoria desejada.
        if record["primary_subject"] != args.category:
            continue

        # Defesa adicional: não incluir resultados fora do intervalo.
        submission_day = record["submission_date"][:10].replace("-", "")
        if not args.start_date <= submission_day <= args.end_date:
            continue

        if record["arxiv_id"] in seen_ids:
            continue

        seen_ids.add(record["arxiv_id"])
        records.append(record)

        print(
            f"[{len(records):03d}/{args.limit}] "
            f"{record['arxiv_id']} — {record['title']}"
        )

        if len(records) >= args.limit:
            break

        # O cliente já impõe um intervalo entre requisições.
        time.sleep(0.1)

    if not records:
        raise SystemExit(
            "Nenhum artigo correspondeu aos filtros. "
            "O manifesto anterior foi preservado."
        )

    if len(records) < args.limit:
        print(
            f"Aviso: foram encontrados apenas {len(records)} "
            f"artigos dos {args.limit} solicitados."
        )

    # O manifesto só é substituído depois de a consulta terminar.
    manifest_tmp = manifest_path.with_name(manifest_path.name + ".tmp")

    with manifest_tmp.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "arXiv API",
        "query": query,
        "category": args.category,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "requested_limit": args.limit,
        "record_count": len(records),
        "records_examined": examined,
        "manifest": str(manifest_path),
    }

    # Preparar os metadados antes de substituir o manifesto.
    metadata_tmp = metadata_path.with_name(metadata_path.name + ".tmp")

    with metadata_tmp.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
        file.write("\n")

    manifest_tmp.replace(manifest_path)
    metadata_tmp.replace(metadata_path)

    print(f"\nSalvos {len(records)} artigos.")
    print(f"Manifesto: {manifest_path}")
    print(f"Metadados: {metadata_path}")


if __name__ == "__main__":
    main(lko)
