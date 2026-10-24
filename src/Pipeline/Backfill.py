"""
Preenche lacunas de cobertura temporal no corpus.

Detecta, via banco de dados, quais intervalos de `CorpusBuilder.DATE_RANGES`
estão sem nenhum artigo registrado (um sintoma comum de `build_corpus`, que
para assim que o limite global é atingido e pode nunca alcançar os
intervalos mais recentes) e busca artigos especificamente para eles, via
`CorpusBuilder.backfill`.
"""
from pathlib import Path

from src.Pipeline.Extraction import CorpusBuilder
from src.Storage import Database


def log(message: str) -> None:
    print(message)


def run_backfill(
    limit_per_range: int = 100,
    db_path: str | Path = Database.DEFAULT_DB_PATH,
) -> None:
    conn = Database.get_connection(db_path)
    Database.init_db(conn)

    try:
        empty_ranges = Database.get_empty_date_ranges(conn, CorpusBuilder.DATE_RANGES)
    finally:
        conn.close()

    if not empty_ranges:
        log("Nenhum intervalo de data sem cobertura encontrado. Nada a fazer.")
        return

    log(
        f"Encontrado(s) {len(empty_ranges)} intervalo(s) sem cobertura: "
        + ", ".join(f"{s}..{e}" for s, e in empty_ranges)
    )

    builder = CorpusBuilder(db_path=db_path)
    builder.backfill(empty_ranges, limit_per_range=limit_per_range)

    # Reimprime o relatório para mostrar o efeito do backfill.
    conn = Database.get_connection(db_path)
    Database.init_db(conn)
    try:
        log("")
        log(Database.build_report(conn, date_ranges=CorpusBuilder.DATE_RANGES))
    finally:
        conn.close()


def main() -> None:
    run_backfill()


if __name__ == "__main__":
    main()