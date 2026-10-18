#!/usr/bin/env python3
"""
Gerencia o download incremental do corpus arXiv.

Fonte de metadados:
    data/corpus/manifest.jsonl

PDFs:
    data/pdfs/<arxiv_id>.pdf

Banco de dados:
    data/corpus/pipeline.sqlite3

Exemplos:
    python -m examples.ingest_corpus run --limit 5
    python -m examples.ingest_corpus run --limit 10
    python -m examples.ingest_corpus run --limit 10 --retry-failures
    python -m examples.ingest_corpus sync
    python -m examples.ingest_corpus status
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sqlite3
import sys
import time

from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

def project_path(value: str) -> Path:
    """Resolve caminhos relativos a partir da raiz do projeto."""
    path = Path(value).expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()
    
PROJECT_ROOT = project_path(".")

DEFAULT_TIMEOUT = 120
DEFAULT_ATTEMPTS = 3
MIN_PDF_SIZE = 1024

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("corpus")


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def utc_now() -> str:
    """Retorna um timestamp UTC em formato ISO 8601."""
    return datetime.now(timezone.utc).isoformat()


def normalize_arxiv_id(value: Any) -> str:
    """
    Normaliza identificadores encontrados nos metadados.

    Aceita, por exemplo:
        2501.12345
        2501.12345v2
        https://arxiv.org/abs/2501.12345
        https://arxiv.org/pdf/2501.12345.pdf

    A versão do artigo é preservada quando presente.
    """
    if value is None:
        raise ValueError("Registro sem identificador arXiv.")

    value = str(value).strip()

    if "/abs/" in value:
        value = value.split("/abs/", 1)[1]
    elif "/pdf/" in value:
        value = value.split("/pdf/", 1)[1]

    value = value.removesuffix(".pdf")

    match = re.search(
        r"(?<!\d)(\d{4}\.\d{4,5}(?:v\d+)?|\w+(?:\.\w+)?/\d{7}(?:v\d+)?)(?!\w)",
        value,
    )

    if not match:
        raise ValueError(f"Identificador arXiv inválido: {value!r}")

    return match.group(1)


def safe_filename(paper_id: str) -> str:
    """Produz um nome de arquivo seguro a partir do identificador."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", paper_id)


def get_field(
    record: dict[str, Any],
    *names: str,
    default: Any = None,
) -> Any:
    """Retorna o primeiro campo disponível entre os nomes fornecidos."""
    for name in names:
        value = record.get(name)
        if value is not None and value != "":
            return value

    return default


def serialize_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


# ---------------------------------------------------------------------------
# Banco de dados
# ---------------------------------------------------------------------------

def connect_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(
        path,
        timeout=30,
    )

    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")
    connection.execute("PRAGMA journal_mode = WAL")

    return connection


def initialize_database(connection: sqlite3.Connection) -> None:
    """
    Cria as tabelas de controle.

    O campo status representa a etapa atual do pipeline.
    Os caminhos dos artefatos são mantidos separadamente.
    """
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS papers (
            arxiv_id TEXT PRIMARY KEY,

            corpus_index INTEGER NOT NULL UNIQUE,

            title TEXT NOT NULL DEFAULT '',
            authors_json TEXT NOT NULL DEFAULT '[]',
            abstract TEXT NOT NULL DEFAULT '',
            published TEXT,
            categories_json TEXT NOT NULL DEFAULT '[]',

            metadata_json TEXT NOT NULL,

            status TEXT NOT NULL DEFAULT 'pending'
                CHECK (
                    status IN (
                        'pending',
                        'downloading',
                        'downloaded',
                        'extracting',
                        'extracted',
                        'processing',
                        'processed',
                        'failed_download',
                        'failed_extraction',
                        'failed_processing'
                    )
                ),

            pdf_path TEXT,
            pdf_size INTEGER,
            pdf_sha256 TEXT,

            extracted_path TEXT,
            processed_path TEXT,

            download_attempts INTEGER NOT NULL DEFAULT 0,
            extraction_attempts INTEGER NOT NULL DEFAULT 0,
            processing_attempts INTEGER NOT NULL DEFAULT 0,

            last_error TEXT,

            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            downloaded_at TEXT,
            extracted_at TEXT,
            processed_at TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_papers_status
            ON papers(status);

        CREATE INDEX IF NOT EXISTS idx_papers_corpus_index
            ON papers(corpus_index);

        CREATE TABLE IF NOT EXISTS pipeline_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS pipeline_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            arxiv_id TEXT NOT NULL,
            event TEXT NOT NULL,
            details TEXT,
            created_at TEXT NOT NULL,

            FOREIGN KEY (arxiv_id)
                REFERENCES papers(arxiv_id)
        );
        """
    )

    connection.execute(
        """
        INSERT OR IGNORE INTO pipeline_state (key, value, updated_at)
        VALUES ('next_index', '0', ?)
        """,
        (utc_now(),),
    )

    connection.commit()


def log_event(
    connection: sqlite3.Connection,
    paper_id: str,
    event: str,
    details: str | None = None,
) -> None:
    connection.execute(
        """
        INSERT INTO pipeline_events (
            arxiv_id,
            event,
            details,
            created_at
        )
        VALUES (?, ?, ?, ?)
        """,
        (paper_id, event, details, utc_now()),
    )


def set_status(
    connection: sqlite3.Connection,
    paper_id: str,
    status: str,
    *,
    error: str | None = None,
) -> None:
    connection.execute(
        """
        UPDATE papers
        SET status = ?,
            last_error = ?,
            updated_at = ?
        WHERE arxiv_id = ?
        """,
        (status, error, utc_now(), paper_id),
    )


def get_next_index(connection: sqlite3.Connection) -> int:
    row = connection.execute(
        "SELECT value FROM pipeline_state WHERE key = 'next_index'"
    ).fetchone()

    return int(row["value"]) if row else 0


def advance_index(
    connection: sqlite3.Connection,
    corpus_index: int,
) -> None:
    """
    Avança o cursor sem nunca retrocedê-lo.

    O cursor é atualizado na mesma transação da operação concluída.
    """
    current = get_next_index(connection)
    next_value = max(current, corpus_index + 1)

    connection.execute(
        """
        INSERT INTO pipeline_state (key, value, updated_at)
        VALUES ('next_index', ?, ?)
        ON CONFLICT(key)
        DO UPDATE SET
            value = excluded.value,
            updated_at = excluded.updated_at
        """,
        (str(next_value), utc_now()),
    )


# ---------------------------------------------------------------------------
# Manifesto
# ---------------------------------------------------------------------------

def load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Manifesto não encontrado: {path}"
        )

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON inválido no manifesto, linha {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Esperado um objeto JSON na linha {line_number}."
                )

            records.append(record)

    return records


def extract_metadata(
    record: dict[str, Any],
    corpus_index: int,
) -> dict[str, Any]:
    raw_id = get_field(
        record,
        "arxiv_id",
        "paper_id",
        "id",
        "entry_id",
    )

    paper_id = normalize_arxiv_id(raw_id)

    title = str(
        get_field(record, "title", default="")
    ).strip()

    authors = get_field(
        record,
        "authors",
        "author",
        default=[],
    )

    if isinstance(authors, str):
        authors = [authors] if authors else []

    categories = get_field(
        record,
        "categories",
        "subjects",
        default=[],
    )

    if isinstance(categories, str):
        categories = [categories] if categories else []

    abstract = str(
        get_field(
            record,
            "abstract",
            "summary",
            default="",
        )
    ).strip()

    published = get_field(
        record,
        "published",
        "published_date",
        "date",
        "submission_date",
    )

    return {
        "arxiv_id": paper_id,
        "corpus_index": corpus_index,
        "title": title,
        "authors_json": serialize_json(authors),
        "abstract": abstract,
        "published": str(published) if published else None,
        "categories_json": serialize_json(categories),
        "metadata_json": serialize_json(record),
    }


def import_manifest(
    connection: sqlite3.Connection,
    manifest_path: Path,
) -> int:
    """
    Registra os metadados sem sobrescrever o progresso de artigos existentes.

    O índice de um artigo já cadastrado permanece estável.
    Novos artigos recebem o índice da sua posição no manifesto.
    """
    records = load_manifest(manifest_path)
    inserted = 0

    for corpus_index, record in enumerate(records):
        metadata = extract_metadata(record, corpus_index)

        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO papers (
                arxiv_id,
                corpus_index,
                title,
                authors_json,
                abstract,
                published,
                categories_json,
                metadata_json,
                status,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
            """,
            (
                metadata["arxiv_id"],
                metadata["corpus_index"],
                metadata["title"],
                metadata["authors_json"],
                metadata["abstract"],
                metadata["published"],
                metadata["categories_json"],
                metadata["metadata_json"],
                utc_now(),
                utc_now(),
            ),
        )

        if cursor.rowcount:
            inserted += 1
            log_event(
                connection,
                metadata["arxiv_id"],
                "registered",
                f"corpus_index={corpus_index}",
            )

    connection.commit()

    logger.info(
        "Manifesto: %d registros; %d novos artigos importados.",
        len(records),
        inserted,
    )

    return inserted


# ---------------------------------------------------------------------------
# Validação de PDFs
# ---------------------------------------------------------------------------

def validate_pdf(path: Path) -> bool:
    """
    Valida o arquivo antes de considerá-lo baixado.

    Quando PyMuPDF está instalado, verifica também se o documento pode
    ser aberto e contém pelo menos uma página.
    """
    try:
        if not path.is_file():
            return False

        if path.stat().st_size < MIN_PDF_SIZE:
            return False

        with path.open("rb") as file:
            if file.read(5) != b"%PDF-":
                return False

        try:
            from typing import TYPE_CHECKING

            if TYPE_CHECKING:
                # For type checkers only
                import fitz  # type: ignore
            else:
                try:
                    import fitz  # type: ignore
                except Exception:  # pragma: no cover - runtime may lack PyMuPDF
                    fitz = None  # type: ignore

            with fitz.open(path) as document:
                if document.page_count < 1:
                    return False

        except ImportError:
            logger.debug(
                "PyMuPDF não instalado; validação limitada à assinatura PDF."
            )

        return True

    except Exception as exc:
        logger.debug("PDF inválido (%s): %s", path, exc)
        return False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def reconcile_existing_pdfs(
    connection: sqlite3.Connection,
    pdf_dir: Path,
) -> int:
    """
    Reconcilia o banco com os arquivos já presentes no disco.

    Não rebaixa artigos que já foram extraídos ou processados.
    """
    rows = connection.execute(
        """
        SELECT arxiv_id, status
        FROM papers
        ORDER BY corpus_index
        """
    ).fetchall()

    recovered = 0

    for row in rows:
        paper_id = row["arxiv_id"]
        status = row["status"]

        pdf_path = pdf_dir / f"{safe_filename(paper_id)}.pdf"

        if not pdf_path.exists():
            continue

        if not validate_pdf(pdf_path):
            logger.warning("PDF existente inválido: %s", pdf_path)
            continue

        if status in ("pending", "downloading", "failed_download"):
            size = pdf_path.stat().st_size
            digest = sha256_file(pdf_path)

            connection.execute(
                """
                UPDATE papers
                SET status = 'downloaded',
                    pdf_path = ?,
                    pdf_size = ?,
                    pdf_sha256 = ?,
                    downloaded_at = ?,
                    updated_at = ?,
                    last_error = NULL
                WHERE arxiv_id = ?
                """,
                (
                    str(pdf_path),
                    size,
                    digest,
                    utc_now(),
                    utc_now(),
                    paper_id,
                ),
            )

            log_event(
                connection,
                paper_id,
                "pdf_reconciled",
                f"size={size}",
            )

            recovered += 1

    connection.commit()

    if recovered:
        logger.info(
            "%d PDFs existentes foram reconhecidos pelo banco.",
            recovered,
        )

    return recovered


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def pdf_url_for(paper: sqlite3.Row) -> str:
    """
    Usa uma URL explícita do manifesto quando disponível.
    Caso contrário, utiliza o endpoint oficial do arXiv.
    """
    metadata = json.loads(paper["metadata_json"])

    url = get_field(
        metadata,
        "pdf_url",
        "pdf",
        default=None,
    )

    if url:
        parsed = urlparse(str(url))

        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"URL de PDF inválida: {url!r}")

        return str(url)

    return f"https://arxiv.org/pdf/{paper['arxiv_id']}"


def download_pdf(
    connection: sqlite3.Connection,
    paper: sqlite3.Row,
    pdf_dir: Path,
    *,
    timeout: int,
    attempts: int,
) -> bool:
    paper_id = paper["arxiv_id"]
    corpus_index = paper["corpus_index"]

    destination = pdf_dir / f"{safe_filename(paper_id)}.pdf"
    temporary = destination.with_suffix(".pdf.part")

    pdf_dir.mkdir(parents=True, exist_ok=True)

    # Verificação imediatamente antes do download.
    # Protege contra arquivos já existentes, inclusive quando o banco
    # ainda não foi sincronizado com o diretório.
    if validate_pdf(destination):
        size = destination.stat().st_size
        digest = sha256_file(destination)

        connection.execute(
            """
            UPDATE papers
            SET status = CASE
                    WHEN status IN (
                        'pending',
                        'downloading',
                        'failed_download'
                    ) THEN 'downloaded'
                    ELSE status
                END,
                pdf_path = ?,
                pdf_size = ?,
                pdf_sha256 = ?,
                downloaded_at = COALESCE(downloaded_at, ?),
                updated_at = ?,
                last_error = NULL
            WHERE arxiv_id = ?
            """,
            (
                str(destination),
                size,
                digest,
                utc_now(),
                utc_now(),
                paper_id,
            ),
        )

        log_event(connection, paper_id, "pdf_already_present")
        advance_index(connection, corpus_index)
        connection.commit()

        logger.info(
            "[%d] %s — PDF já existe; download ignorado.",
            corpus_index,
            paper_id,
        )

        return True

    # Um arquivo parcial não é um download válido.
    temporary.unlink(missing_ok=True)

    connection.execute(
        """
        UPDATE papers
        SET status = 'downloading',
            download_attempts = download_attempts + 1,
            updated_at = ?,
            last_error = NULL
        WHERE arxiv_id = ?
        """,
        (utc_now(), paper_id),
    )

    log_event(connection, paper_id, "download_started")
    connection.commit()

    url = pdf_url_for(paper)
    last_error: str | None = None

    for attempt in range(1, attempts + 1):
        try:
            logger.info(
                "[%d] %s — download (tentativa %d/%d)",
                corpus_index,
                paper_id,
                attempt,
                attempts,
            )

            with requests.get(
                if fitz is None:
                    logger.debug(
                        "PyMuPDF não instalado; validação limitada à assinatura PDF."
                    )
                else:
                    try:
                        with fitz.open(path) as document:
                            if document.page_count < 1:
                                return False
                    except Exception as exc:
                        logger.debug("Erro abrindo PDF com PyMuPDF (%s): %s", path, exc)
                        return False

                with temporary.open("wb") as file:
                    for chunk in response.iter_content(
                        chunk_size=1024 * 128
                    ):
                        if chunk:
                            file.write(chunk)

            if not validate_pdf(temporary):
                raise ValueError(
                    "A resposta recebida não é um PDF válido."
                )

            size = temporary.stat().st_size
            digest = sha256_file(temporary)

            # Só publicamos o arquivo depois da validação.
            temporary.replace(destination)

            connection.execute(
                """
                UPDATE papers
                SET status = 'downloaded',
                    pdf_path = ?,
                    pdf_size = ?,
                    pdf_sha256 = ?,
                    downloaded_at = ?,
                    updated_at = ?,
                    last_error = NULL
                WHERE arxiv_id = ?
                """,
                (
                    str(destination),
                    size,
                    digest,
                    utc_now(),
                    utc_now(),
                    paper_id,
                ),
            )

            log_event(
                connection,
                paper_id,
                "download_succeeded",
                f"bytes={size}; sha256={digest}",
            )

            advance_index(connection, corpus_index)
            connection.commit()

            logger.info(
                "[%d] %s — concluído (%.1f KiB).",
                corpus_index,
                paper_id,
                size / 1024,
            )

            return True

        except Exception as exc:
            last_error = str(exc)

            logger.warning(
                "%s — tentativa %d falhou: %s",
                paper_id,
                attempt,
                exc,
            )

            temporary.unlink(missing_ok=True)

            if attempt < attempts:
                time.sleep(min(2 ** (attempt - 1), 8))

    connection.execute(
        """
        UPDATE papers
        SET status = 'failed_download',
            last_error = ?,
            updated_at = ?
        WHERE arxiv_id = ?
        """,
        (last_error, utc_now(), paper_id),
    )

    log_event(
        connection,
        paper_id,
        "download_failed",
        last_error,
    )

    advance_index(connection, corpus_index)
    connection.commit()

    logger.error(
        "[%d] %s — falha definitiva nesta execução.",
        corpus_index,
        paper_id,
    )

    return False


# ---------------------------------------------------------------------------
# Comandos
# ---------------------------------------------------------------------------

def command_run(args: argparse.Namespace) -> None:
    if args.limit < 1:
        raise ValueError("--limit deve ser maior que zero.")

    connection = connect_database(args.db)

    try:
        initialize_database(connection)
        import_manifest(connection, args.manifest)

        reconcile_existing_pdfs(connection, args.pdf_dir)

        cursor = get_next_index(connection)

        logger.info("Próximo índice registrado: %d", cursor)

        if args.retry_failures:
            rows = connection.execute(
                """
                SELECT *
                FROM papers
                WHERE (
                    status = 'pending'
                    AND corpus_index >= ?
                )
                OR status = 'failed_download'
                ORDER BY corpus_index
                LIMIT ?
                """,
                (cursor, args.limit),
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT *
                FROM papers
                WHERE status = 'pending'
                  AND corpus_index >= ?
                ORDER BY corpus_index
                LIMIT ?
                """,
                (cursor, args.limit),
            ).fetchall()

        if not rows:
            logger.info("Nenhum artigo elegível para download.")
            print_status(connection)
            return

        logger.info(
            "Processando %d artigos nesta execução.",
            len(rows),
        )

        succeeded = 0
        failed = 0

        for row in rows:
            # Reconsulta para obter o estado mais recente.
            current = connection.execute(
                "SELECT * FROM papers WHERE arxiv_id = ?",
                (row["arxiv_id"],),
            ).fetchone()

            if current is None:
                continue

            # Se outro processo já o concluiu, não o baixa novamente.
            if current["status"] not in (
                "pending",
                "failed_download",
            ):
                continue

            ok = download_pdf(
                connection,
                current,
                args.pdf_dir,
                timeout=args.timeout,
                attempts=args.attempts,
            )

            if ok:
                succeeded += 1
            else:
                failed += 1

        print()
        logger.info(
            "Execução encerrada: %d baixados/recuperados; %d falhas.",
            succeeded,
            failed,
        )

        print_status(connection)

    finally:
        connection.close()


def command_sync(args: argparse.Namespace) -> None:
    connection = connect_database(args.db)

    try:
        initialize_database(connection)
        import_manifest(connection, args.manifest)
        reconcile_existing_pdfs(connection, args.pdf_dir)
        print_status(connection)

    finally:
        connection.close()


def print_status(connection: sqlite3.Connection) -> None:
    rows = connection.execute(
        """
        SELECT status, COUNT(*) AS total
        FROM papers
        GROUP BY status
        ORDER BY status
        """
    ).fetchall()

    total = connection.execute(
        "SELECT COUNT(*) FROM papers"
    ).fetchone()[0]

    next_index = get_next_index(connection)

    print("\n=== Estado do corpus ===")
    print(f"Artigos registrados : {total}")
    print(f"Próximo índice      : {next_index}")

    for row in rows:
        print(f"{row['status']:20s}: {row['total']}")

    print("========================\n")


def command_status(args: argparse.Namespace) -> None:
    if not args.db.exists():
        print(
            "Banco ainda não inicializado. "
            "Execute primeiro o comando 'run' ou 'sync'."
        )
        return

    with closing(connect_database(args.db)) as connection:
        initialize_database(connection)
        print_status(connection)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gerenciador incremental do corpus arXiv."
    )

    parser.add_argument(
        "--manifest",
        type=project_path,
        default=PROJECT_ROOT / "data" / "corpus" / "manifest.jsonl",
        help="Caminho do manifesto JSONL a ser processado.",
    )

    parser.add_argument(
        "--db",
        type=project_path,
        default=PROJECT_ROOT / "data" / "corpus" / "pipeline.sqlite3",
        help="Banco SQLite que registra o estado da ingestão.",
    )

    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=DEFAULT_PDF_DIR,
        help="Diretório onde os PDFs são armazenados.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Importa metadados e baixa os próximos artigos pendentes.",
    )

    run_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Número máximo de artigos por execução (padrão: 5).",
    )

    run_parser.add_argument(
        "--retry-failures",
        action="store_true",
        help="Inclui artigos que falharam em downloads anteriores.",
    )

    run_parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Timeout de cada requisição em segundos.",
    )

    run_parser.add_argument(
        "--attempts",
        type=int,
        default=DEFAULT_ATTEMPTS,
        help="Tentativas por artigo.",
    )

    run_parser.set_defaults(func=command_run)

    sync_parser = subparsers.add_parser(
        "sync",
        help="Importa metadados e reconhece PDFs existentes, sem baixar.",
    )

    sync_parser.set_defaults(func=command_sync)

    status_parser = subparsers.add_parser(
        "status",
        help="Mostra o estado atual do corpus.",
    )

    status_parser.set_defaults(func=command_status)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        args.func(args)
        return 0

    except KeyboardInterrupt:
        logger.warning(
            "Interrompido pelo usuário. "
            "Downloads incompletos não serão tratados como válidos."
        )
        return 130

    except Exception as exc:
        logger.exception("Erro fatal: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
