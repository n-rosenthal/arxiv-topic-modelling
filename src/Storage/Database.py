"""src.Storage.Database
Camada de persistência em SQLite para o pipeline de artigos.

Complementa (não substitui) o manifesto JSONL do CorpusBuilder: o
manifesto continua sendo a fonte de verdade portátil e legível
linha a linha; o banco de dados existe para consultas rápidas de
estado — quais arxiv_ids já são conhecidos, quais já têm PDF
baixado, quais já têm texto extraído — sem precisar reler e
reconstruir o manifesto inteiro a cada execução.

Três tabelas:
    papers:
        Metadados normalizados do arXiv (imutáveis após a coleta).
        Espelham diretamente a saída de `normalize_paper()`.
    processing:
        Estado mutável do pipeline por artigo: caminho do PDF,
        caminho do texto extraído, timestamps, erros, estágio.
    paper_features:
        Tabela extensível para features de PLN (bag-of-words,
        tf-idf, embeddings, ...). Guarda apenas ponteiros de
        arquivo (ex.: .npy/.npz/.json), não os vetores em si —
        SQLite não é um bom lugar para matrizes grandes, e um
        ponteiro evita migração de schema a cada novo tipo de
        feature.

Este módulo é deliberadamente desacoplado do dataclass `Paper`
(cujos `SourceInfo`/`ProcessingInfo`/`TopicAssignment` internos não
são conhecidos aqui): ele trabalha com os dicts normalizados do
arXiv e com `arxiv_id` como chave, sincronizando nas fronteiras do
pipeline em vez de espelhar campos de classes que podem mudar.
"""

from __future__ import annotations

import json
import sqlite3

from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional


CorpusRecord = dict[str, Any]


DEFAULT_DB_PATH = Path("data") / "corpus" / "papers.db"


SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id            TEXT PRIMARY KEY,
    versioned_arxiv_id  TEXT NOT NULL,
    title               TEXT NOT NULL,
    authors             TEXT NOT NULL,   -- JSON list[str]
    abstract            TEXT NOT NULL,
    primary_subject     TEXT,
    subjects            TEXT,            -- JSON list[str]
    submission_date     TEXT,
    updated_date        TEXT,
    doi                 TEXT,
    comment             TEXT,
    pdf_url             TEXT,
    abstract_url        TEXT,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS processing (
    arxiv_id              TEXT PRIMARY KEY REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    stage                 TEXT NOT NULL DEFAULT 'DOWNLOAD_PENDING',
    pdf_path              TEXT,
    pdf_downloaded_at     TEXT,
    pdf_download_error    TEXT,
    text_path             TEXT,
    text_extracted_at     TEXT,
    text_extraction_error TEXT,
    updated_at            TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_features (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    arxiv_id       TEXT NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    feature_name   TEXT NOT NULL,     -- ex.: 'bow', 'tfidf', 'embedding_minilm'
    feature_format TEXT NOT NULL,     -- ex.: 'npy', 'npz', 'json'
    feature_path   TEXT NOT NULL,
    dim            INTEGER,
    created_at     TEXT NOT NULL,
    UNIQUE(arxiv_id, feature_name)
);

CREATE INDEX IF NOT EXISTS idx_processing_stage ON processing(stage);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_connection(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Abre (criando o diretório, se necessário) a conexão com o banco."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


# Colunas de `processing` acrescentadas depois da primeira versão do schema.
# `_migrate` as cria em bancos antigos (ALTER TABLE) e é um no-op quando já
# existem; assim `init_db` continua idempotente e não exige recriar o banco.
_PROCESSING_EXTRA_COLUMNS: dict[str, str] = {
    "tei_path": "TEXT",                                # TEI XML bruto do GROBID
    "text_char_count": "INTEGER",
    "pdf_removed_at": "TEXT",                          # PDF apagado após extrair o texto
    "download_attempts": "INTEGER NOT NULL DEFAULT 0",
    "extraction_attempts": "INTEGER NOT NULL DEFAULT 0",
    "token_count": "INTEGER",                          # tokens após limpeza + NLPPreprocessor
    "section_count": "INTEGER",
    "reference_count": "INTEGER",
}


def _migrate(conn: sqlite3.Connection) -> None:
    """Adiciona a `processing` as colunas que um banco antigo ainda não tem."""
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(processing)")}

    for column, ddl in _PROCESSING_EXTRA_COLUMNS.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE processing ADD COLUMN {column} {ddl}")

    conn.commit()


def init_db(conn: sqlite3.Connection) -> None:
    """Cria as tabelas, se ainda não existirem, e aplica migrações. Idempotente."""
    with closing(conn.cursor()) as cur:
        cur.executescript(SCHEMA)
    conn.commit()
    _migrate(conn)


# ---------------------------------------------------------------------------
# papers
# ---------------------------------------------------------------------------

def upsert_paper(conn: sqlite3.Connection, record: CorpusRecord) -> None:
    """
    Insere ou atualiza um registro normalizado do arXiv.

    Aceita diretamente a saída de `normalize_paper()`. `created_at`
    é preservado em atualizações; `updated_at` é sempre renovado.
    Também garante a linha correspondente em `processing`, para que
    as consultas de pendência funcionem mesmo antes de qualquer
    tentativa de download.
    """
    now = _now()

    conn.execute(
        """
        INSERT INTO papers (
            arxiv_id, versioned_arxiv_id, title, authors, abstract,
            primary_subject, subjects, submission_date, updated_date,
            doi, comment, pdf_url, abstract_url, created_at, updated_at
        ) VALUES (
            :arxiv_id, :versioned_arxiv_id, :title, :authors, :abstract,
            :primary_subject, :subjects, :submission_date, :updated_date,
            :doi, :comment, :pdf_url, :abstract_url, :created_at, :updated_at
        )
        ON CONFLICT(arxiv_id) DO UPDATE SET
            versioned_arxiv_id = excluded.versioned_arxiv_id,
            title              = excluded.title,
            authors            = excluded.authors,
            abstract           = excluded.abstract,
            primary_subject    = excluded.primary_subject,
            subjects           = excluded.subjects,
            submission_date    = excluded.submission_date,
            updated_date       = excluded.updated_date,
            doi                = excluded.doi,
            comment            = excluded.comment,
            pdf_url            = excluded.pdf_url,
            abstract_url       = excluded.abstract_url,
            updated_at         = excluded.updated_at
        """,
        {
            "arxiv_id": record["arxiv_id"],
            "versioned_arxiv_id": record.get("versioned_arxiv_id"),
            "title": record.get("title"),
            "authors": json.dumps(record.get("authors") or [], ensure_ascii=False),
            "abstract": record.get("abstract"),
            "primary_subject": record.get("primary_subject"),
            "subjects": json.dumps(record.get("subjects") or [], ensure_ascii=False),
            "submission_date": record.get("submission_date"),
            "updated_date": record.get("updated_date"),
            "doi": record.get("doi"),
            "comment": record.get("comment"),
            "pdf_url": record.get("pdf_url"),
            "abstract_url": record.get("abstract_url"),
            "created_at": now,
            "updated_at": now,
        },
    )

    conn.execute(
        """
        INSERT INTO processing (arxiv_id, updated_at)
        VALUES (:arxiv_id, :updated_at)
        ON CONFLICT(arxiv_id) DO NOTHING
        """,
        {"arxiv_id": record["arxiv_id"], "updated_at": now},
    )


def upsert_many(conn: sqlite3.Connection, records: Iterable[CorpusRecord]) -> None:
    """Insere ou atualiza vários registros em uma única transação."""
    with conn:
        for record in records:
            upsert_paper(conn, record)


def get_known_ids(conn: sqlite3.Connection) -> set[str]:
    """Retorna todos os arxiv_ids já conhecidos pelo banco."""
    cur = conn.execute("SELECT arxiv_id FROM papers")
    return {row["arxiv_id"] for row in cur.fetchall()}


def get_paper(conn: sqlite3.Connection, arxiv_id: str) -> Optional[sqlite3.Row]:
    cur = conn.execute("SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,))
    return cur.fetchone()


def get_processing(conn: sqlite3.Connection, arxiv_id: str) -> Optional[sqlite3.Row]:
    """Linha de estado (`processing`) de um artigo, ou None."""
    cur = conn.execute("SELECT * FROM processing WHERE arxiv_id = ?", (arxiv_id,))
    return cur.fetchone()


def row_to_record(row: sqlite3.Row) -> CorpusRecord:
    """
    Converte uma linha de `papers` de volta no dict normalizado que
    `normalize_paper()` produz (mesmas chaves, listas decodificadas).
    """
    record = {
        key: row[key]
        for key in row.keys()
        if key not in ("created_at", "updated_at")
    }
    record["authors"] = json.loads(record["authors"] or "[]")
    record["subjects"] = json.loads(record["subjects"] or "[]")
    return record


def get_record(conn: sqlite3.Connection, arxiv_id: str) -> Optional[CorpusRecord]:
    """Registro normalizado de um artigo, no formato do manifesto."""
    row = get_paper(conn, arxiv_id)
    return row_to_record(row) if row is not None else None


def get_work_queue(
    conn: sqlite3.Connection,
    limit: Optional[int] = None,
    max_attempts: int = 3,
) -> list[str]:
    """
    arxiv_ids que ainda precisam de texto, na ordem de processamento.

    Primeiro vêm os que já têm PDF em disco (falta só extrair, sem
    gastar o intervalo entre requisições do download); depois os que
    ainda precisam ser baixados. Quem já esgotou `max_attempts` de
    download ou de extração fica de fora, para que um PDF
    irrecuperável não seja retentado a cada execução.
    """
    cur = conn.execute(
        """
        SELECT arxiv_id FROM processing
        WHERE text_path IS NULL
          AND download_attempts < :max_attempts
          AND extraction_attempts < :max_attempts
        ORDER BY (pdf_path IS NULL), arxiv_id
        LIMIT :limit
        """,
        {"max_attempts": max_attempts, "limit": -1 if limit is None else limit},
    )
    return [row["arxiv_id"] for row in cur.fetchall()]


def get_prunable_pdfs(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """
    (arxiv_id, pdf_path) dos PDFs que sobraram em disco de papers que já
    têm texto e TEI registrados — ou seja, que podem ser refeitos sem o PDF.
    """
    cur = conn.execute(
        """
        SELECT arxiv_id, pdf_path FROM processing
        WHERE pdf_path IS NOT NULL
          AND text_path IS NOT NULL
          AND tei_path IS NOT NULL
        ORDER BY arxiv_id
        """
    )
    return [(row["arxiv_id"], row["pdf_path"]) for row in cur.fetchall()]


def get_reprocessable(
    conn: sqlite3.Connection,
    limit: Optional[int] = None,
) -> list[str]:
    """
    arxiv_ids com texto pronto e TEI registrado: podem ter limpeza e
    tokens refeitos a partir do TEI, sem PDF e sem GROBID (útil depois
    de mudar o limpador, as stopwords ou a lematização).
    """
    cur = conn.execute(
        """
        SELECT arxiv_id FROM processing
        WHERE text_path IS NOT NULL AND tei_path IS NOT NULL
        ORDER BY arxiv_id
        LIMIT :limit
        """,
        {"limit": -1 if limit is None else limit},
    )
    return [row["arxiv_id"] for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# processing / PDFs
# ---------------------------------------------------------------------------

def mark_pdf_downloaded(conn: sqlite3.Connection, arxiv_id: str, pdf_path: Path | str) -> None:
    now = _now()
    with conn:
        conn.execute(
            """
            UPDATE processing
            SET pdf_path = ?, pdf_downloaded_at = ?, pdf_download_error = NULL,
                stage = 'PDF_AVAILABLE', updated_at = ?
            WHERE arxiv_id = ?
            """,
            (str(pdf_path), now, now, arxiv_id),
        )


def mark_pdf_failed(conn: sqlite3.Connection, arxiv_id: str, error: str) -> None:
    now = _now()
    with conn:
        conn.execute(
            """
            UPDATE processing
            SET pdf_download_error = ?, stage = 'INVALID_DOWNLOAD', updated_at = ?,
                download_attempts = download_attempts + 1
            WHERE arxiv_id = ?
            """,
            (error, now, arxiv_id),
        )


def get_pending_pdf_downloads(
    conn: sqlite3.Connection,
    arxiv_ids: Optional[Iterable[str]] = None,
    max_attempts: Optional[int] = None,
) -> list[str]:
    """
    Retorna os arxiv_ids que ainda precisam de download: sem texto
    extraído e sem PDF registrado.

    Um paper cujo texto já foi extraído nunca é pendente, mesmo com
    `pdf_path` vazio — o PDF é removido de propósito depois da
    extração, e baixá-lo de novo seria trabalho jogado fora.

    Args:
        arxiv_ids:
            Restringe a checagem a este subconjunto (ex.: os papers
            da execução atual). Caso contrário, considera todo o banco.
        max_attempts:
            Ignora papers cujo download já falhou esta quantidade
            de vezes (ou mais).

    Esta função confia apenas no registro em banco; chame
    `reconcile_pdfs` antes para detectar PDFs apagados à mão.
    """
    conditions = ["text_path IS NULL", "pdf_path IS NULL"]
    params: list[Any] = []

    if max_attempts is not None:
        conditions.append("download_attempts < ?")
        params.append(max_attempts)

    if arxiv_ids is not None:
        ids = list(arxiv_ids)
        if not ids:
            return []

        conditions.append(f"arxiv_id IN ({','.join('?' for _ in ids)})")
        params.extend(ids)

    cur = conn.execute(
        f"SELECT arxiv_id FROM processing WHERE {' AND '.join(conditions)}",
        params,
    )
    return [row["arxiv_id"] for row in cur.fetchall()]


def verify_pdf_on_disk(conn: sqlite3.Connection, arxiv_id: str) -> bool:
    """
    Confirma que o PDF registrado para `arxiv_id` ainda existe no
    disco. Se o registro apontar para um arquivo ausente (ex.:
    apagado manualmente), o campo é limpo e a função retorna False,
    fazendo o paper voltar a ser considerado pendente.
    """
    row = conn.execute(
        "SELECT pdf_path FROM processing WHERE arxiv_id = ?",
        (arxiv_id,),
    ).fetchone()

    if row is None or row["pdf_path"] is None:
        return False

    if Path(row["pdf_path"]).exists():
        return True

    with conn:
        conn.execute(
            """
            UPDATE processing
            SET pdf_path = NULL, pdf_downloaded_at = NULL, stage = 'DOWNLOAD_PENDING'
            WHERE arxiv_id = ?
            """,
            (arxiv_id,),
        )
    return False


def reconcile_pdfs(conn: sqlite3.Connection) -> int:
    """
    Faz o banco concordar com o disco: papers sem texto cujo PDF
    registrado não existe mais (apagado à mão, disco trocado, ...)
    voltam a ficar pendentes de download.

    Papers com texto já extraído são ignorados — para eles a ausência
    do PDF é o estado esperado. Retorna quantos registros foram
    corrigidos.
    """
    rows = conn.execute(
        "SELECT arxiv_id, pdf_path FROM processing "
        "WHERE pdf_path IS NOT NULL AND text_path IS NULL"
    ).fetchall()

    missing = [row["arxiv_id"] for row in rows if not Path(row["pdf_path"]).is_file()]

    if missing:
        now = _now()
        with conn:
            conn.executemany(
                """
                UPDATE processing
                SET pdf_path = NULL, pdf_downloaded_at = NULL,
                    stage = 'DOWNLOAD_PENDING', updated_at = ?
                WHERE arxiv_id = ?
                """,
                [(now, arxiv_id) for arxiv_id in missing],
            )

    return len(missing)


def mark_pdf_removed(conn: sqlite3.Connection, arxiv_id: str) -> None:
    """Registra que o PDF local foi apagado depois de extrair o texto."""
    now = _now()
    with conn:
        conn.execute(
            """
            UPDATE processing
            SET pdf_path = NULL, pdf_removed_at = ?, updated_at = ?
            WHERE arxiv_id = ?
            """,
            (now, now, arxiv_id),
        )


# ---------------------------------------------------------------------------
# processing / texto extraído
# ---------------------------------------------------------------------------

def mark_text_extracted(
    conn: sqlite3.Connection,
    arxiv_id: str,
    text_path: Path | str,
    tei_path: Optional[Path | str] = None,
    char_count: Optional[int] = None,
    token_count: Optional[int] = None,
    section_count: Optional[int] = None,
    reference_count: Optional[int] = None,
) -> None:
    """
    Registra o texto extraído. `text_path` aponta o artefato de texto
    principal (hoje, o JSON de seções do `ScientificPaperProcessor`,
    com texto bruto, texto limpo e tokens por seção). `tei_path` é
    opcional e só é preenchido por quem guarda o TEI do GROBID à parte.

    As contagens são verificações baratas de sanidade (um paper com 0
    seções ou 0 tokens quase sempre indica extração ruim).
    """
    now = _now()
    with conn:
        conn.execute(
            """
            UPDATE processing
            SET text_path = ?, tei_path = ?, text_char_count = ?,
                token_count = ?, section_count = ?, reference_count = ?,
                text_extracted_at = ?, text_extraction_error = NULL,
                stage = 'TEXT_AVAILABLE', updated_at = ?
            WHERE arxiv_id = ?
            """,
            (
                str(text_path),
                str(tei_path) if tei_path is not None else None,
                char_count,
                token_count,
                section_count,
                reference_count,
                now,
                now,
                arxiv_id,
            ),
        )


def mark_text_extraction_failed(conn: sqlite3.Connection, arxiv_id: str, error: str) -> None:
    now = _now()
    with conn:
        conn.execute(
            """
            UPDATE processing
            SET text_extraction_error = ?, updated_at = ?,
                extraction_attempts = extraction_attempts + 1
            WHERE arxiv_id = ?
            """,
            (error, now, arxiv_id),
        )


def get_pending_text_extractions(conn: sqlite3.Connection) -> list[str]:
    """arxiv_ids com PDF disponível mas sem texto extraído ainda."""
    cur = conn.execute(
        """
        SELECT arxiv_id FROM processing
        WHERE pdf_path IS NOT NULL AND text_path IS NULL
        """
    )
    return [row["arxiv_id"] for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# paper_features (bag-of-words, tf-idf, embeddings, ...)
# ---------------------------------------------------------------------------

def upsert_feature(
    conn: sqlite3.Connection,
    arxiv_id: str,
    feature_name: str,
    feature_format: str,
    feature_path: Path | str,
    dim: Optional[int] = None,
) -> None:
    """
    Registra (ou substitui) uma feature de PLN para um artigo.

    O vetor/matriz em si não é armazenado no banco — apenas o
    caminho do arquivo (.npy, .npz, .json, ...) e sua dimensão,
    quando aplicável. `feature_name` identifica a versão/método
    (ex.: 'tfidf_v1', 'embedding_minilm_l6'), permitindo registrar
    várias features diferentes para o mesmo artigo.
    """
    now = _now()
    with conn:
        conn.execute(
            """
            INSERT INTO paper_features (
                arxiv_id, feature_name, feature_format, feature_path, dim, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(arxiv_id, feature_name) DO UPDATE SET
                feature_format = excluded.feature_format,
                feature_path   = excluded.feature_path,
                dim            = excluded.dim,
                created_at     = excluded.created_at
            """,
            (arxiv_id, feature_name, feature_format, str(feature_path), dim, now),
        )


def get_features(conn: sqlite3.Connection, arxiv_id: str) -> list[sqlite3.Row]:
    cur = conn.execute(
        "SELECT * FROM paper_features WHERE arxiv_id = ?",
        (arxiv_id,),
    )
    return cur.fetchall()


# ---------------------------------------------------------------------------
# Relatório de estado
# ---------------------------------------------------------------------------

def get_summary(conn: sqlite3.Connection) -> dict[str, int]:
    """
    Visão geral rápida do estado do corpus: total de artigos
    conhecidos e progresso de download de PDF / extração de texto.
    """
    def count(where: str) -> int:
        return conn.execute(
            f"SELECT COUNT(*) AS n FROM processing WHERE {where}"
        ).fetchone()["n"]

    total = conn.execute("SELECT COUNT(*) AS n FROM papers").fetchone()["n"]
    text_done = count("text_path IS NOT NULL")

    # Como o PDF é apagado depois da extração, "baixado" deixou de ser
    # uma medida de progresso: o que importa é ter texto. PDFs em disco
    # de papers que já têm texto sobram de execuções com --keep-pdfs
    # (`pdf_on_disk_done`) e podem ser apagados com --prune-pdfs.
    return {
        "total_papers": total,
        "text_extracted": text_done,
        "text_pending": total - text_done,
        "text_failed": count(
            "text_path IS NULL AND text_extraction_error IS NOT NULL"
        ),
        "pdf_on_disk": count("pdf_path IS NOT NULL"),
        "pdf_on_disk_done": count(
            "pdf_path IS NOT NULL AND text_path IS NOT NULL"
        ),
        "pdf_pending": count("text_path IS NULL AND pdf_path IS NULL"),
        "pdf_failed": count(
            "text_path IS NULL AND pdf_path IS NULL "
            "AND pdf_download_error IS NOT NULL"
        ),
    }


def get_category_counts(conn: sqlite3.Connection) -> list[tuple[str, int]]:
    """Quantidade de artigos por `primary_subject`, do maior para o menor."""
    cur = conn.execute(
        """
        SELECT COALESCE(primary_subject, '(sem categoria)') AS category, COUNT(*) AS n
        FROM papers
        GROUP BY category
        ORDER BY n DESC
        """
    )
    return [(row["category"], row["n"]) for row in cur.fetchall()]


def _range_label(start: str, end: str) -> str:
    """
    Rótulo curto para um intervalo 'YYYYMMDD'/'YYYYMMDD', usado como
    cabeçalho de coluna no relatório (ex.: '2000-2003').

    Quando o intervalo não começa/termina em 1º de janeiro / 31 de
    dezembro, os meses e dias são preservados para não esconder a
    granularidade real do intervalo (ex.: '20230601-20230831' vira
    '2023/06/01-2023/08/31').
    """
    def compact_to_label(value: str) -> str:
        year, month, day = value[:4], value[4:6], value[6:8]
        if (month, day) in (("01", "01"), ("12", "31")):
            return year
        return f"{year}/{month}/{day}"

    return f"{compact_to_label(start)}-{compact_to_label(end)}"


def _bucket_for_date(
    date_ranges: list[tuple[str, str]],
    submission_date: Optional[str],
) -> str:
    """
    Classifica um `submission_date` ISO 8601 (ex.:
    '2023-05-10T00:00:00+00:00') em um dos intervalos 'YYYYMMDD'
    de `date_ranges` (o mesmo formato de `CorpusBuilder.DATE_RANGES`).

    Retorna o rótulo curto (ver `_range_label`) do intervalo
    correspondente, 'desconhecido' se não houver data, ou 'outro' se
    a data não cair em nenhum dos intervalos configurados.
    """
    if not submission_date:
        return "desconhecido"

    compact = submission_date[:10].replace("-", "")

    for start, end in date_ranges:
        if start <= compact <= end:
            return _range_label(start, end)

    return "outro"


def get_category_date_matrix(
    conn: sqlite3.Connection,
    date_ranges: list[tuple[str, str]],
) -> tuple[list[str], dict[str, dict[str, int]]]:
    """
    Cruza categoria (`primary_subject`) com intervalo de submissão.

    Retorna (colunas, matriz): `colunas` é a lista de rótulos de
    intervalo, na ordem de `date_ranges`, seguida de 'outro' e/ou
    'desconhecido' quando aplicável; `matriz` mapeia
    categoria -> rótulo de coluna -> contagem.
    """
    columns = [_range_label(s, e) for s, e in date_ranges]
    extra_columns: list[str] = []

    cur = conn.execute(
        "SELECT COALESCE(primary_subject, '(sem categoria)') AS category, "
        "submission_date FROM papers"
    )

    matrix: dict[str, dict[str, int]] = {}

    for row in cur.fetchall():
        category = row["category"]
        label = _bucket_for_date(date_ranges, row["submission_date"])

        if label not in columns and label not in extra_columns:
            extra_columns.append(label)

        matrix.setdefault(category, {})
        matrix[category][label] = matrix[category].get(label, 0) + 1

    return columns + extra_columns, matrix


def get_date_range_totals(
    conn: sqlite3.Connection,
    date_ranges: list[tuple[str, str]],
) -> dict[tuple[str, str], int]:
    """
    Conta quantos artigos caem em cada intervalo de `date_ranges`,
    somando todas as categorias.

    Útil para identificar lacunas de cobertura temporal: como
    `CorpusBuilder.build_corpus` para assim que o `limit` global é
    atingido, intervalos percorridos por último (os mais recentes,
    já que `DATE_RANGES` é cronológico ascendente) podem nunca ter
    sido consultados e aparecem aqui com contagem zero.
    """
    totals: dict[tuple[str, str], int] = {(s, e): 0 for s, e in date_ranges}

    cur = conn.execute("SELECT submission_date FROM papers")

    for row in cur.fetchall():
        compact = (row["submission_date"] or "")[:10].replace("-", "")

        for start, end in date_ranges:
            if start <= compact <= end:
                totals[(start, end)] += 1
                break

    return totals


def get_empty_date_ranges(
    conn: sqlite3.Connection,
    date_ranges: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """
    Subconjunto de `date_ranges` sem nenhum artigo registrado no
    banco — candidatos naturais para `CorpusBuilder.backfill`.
    """
    totals = get_date_range_totals(conn, date_ranges)
    return [range_ for range_, count in totals.items() if count == 0]


def build_report(
    conn: sqlite3.Connection,
    date_ranges: Optional[list[tuple[str, str]]] = None,
) -> str:
    """
    Monta um relatório textual do estado atual do corpus: totais,
    progresso de PDF/texto, contagem por categoria e, se
    `date_ranges` for informado, a matriz categoria x intervalo de
    submissão.
    """
    summary = get_summary(conn)

    lines: list[str] = []
    lines.append("=== Estado do corpus ===")
    lines.append(f"Artigos conhecidos         : {summary['total_papers']}")
    lines.append(
        f"Textos extraídos (GROBID)  : {summary['text_extracted']} "
        f"({summary['text_pending']} pendente(s), "
        f"{summary['text_failed']} com erro)"
    )
    awaiting = summary["pdf_on_disk"] - summary["pdf_on_disk_done"]
    lines.append(
        f"PDFs em disco              : {summary['pdf_on_disk']} "
        f"({awaiting} aguardando extração, "
        f"{summary['pdf_on_disk_done']} de papers já processados)"
    )
    lines.append(
        f"PDFs a baixar              : {summary['pdf_pending']} "
        f"({summary['pdf_failed']} com erro de download)"
    )
    lines.append("")

    lines.append("=== Por categoria ===")
    for category, n in get_category_counts(conn):
        lines.append(f"  {category:<20} {n:>5}")
    lines.append("")

    if date_ranges:
        lines.append("=== Categoria x intervalo de submissão ===")

        columns, matrix = get_category_date_matrix(conn, date_ranges)

        col_width = 12
        header = (
            "categoria".ljust(20)
            + "".join(col.rjust(col_width) for col in columns)
            + "total".rjust(8)
        )
        lines.append(header)

        col_totals = {col: 0 for col in columns}
        grand_total = 0

        for category in sorted(matrix.keys()):
            row_counts = matrix[category]
            row_total = sum(row_counts.values())
            grand_total += row_total

            cells = []
            for col in columns:
                n = row_counts.get(col, 0)
                col_totals[col] += n
                cells.append(str(n).rjust(col_width))

            lines.append(
                category.ljust(20) + "".join(cells) + str(row_total).rjust(8)
            )

        totals_row = (
            "TOTAL".ljust(20)
            + "".join(str(col_totals[col]).rjust(col_width) for col in columns)
            + str(grand_total).rjust(8)
        )
        lines.append(totals_row)

    return "\n".join(lines)
