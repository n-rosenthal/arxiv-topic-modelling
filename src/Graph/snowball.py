"""
Expansão do corpus por citações ("snowball"), recursiva e com orçamento.

Rodada:
  1. Para cada paper do corpus ainda sem referências buscadas, pergunta ao
     Semantic Scholar quais referências têm arXiv ID (uma chamada por paper,
     sem baixar PDF) e soma os IDs que o GROBID já tinha extraído.
     Tudo vai para `citation_links` (cache: nunca busca duas vezes).
  2. Candidatos = IDs citados que ainda não estão no banco, ordenados por
     quantos papers do corpus os citam (--min-citations filtra o ruído).
  3. Busca os metadados no arXiv, filtra por categoria primária e insere em
     `papers` (entram na fila normal de download/extração do pipeline).
  4. Repete até --depth rodadas, respeitando --max-new por rodada.

Depois, rode `python -m src.pipe` para baixar e extrair os papers novos.

Uso:
    python -m src.Graph.snowball --depth 1 --min-citations 2 --max-new 500 --dry-run
    python -m src.Graph.snowball --depth 2 --min-citations 3 --max-new 1000

Chave do Semantic Scholar (opcional, recomendada): variável S2_API_KEY.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from src.Graph.citations import norm_arxiv
from src.Storage import Database

S2_URL = "https://api.semanticscholar.org/graph/v1"

SCHEMA = """
CREATE TABLE IF NOT EXISTS citation_links (
    source  TEXT NOT NULL,          -- paper do corpus que cita
    cited   TEXT NOT NULL,          -- arXiv ID citado (pode não estar no corpus)
    origin  TEXT NOT NULL,          -- 's2' | 'grobid'
    PRIMARY KEY (source, cited)
);
CREATE INDEX IF NOT EXISTS idx_citation_links_cited ON citation_links(cited);

CREATE TABLE IF NOT EXISTS reference_fetch (
    arxiv_id   TEXT PRIMARY KEY,
    fetched_at TEXT NOT NULL,
    n_refs     INTEGER,             -- total de referências conhecidas pelo S2
    n_arxiv    INTEGER,             -- quantas têm arXiv ID
    error      TEXT
);

CREATE TABLE IF NOT EXISTS paper_origin (
    arxiv_id TEXT PRIMARY KEY,      -- só papers descobertos por citação
    depth    INTEGER NOT NULL,
    cited_by INTEGER NOT NULL       -- nº de papers do corpus que o citavam
);

CREATE TABLE IF NOT EXISTS snowball_skipped (
    arxiv_id TEXT PRIMARY KEY,
    reason   TEXT NOT NULL          -- 'category:<x>' | 'not_found'
);
"""


def log(message: str) -> None:
    print(f"[snowball] {message}", flush=True)


def ensure_schema(conn) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------- Semantic Scholar

class S2Error(RuntimeError):
    pass


def fetch_s2_references(session, arxiv_id: str, api_key: str | None = None,
                        retries: int = 4, sleep: Callable[[float], None] = time.sleep):
    """
    Retorna (n_refs, [arxiv_ids]) ou None se o S2 não conhece o paper (404).
    Pagina de 1000 em 1000 e faz backoff em 429/5xx.
    """
    headers = {"x-api-key": api_key} if api_key else {}
    found: list[str] = []
    total, offset = 0, 0
    while True:
        for attempt in range(retries):
            response = session.get(
                f"{S2_URL}/paper/ARXIV:{arxiv_id}/references",
                params={"fields": "externalIds", "limit": 1000, "offset": offset},
                headers=headers, timeout=60,
            )
            if response.status_code == 404:
                return None
            if response.status_code in (429, 500, 502, 503, 504):
                retry_after = response.headers.get("Retry-After", "")
                wait = float(retry_after) if retry_after.replace(".", "", 1).isdigit() else 2 ** attempt * 2
                sleep(min(30, wait))
                continue
            response.raise_for_status()
            break
        else:
            raise S2Error(f"Semantic Scholar kept failing for {arxiv_id} (last HTTP {response.status_code})")

        payload = response.json()
        for item in payload.get("data") or []:
            total += 1
            external = (item.get("citedPaper") or {}).get("externalIds") or {}
            cited = norm_arxiv(external.get("ArXiv"))
            if cited:
                found.append(cited)
        if payload.get("next") is None:
            return total, found
        offset = payload["next"]


def grobid_arxiv_ids(sections_json: str | Path | None) -> list[str]:
    if not sections_json or not Path(sections_json).is_file():
        return []
    data = json.loads(Path(sections_json).read_text(encoding="utf-8"))
    ids = (norm_arxiv(r.get("arxiv_id")) for r in data.get("references", []))
    return [i for i in ids if i]


# ---------------------------------------------------------------- arXiv

def fetch_arxiv_metadata(ids: list[str], delay_seconds: float = 3.0) -> list[dict]:
    """Registros normalizados (mesmo formato do corpus) para uma lista de IDs."""
    import arxiv
    from src.Pipeline.Extraction.Extractor import normalize_paper

    client = arxiv.Client(page_size=100, delay_seconds=delay_seconds, num_retries=5)
    records = []
    for start in range(0, len(ids), 100):
        batch = ids[start:start + 100]
        search = arxiv.Search(id_list=batch, max_results=len(batch))
        records.extend(normalize_paper(r) for r in client.results(search))
    return records


# ---------------------------------------------------------------- rodadas

def papers_needing_references(conn, max_depth: int, limit: int | None = None) -> list[tuple[str, str | None]]:
    """Papers do corpus (depth < max_depth) cujas referências ainda não foram buscadas."""
    sql = """
        SELECT p.arxiv_id, pr.text_path
        FROM papers p
        LEFT JOIN processing pr ON pr.arxiv_id = p.arxiv_id
        LEFT JOIN paper_origin o ON o.arxiv_id = p.arxiv_id
        WHERE p.arxiv_id NOT IN (SELECT arxiv_id FROM reference_fetch)
          AND COALESCE(o.depth, 0) < ?
        ORDER BY COALESCE(o.depth, 0), p.arxiv_id
    """
    params: list = [max_depth]
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    return [tuple(r) for r in conn.execute(sql, params)]


def collect_references(conn, session, max_depth: int, api_key=None, interval: float = 1.0,
                       limit: int | None = None, fetcher=fetch_s2_references,
                       sleep: Callable[[float], None] = time.sleep, max_consecutive_errors: int = 5) -> dict:
    stats = {"fetched": 0, "unknown_to_s2": 0, "errors": 0}
    consecutive = 0
    todo = papers_needing_references(conn, max_depth, limit)
    log(f"{len(todo)} paper(s) without fetched references")
    for i, (arxiv_id, text_path) in enumerate(todo, 1):
        links = {(arxiv_id, c): "grobid" for c in grobid_arxiv_ids(text_path) if c != arxiv_id}
        n_refs = n_arxiv = error = None
        try:
            result = fetcher(session, arxiv_id, api_key)
        except Exception as exc:                       # erro de rede/S2: não grava, tenta na próxima execução
            stats["errors"] += 1
            consecutive += 1
            log(f"{arxiv_id}: {type(exc).__name__}: {exc}")
            if consecutive >= max_consecutive_errors:
                log(f"{consecutive} falhas seguidas (limite de taxa?). Parando; o que já foi buscado está salvo. "
                    f"Rode de novo mais tarde, de preferência com S2_API_KEY e --interval maior.")
                stats["aborted"] = True
                return stats
            sleep(interval)
            continue
        consecutive = 0
        if result is None:
            stats["unknown_to_s2"] += 1
            error = "not_in_s2"
        else:
            n_refs, ids = result
            n_arxiv = len(ids)
            for c in ids:
                if c != arxiv_id:
                    links[(arxiv_id, c)] = "s2"
            stats["fetched"] += 1

        with conn:
            for (s, c), origin in links.items():
                conn.execute("INSERT INTO citation_links (source, cited, origin) VALUES (?,?,?) "
                             "ON CONFLICT(source, cited) DO NOTHING", (s, c, origin))
            conn.execute("INSERT OR REPLACE INTO reference_fetch VALUES (?,?,?,?,?)",
                         (arxiv_id, _now(), n_refs, n_arxiv, error))
        if i % 100 == 0:
            log(f"  {i}/{len(todo)} referências buscadas")
        sleep(interval)
    return stats


def candidates(conn, min_citations: int, max_new: int) -> list[tuple[str, int, int]]:
    """(arxiv_id, nº de papers do corpus que o citam, profundidade nova)."""
    rows = conn.execute(
        """
        SELECT l.cited, COUNT(DISTINCT l.source) AS n,
               MIN(COALESCE(o.depth, 0)) + 1 AS depth
        FROM citation_links l
        LEFT JOIN paper_origin o ON o.arxiv_id = l.source
        WHERE l.cited NOT IN (SELECT arxiv_id FROM papers)
          AND l.cited NOT IN (SELECT arxiv_id FROM snowball_skipped)
        GROUP BY l.cited HAVING n >= ?
        ORDER BY n DESC, l.cited LIMIT ?
        """,
        (min_citations, max_new),
    ).fetchall()
    return [tuple(r) for r in rows]


def add_candidates(conn, cands, categories: tuple[str, ...] | None,
                   metadata_fetcher: Callable[[list[str]], Iterable[dict]] = fetch_arxiv_metadata) -> dict:
    stats = {"added": 0, "skipped_category": 0, "not_found": 0}
    info = {a: (n, d) for a, n, d in cands}
    records = list(metadata_fetcher(list(info)))
    returned = set()
    to_add = []
    with conn:
        for rec in records:
            arxiv_id = norm_arxiv(rec["arxiv_id"])
            if arxiv_id not in info:
                continue
            returned.add(arxiv_id)
            primary = rec.get("primary_subject") or ""
            if categories and not any(primary == c or primary.startswith(c + ".") or primary.startswith(c)
                                      for c in categories):
                conn.execute("INSERT OR REPLACE INTO snowball_skipped VALUES (?,?)",
                             (arxiv_id, f"category:{primary}"))
                stats["skipped_category"] += 1
                continue
            to_add.append((rec, arxiv_id))
        for arxiv_id in info.keys() - returned:
            conn.execute("INSERT OR REPLACE INTO snowball_skipped VALUES (?,?)", (arxiv_id, "not_found"))
            stats["not_found"] += 1

    Database.upsert_many(conn, [rec for rec, _ in to_add])
    with conn:
        for _, arxiv_id in to_add:
            n, depth = info[arxiv_id]
            conn.execute("INSERT OR REPLACE INTO paper_origin VALUES (?,?,?)", (arxiv_id, depth, n))
            stats["added"] += 1
    return stats


def run(db_path=Database.DEFAULT_DB_PATH, depth: int = 1, min_citations: int = 2, max_new: int = 500,
        categories: tuple[str, ...] | None = ("cs", "stat", "eess"), dry_run: bool = False,
        limit_refs: int | None = None, interval: float = 1.0, session=None, api_key=None,
        fetcher=fetch_s2_references, metadata_fetcher=fetch_arxiv_metadata,
        sleep: Callable[[float], None] = time.sleep) -> list[dict]:
    if session is None:
        import requests
        session = requests.Session()
    api_key = api_key or os.environ.get("S2_API_KEY")

    conn = Database.get_connection(db_path)
    Database.init_db(conn)
    ensure_schema(conn)

    history = []
    for round_number in range(1, depth + 1):
        log(f"=== rodada {round_number}/{depth} ===")
        refs = collect_references(conn, session, round_number, api_key, interval, limit_refs, fetcher, sleep)
        if refs.get("aborted"):
            log("interrompido por limite de taxa; nenhum paper foi adicionado nesta execução")
            history.append({"round": round_number, **refs})
            break
        cands = candidates(conn, min_citations, max_new)
        summary = {"round": round_number, **refs, "candidates": len(cands),
                   "citations_of_candidates": sum(n for _, n, _ in cands)}
        if dry_run:
            summary["top"] = cands[:10]
        elif cands:
            summary.update(add_candidates(conn, cands, categories, metadata_fetcher))
        log(json.dumps(summary, ensure_ascii=False))
        history.append(summary)
        if dry_run or (not cands and not refs["fetched"]):
            break
    conn.close()
    return history


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default=str(Database.DEFAULT_DB_PATH))
    p.add_argument("--depth", type=int, default=1, help="nº de rodadas de expansão")
    p.add_argument("--min-citations", type=int, default=2, help="mínimo de papers do corpus que citam o candidato")
    p.add_argument("--max-new", type=int, default=500, help="máximo de papers novos por rodada")
    p.add_argument("--categories", nargs="*", default=["cs", "stat", "eess"],
                   help="prefixos da categoria primária aceitos (vazio = todas)")
    p.add_argument("--limit-refs", type=int, default=None, help="máx. de papers cujas referências buscar por rodada")
    p.add_argument("--interval", type=float, default=1.0, help="segundos entre chamadas ao Semantic Scholar")
    p.add_argument("--dry-run", action="store_true", help="busca referências e mostra candidatos, sem inserir papers")
    a = p.parse_args()
    run(a.db, a.depth, a.min_citations, a.max_new, tuple(a.categories) or None, a.dry_run,
        a.limit_refs, a.interval)


if __name__ == "__main__":
    main()