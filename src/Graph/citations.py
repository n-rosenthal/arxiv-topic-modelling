"""
Arestas de citação ENTRE papers do corpus.

Cada referência extraída pelo GROBID é ligada a um paper do corpus por, em
ordem: (1) arXiv ID, (2) DOI, (3) título normalizado (+ ano, se ambos tiverem).
Imprime estatísticas (quantas referências, quantas casam, grau, componentes
conexas) e grava as arestas em data/graph/citation_edges.csv.

Uso:
    python -m src.Graph.citations
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from src.Storage import Database

_VERSION = re.compile(r"v\d+$")


def norm_arxiv(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip().lower()
    value = re.sub(r"^(arxiv:|https?://arxiv\.org/(abs|pdf)/)", "", value)
    value = re.sub(r"\.pdf$", "", value)
    return _VERSION.sub("", value) or None


def norm_doi(value: str | None) -> str | None:
    return value.strip().lower().rstrip(".") if value and value.strip() else None


def norm_title(value: str | None) -> str | None:
    if not value:
        return None
    t = re.sub(r"[^a-z0-9 ]+", " ", value.lower())
    t = " ".join(t.split())
    return t if len(t) >= 20 else None        # títulos curtos casam por acaso


def build_index(db):
    by_arxiv, by_doi, by_title = {}, {}, defaultdict(list)
    for arxiv_id, doi, title, year in db.execute(
        "SELECT arxiv_id, doi, title, substr(submission_date,1,4) FROM papers"
    ):
        by_arxiv[norm_arxiv(arxiv_id)] = arxiv_id
        if norm_doi(doi):
            by_doi[norm_doi(doi)] = arxiv_id
        if norm_title(title):
            by_title[norm_title(title)].append((arxiv_id, int(year) if year else None))
    return by_arxiv, by_doi, by_title


def find_edges(db) -> tuple[list[tuple[str, str, str]], Counter]:
    by_arxiv, by_doi, by_title = build_index(db)
    rows = db.execute(
        "SELECT arxiv_id, text_path FROM processing WHERE text_path LIKE '%.json'"
    ).fetchall()
    edges: dict[tuple[str, str], str] = {}
    stats: Counter = Counter()
    for source, path in rows:
        if not Path(path).is_file():
            continue
        refs = json.loads(Path(path).read_text(encoding="utf-8")).get("references", [])
        stats["papers"] += 1
        for ref in refs:
            stats["references"] += 1
            target, how = None, None
            if (a := norm_arxiv(ref.get("arxiv_id"))) and a in by_arxiv:
                target, how = by_arxiv[a], "arxiv"
            elif (d := norm_doi(ref.get("doi"))) and d in by_doi:
                target, how = by_doi[d], "doi"
            elif (t := norm_title(ref.get("title"))) and t in by_title:
                year = ref.get("year")
                cands = [i for i, y in by_title[t] if not (year and y and abs(year - y) > 1)]
                if len(cands) == 1:
                    target, how = cands[0], "title"
            if ref.get("arxiv_id"):
                stats["refs_with_arxiv_id"] += 1
            if target and target != source and (source, target) not in edges:
                edges[(source, target)] = how
                stats[f"edges_by_{how}"] += 1
    return [(s, t, h) for (s, t), h in edges.items()], stats


def graph_summary(edges, all_ids) -> dict:
    out_deg, in_deg, adj = Counter(), Counter(), defaultdict(set)
    for s, t, _ in edges:
        out_deg[s] += 1
        in_deg[t] += 1
        adj[s].add(t)
        adj[t].add(s)
    isolated = sum(1 for n in all_ids if n not in adj)   # antes da busca: adj[n] cria chaves
    seen, components = set(), []
    for node in all_ids:
        if node in seen:
            continue
        stack, comp = [node], 0
        seen.add(node)
        while stack:
            n = stack.pop()
            comp += 1
            for m in adj.get(n, set()) - seen:
                seen.add(m)
                stack.append(m)
        components.append(comp)
    components.sort(reverse=True)
    return {
        "nodes": len(all_ids), "edges": len(edges),
        "isolated_nodes": isolated,
        "largest_component": components[0] if components else 0,
        "components": len(components),
        "mean_out_degree": round(len(edges) / max(len(all_ids), 1), 3),
        "max_in_degree": max(in_deg.values(), default=0),
    }


def run(db_path=Database.DEFAULT_DB_PATH, out_dir="data/graph") -> dict:
    db = Database.get_connection(db_path)
    edges, stats = find_edges(db)
    ids = [r[0] for r in db.execute("SELECT arxiv_id FROM processing WHERE text_path LIKE '%.json'")]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "citation_edges.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "target", "matched_by"])
        w.writerows(edges)
    result = {**dict(stats), **graph_summary(edges, ids)}
    db.close()
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=str(Database.DEFAULT_DB_PATH))
    p.add_argument("--out", default="data/graph")
    a = p.parse_args()
    for k, v in run(a.db, a.out).items():
        print(f"{k:22s}: {v}")


if __name__ == "__main__":
    main()
