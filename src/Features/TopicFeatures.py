"""
Bag-of-words e tf-idf em nível de PAPER, a partir dos tokens já salvos em
data/sections/*.json (nenhum PDF/GROBID/spaCy envolvido).

Saída em `out_dir`:
    bow.npz / tfidf.npz   matrizes esparsas (papers x vocabulário), CSR
    vocabulary.json       lista de termos (coluna j = vocabulary[j])
    arxiv_ids.json        lista de IDs (linha i = arxiv_ids[i])

No banco, cada paper recebe uma linha em `paper_features` apontando para a
matriz do corpus; a linha do paper na matriz é o índice do ID em arxiv_ids.json.

Uso:
    python -m src.Features.TopicFeatures --min-df 5 --max-df 0.5
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from src.Storage import Database


def paper_tokens(sections_json: Path, skip_titles: tuple[str, ...] = ()) -> list[str]:
    """Concatena os tokens de todas as seções (opcionalmente ignora títulos)."""
    data = json.loads(Path(sections_json).read_text(encoding="utf-8"))
    tokens: list[str] = []
    for section in data["sections"]:
        if section.get("title", "").strip().lower() in skip_titles:
            continue
        tokens.extend(section.get("tokens", []))
    return tokens


# Detecção de idioma pelo TEXTO BRUTO (os tokens já perderam as stopwords, e
# palavras como "per", "non" ou "de" aparecem em inglês técnico). Textos em
# inglês têm cerca de 20-30% das palavras nesta lista; francês, alemão,
# espanhol, português e russo ficam perto de zero.
_ENGLISH_FUNCTION_WORDS = frozenset(
    "the of and to in is that for with as we are this by on be an it from at which can "
    "or our not these have has was were been their also each such more than when where "
    "then thus how its into both using used".split()
)
_SAMPLE_CHARS = 30000
_MIN_ENGLISH_RATIO = 0.08


def english_ratio(text: str) -> float | None:
    """Fração de palavras funcionais do inglês; None se o texto é curto demais para julgar."""
    sample = text[:_SAMPLE_CHARS]
    words = re.findall(r"[^\W\d_]+", sample.lower())
    if len(words) < 150:
        return None
    return sum(w in _ENGLISH_FUNCTION_WORDS for w in words) / len(words)


def non_latin_share(text: str) -> float:
    letters = [c for c in text[:_SAMPLE_CHARS] if c.isalpha()]
    return sum(ord(c) > 0x2FF for c in letters) / len(letters) if letters else 0.0


def language_check(raw_text: str) -> tuple[bool, float | None]:
    """(parece_inglês, razão). Na dúvida (texto curto), mantém o paper."""
    if non_latin_share(raw_text) > 0.05:
        return False, 0.0
    ratio = english_ratio(raw_text)
    return (True if ratio is None else ratio >= _MIN_ENGLISH_RATIO), ratio


def read_paper(sections_json: Path, skip_titles: tuple[str, ...] = ()) -> tuple[list[str], str]:
    """(tokens, texto bruto) de um paper, a partir do JSON de seções."""
    data = json.loads(Path(sections_json).read_text(encoding="utf-8"))
    tokens: list[str] = []
    raw: list[str] = []
    for section in data["sections"]:
        if section.get("title", "").strip().lower() in skip_titles:
            continue
        tokens.extend(section.get("tokens", []))
        raw.append(section.get("raw_text", ""))
    return tokens, "\n".join(raw)


def load_corpus(
    db, skip_titles: tuple[str, ...] = (), drop_non_english: bool = True, dropped: list | None = None,
) -> tuple[list[str], list[list[str]]]:
    rows = db.execute(
        "SELECT arxiv_id, text_path FROM processing "
        "WHERE text_path LIKE '%.json' ORDER BY arxiv_id"
    ).fetchall()
    ids, docs = [], []
    for arxiv_id, path in rows:
        if not Path(path).is_file():
            continue
        tokens, raw = read_paper(path, skip_titles)
        if not tokens:
            continue
        if drop_non_english:
            is_english, ratio = language_check(raw)
            if not is_english:
                if dropped is not None:
                    dropped.append((arxiv_id, ratio))
                continue
        ids.append(arxiv_id)
        docs.append(tokens)
    return ids, docs


def build_features(
    docs: list[list[str]],
    min_df: int | float = 5,
    max_df: int | float = 0.5,
    max_features: int | None = None,
    ngram_range: tuple[int, int] = (1, 1),
):
    """
    Retorna (bow, tfidf, vocabulary). `docs` já são listas de tokens, então o
    vetorizador não tokeniza nem normaliza nada: `analyzer` só (opcionalmente)
    forma n-gramas.
    """
    def analyzer(tokens):
        lo, hi = ngram_range
        out = list(tokens) if lo == 1 else []
        for n in range(max(lo, 2), hi + 1):
            out += [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        return out

    vectorizer = CountVectorizer(
        analyzer=analyzer, min_df=min_df, max_df=max_df, max_features=max_features,
    )
    bow = vectorizer.fit_transform(docs).astype(np.float32).tocsr()
    tfidf = TfidfTransformer(norm="l2", sublinear_tf=True, smooth_idf=True).fit_transform(bow)
    return bow, tfidf.astype(np.float32).tocsr(), vectorizer.get_feature_names_out().tolist()


def run(db_path=Database.DEFAULT_DB_PATH, out_dir="data/features", version="v1",
        drop_non_english=True, extra_stopwords=(), **kwargs) -> dict:
    db = Database.get_connection(db_path)
    dropped: list = []
    ids, docs = load_corpus(db, drop_non_english=drop_non_english, dropped=dropped)
    if extra_stopwords:
        stop = set(extra_stopwords)
        docs = [[t for t in d if t not in stop] for d in docs]
    if not docs:
        raise SystemExit("nenhum paper com JSON de seções encontrado")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if dropped:
        with open(out / "dropped_non_english.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["arxiv_id", "english_function_word_ratio"])
            writer.writerows(sorted(dropped, key=lambda row: (row[1] is None, row[1])))
        print(f"descartados por idioma: {len(dropped)} (lista em {out / 'dropped_non_english.csv'}; "
              f"limiar {_MIN_ENGLISH_RATIO})")

    bow, tfidf, vocabulary = build_features(docs, **kwargs)

    sp.save_npz(out / f"bow_{version}.npz", bow)
    sp.save_npz(out / f"tfidf_{version}.npz", tfidf)
    (out / f"vocabulary_{version}.json").write_text(json.dumps(vocabulary, ensure_ascii=False), encoding="utf-8")
    (out / f"arxiv_ids_{version}.json").write_text(json.dumps(ids), encoding="utf-8")

    for name, path in ((f"bow_{version}", out / f"bow_{version}.npz"),
                       (f"tfidf_{version}", out / f"tfidf_{version}.npz")):
        for arxiv_id in ids:
            Database.upsert_feature(db, arxiv_id, name, "npz", path, dim=len(vocabulary))
    db.close()
    return {"papers": len(ids), "vocabulary": len(vocabulary), "nnz": int(tfidf.nnz)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=str(Database.DEFAULT_DB_PATH))
    p.add_argument("--out", default="data/features")
    p.add_argument("--version", default="v1")
    p.add_argument("--min-df", type=float, default=5, help=">=1: nº de papers; <1: fração")
    p.add_argument("--max-df", type=float, default=0.5)
    p.add_argument("--max-features", type=int, default=None)
    p.add_argument("--keep-non-english", action="store_true", help="não descarta papers que não parecem inglês")
    p.add_argument("--stopwords", nargs="*", default=[], help="stopwords extras do domínio (ex.: fig across enabling)")
    p.add_argument("--ngrams", type=int, default=1, help="1 = só unigramas; 2 = + bigramas")
    a = p.parse_args()
    as_num = lambda v: int(v) if v >= 1 else v
    print(run(a.db, a.out, a.version, min_df=as_num(a.min_df), max_df=as_num(a.max_df),
              max_features=a.max_features, drop_non_english=not a.keep_non_english,
              extra_stopwords=a.stopwords, ngram_range=(1, a.ngrams)))


if __name__ == "__main__":
    main()
