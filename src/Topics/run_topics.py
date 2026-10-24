"""
Experimento de modelagem de tópicos sobre as features de TopicFeatures.

Para cada k: ajusta NMF (sobre tf-idf) e LDA (sobre bag-of-words), mostra os
termos principais, e calcula
  - coerência NPMI (co-ocorrência de termos nos próprios documentos; maior = melhor)
  - NMI entre o tópico dominante e a categoria primária (só categorias com
    >= --min-cat papers), como sanidade externa, não como "verdade"
Salva a matriz documento-tópico em data/topics/{modelo}_k{k}.npy.

Uso:
    python -m src.Topics.run_topics --ks 10 20 30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.metrics import normalized_mutual_info_score

from src.Storage import Database


def top_terms(components: np.ndarray, vocabulary: list[str], n: int = 10) -> list[list[str]]:
    return [[vocabulary[j] for j in row.argsort()[::-1][:n]] for row in components]


def npmi_coherence(binary_docs: sp.csc_matrix, topics: list[list[int]]) -> float:
    """NPMI médio entre pares dos termos principais de cada tópico."""
    n_docs = binary_docs.shape[0]
    df = np.asarray(binary_docs.sum(axis=0)).ravel()
    scores = []
    for terms in topics:
        sub = binary_docs[:, terms]
        co = (sub.T @ sub).toarray()
        vals = []
        for a in range(len(terms)):
            for b in range(a + 1, len(terms)):
                p_ab = co[a, b] / n_docs
                p_a, p_b = df[terms[a]] / n_docs, df[terms[b]] / n_docs
                if p_ab == 0:
                    vals.append(-1.0)
                else:
                    vals.append(np.log(p_ab / (p_a * p_b)) / -np.log(p_ab))
        scores.append(np.mean(vals))
    return float(np.mean(scores))


def category_nmi(doc_topic: np.ndarray, categories: list[str], min_cat: int) -> float:
    cats = np.array(categories)
    values, counts = np.unique(cats, return_counts=True)
    keep = np.isin(cats, values[counts >= min_cat])
    return float(normalized_mutual_info_score(cats[keep], doc_topic[keep].argmax(axis=1)))


def run(features_dir="data/features", out_dir="data/topics", db_path=Database.DEFAULT_DB_PATH,
        version="v1", ks=(10, 20, 30), min_cat=30, n_terms=10, seed=0) -> list[dict]:
    fdir = Path(features_dir)
    bow = sp.load_npz(fdir / f"bow_{version}.npz").tocsr()
    tfidf = sp.load_npz(fdir / f"tfidf_{version}.npz").tocsr()
    vocabulary = json.loads((fdir / f"vocabulary_{version}.json").read_text(encoding="utf-8"))
    ids = json.loads((fdir / f"arxiv_ids_{version}.json").read_text())

    db = Database.get_connection(db_path)
    subject = dict(db.execute("SELECT arxiv_id, primary_subject FROM papers").fetchall())
    categories = [subject[i] for i in ids]
    binary = (bow > 0).astype(np.float32).tocsc()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for k in ks:
        models = {
            "nmf": (NMF(n_components=k, init="nndsvd", random_state=seed, max_iter=400), tfidf),
            "lda": (LatentDirichletAllocation(n_components=k, random_state=seed,
                                              learning_method="online", n_jobs=-1), bow),
        }
        for name, (model, matrix) in models.items():
            doc_topic = model.fit_transform(matrix)
            comps = model.components_
            idx = [row.argsort()[::-1][:n_terms].tolist() for row in comps]
            res = {"model": name, "k": k,
                   "npmi": round(npmi_coherence(binary, idx), 4),
                   "nmi_category": round(category_nmi(doc_topic, categories, min_cat), 4)}
            results.append(res)
            np.save(out / f"{name}_k{k}.npy", doc_topic.astype(np.float32))
            (out / f"{name}_k{k}_terms.json").write_text(
                json.dumps(top_terms(comps, vocabulary, n_terms), ensure_ascii=False, indent=1),
                encoding="utf-8")
            print(res)
            for t, terms in enumerate(top_terms(comps, vocabulary, n_terms)):
                print(f"  [{t:02d}] " + " ".join(terms))
    (out / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    db.close()
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="data/features")
    p.add_argument("--out", default="data/topics")
    p.add_argument("--db", default=str(Database.DEFAULT_DB_PATH))
    p.add_argument("--version", default="v1")
    p.add_argument("--ks", type=int, nargs="+", default=[10, 20, 30])
    p.add_argument("--min-cat", type=int, default=30)
    a = p.parse_args()
    run(a.features, a.out, a.db, a.version, tuple(a.ks), a.min_cat)


if __name__ == "__main__":
    main()
