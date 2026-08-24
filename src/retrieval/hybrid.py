"""
Hybrid retrieval: RRF fusion of BM25 + Semantic (Q3 extension)
================================================================
Combines BM25 and semantic ranked lists using Reciprocal Rank Fusion.
RRF is more robust than score-level blending because it doesn't require
normalising BM25 scores (which have different scales from cosine similarity).

Usage:
    python -m src.retrieval.hybrid --dataset mind --split val --k 100
"""

import argparse
import json
from pathlib import Path

import polars as pl
from tqdm import tqdm

from src.retrieval.bm25 import (
    BM25Retriever, build_query_from_history, recall_at_k,
    _article_text, PROCESSED_DIR, INDEX_DIR as BM25_INDEX_DIR
)
from src.retrieval.semantic import (
    SemanticRetriever, load_or_compute_embeddings, build_user_vector,
    INDEX_DIR as SEM_INDEX_DIR
)

RESULTS_DIR = Path("data/results")


def rrf_fusion(list1: list, list2: list, k: int = 60) -> list:
    """Reciprocal Rank Fusion. Returns merged list sorted by RRF score."""
    scores = {}
    for rank, aid in enumerate(list1, 1):
        scores[aid] = scores.get(aid, 0.0) + 1.0 / (k + rank)
    for rank, aid in enumerate(list2, 1):
        scores[aid] = scores.get(aid, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda x: -scores[x])


def evaluate_hybrid(
    dataset: str,
    split: str = "val",
    k_values: list = None,
    max_history: int = 10,
    rrf_k: int = 60,
) -> dict:
    """
    Evaluate hybrid BM25 + semantic retrieval via RRF.

    Parameters
    ----------
    dataset     : "mind" or "ebnerd"
    split       : "train" / "val" / "test"
    k_values    : Recall@K cutoffs (default [50, 100, 200])
    max_history : max recent articles for query/user vector
    rrf_k       : RRF smoothing constant (default 60)

    Returns
    -------
    dict with recall@K metrics
    """
    if k_values is None:
        k_values = [50, 100, 200]

    articles_path = PROCESSED_DIR / f"articles_{dataset}.parquet"
    behaviors_path = PROCESSED_DIR / f"behaviors_{dataset}_{split}.parquet"

    articles = pl.read_parquet(articles_path)
    behaviors = pl.read_parquet(behaviors_path)

    # BM25 setup
    bm25_path = BM25_INDEX_DIR / dataset
    bm25 = BM25Retriever()
    if (bm25_path / "bm25_index.pkl").exists():
        bm25.load(bm25_path)
    else:
        bm25.build(articles)
        bm25.save(bm25_path)
    art_dict = articles.select(["article_id", "title", "subtitle"]).to_dicts()
    article_text_map = {r["article_id"]: _article_text(r) for r in art_dict}

    # Semantic setup
    embs, ids = load_or_compute_embeddings(articles, dataset)
    embedding_map = dict(zip(ids, embs))
    sem_path = SEM_INDEX_DIR / dataset
    sem = SemanticRetriever()
    if (sem_path / "faiss.index").exists():
        sem.load(sem_path)
    else:
        sem.build(embs, ids)
        sem.save(sem_path)

    max_k = max(k_values)
    recall_sums = {k: 0.0 for k in k_values}
    n_evaluated = 0
    has_history = "history" in behaviors.columns

    print(f"\n[Hybrid RRF] Evaluating {dataset}/{split} ...")
    for row in tqdm(behaviors.iter_rows(named=True), total=len(behaviors)):
        impressions = row["impressions"] or []
        labels = row["labels"] or []
        ground_truth = {aid for aid, lbl in zip(impressions, labels) if lbl == 1}
        if not ground_truth:
            continue

        history = row["history"] if has_history else []
        if isinstance(history, str):
            history = history.split()
        if not history:
            history = list(ground_truth)

        query = build_query_from_history(history, article_text_map, max_articles=max_history)
        user_vec = build_user_vector(history, embedding_map, max_articles=max_history)

        bm25_list = bm25.retrieve(query, k=max_k) if query else []
        sem_list = sem.retrieve_by_vector(user_vec, k=max_k) if user_vec is not None else []

        fused = rrf_fusion(bm25_list, sem_list, k=rrf_k)

        for k in k_values:
            recall_sums[k] += recall_at_k(fused, ground_truth, k)
        n_evaluated += 1

    results = {}
    if n_evaluated > 0:
        results = {f"recall@{k}": round(recall_sums[k] / n_evaluated, 4) for k in k_values}

    print(f"\n[Hybrid RRF] {dataset}/{split} (n={n_evaluated}):")
    for m, v in results.items():
        print(f"  {m}: {v:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Hybrid BM25+Semantic retrieval (RRF)")
    parser.add_argument("--dataset", choices=["mind", "ebnerd"], default="mind")
    parser.add_argument("--split", default="val")
    parser.add_argument("--k", type=int, nargs="+", default=[50, 100, 200])
    parser.add_argument("--max-history", type=int, default=10)
    parser.add_argument("--rrf-k", type=int, default=60)
    args = parser.parse_args()

    results = evaluate_hybrid(
        dataset=args.dataset,
        split=args.split,
        k_values=args.k,
        max_history=args.max_history,
        rrf_k=args.rrf_k,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"hybrid_{args.dataset}_{args.split}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out}")


if __name__ == "__main__":
    main()
