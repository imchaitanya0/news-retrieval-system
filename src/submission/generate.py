"""
Codabench Submission Generator (Q5)
=====================================
Generates prediction files for both MIND and EB-NeRD leaderboards.

MIND format (one line per impression):
    {impression_id} [{article_id}-{rank} {article_id}-{rank} ...]

EB-NeRD format (one line per impression):
    {impression_id} [{article_id}-{rank} ...]

Scoring strategy (configurable):
  "bm25"     — use BM25 Recall order
  "semantic" — use semantic ANN order
  "hybrid"   — RRF fusion of BM25 + semantic (recommended)
  "random"   — random baseline
  "oracle"   — use labels (only valid on val, not test)

Usage:
    python -m src.submission.generate --dataset mind --strategy hybrid
    python -m src.submission.generate --dataset ebnerd --strategy hybrid
"""

import argparse
import re
from pathlib import Path

import polars as pl
import numpy as np

PROCESSED_DIR = Path("data/processed")
SUBMISSION_DIR = Path("data/submissions")

# ------------------------------------------------------------------ #
#  Ranking helpers                                                    #
# ------------------------------------------------------------------ #

def rrf_fusion(list1: list, list2: list, k: int = 60) -> list:
    """
    Reciprocal Rank Fusion of two ranked lists.
    RRF(d) = sum_i 1 / (k + rank_i(d))
    Returns merged list sorted by descending RRF score.
    """
    scores = {}
    for rank, aid in enumerate(list1, 1):
        scores[aid] = scores.get(aid, 0.0) + 1.0 / (k + rank)
    for rank, aid in enumerate(list2, 1):
        scores[aid] = scores.get(aid, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda x: -scores[x])


def rank_impressions(
    impressions: list,
    bm25_order: list = None,
    semantic_order: list = None,
    strategy: str = "hybrid",
    labels: list = None,
) -> list:
    """
    Rank a list of candidate article IDs using the given strategy.

    Parameters
    ----------
    impressions : list of article_id (all candidates in the impression)
    bm25_order : list of article_id sorted by BM25 score (descending)
    semantic_order : same for semantic similarity
    strategy : "hybrid", "bm25", "semantic", "random", "oracle"
    labels : ground-truth labels (only used for oracle strategy)

    Returns
    -------
    list of article_id in predicted rank order (best first)
    """
    imp_set = set(impressions)

    if strategy == "oracle" and labels is not None:
        # Sort by label descending (only valid on val)
        return [aid for _, aid in sorted(
            zip(labels, impressions), reverse=True
        )]

    if strategy == "random":
        order = list(impressions)
        np.random.shuffle(order)
        return order

    if strategy == "bm25" and bm25_order:
        # Filter BM25 global ranking to only this impression's candidates
        ranked = [a for a in bm25_order if a in imp_set]
        missing = [a for a in impressions if a not in set(ranked)]
        return ranked + missing

    if strategy == "semantic" and semantic_order:
        ranked = [a for a in semantic_order if a in imp_set]
        missing = [a for a in impressions if a not in set(ranked)]
        return ranked + missing

    if strategy == "hybrid":
        b = bm25_order or []
        s = semantic_order or []
        if b and s:
            fused = rrf_fusion(b, s)
            ranked = [a for a in fused if a in imp_set]
        elif b:
            ranked = [a for a in b if a in imp_set]
        elif s:
            ranked = [a for a in s if a in imp_set]
        else:
            ranked = list(impressions)
        missing = [a for a in impressions if a not in set(ranked)]
        return ranked + missing

    # Fallback: return as-is
    return list(impressions)


# ------------------------------------------------------------------ #
#  BM25 pre-ranking (global retrieval to get order over candidates)  #
# ------------------------------------------------------------------ #

def bm25_rank_candidates(
    row: dict,
    retriever,
    article_text_map: dict,
    max_history: int = 10,
) -> list:
    """Return globally BM25-ranked article IDs, filtered to impression candidates."""
    from src.retrieval.bm25 import build_query_from_history
    history = row.get("history") or []
    if isinstance(history, str):
        history = history.split()
    if not history:
        return []
    query = build_query_from_history(history, article_text_map, max_articles=max_history)
    if not query:
        return []
    return retriever.retrieve(query, k=len(row["impressions"]) * 3)


def semantic_rank_candidates(
    row: dict,
    retriever,
    embedding_map: dict,
    max_history: int = 10,
) -> list:
    """Return semantically ranked article IDs, filtered to impression candidates."""
    from src.retrieval.semantic import build_user_vector
    history = row.get("history") or []
    if isinstance(history, str):
        history = history.split()
    if not history:
        return []
    user_vec = build_user_vector(history, embedding_map, max_articles=max_history)
    if user_vec is None:
        return []
    return retriever.retrieve_by_vector(user_vec, k=len(row["impressions"]) * 3)


# ------------------------------------------------------------------ #
#  Submission file writers                                            #
# ------------------------------------------------------------------ #

def write_mind_submission(ranked_impressions: list, out_path: Path) -> None:
    """
    Write MIND Codabench format.
    Format: {impression_id} [{rank1},{rank2},...] (NO SPACES inside brackets)
    Ranks must correspond exactly to the order of candidates in the original impression log.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for imp_id, original_impressions, ranked_ids in ranked_impressions:
            rank_map = {aid: rank for rank, aid in enumerate(ranked_ids, 1)}
            ranks = [str(rank_map.get(aid, len(ranked_ids)+1)) for aid in original_impressions]
            f.write(f"{imp_id} [{','.join(ranks)}]\n")
    print(f"MIND submission saved → {out_path} ({len(ranked_impressions)} impressions)")


def write_ebnerd_submission(ranked_impressions: list, out_path: Path) -> None:
    """
    Write EB-NeRD / RecSys 2024 Codabench format.

    Format: {impression_id} [{N1}-1 {N2}-2 ...]
    (same structure as MIND)
    """
    write_mind_submission(ranked_impressions, out_path)
    print(f"EB-NeRD submission saved → {out_path}")


# ------------------------------------------------------------------ #
#  Main generation pipeline                                           #
# ------------------------------------------------------------------ #

def generate_submission(
    dataset: str,
    split: str = "test",
    strategy: str = "hybrid",
    max_history: int = 10,
) -> None:
    """
    Generate a full submission file for the given dataset/split.

    Parameters
    ----------
    dataset   : "mind" or "ebnerd"
    split     : usually "test" for final submission, "val" for local check
    strategy  : "hybrid", "bm25", "semantic", "random", "oracle"
    """
    behaviors_path = PROCESSED_DIR / f"behaviors_{dataset}_{split}.parquet"
    articles_path = PROCESSED_DIR / f"articles_{dataset}.parquet"

    if not behaviors_path.exists():
        raise FileNotFoundError(f"Run build_pipeline.py first: {behaviors_path}")
    if not articles_path.exists():
        raise FileNotFoundError(f"Run build_pipeline.py first: {articles_path}")

    print(f"\n[Submit] Generating {dataset} / {split} with strategy={strategy}")
    behaviors = pl.read_parquet(behaviors_path)
    articles = pl.read_parquet(articles_path)

    retriever_bm25 = None
    retriever_sem = None
    article_text_map = {}
    embedding_map = {}

    if strategy in ("bm25", "hybrid"):
        from src.retrieval.bm25 import BM25Retriever, _article_text
        index_path = Path("data/feature_store/bm25") / dataset
        retriever_bm25 = BM25Retriever()
        if (index_path / "bm25_index.pkl").exists():
            retriever_bm25.load(index_path)
        else:
            retriever_bm25.build(articles)
            retriever_bm25.save(index_path)
        art_rows = articles.select(["article_id", "title", "subtitle"]).to_dicts()
        article_text_map = {r["article_id"]: _article_text(r) for r in art_rows}

    if strategy in ("semantic", "hybrid"):
        from src.retrieval.semantic import SemanticRetriever, load_or_compute_embeddings
        embs, ids = load_or_compute_embeddings(articles, dataset)
        embedding_map = dict(zip(ids, embs))
        index_path = Path("data/feature_store/semantic") / dataset
        retriever_sem = SemanticRetriever()
        if (index_path / "faiss.index").exists():
            retriever_sem.load(index_path)
        else:
            retriever_sem.build(embs, ids)
            retriever_sem.save(index_path)

    from src.retrieval.bm25 import build_query_from_history
    from src.retrieval.semantic import build_user_vector
    
    ranked_impressions = []
    has_history = "history" in behaviors.columns

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **kw): return it

    # Convert to list of dicts for faster iteration than iter_rows()
    behaviors_dicts = behaviors.to_dicts()
    batch_size = 10000

    for i in tqdm(range(0, len(behaviors_dicts), batch_size), desc="Generating predictions (batched)"):
        batch = behaviors_dicts[i:i+batch_size]
        
        bm25_queries = []
        sem_queries = []
        
        for row in batch:
            hist = row.get("history") or []
            if isinstance(hist, str):
                hist = hist.split()
                
            if retriever_bm25 and has_history:
                bm25_queries.append(build_query_from_history(hist, article_text_map, max_history))
            else:
                bm25_queries.append("")
                
            if retriever_sem and has_history:
                sem_queries.append(build_user_vector(hist, embedding_map, max_history))
            else:
                sem_queries.append(None)
                
        # Batch Retrieval BM25
        batch_bm25_results = [[] for _ in range(len(batch))]
        if retriever_bm25 and has_history:
            import bm25s
            tokens = bm25s.tokenize(bm25_queries, lower=True, show_progress=False)
            # Retrieve max required k (e.g. 150 for 50 candidates * 3)
            # We'll just retrieve 200 globally to be safe
            k_val = min(200, len(retriever_bm25.article_ids))
            res_idx, _ = retriever_bm25._index.retrieve(tokens, k=k_val, show_progress=False)
            for r_idx, indices in enumerate(res_idx):
                if bm25_queries[r_idx]:
                    batch_bm25_results[r_idx] = [retriever_bm25.article_ids[idx] for idx in indices]
                    
        # Batch Retrieval FAISS
        batch_sem_results = [[] for _ in range(len(batch))]
        if retriever_sem and has_history:
            valid_vecs = []
            valid_idx_map = []
            for r_idx, vec in enumerate(sem_queries):
                if vec is not None:
                    valid_vecs.append(vec)
                    valid_idx_map.append(r_idx)
            if valid_vecs:
                vec_array = np.vstack(valid_vecs)
                k_val = min(200, len(retriever_sem.article_ids))
                _, I = retriever_sem._index.search(vec_array, k_val)
                for map_idx, indices in enumerate(I):
                    r_idx = valid_idx_map[map_idx]
                    batch_sem_results[r_idx] = [retriever_sem.article_ids[idx] for idx in indices]

        # Fusion & Rank for the batch
        for r_idx, row in enumerate(batch):
            imp_id = row["impression_id"]
            impressions = row.get("impressions") or []
            if isinstance(impressions, str):
                impressions = impressions.split()
            if not impressions:
                continue
                
            ranked = rank_impressions(
                impressions=impressions,
                bm25_order=batch_bm25_results[r_idx],
                semantic_order=batch_sem_results[r_idx],
                strategy=strategy,
                labels=row.get("labels")
            )
            ranked_impressions.append((imp_id, impressions, ranked))

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SUBMISSION_DIR / f"{dataset}_{split}_{strategy}.txt"
    if dataset == "mind":
        write_mind_submission(ranked_impressions, out_path)
    else:
        write_ebnerd_submission(ranked_impressions, out_path)


# ------------------------------------------------------------------ #
#  CLI                                                                #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Submission file generator (Q5)")
    parser.add_argument("--dataset", choices=["mind", "ebnerd"], required=True)
    parser.add_argument("--split", default="test",
                        help="Split to generate predictions for (default: test)")
    parser.add_argument("--strategy", default="hybrid",
                        choices=["hybrid", "bm25", "semantic", "random", "oracle"],
                        help="Ranking strategy (default: hybrid)")
    parser.add_argument("--max-history", type=int, default=10)
    args = parser.parse_args()

    generate_submission(
        dataset=args.dataset,
        split=args.split,
        strategy=args.strategy,
        max_history=args.max_history,
    )


if __name__ == "__main__":
    main()
