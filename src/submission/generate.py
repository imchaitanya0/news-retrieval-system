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
    articles = pl.read_parquet(articles_path)
    # Read behaviors ONCE into columnar RAM (~500MB polars format, not Python dicts)
    behaviors = pl.read_parquet(behaviors_path)
    n_behaviors = len(behaviors)
    print(f"  Total impressions to process: {n_behaviors:,}")

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
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **kw): return it

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SUBMISSION_DIR / f"{dataset}_{split}_{strategy}.txt"

    # Stream in 5000-row slices — polars .slice() is zero-copy, no RAM overhead
    chunk_size = 5000
    n_written = 0
    with open(out_path, "w") as out_f:
        for chunk_start in tqdm(range(0, n_behaviors, chunk_size), desc="Generating predictions"):
            batch = behaviors.slice(chunk_start, chunk_size).to_dicts()

            for row in batch:
                imp_id = row["impression_id"]
                impressions = row.get("impressions") or []
                if isinstance(impressions, str):
                    impressions = impressions.split()
                if not impressions:
                    continue

                hist = row.get("history") or []

                sem_scores = {}
                bm25_scores = {}

                # --- Direct semantic scoring of candidates ---
                if retriever_sem and hist:
                    user_vec = build_user_vector(hist, embedding_map, max_history)
                    if user_vec is not None:
                        # Collect embeddings for only the impression candidates
                        cand_vecs = []
                        cand_ids = []
                        for aid in impressions:
                            if aid in embedding_map:
                                cand_vecs.append(embedding_map[aid])
                                cand_ids.append(aid)
                        if cand_vecs:
                            # Dot product = cosine sim (vecs are L2-normalized)
                            scores = np.dot(np.array(cand_vecs), user_vec)
                            for aid, sc in zip(cand_ids, scores):
                                sem_scores[aid] = float(sc)

                # --- Direct BM25 scoring of candidates ---
                if retriever_bm25 and hist:
                    query = build_query_from_history(hist, article_text_map, max_history)
                    if query:
                        import bm25s
                        q_tokens = bm25s.tokenize([query], lower=True, show_progress=False)
                        # Score only the candidate documents
                        cand_texts = [article_text_map.get(aid, "") for aid in impressions]
                        c_tokens = bm25s.tokenize(cand_texts, lower=True, show_progress=False)
                        # Get scores via inner product of query against corpus of candidates
                        # Rebuild mini-index for just this impression's candidates
                        mini_idx = bm25s.BM25()
                        mini_idx.index(c_tokens, show_progress=False)
                        res, sc = mini_idx.retrieve(q_tokens, k=len(impressions), show_progress=False)
                        for idx, score in zip(res[0], sc[0]):
                            bm25_scores[impressions[idx]] = float(score)

                # --- Fuse scores and rank ---
                if strategy == "hybrid" and (sem_scores or bm25_scores):
                    # Normalize each to [0,1] then combine
                    def norm(d):
                        if not d: return d
                        mn, mx = min(d.values()), max(d.values())
                        rng = mx - mn if mx != mn else 1.0
                        return {k: (v - mn) / rng for k, v in d.items()}
                    ns = norm(sem_scores)
                    nb = norm(bm25_scores)
                    combined = {}
                    for aid in impressions:
                        combined[aid] = 0.6 * ns.get(aid, 0.0) + 0.4 * nb.get(aid, 0.0)
                    ranked = sorted(impressions, key=lambda a: -combined.get(a, 0.0))
                elif strategy == "semantic" and sem_scores:
                    ranked = sorted(impressions, key=lambda a: -sem_scores.get(a, 0.0))
                elif strategy == "bm25" and bm25_scores:
                    ranked = sorted(impressions, key=lambda a: -bm25_scores.get(a, 0.0))
                else:
                    ranked = list(impressions)

                rank_map = {aid: rank for rank, aid in enumerate(ranked, 1)}
                ranks = [str(rank_map.get(aid, len(ranked)+1)) for aid in impressions]
                out_f.write(f"{imp_id} [{','.join(ranks)}]\n")
                n_written += 1

    print(f"Submission saved → {out_path} ({n_written:,} impressions)")
    return



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
