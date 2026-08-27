"""
Feature Store (Q5 / LightGBM Reranker)
========================================
Computes per-(impression, candidate) features used to train and inference
the LightGBM learning-to-rank model.

Features computed:
  1. sem_score       – cosine similarity between user_vec and article_vec
  2. bm25_in_top200  – 1 if article appears in BM25 global top-200 retrieval
  3. popularity      – global click count from training behaviors (log-scaled)
  4. hist_len        – number of articles in user's click history
  5. cat_match       – 1 if candidate category == user's most-clicked category
  6. position        – original position of candidate in impression (1-indexed)

Usage:
    from src.features.feature_store import build_features
    X, y, groups = build_features(behaviors, articles, embedding_map, retriever_bm25)
"""

from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl


def _user_category(history: list, category_map: dict) -> Optional[str]:
    """Return the most-frequent category in a user's click history."""
    if not history:
        return None
    cats = [category_map.get(aid) for aid in history if category_map.get(aid)]
    if not cats:
        return None
    return max(set(cats), key=cats.count)


def build_features(
    behaviors: pl.DataFrame,
    articles:  pl.DataFrame,
    embedding_map: dict,
    retriever_bm25=None,
    article_text_map: dict = None,
    popularity_map: dict = None,
    max_history: int = 10,
    bm25_k: int = 200,
) -> tuple:
    """
    Build feature matrix for all (impression, candidate) pairs.

    Returns
    -------
    X      : np.ndarray shape (N_pairs, N_features)
    y      : np.ndarray shape (N_pairs,)  — binary labels (0/1), -1 for test
    groups : np.ndarray shape (N_impressions,) — number of candidates per impression
    imp_ids: list of impression_id repeated for each candidate row
    art_ids: list of article_id for each candidate row
    """
    from src.retrieval.semantic import build_user_vector
    from src.retrieval.bm25 import build_query_from_history

    # Article lookups
    art_rows = articles.select(["article_id", "category"]).to_dicts()
    category_map = {r["article_id"]: r["category"] for r in art_rows}

    if popularity_map is None:
        popularity_map = {}

    X_rows  = []
    y_rows  = []
    groups  = []
    imp_ids = []
    art_ids = []

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **kw): return it

    for row in tqdm(behaviors.iter_rows(named=True), total=len(behaviors), desc="Building Features"):
        impressions = row.get("impressions") or []
        labels      = row.get("labels")      or []
        history     = row.get("history")     or []
        imp_id      = row["impression_id"]

        if not impressions:
            continue

        n_cands = len(impressions)

        # User-level features (same for all candidates in this impression)
        hist_len  = len(history)
        user_cat  = _user_category(history, category_map)

        # User vector
        user_vec = None
        if embedding_map and history:
            user_vec = build_user_vector(history, embedding_map, max_history)

        # BM25 top-K set
        bm25_top = set()
        if retriever_bm25 and article_text_map and history:
            import bm25s
            query  = build_query_from_history(history, article_text_map, max_history)
            if query:
                k_val  = min(bm25_k, len(retriever_bm25.article_ids))
                tokens = bm25s.tokenize([query], lower=True, show_progress=False)
                res, _ = retriever_bm25._index.retrieve(tokens, k=k_val, show_progress=False)
                bm25_top = {retriever_bm25.article_ids[idx] for idx in res[0]}

        for pos, (aid, lbl) in enumerate(
            zip(impressions, labels if labels else [-1] * n_cands), start=1
        ):
            # Feature 1: semantic similarity
            sem = 0.0
            if user_vec is not None and aid in embedding_map:
                sem = float(np.dot(user_vec, embedding_map[aid]))

            # Feature 2: BM25 global overlap
            bm25_in = 1 if aid in bm25_top else 0

            # Feature 3: log-scaled popularity
            pop = np.log1p(popularity_map.get(aid, 0))

            # Feature 4: user history length (warm vs cold)
            hl = min(hist_len, 50)  # cap at 50 to avoid outlier influence

            # Feature 5: category match
            cat_m = 1 if (user_cat and category_map.get(aid) == user_cat) else 0

            # Feature 6: original position in impression (recency / platform signal)
            pos_feat = 1.0 / pos   # inverse position (earlier = higher)

            X_rows.append([sem, bm25_in, pop, hl, cat_m, pos_feat])
            y_rows.append(int(lbl) if lbl != -1 else -1)
            imp_ids.append(imp_id)
            art_ids.append(aid)

        groups.append(n_cands)

    X      = np.array(X_rows, dtype=np.float32)
    y      = np.array(y_rows, dtype=np.int32)
    groups = np.array(groups, dtype=np.int32)

    return X, y, groups, imp_ids, art_ids


FEATURE_NAMES = [
    "sem_score",
    "bm25_in_top200",
    "log_popularity",
    "hist_len",
    "cat_match",
    "inv_position",
]
