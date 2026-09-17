"""
Feature Store (L2 Ranking - LightGBM Reranker)
================================================
Computes per-(impression, candidate) features in GPU-accelerated batches.

Architecture note (L1 vs L2):
  In industry-standard 2-stage recommendation systems:
  - L1 (Retrieval): Uses global indices like FAISS/Elasticsearch to narrow
    120K articles → ~200 candidates per user. This is what bm25.py/semantic.py does.
  - L2 (Ranking): Takes those candidates and scores them with a heavy ML model.
    We do NOT query a global Elasticsearch index here — because we already know
    the candidates, we compute fast in-memory features instead.
    This is exactly how Google/YouTube rank their retrieved candidates.

Features computed (all local, no global index query needed):
  1. sem_score         – GPU batched dot product: user_vec · candidate_vec
                        user_vec = RECENCY-WEIGHTED mean pool of history embeddings
  2. lex_overlap       – Jaccard overlap between user keyword tokens & candidate tokens
  3. log_popularity    – log(1 + global click count) for the candidate article
  4. hist_len          – number of articles in user's click history (cold/warm signal)
  5. cat_match         – 1 if candidate category == user's dominant category
  6. inv_position      – 1/position (editorial placement signal from the platform)
  7. freshness         – exp(-hours_since_publish/24); news older than 1 week ≈ 0
  8. recency_sem_score – sem_score using ONLY the most recent 3 clicked articles
                        (captures short-term interest shift, independent of Feature 1)

Why lexical_overlap instead of BM25:
  BM25 requires a global inverted index search per user (O(V) per query).
  For 6M users, this takes >2 hours on a single CPU.
  Lexical overlap is a pure dictionary lookup — O(|history_tokens|) — and
  captures the exact same signal (e.g., "Messi" appearing in the article text).

Why recency-weighted mean pool:
  Flat mean pooling gives equal weight to a click from 30 days ago and a click
  from 5 minutes ago. News interests are highly dynamic — using exponential decay
  (recent clicks weighted ~2.7× more per step) makes the user vector represent
  current intent rather than historical averages.

Usage:
    from src.features.feature_store import build_features, FEATURE_NAMES
    X, y, groups, imp_ids, art_ids = build_features(
        behaviors, articles, embedding_map, article_text_map, popularity_map,
        article_publish_map=article_publish_map
    )
"""

from pathlib import Path
from typing import Optional
import datetime

import numpy as np
import polars as pl

FEATURE_NAMES = [
    "sem_score",         # 1. Recency-weighted semantic similarity
    "lex_overlap",       # 2. Jaccard lexical overlap
    "log_popularity",    # 3. Log(1 + click count)
    "hist_len",          # 4. Click history length
    "cat_match",         # 5. Category match
    "inv_position",      # 6. Inverse editorial position
    "freshness",         # 7. Temporal freshness (exp decay)
    "recency_sem_score", # 8. Short-term interest (last 3 clicks)
]

_RECENCY_DECAY = 0.15   # exponential decay rate for history weighting
_SHORT_WINDOW  = 3      # number of most recent clicks for recency_sem_score
_FRESH_HALF_LIFE_HOURS = 24.0  # freshness halves every 24 hours


def _tokenize(text: str) -> set:
    """Simple whitespace tokenizer producing a lowercase token set."""
    if not text:
        return set()
    return set(text.lower().split())


def _user_category(history: list, category_map: dict) -> Optional[str]:
    """Return the most-frequent category in a user's click history."""
    if not history:
        return None
    cats = [category_map.get(aid) for aid in history if category_map.get(aid)]
    if not cats:
        return None
    return max(set(cats), key=cats.count)


def _recency_weighted_mean(vecs: list, decay: float = _RECENCY_DECAY) -> np.ndarray:
    """
    Compute exponential-decay weighted mean of embeddings.
    Most recent click (last in list) gets weight 1.0.
    Older clicks decay: weight_i = exp(-decay * (n - 1 - i)).

    Parameters
    ----------
    vecs  : list of np.ndarray(384,), ordered oldest→newest
    decay : float, exponential decay rate (higher = more recency bias)

    Returns
    -------
    np.ndarray(384,), L2-normalized
    """
    if not vecs:
        return None
    n = len(vecs)
    # Position 0 = oldest, position n-1 = newest
    weights = np.exp(-decay * np.arange(n - 1, -1, -1)).astype(np.float32)
    weights /= weights.sum()
    result = np.average(vecs, axis=0, weights=weights).astype(np.float32)
    # L2 normalize for cosine similarity compatibility
    norm = np.linalg.norm(result)
    if norm > 0:
        result /= norm
    return result


def _compute_freshness(published_time, impression_time) -> float:
    """
    Freshness = exp(-hours_since_publish / half_life).
    Returns 1.0 for very fresh articles, ~0.0 for week-old ones.
    Returns 0.5 as default if timestamps are unavailable.
    """
    if published_time is None or impression_time is None:
        return 0.5
    try:
        # Handle both datetime objects and epoch ints/floats
        if isinstance(published_time, (int, float)):
            published_time = datetime.datetime.fromtimestamp(published_time, tz=datetime.timezone.utc)
        if isinstance(impression_time, (int, float)):
            impression_time = datetime.datetime.fromtimestamp(impression_time, tz=datetime.timezone.utc)
        delta_hours = max(0.0, (impression_time - published_time).total_seconds() / 3600.0)
        return float(np.exp(-delta_hours / _FRESH_HALF_LIFE_HOURS))
    except Exception:
        return 0.5


def build_features(
    behaviors: pl.DataFrame,
    articles: pl.DataFrame,
    embedding_map: dict,
    article_text_map: dict = None,
    popularity_map: dict = None,
    article_publish_map: dict = None,
    max_history: int = 10,
    chunk_size: int = 1500,
) -> tuple:
    """
    Build feature matrix for all (impression, candidate) pairs via GPU batching.

    Parameters
    ----------
    behaviors           : DataFrame with impression_id, user_id, history,
                          impressions, labels, impression_time
    articles            : DataFrame with article_id, category, published_time
    embedding_map       : dict {article_id -> np.ndarray(384,)}
    article_text_map    : dict {article_id -> str}  — for lexical overlap
    popularity_map      : dict {article_id -> int}  — click counts from train
    article_publish_map : dict {article_id -> datetime} — publication time for freshness
    max_history         : number of recent articles to use for user vector
    chunk_size          : batch size for GPU matrix multiplication

    Returns
    -------
    X      : np.ndarray (N_pairs, 8)     — feature matrix
    y      : np.ndarray (N_pairs,)       — binary labels; -1 for test set
    groups : np.ndarray (N_impressions,) — candidates per impression (for LightGBM)
    imp_ids: list[int]
    art_ids: list[int]
    """
    try:
        import torch
        use_gpu = torch.cuda.is_available()
    except ImportError:
        use_gpu = False

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **kw): return it

    if popularity_map is None:
        popularity_map = {}
    if article_text_map is None:
        article_text_map = {}
    if article_publish_map is None:
        article_publish_map = {}

    # ------------------------------------------------------------------
    # Build article lookup maps
    # ------------------------------------------------------------------
    art_rows     = articles.select(["article_id", "category"]).to_dicts()
    category_map = {r["article_id"]: r["category"] for r in art_rows}

    # Pre-tokenize all article texts once (avoids re-tokenizing per impression)
    article_tokens: dict = {}
    for aid, text in article_text_map.items():
        article_tokens[aid] = _tokenize(text)

    # ------------------------------------------------------------------
    # Build embedding matrix for GPU batch matmul
    # ------------------------------------------------------------------
    DIM = 384
    emb_id_list   = list(embedding_map.keys())
    emb_id_to_idx = {aid: i for i, aid in enumerate(emb_id_list)}
    emb_np = np.array([embedding_map[aid] for aid in emb_id_list], dtype=np.float32)

    if use_gpu:
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        emb_gpu = torch.tensor(emb_np).cuda()  # (N_articles, 384)
        print(f"  GPU embedding matrix: {emb_gpu.shape} on CUDA ✓")
        free_gb = (torch.cuda.get_device_properties(0).total_memory
                   - torch.cuda.memory_allocated()) / 1e9
        print(f"  GPU VRAM free after loading matrix: {free_gb:.1f} GB")
    else:
        emb_gpu = emb_np
        print("  Running on CPU (GPU not available)")

    # ------------------------------------------------------------------
    # Collect all rows, build user vectors with RECENCY DECAY
    # ------------------------------------------------------------------
    n_behaviors = len(behaviors)
    all_rows    = behaviors.to_dicts()

    # Detect if impression_time column exists
    has_impression_time = "impression_time" in behaviors.columns

    print("  Pre-computing recency-weighted user vectors...")
    user_vecs_main   = np.zeros((n_behaviors, DIM), dtype=np.float32)  # full history
    user_vecs_recent = np.zeros((n_behaviors, DIM), dtype=np.float32)  # last 3 only
    hist_lens        = np.zeros(n_behaviors, dtype=np.int32)
    user_cats        = [None] * n_behaviors
    user_token_sets  = [None] * n_behaviors
    impression_times = [None] * n_behaviors

    for i, row in enumerate(tqdm(all_rows, desc="Building User Vectors")):
        history  = row.get("history") or []
        hist_len = len(history)
        hist_lens[i] = min(hist_len, 50)
        user_cats[i] = _user_category(history, category_map)

        if has_impression_time:
            impression_times[i] = row.get("impression_time")

        if history:
            recent = history[-max_history:]  # cap to last max_history clicks

            # --- Full recency-weighted vector ---
            vecs_all = [embedding_map[aid] for aid in recent if aid in embedding_map]
            rv = _recency_weighted_mean(vecs_all, decay=_RECENCY_DECAY)
            if rv is not None:
                user_vecs_main[i] = rv

            # --- Short-term (last 3) vector for recency_sem_score feature ---
            short_window = history[-_SHORT_WINDOW:]
            vecs_short = [embedding_map[aid] for aid in short_window if aid in embedding_map]
            rv_short = _recency_weighted_mean(vecs_short, decay=_RECENCY_DECAY)
            if rv_short is not None:
                user_vecs_recent[i] = rv_short

            # --- Aggregate user's keyword tokens from history text ---
            tok_set = set()
            for aid in recent:
                tok_set.update(article_tokens.get(aid, set()))
            user_token_sets[i] = tok_set

    # ------------------------------------------------------------------
    # GPU batch matmul → extract candidate semantic scores inline
    # ------------------------------------------------------------------
    print("  GPU batch semantic scoring (full + recent windows)...")

    all_cand_indices = []
    for row in all_rows:
        impressions = row.get("impressions") or []
        all_cand_indices.append([emb_id_to_idx.get(aid, -1) for aid in impressions])

    all_sem_main   = []  # per impression: list of float
    all_sem_recent = []  # per impression: list of float

    for chunk_start in tqdm(range(0, n_behaviors, chunk_size), desc="GPU Batches"):
        chunk_end = min(chunk_start + chunk_size, n_behaviors)

        if use_gpu:
            # Full history semantic scores
            uv_t   = torch.tensor(user_vecs_main[chunk_start:chunk_end]).cuda()
            sc_all = torch.mm(uv_t, emb_gpu.T).cpu().numpy()
            del uv_t

            # Short-term semantic scores
            uv_r   = torch.tensor(user_vecs_recent[chunk_start:chunk_end]).cuda()
            sc_rec = torch.mm(uv_r, emb_gpu.T).cpu().numpy()
            del uv_r
        else:
            sc_all = np.dot(user_vecs_main[chunk_start:chunk_end], emb_gpu.T)
            sc_rec = np.dot(user_vecs_recent[chunk_start:chunk_end], emb_gpu.T)

        for i in range(chunk_end - chunk_start):
            cand_idxs = all_cand_indices[chunk_start + i]
            sem_row        = [float(sc_all[i, idx]) if idx >= 0 else 0.0 for idx in cand_idxs]
            sem_recent_row = [float(sc_rec[i, idx]) if idx >= 0 else 0.0 for idx in cand_idxs]
            all_sem_main.append(sem_row)
            all_sem_recent.append(sem_recent_row)

        del sc_all, sc_rec

    if use_gpu:
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Build final (N_pairs, 8) feature matrix
    # ------------------------------------------------------------------
    print("  Assembling feature rows...")
    X_rows  = []
    y_rows  = []
    groups  = []
    imp_ids = []
    art_ids = []

    for i, row in enumerate(tqdm(all_rows, desc="Feature Assembly")):
        impressions = row.get("impressions") or []
        labels      = row.get("labels")      or []
        imp_id      = row["impression_id"]

        if not impressions:
            continue

        n_cands  = len(impressions)
        hl       = int(hist_lens[i])
        uc       = user_cats[i]
        u_tokens = user_token_sets[i] or set()
        imp_time = impression_times[i]
        sem_vec  = all_sem_main[i]
        rec_vec  = all_sem_recent[i]

        for pos, (aid, lbl) in enumerate(
            zip(impressions, labels if labels else [-1] * n_cands), start=1
        ):
            # Feature 1: Recency-weighted semantic similarity
            sem = sem_vec[pos - 1] if (pos - 1) < len(sem_vec) else 0.0

            # Feature 2: Lexical overlap (Jaccard)
            cand_tokens = article_tokens.get(aid, set())
            if u_tokens and cand_tokens:
                intersection = len(u_tokens & cand_tokens)
                union        = len(u_tokens | cand_tokens)
                lex_overlap  = intersection / union if union > 0 else 0.0
            else:
                lex_overlap = 0.0

            # Feature 3: Log-scaled popularity
            pop = float(np.log1p(popularity_map.get(aid, 0)))

            # Feature 4: User history length (cold=0 clicks, warm=50 clicks)
            # (already computed as hl)

            # Feature 5: Category match (1 if candidate in user's top category)
            cat_m = 1 if (uc and category_map.get(aid) == uc) else 0

            # Feature 6: Inverse position (editorial placement signal)
            pos_feat = 1.0 / pos

            # Feature 7: Freshness — how recent is this article?
            pub_time  = article_publish_map.get(aid)
            freshness = _compute_freshness(pub_time, imp_time)

            # Feature 8: Short-term (last 3 clicks) semantic score
            rec_sem = rec_vec[pos - 1] if (pos - 1) < len(rec_vec) else 0.0

            X_rows.append([sem, lex_overlap, pop, hl, cat_m, pos_feat, freshness, rec_sem])
            y_rows.append(int(lbl) if lbl != -1 else -1)
            imp_ids.append(imp_id)
            art_ids.append(aid)

        groups.append(n_cands)

    X      = np.array(X_rows, dtype=np.float32)
    y      = np.array(y_rows,  dtype=np.int32)
    groups = np.array(groups,  dtype=np.int32)

    print(f"  Features built: {X.shape[0]:,} pairs, {X.shape[1]} features")
    return X, y, groups, imp_ids, art_ids
