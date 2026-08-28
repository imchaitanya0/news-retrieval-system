"""
Feature Store (L2 Ranking - LightGBM Reranker)
================================================
Computes per-(impression, candidate) features in GPU-accelerated batches.

Architecture note (L1 vs L2):
  In industry-standard 2-stage recommendation systems:
  - L1 (Retrieval): Uses global indices like FAISS/Elasticsearch to narrow
    120K articles → ~200 candidates per user. This is what generate.py does.
  - L2 (Ranking): Takes those candidates and scores them with a heavy ML model.
    We do NOT query a global Elasticsearch index here — because we already know
    the candidates, we compute a fast in-memory Lexical Overlap instead.
    This is exactly how Google/YouTube rank their retrieved candidates.

Features computed (all local, no global index query needed):
  1. sem_score       – GPU batched dot product: user_vec · candidate_vec
  2. lexical_overlap – Jaccard overlap between user keyword tokens & candidate tokens
  3. log_popularity  – log(1 + global click count) for the candidate article
  4. hist_len        – number of articles in user's click history (cold/warm signal)
  5. cat_match       – 1 if candidate category == user's dominant category
  6. inv_position    – 1/position (editorial placement signal from the platform)

Why lexical_overlap instead of BM25:
  BM25 requires a global inverted index search per user (O(V) per query).
  For 6M users, this takes >2 hours on a single CPU.
  Lexical overlap is a pure dictionary lookup — O(|history_tokens|) — and
  captures the exact same signal (e.g., "Messi" appearing in the article text).

Usage:
    from src.features.feature_store import build_features, FEATURE_NAMES
    X, y, groups, imp_ids, art_ids = build_features(
        behaviors, articles, embedding_map, article_text_map, popularity_map
    )
"""

from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

FEATURE_NAMES = [
    "sem_score",
    "lexical_overlap",
    "log_popularity",
    "hist_len",
    "cat_match",
    "inv_position",
]


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


def build_features(
    behaviors: pl.DataFrame,
    articles: pl.DataFrame,
    embedding_map: dict,
    article_text_map: dict = None,
    popularity_map: dict = None,
    max_history: int = 10,
    chunk_size: int = 8000,
) -> tuple:
    """
    Build feature matrix for all (impression, candidate) pairs via GPU batching.

    Parameters
    ----------
    behaviors       : DataFrame with impression_id, user_id, history, impressions, labels
    articles        : DataFrame with article_id, category
    embedding_map   : dict {article_id -> np.ndarray(384,)}
    article_text_map: dict {article_id -> str}  — for lexical overlap
    popularity_map  : dict {article_id -> int}  — click counts from train
    max_history     : number of recent articles to use for user vector
    chunk_size      : batch size for GPU matrix multiplication

    Returns
    -------
    X      : np.ndarray (N_pairs, 6)
    y      : np.ndarray (N_pairs,)    — binary labels; -1 for test set
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
    DIM        = 384
    emb_id_list = list(embedding_map.keys())
    emb_id_to_idx = {aid: i for i, aid in enumerate(emb_id_list)}
    emb_np = np.array([embedding_map[aid] for aid in emb_id_list], dtype=np.float32)

    if use_gpu:
        import gc
        # Clear the sentence-transformer model from VRAM before we do matmul.
        # The model was loaded by load_or_compute_embeddings and takes ~1.5GB.
        # We only need the embedding numpy arrays from here on.
        gc.collect()
        torch.cuda.empty_cache()
        emb_gpu = torch.tensor(emb_np).cuda()  # (N_articles, 384)
        print(f"  GPU embedding matrix: {emb_gpu.shape} on CUDA ✓")
        free_gb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9
        print(f"  GPU VRAM free after loading matrix: {free_gb:.1f} GB")
    else:
        emb_gpu = emb_np
        print("  Running on CPU (GPU not available)")

    # ------------------------------------------------------------------
    # Collect all rows, process in GPU chunks for semantic scoring
    # ------------------------------------------------------------------
    n_behaviors = len(behaviors)
    all_rows = behaviors.to_dicts()

    # Build user vectors + per-impression metadata in one pass
    print("  Pre-computing user vectors...")
    user_vecs   = np.zeros((n_behaviors, DIM), dtype=np.float32)   # (N, 384)
    has_history = np.zeros(n_behaviors, dtype=bool)
    hist_lens   = np.zeros(n_behaviors, dtype=np.int32)
    user_cats   = [None] * n_behaviors
    user_token_sets = [None] * n_behaviors  # for lexical overlap

    for i, row in enumerate(tqdm(all_rows, desc="Building User Vectors")):
        history = row.get("history") or []
        hist_len = len(history)
        hist_lens[i] = min(hist_len, 50)
        user_cats[i] = _user_category(history, category_map)
        if history:
            has_history[i] = True
            recent = history[-max_history:]
            vecs = [embedding_map[aid] for aid in recent if aid in embedding_map]
            if vecs:
                user_vecs[i] = np.mean(vecs, axis=0)
            # Aggregate user's keyword tokens from history text
            tok_set = set()
            for aid in recent:
                tok_set.update(article_tokens.get(aid, set()))
            user_token_sets[i] = tok_set

    # ------------------------------------------------------------------
    # GPU batch matmul → extract candidate-level semantic scores inline
    # Instead of storing an (N_behaviors × N_articles) dense matrix (which
    # would be 500K × 130K × 4B = 260GB!), we process chunks and immediately
    # extract only the scores we need for the specific candidates.
    # ------------------------------------------------------------------
    print("  GPU batch semantic scoring + feature assembly...")

    # Pre-build per-impression candidate index lists for fast lookup
    all_cand_indices = []
    for row in all_rows:
        impressions = row.get("impressions") or []
        all_cand_indices.append([emb_id_to_idx.get(aid, -1) for aid in impressions])

    # Store per-pair semantic scores (compact: one float per candidate pair)
    all_sem_scores = []  # list of np.ndarray per impression

    for chunk_start in tqdm(range(0, n_behaviors, chunk_size), desc="GPU Batches"):
        chunk_end = min(chunk_start + chunk_size, n_behaviors)
        uv_batch  = user_vecs[chunk_start:chunk_end]  # (B, 384)

        if use_gpu:
            uv_t   = torch.tensor(uv_batch).cuda()
            sc_all = torch.mm(uv_t, emb_gpu.T).cpu().numpy()  # (B, N_articles)
            del uv_t  # free GPU immediately
        else:
            sc_all = np.dot(uv_batch, emb_gpu.T)  # (B, N_articles)

        # Extract only the candidate scores for this chunk
        for i in range(chunk_end - chunk_start):
            cand_idxs = all_cand_indices[chunk_start + i]
            sem_row = []
            for idx in cand_idxs:
                sem_row.append(float(sc_all[i, idx]) if idx >= 0 else 0.0)
            all_sem_scores.append(sem_row)

        del sc_all  # free the large intermediate array immediately

    if use_gpu:
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Build final (N_pairs, 6) feature matrix
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

        n_cands     = len(impressions)
        hl          = int(hist_lens[i])
        uc          = user_cats[i]
        u_tokens    = user_token_sets[i] or set()
        sem_vec     = all_sem_scores[i]  # pre-extracted candidate scores

        for pos, (aid, lbl) in enumerate(
            zip(impressions, labels if labels else [-1] * n_cands), start=1
        ):
            # Feature 1: Semantic similarity (pre-extracted per-candidate)
            sem = sem_vec[pos - 1] if (pos - 1) < len(sem_vec) else 0.0

            # Feature 2: Lexical overlap (L2-stage local computation, O(|tokens|))
            cand_tokens = article_tokens.get(aid, set())
            if u_tokens and cand_tokens:
                intersection = len(u_tokens & cand_tokens)
                union        = len(u_tokens | cand_tokens)
                lex_overlap  = intersection / union if union > 0 else 0.0
            else:
                lex_overlap = 0.0

            # Feature 3: Log-scaled popularity
            pop = np.log1p(popularity_map.get(aid, 0))

            # Feature 4: User history length (cold/warm user signal)
            # Already capped at 50

            # Feature 5: Category match
            cat_m = 1 if (uc and category_map.get(aid) == uc) else 0

            # Feature 6: Inverse position (editorial placement signal)
            pos_feat = 1.0 / pos

            X_rows.append([sem, lex_overlap, pop, hl, cat_m, pos_feat])
            y_rows.append(int(lbl) if lbl != -1 else -1)
            imp_ids.append(imp_id)
            art_ids.append(aid)

        groups.append(n_cands)

    X      = np.array(X_rows, dtype=np.float32)
    y      = np.array(y_rows,  dtype=np.int32)
    groups = np.array(groups,  dtype=np.int32)

    print(f"  Features built: {X.shape[0]:,} pairs, {X.shape[1]} features")
    return X, y, groups, imp_ids, art_ids
