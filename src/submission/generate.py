"""
Codabench Submission Generator (Q5)
=====================================
Generates prediction files for both MIND and EB-NeRD leaderboards.

MIND format   : {impression_id} [{rank1},{rank2},...}]  file: prediction.txt
EB-NeRD format: {impression_id} [{rank1},{rank2},...}]  file: predictions.txt  (note the 's')

Scoring strategy (configurable):
  "hybrid"   — 70% semantic + 20% BM25 overlap + 10% popularity (recommended)
  "semantic" — pure semantic similarity
  "bm25"     — pure BM25
  "random"   — random baseline
  "oracle"   — uses labels (only valid on val set, not test)

GPU acceleration:
  When CUDA is available (Kaggle GPU tier), all user-candidate dot products are
  computed as a single GPU batched matrix multiply (10K users × 120K articles × 384 dims).
  This runs in seconds instead of hours.

Usage:
    python -m src.submission.generate --dataset mind --strategy hybrid
    python -m src.submission.generate --dataset ebnerd --strategy hybrid
"""

import argparse
import shutil
import zipfile
from pathlib import Path

import numpy as np
import polars as pl

PROCESSED_DIR = Path("data/processed")
SUBMISSION_DIR = Path("data/submissions")


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
    Generate a full Codabench submission file for the given dataset/split.

    Parameters
    ----------
    dataset   : "mind" or "ebnerd"
    split     : "test" for final submission, "val" for local evaluation
    strategy  : "hybrid" | "bm25" | "semantic" | "random" | "oracle"
    max_history : number of recent clicks to use for user profile
    """
    behaviors_path = PROCESSED_DIR / f"behaviors_{dataset}_{split}.parquet"
    articles_path  = PROCESSED_DIR / f"articles_{dataset}.parquet"

    if not behaviors_path.exists():
        raise FileNotFoundError(f"Run build_pipeline.py first: {behaviors_path}")
    if not articles_path.exists():
        raise FileNotFoundError(f"Run build_pipeline.py first: {articles_path}")

    print(f"\n[Submit] Generating {dataset} / {split} with strategy={strategy}")
    articles  = pl.read_parquet(articles_path)
    # Sort by impression_id to preserve the original test-file row order
    # (both competitions require this for their evaluators)
    behaviors = pl.read_parquet(behaviors_path).sort("impression_id")
    n_behaviors = len(behaviors)
    print(f"  Total impressions to process: {n_behaviors:,}")

    # ------------------------------------------------------------------ #
    # 1. Build article text map and BM25 global index                     #
    # ------------------------------------------------------------------ #
    article_text_map = {}
    retriever_bm25   = None

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

    # ------------------------------------------------------------------ #
    # 2. Load / compute article embeddings                                #
    # ------------------------------------------------------------------ #
    embedding_map = {}

    if strategy in ("semantic", "hybrid"):
        from src.retrieval.semantic import load_or_compute_embeddings
        embs, ids = load_or_compute_embeddings(articles, dataset)
        embedding_map = dict(zip(ids, embs))

    from src.retrieval.bm25 import build_query_from_history
    from src.retrieval.semantic import build_user_vector

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **kw): return it

    # ------------------------------------------------------------------ #
    # 3. Pre-compute global article popularity (cold-start signal)        #
    #    = how many times each article appears as an impression candidate  #
    # ------------------------------------------------------------------ #
    print("  Pre-computing article popularity...")
    popularity: dict = {}
    for (imps,) in behaviors.select("impressions").iter_rows():
        for aid in (imps or []):
            popularity[aid] = popularity.get(aid, 0) + 1
    pop_max = max(popularity.values()) if popularity else 1
    print(f"  Popularity computed for {len(popularity):,} articles.")

    # ------------------------------------------------------------------ #
    # 4. Build GPU (or CPU) embedding matrix for batch dot-products       #
    # ------------------------------------------------------------------ #
    use_gpu = False
    emb_matrix = None
    emb_id_list: list = []
    emb_id_to_idx: dict = {}

    if embedding_map:
        emb_id_list    = list(embedding_map.keys())
        emb_id_to_idx  = {aid: i for i, aid in enumerate(emb_id_list)}
        emb_np = np.array([embedding_map[aid] for aid in emb_id_list], dtype=np.float32)
        try:
            import torch
            if torch.cuda.is_available():
                emb_matrix = torch.tensor(emb_np).cuda()
                use_gpu = True
                print(f"  GPU embedding matrix: {emb_matrix.shape} on CUDA ✓")
        except Exception:
            pass
        if not use_gpu:
            emb_matrix = emb_np
            print(f"  CPU embedding matrix: {emb_np.shape}")

    # ------------------------------------------------------------------ #
    # 5. Scoring loop — batched, GPU-accelerated                          #
    # ------------------------------------------------------------------ #
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    txt_name = "predictions.txt" if dataset == "ebnerd" else "prediction.txt"
    out_path  = SUBMISSION_DIR / f"{dataset}_{split}_{strategy}.txt"
    txt_path  = SUBMISSION_DIR / txt_name

    chunk_size = 10_000
    n_written  = 0

    with open(out_path, "w") as out_f:
        for chunk_start in tqdm(range(0, n_behaviors, chunk_size), desc="Generating"):
            batch = behaviors.slice(chunk_start, chunk_size).to_dicts()

            # Collect per-row data
            imp_ids    = [row["impression_id"] for row in batch]
            imps_list  = [row.get("impressions") or [] for row in batch]
            hist_list  = [row.get("history") or [] for row in batch]

            # --- BM25 global retrieval for the batch (top-500 from global index) ---
            bm25_sets = [set() for _ in batch]
            if retriever_bm25 and any(hist_list):
                import bm25s
                queries = [
                    build_query_from_history(h, article_text_map, max_history)
                    for h in hist_list
                ]
                non_empty = [(i, q) for i, q in enumerate(queries) if q]
                if non_empty:
                    idxs, qs = zip(*non_empty)
                    k_val = min(500, len(retriever_bm25.article_ids))
                    tokens = bm25s.tokenize(list(qs), lower=True, show_progress=False)
                    res_idx, _ = retriever_bm25._index.retrieve(
                        tokens, k=k_val, show_progress=False
                    )
                    for bi, indices in zip(idxs, res_idx):
                        bm25_sets[bi] = {retriever_bm25.article_ids[idx] for idx in indices}

            # --- Build user vectors (mean of clicked article embeddings) ---
            user_vecs = [None] * len(batch)
            if emb_matrix is not None:
                for i, hist in enumerate(hist_list):
                    uv = build_user_vector(hist, embedding_map, max_history)
                    user_vecs[i] = uv

            # --- GPU/CPU batch semantic scoring ---
            # For each user, compute dot product with ALL article embeddings in one go,
            # then index into the result for each impression's candidates.
            sem_score_rows = [None] * len(batch)
            if emb_matrix is not None:
                valid_idx = [i for i, uv in enumerate(user_vecs) if uv is not None]
                if valid_idx:
                    uvs_np = np.array([user_vecs[i] for i in valid_idx], dtype=np.float32)
                    if use_gpu:
                        import torch
                        uv_t = torch.tensor(uvs_np).cuda()
                        # (n_valid, n_articles) — one GPU matmul, super fast
                        scores_t = torch.mm(uv_t, emb_matrix.T).cpu().numpy()
                    else:
                        scores_t = np.dot(uvs_np, emb_matrix.T)
                    for bi, row_scores in zip(valid_idx, scores_t):
                        sem_score_rows[bi] = row_scores  # shape: (n_articles,)

            # --- Per-impression fusion & write ---
            for r in range(len(batch)):
                imp_id      = imp_ids[r]
                impressions = imps_list[r]
                if not impressions:
                    continue

                scores = {}
                sem_row = sem_score_rows[r]
                b_set   = bm25_sets[r]

                for aid in impressions:
                    s = 0.0
                    # 70% weight: semantic similarity (direct dot product)
                    if sem_row is not None and aid in emb_id_to_idx:
                        s += 0.7 * float(sem_row[emb_id_to_idx[aid]])
                    # 20% weight: BM25 global overlap bonus
                    if aid in b_set:
                        s += 0.2
                    # 10% weight: global popularity (cold-start tiebreaker)
                    s += 0.1 * (popularity.get(aid, 0) / pop_max)
                    scores[aid] = s

                if strategy == "semantic":
                    if sem_row is not None:
                        ranked = sorted(impressions,
                                        key=lambda a: -(float(sem_row[emb_id_to_idx[a]])
                                                        if a in emb_id_to_idx else 0.0))
                    else:
                        ranked = list(impressions)
                elif strategy == "bm25":
                    ordered = [a for a in b_set if a in set(impressions)]
                    rest    = [a for a in impressions if a not in set(ordered)]
                    ranked  = ordered + rest
                elif strategy == "hybrid":
                    ranked = sorted(impressions, key=lambda a: -scores.get(a, 0.0))
                elif strategy == "random":
                    ranked = list(impressions)
                    np.random.shuffle(ranked)
                else:
                    ranked = list(impressions)

                rank_map = {aid: rk for rk, aid in enumerate(ranked, 1)}
                ranks    = [str(rank_map.get(aid, len(ranked) + 1)) for aid in impressions]
                out_f.write(f"{imp_id} [{','.join(ranks)}]\n")
                n_written += 1

    print(f"Submission saved → {out_path} ({n_written:,} impressions)")

    # Auto-create correctly named zip for Codabench upload
    shutil.copy(out_path, txt_path)
    zip_path = SUBMISSION_DIR / f"{dataset}_{split}_{strategy}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(txt_path, arcname=txt_name)
    print(f"Codabench zip → {zip_path}  (contains {txt_name})")


# ------------------------------------------------------------------ #
#  CLI                                                                #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Codabench Submission Generator")
    parser.add_argument("--dataset",  choices=["mind", "ebnerd"], required=True)
    parser.add_argument("--split",    default="test")
    parser.add_argument("--strategy", default="hybrid",
                        choices=["hybrid", "bm25", "semantic", "random", "oracle"])
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
