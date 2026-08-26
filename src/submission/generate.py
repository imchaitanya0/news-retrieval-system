"""
Codabench Submission Generator (Q5)
=====================================
GPU-accelerated direct candidate scoring for MIND and EB-NeRD.

Scoring: 0.8 × semantic_similarity + 0.2 × global_popularity

- Semantic: dot product of user-embedding vs each candidate's embedding (GPU T4 batch matmul)
- Popularity: how often the article appeared as a candidate in the test set (cold-start signal)
- BM25 is NOT used in the hot loop (batch BM25 retrieval of 10K × 120K = too slow for reranking)

MIND format   → prediction.txt   inside zip
EB-NeRD format→ predictions.txt  inside zip  (note the 's')

Format per line:  {impression_id} [{rank_of_cand1},{rank_of_cand2},...]
"""

import argparse
import shutil
import zipfile
from pathlib import Path

import numpy as np
import polars as pl

PROCESSED_DIR = Path("data/processed")
SUBMISSION_DIR = Path("data/submissions")


def generate_submission(
    dataset: str,
    split: str = "test",
    strategy: str = "hybrid",
    max_history: int = 10,
) -> None:
    behaviors_path = PROCESSED_DIR / f"behaviors_{dataset}_{split}.parquet"
    articles_path  = PROCESSED_DIR / f"articles_{dataset}.parquet"

    if not behaviors_path.exists():
        raise FileNotFoundError(f"Run build_pipeline.py first: {behaviors_path}")
    if not articles_path.exists():
        raise FileNotFoundError(f"Run build_pipeline.py first: {articles_path}")

    print(f"\n[Submit] {dataset}/{split} strategy={strategy}")
    articles  = pl.read_parquet(articles_path)
    behaviors = pl.read_parquet(behaviors_path).sort("impression_id")
    n_imp = len(behaviors)
    print(f"  {n_imp:,} impressions")

    # ------------------------------------------------------------------
    # 1. Load embeddings
    # ------------------------------------------------------------------
    from src.retrieval.semantic import load_or_compute_embeddings
    embs, ids = load_or_compute_embeddings(articles, dataset)
    embedding_map   = dict(zip(ids, embs))
    emb_id_list     = list(embedding_map.keys())
    emb_id_to_idx   = {aid: i for i, aid in enumerate(emb_id_list)}
    emb_np = np.array([embedding_map[aid] for aid in emb_id_list], dtype=np.float32)

    # ------------------------------------------------------------------
    # 2. GPU setup — load embedding matrix to CUDA once
    # ------------------------------------------------------------------
    use_gpu = False
    try:
        import torch
        if torch.cuda.is_available():
            emb_gpu = torch.tensor(emb_np).cuda()
            use_gpu = True
            # Free transformer model from GPU memory before matmul
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            free_gb = torch.cuda.mem_get_info()[0] / 1e9
            print(f"  GPU: {emb_gpu.shape} on CUDA | {free_gb:.1f}GB free")
    except Exception:
        pass
    if not use_gpu:
        emb_gpu = emb_np
        print("  CPU fallback for embeddings")

    # ------------------------------------------------------------------
    # 3. Pre-compute global popularity from test impressions (cold-start)
    # ------------------------------------------------------------------
    print("  Computing article popularity...")
    pop: dict = {}
    for (imps,) in behaviors.select("impressions").iter_rows():
        for aid in (imps or []):
            pop[aid] = pop.get(aid, 0) + 1
    pop_max = max(pop.values()) if pop else 1
    print(f"  Done: {len(pop):,} articles tracked.")

    # ------------------------------------------------------------------
    # 4. BM25 (optional, only used for --strategy bm25)
    # ------------------------------------------------------------------
    retriever_bm25   = None
    article_text_map = {}
    if strategy == "bm25":
        from src.retrieval.bm25 import BM25Retriever, _article_text, build_query_from_history
        index_path = Path("data/feature_store/bm25") / dataset
        retriever_bm25 = BM25Retriever()
        if (index_path / "bm25_index.pkl").exists():
            retriever_bm25.load(index_path)
        else:
            retriever_bm25.build(articles)
            retriever_bm25.save(index_path)
        art_rows = articles.select(["article_id", "title", "subtitle"]).to_dicts()
        article_text_map = {r["article_id"]: _article_text(r) for r in art_rows}

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **kw): return it

    # ------------------------------------------------------------------
    # 5. Main scoring loop — GPU batch matmul
    # ------------------------------------------------------------------
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    txt_name = "predictions.txt" if dataset == "ebnerd" else "prediction.txt"
    out_path  = SUBMISSION_DIR / f"{dataset}_{split}_{strategy}.txt"
    txt_path  = SUBMISSION_DIR / txt_name

    DIM        = emb_np.shape[1]
    chunk_size = 2000   # safe for T4: 2000 × 120K × 4B = 960MB well within 14GB
    n_written  = 0

    with open(out_path, "w") as out_f:
        for chunk_start in tqdm(range(0, n_imp, chunk_size), desc="Scoring"):
            batch = behaviors.slice(chunk_start, chunk_size).to_dicts()
            B = len(batch)

            # Build user vectors for entire batch (vectorized numpy)
            uvs = np.zeros((B, DIM), dtype=np.float32)
            has_vec = np.zeros(B, dtype=bool)
            for i, row in enumerate(batch):
                hist = row.get("history") or []
                if not hist:
                    continue
                recent = hist[-max_history:]
                vecs = [embedding_map[aid] for aid in recent if aid in embedding_map]
                if vecs:
                    uvs[i] = np.mean(vecs, axis=0)
                    has_vec[i] = True

            # Single GPU matmul: (B, DIM) × (DIM, N_articles) → (B, N_articles)
            if use_gpu:
                import torch
                uv_t   = torch.tensor(uvs).cuda()
                sc_all = torch.mm(uv_t, emb_gpu.T).cpu().numpy()  # (B, N_articles)
            else:
                sc_all = np.dot(uvs, emb_gpu.T)                   # (B, N_articles)

            # Per-impression write
            for i, row in enumerate(batch):
                imp_id      = row["impression_id"]
                impressions = row.get("impressions") or []
                if not impressions:
                    continue

                # Gather semantic scores for candidates only
                cand_idx = np.array([emb_id_to_idx.get(aid, -1) for aid in impressions])
                sem = np.where(
                    cand_idx >= 0,
                    sc_all[i, np.where(cand_idx >= 0, cand_idx, 0)],
                    0.0
                )

                # Popularity scores
                pop_s = np.array([pop.get(aid, 0) / pop_max for aid in impressions],
                                 dtype=np.float32)

                if strategy == "random":
                    order = np.random.permutation(len(impressions))
                elif strategy == "bm25" and retriever_bm25 and has_vec[i]:
                    from src.retrieval.bm25 import build_query_from_history
                    hist  = row.get("history") or []
                    query = build_query_from_history(hist, article_text_map, max_history)
                    order = list(range(len(impressions)))  # fallback
                    if query:
                        import bm25s
                        k_val  = min(200, len(retriever_bm25.article_ids))
                        tokens = bm25s.tokenize([query], lower=True, show_progress=False)
                        res, _ = retriever_bm25._index.retrieve(tokens, k=k_val,
                                                                show_progress=False)
                        top_ids = {retriever_bm25.article_ids[idx] for idx in res[0]}
                        order   = sorted(range(len(impressions)),
                                         key=lambda j: (impressions[j] not in top_ids,
                                                        -pop_s[j]))
                else:
                    # Hybrid / semantic: weighted sum
                    final = 0.8 * sem + 0.2 * pop_s
                    order = np.argsort(-final)

                # Compute rank of each original candidate
                rank_of = np.empty(len(impressions), dtype=np.int32)
                rank_of[order] = np.arange(1, len(impressions) + 1)

                out_f.write(f"{imp_id} [{','.join(map(str, rank_of))}]\n")
                n_written += 1

    print(f"Saved → {out_path}  ({n_written:,} impressions)")

    # Auto-create Codabench zip
    shutil.copy(out_path, txt_path)
    zip_path = SUBMISSION_DIR / f"{dataset}_{split}_{strategy}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(txt_path, arcname=txt_name)
    print(f"Zip → {zip_path}  [{txt_name}]")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",  choices=["mind", "ebnerd"], required=True)
    p.add_argument("--split",    default="test")
    p.add_argument("--strategy", default="hybrid",
                   choices=["hybrid", "semantic", "bm25", "random", "oracle"])
    p.add_argument("--max-history", type=int, default=10)
    a = p.parse_args()
    generate_submission(dataset=a.dataset, split=a.split,
                        strategy=a.strategy, max_history=a.max_history)


if __name__ == "__main__":
    main()
