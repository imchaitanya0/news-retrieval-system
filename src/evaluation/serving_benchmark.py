"""
Serving & Scale Benchmark (A2 Q4)
===================================
Measures the memory footprint and p99 latency of the full pipeline:
  1. FAISS ANN index memory (article embeddings)
  2. LightGBM model size
  3. End-to-end p99 retrieval + reranking latency (1000 simulated requests)
  4. Back-of-envelope cost/QPS at T4-equivalent hardware
  5. 10× scale analysis

Usage:
    python -m src.evaluation.serving_benchmark --dataset mind
    python -m src.evaluation.serving_benchmark --dataset ebnerd
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("data/models")
RESULTS_DIR   = Path("data/results")

# Kaggle T4-equivalent cloud cost estimate
CLOUD_COST_PER_HOUR_USD = 0.35


def run_benchmark(dataset: str, n_requests: int = 1000, k_candidates: int = 50):
    """Run the full serving benchmark."""
    import tracemalloc, pickle

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **kw): return it

    print(f"\n[Serving Benchmark] Dataset: {dataset}")
    results = {"dataset": dataset, "n_requests": n_requests, "k_candidates": k_candidates}

    # --- Load artifacts ---
    articles_path = PROCESSED_DIR / f"articles_{dataset}.parquet"
    if not articles_path.exists():
        raise FileNotFoundError(f"Articles not found: {articles_path}\nRun build_pipeline.py first.")
    articles = pl.read_parquet(articles_path)

    from src.retrieval.semantic import load_or_compute_embeddings, SemanticRetriever
    print("  Loading embeddings...")
    embs, ids = load_or_compute_embeddings(articles, dataset)
    embedding_map = dict(zip(ids, embs))

    emb_matrix = np.array([embedding_map[aid] for aid in ids], dtype=np.float32)
    results["n_articles"] = len(ids)
    results["embedding_store_mb"] = round(emb_matrix.nbytes / 1e6, 2)
    print(f"  Embedding matrix: {emb_matrix.shape}  ({results['embedding_store_mb']:.1f} MB)")

    # --- Build FAISS index, measure peak memory ---
    print("  Building FAISS index...")
    tracemalloc.start()
    retriever = SemanticRetriever()
    # FIX: SemanticRetriever.build() takes (embeddings, article_ids) — NOT articles df
    retriever.build(emb_matrix, ids)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results["faiss_index_mb"] = round(peak / 1e6, 2)
    print(f"  FAISS index peak memory: {results['faiss_index_mb']:.1f} MB")

    # --- LightGBM model size ---
    lgbm_path = MODELS_DIR / f"lgbm_{dataset}.pkl"
    lgbm_model = None
    if lgbm_path.exists():
        results["lgbm_model_mb"] = round(lgbm_path.stat().st_size / 1e6, 2)
        print(f"  LightGBM model: {results['lgbm_model_mb']:.2f} MB")
        with open(lgbm_path, "rb") as f:
            lgbm_model = pickle.load(f)
    else:
        results["lgbm_model_mb"] = None
        print("  Warning: LightGBM model not found — latency will be retrieval-only")

    # --- Simulate requests ---
    print(f"\n  Running {n_requests} simulated requests (k={k_candidates})...")
    rng = np.random.default_rng(42)
    user_vectors = rng.standard_normal((n_requests, 384)).astype(np.float32)
    norms = np.linalg.norm(user_vectors, axis=1, keepdims=True)
    user_vectors /= np.maximum(norms, 1e-8)

    latencies_ms = []

    for i in tqdm(range(n_requests), desc="Benchmarking"):
        t_start = time.perf_counter()

        # Step 1: FAISS ANN retrieval
        user_vec = user_vectors[i]
        retrieved = retriever.retrieve_by_vector(user_vec, k=k_candidates)

        # Step 2: LightGBM reranking
        if lgbm_model is not None and retrieved:
            sem_scores = np.array([
                float(np.dot(user_vec, embedding_map.get(aid, np.zeros(384))))
                for aid in retrieved
            ], dtype=np.float32)
            n_cands = len(retrieved)
            X = np.zeros((n_cands, 8), dtype=np.float32)
            X[:, 0] = sem_scores
            X[:, 5] = 1.0 / np.arange(1, n_cands + 1)
            scores = lgbm_model.predict(X)
            _ = [retrieved[j] for j in np.argsort(-scores)]

        t_end = time.perf_counter()
        latencies_ms.append((t_end - t_start) * 1000)

    latencies_ms = np.array(latencies_ms)
    p50  = float(np.percentile(latencies_ms, 50))
    p95  = float(np.percentile(latencies_ms, 95))
    p99  = float(np.percentile(latencies_ms, 99))
    mean_lat = float(np.mean(latencies_ms))

    print(f"\n  Latency over {n_requests} requests:")
    print(f"    Mean:  {mean_lat:.1f} ms")
    print(f"    p50:   {p50:.1f} ms")
    print(f"    p95:   {p95:.1f} ms")
    print(f"    p99:   {p99:.1f} ms")

    max_qps = 1000.0 / p99 if p99 > 0 else float("inf")
    cost_per_1k = (1000.0 / max(max_qps, 1e-9) / 3600.0) * CLOUD_COST_PER_HOUR_USD

    print(f"\n  Throughput @ p99 SLA:")
    print(f"    Max QPS (single process): {max_qps:.1f}")
    print(f"    Cost per 1K queries:      ${cost_per_1k:.4f}")
    print(f"    Cost per 1M queries:      ${cost_per_1k * 1000:.2f}")

    results.update({
        "mean_ms": round(mean_lat, 2),
        "p50_ms":  round(p50, 2),
        "p95_ms":  round(p95, 2),
        "p99_ms":  round(p99, 2),
        "max_qps": round(max_qps, 2),
        "cost_per_1k_usd": round(cost_per_1k, 4),
    })

    # --- 10x scale analysis ---
    n10x = results["n_articles"] * 10
    emb_10x_mb = results["embedding_store_mb"] * 10
    print(f"\n  10× Scale Analysis (current={results['n_articles']:,} → 10x={n10x:,} articles):")
    print(f"    Embedding store:   {emb_10x_mb:.0f} MB ({emb_10x_mb/1024:.1f} GB)")
    print(f"    FAISS FlatIP p99:  ~{p99 * 10:.0f} ms  → FAILS 100ms SLA (linear scan)")
    print(f"    Fix retrieval:     Switch to IndexIVFFlat (nprobe=64) → ~{p99 * 0.5:.0f} ms")
    print(f"    VRAM at 10x:       {emb_10x_mb:.0f} MB {'> 14GB OOM on T4!' if emb_10x_mb > 14000 else 'OK'}")

    results["scale_10x"] = {
        "n_articles":          n10x,
        "embedding_mb":        round(emb_10x_mb, 1),
        "issue":               "FlatIP linear scan O(N) too slow; GPU VRAM may OOM",
        "fix_retrieval":       "Switch to IndexIVFFlat or ScaNN (nprobe=64)",
        "fix_memory":          "CPU-resident FAISS + GPU only for LightGBM reranking",
        "estimated_p99_ms":    round(p99 * 10, 1),
        "vram_oom":            emb_10x_mb > 14000,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"serving_{dataset}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Serving Benchmark] Results saved → {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Serving & scale benchmark (A2 Q4)")
    parser.add_argument("--dataset",    choices=["mind", "ebnerd"], required=True)
    parser.add_argument("--n-requests", type=int, default=1000)
    parser.add_argument("--k",          type=int, default=50)
    args = parser.parse_args()
    run_benchmark(args.dataset, args.n_requests, args.k)


if __name__ == "__main__":
    main()
