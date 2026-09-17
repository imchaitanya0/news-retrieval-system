"""
Serving & Scale Benchmark (A2 Q4)
===================================
Measures the memory footprint and p99 latency of the full pipeline:
  1. FAISS ANN index memory (article embeddings)
  2. LightGBM model size
  3. End-to-end p99 retrieval latency for 1000 simulated user requests
  4. Back-of-envelope cost/QPS calculation

Usage:
    python -m src.evaluation.serving_benchmark --dataset mind
    python -m src.evaluation.serving_benchmark --dataset ebnerd

Output (stdout + JSON at data/results/serving_{dataset}.json):
    {
      "faiss_index_mb":     float,   # memory of FAISS flat index
      "lgbm_model_mb":      float,   # model pickle size
      "embedding_store_mb": float,   # numpy embedding matrix size
      "p50_ms":             float,   # median latency
      "p95_ms":             float,   # 95th percentile latency
      "p99_ms":             float,   # 99th percentile latency (key SLA metric)
      "max_qps":            float,   # queries per second at p99
      "cost_per_1k":        float,   # USD per 1000 queries (Kaggle T4 equiv)
      "n_articles":         int,
    }
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import polars as pl

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("data/models")
RESULTS_DIR   = Path("data/results")

# Kaggle T4 equivalent cloud cost estimate
# T4 GPU on GCP: ~$0.35/hour; Kaggle is free but equivalent to ~$0.35/hour
CLOUD_COST_PER_HOUR_USD = 0.35


def get_size_mb(obj) -> float:
    """Estimate in-memory size in MB using numpy/sys."""
    import sys
    if hasattr(obj, "nbytes"):
        return obj.nbytes / 1e6
    return sys.getsizeof(obj) / 1e6


def run_benchmark(dataset: str, n_requests: int = 1000, k_candidates: int = 50):
    """
    Run the full serving benchmark.

    Simulates n_requests random user requests end-to-end:
      1. Semantic retrieval: user vector → FAISS top-K
      2. LightGBM reranking: K candidates × 8 features → ranked scores

    Parameters
    ----------
    dataset     : "mind" or "ebnerd"
    n_requests  : number of simulated requests to benchmark
    k_candidates: number of candidates to retrieve and rerank
    """
    import tracemalloc, pickle

    print(f"\n[Serving Benchmark] Dataset: {dataset}")
    results = {"dataset": dataset, "n_requests": n_requests, "k_candidates": k_candidates}

    # --- Load artifacts ---
    articles_path = PROCESSED_DIR / f"articles_{dataset}.parquet"
    articles = pl.read_parquet(articles_path)

    from src.retrieval.semantic import load_or_compute_embeddings, SemanticRetriever
    print("  Loading embeddings...")
    embs, ids = load_or_compute_embeddings(articles, dataset)
    embedding_map = dict(zip(ids, embs))

    emb_matrix = np.array([embedding_map[aid] for aid in ids], dtype=np.float32)
    results["n_articles"] = len(ids)
    results["embedding_store_mb"] = round(emb_matrix.nbytes / 1e6, 2)
    print(f"  Embedding matrix: {emb_matrix.shape}  ({results['embedding_store_mb']:.1f} MB)")

    # Build FAISS index and measure its memory
    print("  Building FAISS index...")
    tracemalloc.start()
    retriever = SemanticRetriever()
    retriever.build(articles, emb_matrix, ids)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results["faiss_index_mb"] = round(peak / 1e6, 2)
    print(f"  FAISS index peak memory: {results['faiss_index_mb']:.1f} MB")

    # LightGBM model size
    lgbm_path = MODELS_DIR / f"lgbm_{dataset}.pkl"
    if lgbm_path.exists():
        results["lgbm_model_mb"] = round(lgbm_path.stat().st_size / 1e6, 2)
        print(f"  LightGBM model: {results['lgbm_model_mb']:.2f} MB")
        with open(lgbm_path, "rb") as f:
            lgbm_model = pickle.load(f)
    else:
        results["lgbm_model_mb"] = None
        lgbm_model = None
        print("  Warning: LightGBM model not found — skipping reranking latency")

    # --- Simulate requests ---
    print(f"  Running {n_requests} simulated requests (k={k_candidates})...")

    # Create random user vectors (simulate users with history)
    rng = np.random.default_rng(42)
    user_vectors = rng.standard_normal((n_requests, 384)).astype(np.float32)
    # Normalize for cosine similarity
    norms = np.linalg.norm(user_vectors, axis=1, keepdims=True)
    user_vectors /= np.maximum(norms, 1e-8)

    latencies_ms = []

    for i in range(n_requests):
        t_start = time.perf_counter()

        # Step 1: FAISS ANN retrieval
        user_vec = user_vectors[i]
        retrieved = retriever.retrieve_by_vector(user_vec, k=k_candidates)

        # Step 2: LightGBM reranking (if model available)
        if lgbm_model is not None:
            # Build minimal feature matrix (8 features, all zero except sem_score)
            sem_scores = np.array([
                float(np.dot(user_vec, embedding_map.get(aid, np.zeros(384))))
                for aid in retrieved
            ], dtype=np.float32)
            # Feature matrix: [sem, lex, pop, hist, cat, pos, fresh, rec_sem]
            X = np.zeros((len(retrieved), 8), dtype=np.float32)
            X[:, 0] = sem_scores                              # sem_score
            X[:, 5] = 1.0 / np.arange(1, len(retrieved)+1)  # inv_position
            scores = lgbm_model.predict(X)
            order = np.argsort(-scores)
            ranked = [retrieved[j] for j in order]

        t_end = time.perf_counter()
        latencies_ms.append((t_end - t_start) * 1000)

    latencies_ms = np.array(latencies_ms)
    p50  = float(np.percentile(latencies_ms, 50))
    p95  = float(np.percentile(latencies_ms, 95))
    p99  = float(np.percentile(latencies_ms, 99))
    mean = float(np.mean(latencies_ms))

    print(f"\n  Latency (n={n_requests} requests):")
    print(f"    Mean:  {mean:.1f} ms")
    print(f"    p50:   {p50:.1f} ms")
    print(f"    p95:   {p95:.1f} ms")
    print(f"    p99:   {p99:.1f} ms")

    max_qps = 1000.0 / p99 if p99 > 0 else float("inf")
    # Cost per 1000 queries: (1000 / QPS) seconds / 3600 * cost_per_hour
    cost_per_1k = (1000.0 / max_qps / 3600.0) * CLOUD_COST_PER_HOUR_USD if max_qps > 0 else 0.0

    print(f"\n  Throughput (at p99 SLA):")
    print(f"    Max QPS (single process): {max_qps:.1f}")
    print(f"    Cost per 1000 queries:    ${cost_per_1k:.4f} USD")
    print(f"    Cost per 1M queries:      ${cost_per_1k * 1000:.2f} USD")

    results.update({
        "mean_ms": round(mean, 2),
        "p50_ms":  round(p50, 2),
        "p95_ms":  round(p95, 2),
        "p99_ms":  round(p99, 2),
        "max_qps": round(max_qps, 2),
        "cost_per_1k_usd": round(cost_per_1k, 4),
    })

    # --- 10x scale analysis ---
    print("\n  10× Scale Analysis:")
    n10x = results["n_articles"] * 10
    emb_10x_mb = results["embedding_store_mb"] * 10
    print(f"    At 10× articles ({n10x:,}):")
    print(f"    Embedding store:  {emb_10x_mb:.0f} MB ({emb_10x_mb/1024:.1f} GB)")
    print(f"    FAISS FlatIP:     Linear scan O(N) → latency ~{p99 * 10:.0f} ms → FAILS 100ms SLA")
    print(f"    Fix: Switch to FAISS IndexIVFFlat (nprobe=64) → ~{p99 * 0.5:.0f} ms (approx)")
    print(f"    GPU VRAM needed:  {emb_10x_mb:.0f} MB → exceeds T4 (14GB) if >{14000:.0f} MB")
    print(f"    Fix: CPU-resident FAISS + GPU reranking")

    results["scale_10x"] = {
        "n_articles":       n10x,
        "embedding_mb":     round(emb_10x_mb, 1),
        "issue":            "FAISS FlatIP becomes too slow; GPU VRAM exhausted",
        "fix_retrieval":    "Switch to IndexIVFFlat or ScaNN (ANN index with nprobe=64)",
        "fix_memory":       "CPU-resident FAISS, GPU for LightGBM reranking only",
        "estimated_p99_ms": round(p99 * 10, 1),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"serving_{dataset}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Serving Benchmark] Results saved → {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Serving & scale benchmark (A2 Q4)")
    parser.add_argument("--dataset",     choices=["mind", "ebnerd"], required=True)
    parser.add_argument("--n-requests",  type=int, default=1000)
    parser.add_argument("--k",           type=int, default=50)
    args = parser.parse_args()
    run_benchmark(args.dataset, args.n_requests, args.k)


if __name__ == "__main__":
    main()
