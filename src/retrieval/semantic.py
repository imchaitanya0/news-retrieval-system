"""
Semantic Retriever (Q3)
=======================
Computes or loads article embeddings, builds a FAISS ANN index, and
retrieves the top-K candidates given a user representation (mean pool
of clicked article embeddings).

Strategy
--------
1. Prefer pre-computed EB-NeRD Word2Vec embeddings if available.
2. Otherwise fall back to sentence-transformers
   (paraphrase-multilingual-MiniLM-L12-v2 — fast, multilingual, free tier).
3. User vector = mean of the embeddings of their recently clicked articles.
4. Retrieve top-K via FAISS flat (exact cosine) — sufficient at demo/small scale.

Usage (standalone):
    python -m src.retrieval.semantic --dataset mind --split val --k 100
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import polars as pl

PROCESSED_DIR = Path("data/processed")
EMBED_DIR = Path("data/feature_store/embeddings")
INDEX_DIR = Path("data/feature_store/semantic")

# Sentence-transformer model used when no pre-computed embeddings exist
_ST_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# --------------------------------------------------------------------- #
#  Embedding computation                                                  #
# --------------------------------------------------------------------- #

def compute_st_embeddings(articles: pl.DataFrame, batch_size: int = 256) -> np.ndarray:
    """
    Compute embeddings for all articles using sentence-transformers.

    Encodes 'title + subtitle' for each article.

    Parameters
    ----------
    articles : pl.DataFrame
        Must have columns: title, subtitle
    batch_size : int
        Sentences per batch (tune down if OOM).

    Returns
    -------
    np.ndarray of shape (n_articles, embedding_dim), float32
    """
    from sentence_transformers import SentenceTransformer

    print(f"Loading sentence-transformer: {_ST_MODEL}")
    model = SentenceTransformer(_ST_MODEL)

    texts = [
        f"{(r['title'] or '')} {(r['subtitle'] or '')}".strip()
        for r in articles.select(["title", "subtitle"]).to_dicts()
    ]
    print(f"Encoding {len(texts)} articles (batch_size={batch_size}) ...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # pre-normalise for cosine similarity
    )
    return embeddings.astype(np.float32)


def load_or_compute_embeddings(
    articles: pl.DataFrame,
    dataset: str,
) -> tuple:
    """
    Load cached embeddings from disk, or compute and cache them.

    Returns
    -------
    (embeddings, article_ids) where embeddings is (N, D) float32 ndarray
    and article_ids is a list aligned with the embedding rows.
    """
    cache_dir = EMBED_DIR / dataset
    emb_path = cache_dir / "article_embeddings.npy"
    ids_path = cache_dir / "article_ids.pkl"

    if emb_path.exists() and ids_path.exists():
        print(f"Loading cached embeddings from {cache_dir} ...")
        embeddings = np.load(emb_path)
        with open(ids_path, "rb") as f:
            article_ids = pickle.load(f)
        print(f"  Loaded {len(article_ids)} embeddings, dim={embeddings.shape[1]}")
        return embeddings, article_ids

    # Compute
    embeddings = compute_st_embeddings(articles)
    article_ids = articles["article_id"].to_list()

    # Cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, embeddings)
    with open(ids_path, "wb") as f:
        pickle.dump(article_ids, f, protocol=4)
    print(f"  Embeddings saved to {cache_dir}")
    return embeddings, article_ids


# --------------------------------------------------------------------- #
#  FAISS index                                                            #
# --------------------------------------------------------------------- #

class SemanticRetriever:
    """
    FAISS-backed ANN retriever for article embeddings.

    Uses exact inner-product search (equivalent to cosine when vectors
    are L2-normalised, which sentence-transformers does by default with
    normalize_embeddings=True).
    """

    def __init__(self):
        self.article_ids = []
        self._index = None      # faiss.IndexFlatIP

    # ------------------------------------------------------------------ #
    #  Build                                                               #
    # ------------------------------------------------------------------ #

    def build(self, embeddings: np.ndarray, article_ids: list) -> None:
        """
        Build a FAISS flat inner-product index.

        Parameters
        ----------
        embeddings : np.ndarray (N, D), float32, L2-normalised
        article_ids : list aligned with embedding rows
        """
        import faiss
        import os

        embeddings = np.ascontiguousarray(embeddings.astype('float32'))
        dim = embeddings.shape[1]
        n = embeddings.shape[0]
        print(f"Building FAISS FlatIP index (dim={dim}, n={len(article_ids)}) ...")

        use_gpu = (
            os.environ.get('FAISS_GPU', '1') == '1'
            and hasattr(faiss, 'StandardGpuResources')
            and hasattr(faiss, 'GpuIndexFlatIP')
        )

        if use_gpu:
            try:
                res = faiss.StandardGpuResources()
                cfg = faiss.GpuIndexFlatConfig()
                cfg.device = 0
                cfg.useFloat16 = False
                self._index = faiss.GpuIndexFlatIP(res, dim, cfg)
                self._index.add(embeddings)
                print(f"[semantic] GPU FAISS index — {n:,} vectors, dim={dim}")
            except Exception as e:
                print(f"[semantic] GPU index failed: {e}; using CPU")
                self._index = faiss.IndexFlatIP(dim)
                self._index.add(embeddings)
                print(f"[semantic] CPU FAISS index — {n:,} vectors, dim={dim}")
        else:
            self._index = faiss.IndexFlatIP(dim)
            self._index.add(embeddings)
            print(f"[semantic] CPU FAISS index — {n:,} vectors, dim={dim}")
        self.article_ids = article_ids
        print(f"  Index ready.")

    # ------------------------------------------------------------------ #
    #  Retrieve                                                            #
    # ------------------------------------------------------------------ #

    def retrieve_by_vector(self, query_vec: np.ndarray, k: int = 100) -> list:
        """
        Retrieve top-k article IDs for a query embedding vector.

        Parameters
        ----------
        query_vec : np.ndarray (D,) or (1, D), float32
        k : int

        Returns
        -------
        list of article_id
        """
        if query_vec.ndim == 1:
            query_vec = query_vec[np.newaxis, :]
        # Normalise query (safe guard)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm
        scores, indices = self._index.search(query_vec.astype(np.float32), k)
        return [self.article_ids[i] for i in indices[0] if i >= 0]

    # ------------------------------------------------------------------ #
    #  Save / Load                                                         #
    # ------------------------------------------------------------------ #

    def save(self, path: Path) -> None:
        import faiss
        path.mkdir(parents=True, exist_ok=True)
        
        # GPU index cannot be serialized directly — move to CPU first
        if hasattr(faiss, 'index_gpu_to_cpu') and hasattr(self._index, 'getDevice'):
            cpu_index = faiss.index_gpu_to_cpu(self._index)
            faiss.write_index(cpu_index, str(path / "faiss.index"))
        else:
            faiss.write_index(self._index, str(path / "faiss.index"))
            
        with open(path / "article_ids.pkl", "wb") as f:
            pickle.dump(self.article_ids, f, protocol=4)
        print(f"  FAISS index saved to {path}")

    def load(self, path: Path) -> None:
        import faiss
        import os

        index_file = path / "faiss.index"
        ids_file = path / "article_ids.pkl"

        if not index_file.exists() or not ids_file.exists():
            self._index = None
            self.article_ids = []
            print("  No cached index — will rebuild")
            return

        try:
            cpu_index = faiss.read_index(str(index_file))
        except Exception as e:
            print(f"  Corrupt faiss.index ({e}); deleting and rebuilding")
            index_file.unlink(missing_ok=True)
            self._index = None
            self.article_ids = []
            return

        self._index = cpu_index
        with open(ids_file, "rb") as f:
            self.article_ids = pickle.load(f)
        
        use_gpu = (
            os.environ.get('FAISS_GPU', '1') == '1'
            and hasattr(faiss, 'GpuIndexFlatIP')
        )
        if use_gpu:
            try:
                # Extract raw vectors from CPU index
                n = cpu_index.ntotal
                d = cpu_index.d
                vecs = cpu_index.reconstruct_n(0, n)
                res = faiss.StandardGpuResources()
                cfg = faiss.GpuIndexFlatConfig()
                cfg.device = 0
                cfg.useFloat16 = False
                gpu = faiss.GpuIndexFlatIP(res, d, cfg)
                gpu.add(vecs)
                self._index = gpu
                print(f"  FAISS index loaded & moved to GPU ({len(self.article_ids)} articles)")
            except Exception as e:
                print(f"  FAISS GPU load failed: {e}. Using CPU ({len(self.article_ids)} articles)")
        else:
            print(f"  FAISS index loaded on CPU ({len(self.article_ids)} articles)")


# --------------------------------------------------------------------- #
#  User representation                                                    #
# --------------------------------------------------------------------- #

def build_user_vector(
    history: list,
    embedding_map: dict,
    max_articles: int = 10,
) -> np.ndarray | None:
    """
    Compute user vector as mean of the most recent clicked article embeddings.
    """
    recent = history[-max_articles:] if history else []
    vecs = [embedding_map[aid] for aid in recent if aid in embedding_map]
    if not vecs:
        return None
    user_vec = np.mean(vecs, axis=0).astype(np.float32)
    norm = np.linalg.norm(user_vec)
    if norm > 0:
        user_vec /= norm
    return user_vec


# --------------------------------------------------------------------- #
#  Evaluation                                                             #
# --------------------------------------------------------------------- #

def recall_at_k(retrieved: list, ground_truth: set, k: int) -> float:
    """Recall@K = |retrieved[:k] ∩ ground_truth| / |ground_truth|"""
    if not ground_truth:
        return 0.0
    return len(set(retrieved[:k]) & ground_truth) / len(ground_truth)


def evaluate_semantic(
    dataset: str,
    split: str = "val",
    k_values: list = None,
    max_history: int = 10,
) -> dict:
    if k_values is None:
        k_values = [50, 100, 200]

    articles_path = PROCESSED_DIR / f"articles_{dataset}.parquet"
    behaviors_path = PROCESSED_DIR / f"behaviors_{dataset}_{split}.parquet"

    if not articles_path.exists():
        raise FileNotFoundError(f"Run build_pipeline.py first: {articles_path}")
    if not behaviors_path.exists():
        raise FileNotFoundError(f"Run build_pipeline.py first: {behaviors_path}")

    print(f"\n[Semantic] Evaluating {dataset} / {split}")
    articles = pl.read_parquet(articles_path)
    behaviors = pl.read_parquet(behaviors_path)

    # Load / compute embeddings
    embeddings, article_ids = load_or_compute_embeddings(articles, dataset)

    # Build or load FAISS index
    index_path = INDEX_DIR / dataset
    retriever = SemanticRetriever()
    if (index_path / "faiss.index").exists():
        retriever.load(index_path)
        
    if retriever._index is None:
        retriever.build(embeddings, article_ids)
        retriever.save(index_path)

    # embedding_map: {article_id -> vector}
    embedding_map = {aid: emb for aid, emb in zip(article_ids, embeddings)}

    max_k = max(k_values)
    recall_sums = {k: 0.0 for k in k_values}
    n_evaluated = 0
    n_no_history = 0

    has_history = "history" in behaviors.columns

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **kw): return it

    user_vecs = []
    ground_truths = []

    for row in tqdm(behaviors.iter_rows(named=True), total=len(behaviors), desc="Building queries"):
        impressions = row["impressions"] or []
        labels = row["labels"] or []
        ground_truth = {aid for aid, lbl in zip(impressions, labels) if lbl == 1}
        if not ground_truth:
            continue

        history = row["history"] if has_history else []
        if isinstance(history, str):
            history = history.split() if history else []
        if not history:
            history = list(ground_truth)
            n_no_history += 1

        user_vec = build_user_vector(history, embedding_map, max_articles=max_history)
        if user_vec is None:
            n_no_history += 1
            continue
            
        user_vecs.append(user_vec)
        ground_truths.append(ground_truth)

    if not user_vecs:
        print("  Warning: no impressions could be evaluated.")
        return {f"recall@{k}": 0.0 for k in k_values}

    # Batch FAISS search (drops execution time from hours to seconds)
    print(f"  Running FAISS batched search for {len(user_vecs):,} queries...")
    query_matrix = np.vstack(user_vecs).astype(np.float32)
    
    # Optional chunking to avoid OOM on massive test sets
    chunk_size = 50_000
    all_indices = []
    for i in tqdm(range(0, len(query_matrix), chunk_size), desc="FAISS search"):
        chunk = query_matrix[i:i+chunk_size]
        _, indices = retriever._index.search(chunk, max_k)
        all_indices.append(indices)
    
    all_indices = np.vstack(all_indices)

    print("  Computing Recall@K...")
    for indices, gt in zip(all_indices, ground_truths):
        retrieved = [retriever.article_ids[i] for i in indices if i >= 0]
        for k in k_values:
            recall_sums[k] += recall_at_k(retrieved, gt, k)
        n_evaluated += 1

    results = {f"recall@{k}": recall_sums[k] / n_evaluated for k in k_values}
    print(f"\n[Semantic] Results on {dataset}/{split} (n={n_evaluated}, no_history={n_no_history}):")
    for metric, val in results.items():
        print(f"  {metric}: {val:.4f}")
    return results


# --------------------------------------------------------------------- #
#  CLI                                                                    #
# --------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Semantic retrieval evaluation (Q3)")
    parser.add_argument("--dataset", choices=["mind", "ebnerd"], default="mind")
    parser.add_argument("--split", default="val")
    parser.add_argument("--k", type=int, nargs="+", default=[50, 100, 200])
    parser.add_argument("--max-history", type=int, default=10)
    parser.add_argument("--rebuild", action="store_true",
                        help="Force rebuild embeddings + index")
    args = parser.parse_args()

    if args.rebuild:
        import shutil
        for d in [EMBED_DIR / args.dataset, INDEX_DIR / args.dataset]:
            if d.exists():
                shutil.rmtree(d)
                print(f"Cleared {d}")

    results = evaluate_semantic(
        dataset=args.dataset,
        split=args.split,
        k_values=args.k,
        max_history=args.max_history,
    )

    out_dir = Path("data/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"semantic_{args.dataset}_{args.split}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
