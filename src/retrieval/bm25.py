"""
BM25 Lexical Retriever (Q2)
===========================
Builds an inverted BM25 index over article title + subtitle (abstract).
Given a user's click history, constructs a query from the titles of
recently clicked articles and retrieves the top-K candidates.

Usage (standalone):
    python -m src.retrieval.bm25 --dataset mind --split val --k 100
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Optional

import polars as pl
import numpy as np

try:
    import bm25s                        # fast BM25 (preferred)
    _USE_BM25S = True
except ImportError:
    from rank_bm25 import BM25Okapi     # fallback
    _USE_BM25S = False

PROCESSED_DIR = Path("data/processed")
INDEX_DIR = Path("data/feature_store/bm25")

# --------------------------------------------------------------------- #
#  Text helpers                                                           #
# --------------------------------------------------------------------- #

def _tokenize(text: str) -> list:
    """Lowercase whitespace tokenizer."""
    return text.lower().split()


def _article_text(row: dict) -> str:
    """Concatenate title + subtitle (abstract) into a single string."""
    title = row.get("title") or ""
    subtitle = row.get("subtitle") or ""
    return f"{title} {subtitle}".strip()


# --------------------------------------------------------------------- #
#  Index                                                                  #
# --------------------------------------------------------------------- #

class BM25Retriever:
    """
    Wraps either bm25s or rank_bm25 behind a common interface.

    Attributes
    ----------
    article_ids : list
        Ordered list of article IDs matching the index rows.
    """

    def __init__(self):
        self.article_ids = []
        self._index = None
        self._corpus_tokens = None  # kept for rank_bm25 only

    # ------------------------------------------------------------------ #
    #  Build                                                               #
    # ------------------------------------------------------------------ #

    def build(self, articles: pl.DataFrame) -> None:
        """
        Build BM25 index from an articles DataFrame.

        Parameters
        ----------
        articles : pl.DataFrame
            Must have columns: article_id, title, subtitle
        """
        print("Building BM25 index ...")
        texts = articles.select([
            pl.col("title").fill_null(""),
            pl.col("subtitle").fill_null(""),
        ]).to_dicts()

        corpus = [_tokenize(_article_text(r)) for r in texts]
        self.article_ids = articles["article_id"].to_list()

        if _USE_BM25S:
            self._index = bm25s.BM25(k1=1.5, b=0.75)
            # Create bm25s index
            corpus_tokens = bm25s.tokenize([_article_text(r) for r in texts], lower=True, show_progress=False)
            self._index.index(corpus_tokens, show_progress=False)
        else:
            self._corpus_tokens = corpus
            self._index = BM25Okapi(corpus, k1=1.5, b=0.75)

        print(f"  Indexed {len(self.article_ids)} articles.")

    # ------------------------------------------------------------------ #
    #  Retrieve                                                            #
    # ------------------------------------------------------------------ #

    def retrieve(self, query: str, k: int = 100) -> list:
        """
        Return top-k article IDs for a free-text query.

        Parameters
        ----------
        query : str
            Free-text query (e.g. concatenated recent article titles).
        k : int
            Number of candidates to return.

        Returns
        -------
        list of article_id values
        """
        if not query:
            return []

        if _USE_BM25S:
            results, scores = self._index.retrieve(
                bm25s.tokenize(query, lower=True, show_progress=False),
                k=min(k, len(self.article_ids)),
                show_progress=False
            )
            indices = results[0].tolist()
        else:
            tokens = _tokenize(query)
            scores = self._index.get_scores(tokens)
            indices = np.argsort(scores)[::-1][:k].tolist()

        return [self.article_ids[i] for i in indices]

    # ------------------------------------------------------------------ #
    #  Save / Load                                                         #
    # ------------------------------------------------------------------ #

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "bm25_index.pkl", "wb") as f:
            pickle.dump({
                "article_ids": self.article_ids,
                "use_bm25s": _USE_BM25S,
                "index": self._index,
                "corpus_tokens": self._corpus_tokens,
            }, f, protocol=4)
        print(f"  Index saved to {path}")

    def load(self, path: Path) -> None:
        with open(path / "bm25_index.pkl", "rb") as f:
            data = pickle.load(f)
        self.article_ids = data["article_ids"]
        self._index = data["index"]
        self._corpus_tokens = data.get("corpus_tokens")
        print(f"  Index loaded ({len(self.article_ids)} articles)")


# --------------------------------------------------------------------- #
#  Query construction                                                     #
# --------------------------------------------------------------------- #

def build_query_from_history(
    history: list,
    article_text_map: dict,
    max_articles: int = 10,
) -> str:
    """
    Construct a BM25 query from a user's click history.

    Takes the *most recent* `max_articles` from the history list and
    concatenates their title + subtitle. Recency matters in news — older
    clicks add noise.

    Parameters
    ----------
    history : list
        Ordered list of article_id values (oldest first, MIND/EB-NeRD style).
    article_text_map : dict
        {article_id: "title subtitle"} lookup.
    max_articles : int
        How many recent articles to use (default: 10).

    Returns
    -------
    str : query string
    """
    recent = history[-max_articles:] if history else []
    parts = [article_text_map.get(aid, "") for aid in recent]
    return " ".join(p for p in parts if p).strip()


# --------------------------------------------------------------------- #
#  Evaluation                                                             #
# --------------------------------------------------------------------- #

def recall_at_k(retrieved: list, ground_truth: set, k: int) -> float:
    """Recall@K = |retrieved[:k] ∩ ground_truth| / |ground_truth|"""
    if not ground_truth:
        return 0.0
    hits = len(set(retrieved[:k]) & ground_truth)
    return hits / len(ground_truth)


def evaluate_bm25(
    dataset: str,
    split: str = "val",
    k_values: list = None,
    max_history: int = 10,
) -> dict:
    """
    Run BM25 retrieval on a processed split and report Recall@K.

    Parameters
    ----------
    dataset : str
        "mind" or "ebnerd"
    split : str
        "train", "val", or "test"
    k_values : list
        K values for Recall@K (default: [50, 100, 200])
    max_history : int
        Max recent articles to use per query.

    Returns
    -------
    dict mapping "recall@K" -> float
    """
    if k_values is None:
        k_values = [50, 100, 200]

    articles_path = PROCESSED_DIR / f"articles_{dataset}.parquet"
    behaviors_path = PROCESSED_DIR / f"behaviors_{dataset}_{split}.parquet"

    if not articles_path.exists():
        raise FileNotFoundError(f"Run build_pipeline.py first: {articles_path}")
    if not behaviors_path.exists():
        raise FileNotFoundError(f"Run build_pipeline.py first: {behaviors_path}")

    print(f"\n[BM25] Evaluating {dataset} / {split}")
    articles = pl.read_parquet(articles_path)
    behaviors = pl.read_parquet(behaviors_path)

    # Build or load index
    index_path = INDEX_DIR / dataset
    retriever = BM25Retriever()
    if (index_path / "bm25_index.pkl").exists():
        retriever.load(index_path)
    else:
        retriever.build(articles)
        retriever.save(index_path)

    # Article text lookup: {article_id -> "title subtitle"}
    art_dict = articles.select(["article_id", "title", "subtitle"]).to_dicts()
    article_text_map = {r["article_id"]: _article_text(r) for r in art_dict}

    # Evaluate per impression
    max_k = max(k_values)
    recall_sums = {k: 0.0 for k in k_values}
    n_evaluated = 0

    # MIND has a 'history' column; EB-NeRD does not (use pseudo-history)
    has_history = "history" in behaviors.columns

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **kw): return it  # fallback: no progress bar

    for row in tqdm(behaviors.iter_rows(named=True), total=len(behaviors), desc="BM25 eval"):
        impressions = row["impressions"] or []
        labels = row["labels"] or []
        ground_truth = {aid for aid, lbl in zip(impressions, labels) if lbl == 1}
        if not ground_truth:
            continue

        # Build query from history
        history = row["history"] if has_history else []
        if isinstance(history, str):
            # MIND history may come as a space-separated string if parsing missed it
            history = history.split() if history else []
        if not history:
            # Fallback: treat known positives as pseudo-history
            history = list(ground_truth)

        query = build_query_from_history(history, article_text_map, max_articles=max_history)
        if not query:
            continue

        retrieved = retriever.retrieve(query, k=max_k)
        for k in k_values:
            recall_sums[k] += recall_at_k(retrieved, ground_truth, k)
        n_evaluated += 1

    if n_evaluated == 0:
        print("  Warning: no impressions with positive labels found.")
        return {f"recall@{k}": 0.0 for k in k_values}

    results = {f"recall@{k}": recall_sums[k] / n_evaluated for k in k_values}
    print(f"\n[BM25] Results on {dataset}/{split} (n={n_evaluated}):")
    for metric, val in results.items():
        print(f"  {metric}: {val:.4f}")
    return results


# --------------------------------------------------------------------- #
#  CLI                                                                    #
# --------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="BM25 retrieval evaluation (Q2)")
    parser.add_argument("--dataset", choices=["mind", "ebnerd"], default="mind")
    parser.add_argument("--split", default="val")
    parser.add_argument("--k", type=int, nargs="+", default=[50, 100, 200])
    parser.add_argument("--max-history", type=int, default=10)
    parser.add_argument("--rebuild-index", action="store_true")
    args = parser.parse_args()

    if args.rebuild_index:
        import shutil
        idx_path = INDEX_DIR / args.dataset
        if idx_path.exists():
            shutil.rmtree(idx_path)
            print(f"Cleared cached index at {idx_path}")

    results = evaluate_bm25(
        dataset=args.dataset,
        split=args.split,
        k_values=args.k,
        max_history=args.max_history,
    )

    out_dir = Path("data/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"bm25_{args.dataset}_{args.split}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
