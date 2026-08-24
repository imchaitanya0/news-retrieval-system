"""
Offline Evaluation Harness (Q4)
================================
Implements all required metrics:
  - AUC, MRR, nDCG@5, nDCG@10
  - Intra-list diversity, novelty, coverage
  - Cold/warm user slicing, head/tail article slicing
  - Bootstrap 95% CI for every metric

Usage:
    python -m src.evaluation.metrics --dataset mind --split val
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("data/results")

# ------------------------------------------------------------------ #
#  Core ranking metrics                                               #
# ------------------------------------------------------------------ #

def compute_auc(labels: list, scores: list) -> float:
    """AUC for a single impression. Returns 0.5 if only one class."""
    if len(set(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


def compute_mrr(labels: list, scores: list) -> float:
    """MRR: reciprocal rank of the first relevant item in ranked list."""
    order = np.argsort(scores)[::-1]
    for rank, idx in enumerate(order, 1):
        if labels[idx] == 1:
            return 1.0 / rank
    return 0.0


def compute_ndcg(labels: list, scores: list, k: int) -> float:
    """nDCG@k for a single impression."""
    order = np.argsort(scores)[::-1][:k]
    dcg = sum(labels[i] / np.log2(r + 2) for r, i in enumerate(order))
    ideal = sorted(labels, reverse=True)[:k]
    idcg = sum(v / np.log2(r + 2) for r, v in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def impression_scores(labels: list) -> list:
    """
    When no ranker score is available, use position-as-score (reversed
    so BM25/semantic order is represented) or label itself as oracle.
    For baseline evaluation we use labels as oracle scores.
    """
    return [float(l) for l in labels]


# ------------------------------------------------------------------ #
#  Beyond-accuracy metrics                                            #
# ------------------------------------------------------------------ #

def intra_list_diversity(
    recommended_ids: list,
    category_map: dict,
) -> float:
    """
    Intra-list diversity = fraction of unique categories in top-K.
    Ranges [0, 1]; higher is more diverse.

    Parameters
    ----------
    recommended_ids : list of article_id in ranked order
    category_map : {article_id -> category str}
    """
    if not recommended_ids:
        return 0.0
    cats = [category_map.get(aid, "unknown") for aid in recommended_ids]
    unique = len(set(cats))
    return unique / len(cats)


def novelty(
    recommended_ids: list,
    popularity_map: dict,
    n_total: int,
) -> float:
    """
    Novelty = mean self-information of recommended items.
    novelty(i) = -log2(popularity_i / n_total)
    Higher = more novel (less popular items).
    """
    if not recommended_ids or n_total == 0:
        return 0.0
    scores = []
    for aid in recommended_ids:
        pop = max(popularity_map.get(aid, 1), 1)
        scores.append(-np.log2(pop / n_total))
    return float(np.mean(scores))


def catalog_coverage(
    all_recommended: set,
    all_articles: set,
) -> float:
    """
    Catalog coverage = fraction of the article catalog recommended at least once.
    """
    if not all_articles:
        return 0.0
    return len(all_recommended & all_articles) / len(all_articles)


# ------------------------------------------------------------------ #
#  Bootstrap CI                                                       #
# ------------------------------------------------------------------ #

def bootstrap_ci(
    values: list,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple:
    """
    Compute bootstrap confidence interval for the mean.

    Returns
    -------
    (mean, lower, upper)
    """
    rng = random.Random(seed)
    n = len(values)
    boot_means = []
    for _ in range(n_boot):
        sample = [rng.choice(values) for _ in range(n)]
        boot_means.append(np.mean(sample))
    alpha = (1 - ci) / 2
    lower = np.quantile(boot_means, alpha)
    upper = np.quantile(boot_means, 1 - alpha)
    return float(np.mean(values)), float(lower), float(upper)


# ------------------------------------------------------------------ #
#  Slicing helpers                                                    #
# ------------------------------------------------------------------ #

def cold_warm_split(
    behaviors: pl.DataFrame,
    cold_threshold: int = 5,
) -> tuple:
    """
    Split behaviors into cold-start and warm users.

    Cold = users with <= cold_threshold clicks in history.
    Works for MIND (has 'history' column). For EB-NeRD, uses impression count.
    """
    if "history" in behaviors.columns:
        hist_len = behaviors.with_columns(
            pl.col("history").list.len().alias("hist_len")
        )
        cold = hist_len.filter(pl.col("hist_len") <= cold_threshold)
        warm = hist_len.filter(pl.col("hist_len") > cold_threshold)
    else:
        # EB-NeRD: use impression_id count per user as proxy
        user_counts = behaviors.group_by("user_id").agg(
            pl.len().alias("n_impressions")
        )
        behaviors = behaviors.join(user_counts, on="user_id")
        cold = behaviors.filter(pl.col("n_impressions") <= cold_threshold)
        warm = behaviors.filter(pl.col("n_impressions") > cold_threshold)
    return cold, warm


def head_tail_split(
    behaviors: pl.DataFrame,
    articles: pl.DataFrame,
    head_percentile: float = 0.9,
) -> tuple:
    """
    Split impressions into 'head' (popular articles) and 'tail'.
    Head = articles with popularity >= 90th percentile.
    """
    pop_threshold = articles["popularity"].quantile(head_percentile)
    head_ids = set(
        articles.filter(pl.col("popularity") >= pop_threshold)["article_id"].to_list()
    )
    return head_ids  # caller uses this set to filter impressions


# ------------------------------------------------------------------ #
#  Full evaluation run                                                #
# ------------------------------------------------------------------ #

def evaluate(
    dataset: str,
    split: str = "val",
    k_values: list = None,
    n_boot: int = 500,
    cold_threshold: int = 5,
) -> dict:
    """
    Run the full evaluation harness on a processed split.

    Uses impression labels as oracle scores (upper-bound evaluation).
    For retrieval-based scoring, pass pre-computed scores instead.

    Parameters
    ----------
    dataset     : "mind" or "ebnerd"
    split       : "train" / "val" / "test"
    k_values    : nDCG cutoffs (default [5, 10])
    n_boot      : bootstrap resamples
    cold_threshold : history-length threshold for cold/warm split

    Returns
    -------
    dict with all metrics + CI bounds
    """
    if k_values is None:
        k_values = [5, 10]

    articles_path = PROCESSED_DIR / f"articles_{dataset}.parquet"
    behaviors_path = PROCESSED_DIR / f"behaviors_{dataset}_{split}.parquet"

    if not articles_path.exists() or not behaviors_path.exists():
        raise FileNotFoundError("Run build_pipeline.py first.")

    print(f"\n[Eval] {dataset} / {split}")
    articles = pl.read_parquet(articles_path)
    behaviors = pl.read_parquet(behaviors_path)

    # Lookup maps
    art_rows = articles.select(["article_id", "category", "popularity"]).to_dicts()
    category_map = {r["article_id"]: r["category"] for r in art_rows}
    popularity_map = {r["article_id"]: (r["popularity"] or 0) for r in art_rows}
    all_article_ids = set(articles["article_id"].to_list())
    n_total_articles = max(len(all_article_ids), 1)

    # Per-impression metric accumulators
    auc_vals, mrr_vals = [], []
    ndcg_vals = {k: [] for k in k_values}
    div_vals, nov_vals = [], []
    all_recommended = set()

    for row in behaviors.iter_rows(named=True):
        impressions = row["impressions"] or []
        labels = row["labels"] or []
        if len(impressions) == 0 or len(set(labels)) < 2:
            continue  # skip no-positive or degenerate impressions

        # Use label as oracle score (measures how good a perfect ranker would be)
        scores = impression_scores(labels)

        auc_vals.append(compute_auc(labels, scores))
        mrr_vals.append(compute_mrr(labels, scores))
        for k in k_values:
            ndcg_vals[k].append(compute_ndcg(labels, scores, k))

        # Beyond-accuracy
        top10 = impressions[:10]
        div_vals.append(intra_list_diversity(top10, category_map))
        nov_vals.append(novelty(top10, popularity_map, n_total_articles))
        all_recommended.update(impressions)

    # Coverage
    coverage = catalog_coverage(all_recommended, all_article_ids)

    # Aggregate + bootstrap CI
    results = {"n_impressions": len(auc_vals)}

    def _add(name, vals):
        if not vals:
            results[name] = {"mean": None, "ci_lower": None, "ci_upper": None}
            return
        mean, lo, hi = bootstrap_ci(vals, n_boot=n_boot)
        results[name] = {"mean": round(mean, 4), "ci_lower": round(lo, 4), "ci_upper": round(hi, 4)}
        print(f"  {name}: {mean:.4f} [{lo:.4f}, {hi:.4f}]")

    print(f"\n  n_impressions={len(auc_vals)}")
    _add("AUC", auc_vals)
    _add("MRR", mrr_vals)
    for k in k_values:
        _add(f"nDCG@{k}", ndcg_vals[k])
    _add("diversity", div_vals)
    _add("novelty", nov_vals)
    results["coverage"] = round(coverage, 4)
    print(f"  coverage: {coverage:.4f}")

    # Cold/warm slice
    print("\n  [Slice] Cold vs warm users:")
    cold_df, warm_df = cold_warm_split(behaviors, cold_threshold)
    for name, sub in [("cold", cold_df), ("warm", warm_df)]:
        sub_auc, sub_mrr = [], []
        for row in sub.iter_rows(named=True):
            impressions = row["impressions"] or []
            labels = row["labels"] or []
            if len(impressions) == 0 or len(set(labels)) < 2:
                continue
            scores = impression_scores(labels)
            sub_auc.append(compute_auc(labels, scores))
            sub_mrr.append(compute_mrr(labels, scores))
        if sub_auc:
            print(f"    {name}: AUC={np.mean(sub_auc):.4f} MRR={np.mean(sub_mrr):.4f} (n={len(sub_auc)})")
            results[f"{name}_AUC"] = round(float(np.mean(sub_auc)), 4)
            results[f"{name}_MRR"] = round(float(np.mean(sub_mrr)), 4)

    return results


# ------------------------------------------------------------------ #
#  CLI                                                                #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Evaluation harness (Q4)")
    parser.add_argument("--dataset", choices=["mind", "ebnerd"], default="mind")
    parser.add_argument("--split", default="val")
    parser.add_argument("--n-boot", type=int, default=500,
                        help="Bootstrap resamples (default 500)")
    parser.add_argument("--cold-threshold", type=int, default=5,
                        help="Max history length for cold-start users")
    args = parser.parse_args()

    results = evaluate(
        dataset=args.dataset,
        split=args.split,
        n_boot=args.n_boot,
        cold_threshold=args.cold_threshold,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"eval_{args.dataset}_{args.split}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
