"""
Evaluate retrieval submissions against ground truth (Q4)
=========================================================
Reads the generated prediction.txt and the val behaviors with labels,
computes AUC, MRR, nDCG@5, nDCG@10, diversity, novelty, coverage,
cold/warm split, and bootstrap 95% CI.

Usage (run AFTER generating val predictions):
    python -m src.evaluation.evaluate --dataset mind --strategy hybrid
    python -m src.evaluation.evaluate --dataset ebnerd --strategy hybrid
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from src.evaluation.metrics import (
    compute_auc, compute_mrr, compute_ndcg,
    intra_list_diversity, novelty, catalog_coverage,
    bootstrap_ci, cold_warm_split, PROCESSED_DIR, RESULTS_DIR
)

SUBMISSION_DIR = Path("data/submissions")


def evaluate_submission(dataset: str, strategy: str, split: str = "val") -> dict:
    """
    Evaluate a generated submission file against ground-truth labels.

    For the val split, labels are available.
    Reads prediction ranks from the submission txt, computes scores.
    """
    behaviors_path = PROCESSED_DIR / f"behaviors_{dataset}_{split}.parquet"
    articles_path  = PROCESSED_DIR / f"articles_{dataset}.parquet"
    pred_path      = SUBMISSION_DIR / f"{dataset}_{split}_{strategy}.txt"

    if not behaviors_path.exists():
        raise FileNotFoundError(f"No behaviors: {behaviors_path}")
    if not articles_path.exists():
        raise FileNotFoundError(f"No articles: {articles_path}")
    if not pred_path.exists():
        raise FileNotFoundError(
            f"No prediction file: {pred_path}\n"
            f"Run: python -m src.submission.generate --dataset {dataset} "
            f"--split {split} --strategy {strategy}"
        )

    print(f"\n[Eval] {dataset}/{split} strategy={strategy}")
    behaviors = pl.read_parquet(behaviors_path).sort("impression_id")
    articles  = pl.read_parquet(articles_path)

    # Build article lookup maps
    art_rows      = articles.select(["article_id", "category"]).to_dicts()
    category_map  = {r["article_id"]: r["category"] for r in art_rows}

    # Build popularity map from all impressions
    pop_map: dict = {}
    for row in behaviors.select("impressions").iter_rows():
        for aid in (row[0] or []):
            pop_map[aid] = pop_map.get(aid, 0) + 1
    n_total = max(len(pop_map), 1)

    all_article_ids = set(articles["article_id"].to_list())

    # --- Parse prediction file ---
    # Format: "{imp_id} [{r1},{r2},...}]"
    pred_ranks: dict = {}   # imp_id -> list of int ranks (1-indexed)
    with open(pred_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            imp_id = int(parts[0])
            ranks_str = parts[1].strip("[]")
            pred_ranks[imp_id] = [int(r) for r in ranks_str.split(",")]

    # --- Evaluate ---
    auc_vals, mrr_vals = [], []
    ndcg5_vals, ndcg10_vals = [], []
    div_vals, nov_vals = [], []
    all_recommended = set()

    for row in behaviors.iter_rows(named=True):
        imp_id      = row["impression_id"]
        impressions = row.get("impressions") or []
        labels      = row.get("labels")      or []

        if not impressions or imp_id not in pred_ranks:
            continue

        # Build scores from predicted ranks (lower rank = higher score)
        ranks = pred_ranks[imp_id]
        if len(ranks) != len(impressions):
            continue

        # Convert ranks to scores: score = -rank (so rank 1 → score -1 is highest)
        scores = [-r for r in ranks]
        int_labels = [int(l) for l in labels]

        if len(set(int_labels)) < 2:
            continue   # no positive or all-positive impression

        auc_vals.append(compute_auc(int_labels, scores))
        mrr_vals.append(compute_mrr(int_labels, scores))
        ndcg5_vals.append(compute_ndcg(int_labels, scores, 5))
        ndcg10_vals.append(compute_ndcg(int_labels, scores, 10))

        # Beyond-accuracy: use top-10 predicted articles
        order    = np.argsort(ranks)[:10]   # lowest rank first = best ranked
        top10    = [impressions[i] for i in order]
        div_vals.append(intra_list_diversity(top10, category_map))
        nov_vals.append(novelty(top10, pop_map, n_total))
        all_recommended.update(top10)

    coverage = catalog_coverage(all_recommended, all_article_ids)

    # --- Aggregate + bootstrap CI ---
    results = {
        "dataset":   dataset,
        "split":     split,
        "strategy":  strategy,
        "n_impressions": len(auc_vals),
    }

    print(f"  Evaluated {len(auc_vals):,} impressions")

    def _report(name, vals):
        if not vals:
            results[name] = None
            return
        mean, lo, hi = bootstrap_ci(vals)
        results[name] = {"mean": round(mean, 4), "lower": round(lo, 4), "upper": round(hi, 4)}
        print(f"  {name:<12}: {mean:.4f}  95%CI [{lo:.4f}, {hi:.4f}]")

    _report("AUC",     auc_vals)
    _report("MRR",     mrr_vals)
    _report("nDCG@5",  ndcg5_vals)
    _report("nDCG@10", ndcg10_vals)
    _report("diversity", div_vals)
    _report("novelty",   nov_vals)
    results["coverage"] = round(coverage, 4)
    print(f"  {'coverage':<12}: {coverage:.4f}")

    # --- Cold/warm slice ---
    print("\n  [Cold/Warm split]")
    cold_df, warm_df = cold_warm_split(behaviors)
    cold_ids = set(cold_df["impression_id"].to_list())
    warm_ids = set(warm_df["impression_id"].to_list())

    for slice_name, slice_ids in [("cold", cold_ids), ("warm", warm_ids)]:
        s_auc, s_mrr = [], []
        for row in behaviors.iter_rows(named=True):
            if row["impression_id"] not in slice_ids:
                continue
            imp_id      = row["impression_id"]
            impressions = row.get("impressions") or []
            labels      = row.get("labels")      or []
            if not impressions or imp_id not in pred_ranks:
                continue
            ranks  = pred_ranks[imp_id]
            scores = [-r for r in ranks]
            int_labels = [int(l) for l in labels]
            if len(set(int_labels)) < 2:
                continue
            s_auc.append(compute_auc(int_labels, scores))
            s_mrr.append(compute_mrr(int_labels, scores))

        if s_auc:
            print(f"  {slice_name:<6}: AUC={np.mean(s_auc):.4f}  MRR={np.mean(s_mrr):.4f}  n={len(s_auc):,}")
            results[f"{slice_name}_AUC"] = round(float(np.mean(s_auc)), 4)
            results[f"{slice_name}_MRR"] = round(float(np.mean(s_mrr)), 4)

    return results


def main():
    p = argparse.ArgumentParser(description="Evaluate submission (Q4)")
    p.add_argument("--dataset",  choices=["mind", "ebnerd"], default="mind")
    p.add_argument("--strategy", default="hybrid")
    p.add_argument("--split",    default="val")
    args = p.parse_args()

    results = evaluate_submission(args.dataset, args.strategy, args.split)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"eval_{args.dataset}_{args.split}_{args.strategy}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
