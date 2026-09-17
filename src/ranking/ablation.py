"""
Ablation Study Runner (A2 Q3 — Ablation)
=========================================
Runs LightGBM with progressively more features to isolate
the contribution of each feature group.

Feature sets:
  A: [sem_score]                                          — semantic only
  B: [sem_score, lex_overlap]                             — + lexical
  C: [sem_score, lex_overlap, log_popularity, hist_len,
      cat_match, inv_position]                            — original 6 features
  D: [all 8 features]                                    — + freshness + recency_sem

Reports AUC, MRR, nDCG@5, nDCG@10 with 95% bootstrap CI for each variant.

Usage:
    python -m src.ranking.ablation --dataset mind
    python -m src.ranking.ablation --dataset ebnerd
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import polars as pl

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("data/models")
RESULTS_DIR   = Path("data/results")


def bootstrap_ci(values, n_boot=1000, ci=0.95, seed=42):
    """Return (mean, lower, upper) with 95% CI."""
    rng = random.Random(seed)
    n   = len(values)
    boot = [np.mean([rng.choice(values) for _ in range(n)]) for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    return float(np.mean(values)), float(np.quantile(boot, alpha)), float(np.quantile(boot, 1 - alpha))


def compute_metrics(imp_ids, art_ids, scores, behaviors_df):
    """Compute AUC, MRR, nDCG@5, nDCG@10 from per-pair scores."""
    from sklearn.metrics import roc_auc_score

    # Rebuild label lookup
    label_lookup = {}
    for row in behaviors_df.iter_rows(named=True):
        imp_id      = row["impression_id"]
        impressions = row.get("impressions") or []
        labels      = row.get("labels")      or []
        label_lookup[imp_id] = dict(zip(impressions, labels))

    # Group scores by impression
    from collections import defaultdict
    imp_groups = defaultdict(list)  # imp_id -> list of (score, label)
    for imp_id, aid, score in zip(imp_ids, art_ids, scores):
        lbl = label_lookup.get(imp_id, {}).get(aid, 0)
        imp_groups[imp_id].append((score, lbl))

    auc_vals, mrr_vals, ndcg5_vals, ndcg10_vals = [], [], [], []

    for imp_id, pairs in imp_groups.items():
        pairs_arr = np.array(pairs)
        sc = pairs_arr[:, 0]
        lb = pairs_arr[:, 1].astype(int)
        if len(set(lb)) < 2:
            continue
        try:
            auc_vals.append(float(roc_auc_score(lb, sc)))
        except Exception:
            pass

        order = np.argsort(sc)[::-1]
        for rank, idx in enumerate(order, 1):
            if lb[idx] == 1:
                mrr_vals.append(1.0 / rank)
                break
        else:
            mrr_vals.append(0.0)

        for k, store in [(5, ndcg5_vals), (10, ndcg10_vals)]:
            top = order[:k]
            dcg  = sum(lb[i] / np.log2(r + 2) for r, i in enumerate(top))
            idcg = sum(v / np.log2(r + 2) for r, v in enumerate(sorted(lb, reverse=True)[:k]))
            store.append(dcg / idcg if idcg > 0 else 0.0)

    return {
        "AUC":     bootstrap_ci(auc_vals)   if auc_vals   else (0, 0, 0),
        "MRR":     bootstrap_ci(mrr_vals)   if mrr_vals   else (0, 0, 0),
        "nDCG@5":  bootstrap_ci(ndcg5_vals) if ndcg5_vals else (0, 0, 0),
        "nDCG@10": bootstrap_ci(ndcg10_vals) if ndcg10_vals else (0, 0, 0),
        "n":       len(auc_vals),
    }


def run_ablation(dataset: str, max_val_rows: int = 30_000):
    """
    Run ablation study: train 4 LightGBM variants with different feature subsets.
    Uses pre-built features from feature_store.py (build all 8 features once).
    """
    from src.features.feature_store import build_features, FEATURE_NAMES
    from src.retrieval.semantic import load_or_compute_embeddings
    from src.retrieval.bm25 import _article_text
    from src.ranking.train_lgbm import LGBMRanker, build_popularity_map
    import lightgbm as lgb

    print(f"\n[Ablation] Dataset: {dataset}")

    articles_path = PROCESSED_DIR / f"articles_{dataset}.parquet"
    train_path    = PROCESSED_DIR / f"behaviors_{dataset}_train.parquet"
    val_path      = PROCESSED_DIR / f"behaviors_{dataset}_val.parquet"

    articles        = pl.read_parquet(articles_path)
    behaviors_train = pl.read_parquet(train_path).sample(min(200_000, len(pl.read_parquet(train_path))), seed=42)
    behaviors_val   = pl.read_parquet(val_path).filter(
        pl.col("labels").list.sum() > 0
    ).head(max_val_rows)

    # Load embeddings
    embs, ids = load_or_compute_embeddings(articles, dataset)
    embedding_map = dict(zip(ids, embs))

    # Article text map
    art_rows = articles.select(["article_id", "title", "subtitle"]).to_dicts()
    article_text_map = {r["article_id"]: _article_text(r) for r in art_rows}

    # Article publish time map
    pub_col = "published_time" if "published_time" in articles.columns else None
    article_publish_map = {}
    if pub_col:
        article_publish_map = {
            r["article_id"]: r[pub_col]
            for r in articles.select(["article_id", pub_col]).to_dicts()
        }

    # Popularity map
    pop_map = build_popularity_map(behaviors_train)

    # Build ALL features once (expensive but do it once)
    print("  Building train features (all 8)...")
    X_tr, y_tr, g_tr, _, _ = build_features(
        behaviors_train, articles, embedding_map,
        article_text_map=article_text_map,
        popularity_map=pop_map,
        article_publish_map=article_publish_map,
    )
    mask = y_tr >= 0
    X_tr, y_tr = X_tr[mask], y_tr[mask]

    print("  Building val features (all 8)...")
    X_vl, y_vl, g_vl, vl_imp_ids, vl_art_ids = build_features(
        behaviors_val, articles, embedding_map,
        article_text_map=article_text_map,
        popularity_map=pop_map,
        article_publish_map=article_publish_map,
    )

    # Feature subsets — FEATURE_NAMES order: sem, lex, pop, hist, cat, pos, fresh, rec_sem
    # Index:                                    0     1    2     3    4    5    6      7
    FEATURE_SETS = {
        "A_semantic_only":   [0],                     # sem_score
        "B_sem_lex":         [0, 1],                  # + lex_overlap
        "C_original_6":      [0, 1, 2, 3, 4, 5],     # original 6 features
        "D_full_8":          [0, 1, 2, 3, 4, 5, 6, 7], # all 8 features (best)
    }

    params = {
        "objective":      "lambdarank",
        "metric":         "ndcg",
        "ndcg_eval_at":   [5, 10],
        "learning_rate":  0.05,
        "num_leaves":     63,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":   5,
        "label_gain":     [0, 1],
        "verbose":        -1,
        "n_jobs":         -1,
    }

    all_results = {}

    for set_name, feat_idxs in FEATURE_SETS.items():
        feat_names_subset = [FEATURE_NAMES[i] for i in feat_idxs]
        print(f"\n  [Ablation] Feature set {set_name}: {feat_names_subset}")

        X_tr_sub = X_tr[:, feat_idxs]
        X_vl_sub = X_vl[:, feat_idxs]

        dtrain = lgb.Dataset(X_tr_sub, label=y_tr, group=g_tr,
                             feature_name=feat_names_subset, free_raw_data=False)
        dval   = lgb.Dataset(X_vl_sub, label=y_vl, group=g_vl,
                             feature_name=feat_names_subset, reference=dtrain, free_raw_data=False)

        model = lgb.train(
            params, dtrain,
            num_boost_round=200,
            valid_sets=[dtrain, dval],
            valid_names=["train", "val"],
            callbacks=[lgb.log_evaluation(50), lgb.early_stopping(30)],
        )

        val_scores = model.predict(X_vl_sub)
        metrics = compute_metrics(vl_imp_ids, vl_art_ids, val_scores, behaviors_val)

        print(f"    AUC:     {metrics['AUC'][0]:.4f}  [{metrics['AUC'][1]:.4f}, {metrics['AUC'][2]:.4f}]")
        print(f"    MRR:     {metrics['MRR'][0]:.4f}  [{metrics['MRR'][1]:.4f}, {metrics['MRR'][2]:.4f}]")
        print(f"    nDCG@5:  {metrics['nDCG@5'][0]:.4f}  [{metrics['nDCG@5'][1]:.4f}, {metrics['nDCG@5'][2]:.4f}]")
        print(f"    nDCG@10: {metrics['nDCG@10'][0]:.4f}  [{metrics['nDCG@10'][1]:.4f}, {metrics['nDCG@10'][2]:.4f}]")

        all_results[set_name] = {
            "features": feat_names_subset,
            "n_features": len(feat_idxs),
            "n_val_impressions": metrics["n"],
            "AUC":     {"mean": metrics["AUC"][0],    "lower": metrics["AUC"][1],    "upper": metrics["AUC"][2]},
            "MRR":     {"mean": metrics["MRR"][0],    "lower": metrics["MRR"][1],    "upper": metrics["MRR"][2]},
            "nDCG@5":  {"mean": metrics["nDCG@5"][0], "lower": metrics["nDCG@5"][1], "upper": metrics["nDCG@5"][2]},
            "nDCG@10": {"mean": metrics["nDCG@10"][0],"lower": metrics["nDCG@10"][1],"upper": metrics["nDCG@10"][2]},
        }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"ablation_{dataset}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Ablation] Results saved → {out_path}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Ablation study (A2 Q3)")
    parser.add_argument("--dataset", choices=["mind", "ebnerd"], required=True)
    parser.add_argument("--max-val-rows", type=int, default=30_000)
    args = parser.parse_args()
    run_ablation(args.dataset, args.max_val_rows)


if __name__ == "__main__":
    main()
