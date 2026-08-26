"""
LightGBM Learning-to-Rank Trainer & Inference (Q5)
====================================================
Trains a LambdaMART ranker on the training behaviors, then uses it
to rerank test impression candidates.

Training flow:
  1. Load training behaviors (article_ids_inview + click labels)
  2. Build 6 features per (user, candidate) pair via feature_store.py
  3. Train LightGBM with objective=lambdarank, eval_metric=ndcg
  4. Save model to data/models/lgbm_{dataset}.pkl

Inference flow:
  1. Load saved model
  2. Build features for test behaviors
  3. Predict scores → rerank candidates
  4. Write Codabench submission file

Usage:
    # Training
    python -m src.ranking.train_lgbm --dataset mind

    # Inference (called automatically by generate.py with --strategy lgbm)
    from src.ranking.train_lgbm import LGBMRanker
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import polars as pl

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("data/models")


class LGBMRanker:
    """Thin wrapper around lightgbm.Booster for LTR."""

    def __init__(self):
        self.model = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        groups_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        groups_val: np.ndarray = None,
        feature_names: list = None,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 63,
        label_gain: list = None,
    ):
        import lightgbm as lgb

        if label_gain is None:
            label_gain = [0, 1]  # binary labels

        params = {
            "objective":      "lambdarank",
            "metric":         "ndcg",
            "ndcg_eval_at":   [5, 10],
            "learning_rate":  learning_rate,
            "num_leaves":     num_leaves,
            "min_data_in_leaf": 20,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq":   5,
            "label_gain":     label_gain,
            "verbose":        -1,
            "n_jobs":         -1,
        }

        dtrain = lgb.Dataset(
            X_train, label=y_train, group=groups_train,
            feature_name=feature_names or [f"f{i}" for i in range(X_train.shape[1])],
            free_raw_data=False,
        )

        callbacks = [lgb.log_evaluation(50), lgb.early_stopping(30)]
        valid_sets = [dtrain]
        valid_names = ["train"]

        if X_val is not None and y_val is not None and groups_val is not None:
            dval = lgb.Dataset(
                X_val, label=y_val, group=groups_val,
                feature_name=feature_names or [f"f{i}" for i in range(X_val.shape[1])],
                reference=dtrain,
                free_raw_data=False,
            )
            valid_sets  = [dtrain, dval]
            valid_names = ["train", "val"]

        self.model = lgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        print("\nFeature importances:")
        for name, imp in sorted(
            zip(self.model.feature_name(), self.model.feature_importance("gain")),
            key=lambda x: -x[1]
        ):
            print(f"  {name}: {imp:.1f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return raw LightGBM scores (higher = more relevant)."""
        assert self.model is not None, "Train or load model first."
        return self.model.predict(X)

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"Model saved → {path}")

    def load(self, path: Path):
        with open(Path(path), "rb") as f:
            self.model = pickle.load(f)
        print(f"Model loaded ← {path}")


def build_popularity_map(behaviors_train: pl.DataFrame) -> dict:
    """Count article click frequency from training behaviors."""
    pop = {}
    for row in behaviors_train.iter_rows(named=True):
        impressions = row.get("impressions") or []
        labels      = row.get("labels")      or []
        for aid, lbl in zip(impressions, labels):
            if lbl == 1:
                pop[aid] = pop.get(aid, 0) + 1
    return pop


def train_pipeline(dataset: str, max_train_rows: int = 500_000):
    """
    Full training pipeline.
    
    Parameters
    ----------
    dataset       : "mind" or "ebnerd"
    max_train_rows: cap on training impressions to fit in RAM (default 500K)
    """
    from src.features.feature_store import build_features, FEATURE_NAMES
    from src.retrieval.semantic import load_or_compute_embeddings
    from src.retrieval.bm25 import BM25Retriever, _article_text

    articles_path   = PROCESSED_DIR / f"articles_{dataset}.parquet"
    train_path      = PROCESSED_DIR / f"behaviors_{dataset}_train.parquet"
    val_path        = PROCESSED_DIR / f"behaviors_{dataset}_val.parquet"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {train_path}\n"
            f"Download MINDlarge_train and run build_pipeline.py first."
        )

    print(f"[LGBM] Training on {dataset} ...")
    articles = pl.read_parquet(articles_path)

    # Load training behaviors (cap to save RAM)
    behaviors_train = pl.read_parquet(train_path)
    if len(behaviors_train) > max_train_rows:
        behaviors_train = behaviors_train.sample(max_train_rows, seed=42)
    print(f"  Train impressions: {len(behaviors_train):,}")

    # Load val behaviors
    behaviors_val = None
    if val_path.exists():
        behaviors_val = pl.read_parquet(val_path)
        # Filter to impressions with at least one positive label
        behaviors_val = behaviors_val.filter(
            pl.col("labels").list.sum() > 0
        )
        behaviors_val = behaviors_val.head(50_000)
        print(f"  Val impressions:   {len(behaviors_val):,}")

    # --- Embeddings ---
    print("  Loading embeddings...")
    embs, ids = load_or_compute_embeddings(articles, dataset)
    embedding_map = dict(zip(ids, embs))

    # --- BM25 ---
    print("  Loading BM25 index...")
    index_path = Path("data/feature_store/bm25") / dataset
    retriever_bm25 = BM25Retriever()
    if (index_path / "bm25_index.pkl").exists():
        retriever_bm25.load(index_path)
    else:
        retriever_bm25.build(articles)
        retriever_bm25.save(index_path)

    art_rows = articles.select(["article_id", "title", "subtitle"]).to_dicts()
    article_text_map = {r["article_id"]: _article_text(r) for r in art_rows}

    # --- Popularity from train ---
    print("  Computing popularity...")
    # Filter to positive clicks only
    pop_map = build_popularity_map(behaviors_train)
    print(f"  {len(pop_map):,} articles with click counts.")

    # --- Build features ---
    print("  Building train features...")
    X_tr, y_tr, g_tr, _, _ = build_features(
        behaviors_train, articles, embedding_map,
        retriever_bm25, article_text_map, pop_map,
    )
    # Remove rows with label=-1 (test set rows, shouldn't appear in train)
    mask = y_tr >= 0
    X_tr, y_tr, = X_tr[mask], y_tr[mask]
    # Recompute groups after filtering (groups must stay consistent)
    # Simpler: just use the full groups array (all train rows have real labels)

    X_vl = y_vl = g_vl = None
    if behaviors_val is not None:
        print("  Building val features...")
        X_vl, y_vl, g_vl, _, _ = build_features(
            behaviors_val, articles, embedding_map,
            retriever_bm25, article_text_map, pop_map,
        )

    # --- Train ---
    ranker = LGBMRanker()
    ranker.train(
        X_tr, y_tr, g_tr,
        X_vl, y_vl, g_vl,
        feature_names=FEATURE_NAMES,
    )

    # --- Save ---
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"lgbm_{dataset}.pkl"
    ranker.save(model_path)
    return ranker


def inference(
    ranker: LGBMRanker,
    dataset: str,
    split: str = "test",
    out_path: Path = None,
) -> Path:
    """
    Run LGBM inference on test behaviors and write Codabench submission.
    Called by generate.py when --strategy lgbm.
    """
    import shutil, zipfile
    from src.features.feature_store import build_features, FEATURE_NAMES
    from src.retrieval.semantic import load_or_compute_embeddings
    from src.retrieval.bm25 import BM25Retriever, _article_text

    articles_path  = PROCESSED_DIR / f"articles_{dataset}.parquet"
    beh_path       = PROCESSED_DIR / f"behaviors_{dataset}_{split}.parquet"

    articles  = pl.read_parquet(articles_path)
    behaviors = pl.read_parquet(beh_path).sort("impression_id")

    embs, ids = load_or_compute_embeddings(articles, dataset)
    embedding_map = dict(zip(ids, embs))

    index_path = Path("data/feature_store/bm25") / dataset
    retriever_bm25 = BM25Retriever()
    if (index_path / "bm25_index.pkl").exists():
        retriever_bm25.load(index_path)
    else:
        retriever_bm25.build(articles)
        retriever_bm25.save(index_path)
    art_rows = articles.select(["article_id", "title", "subtitle"]).to_dicts()
    article_text_map = {r["article_id"]: _article_text(r) for r in art_rows}

    # Load popularity from saved train data if available
    pop_map = {}
    train_path = PROCESSED_DIR / f"behaviors_{dataset}_train.parquet"
    if train_path.exists():
        pop_map = build_popularity_map(pl.read_parquet(train_path).head(200_000))

    print(f"  Building features for {len(behaviors):,} test impressions...")
    X, y, groups, imp_ids, art_ids = build_features(
        behaviors, articles, embedding_map,
        retriever_bm25, article_text_map, pop_map,
    )

    print("  Running LGBM inference...")
    scores = ranker.predict(X)

    # Reconstruct per-impression rankings
    from pathlib import Path as P
    SUBMISSION_DIR = P("data/submissions")
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    txt_name = "predictions.txt" if dataset == "ebnerd" else "prediction.txt"
    if out_path is None:
        out_path = SUBMISSION_DIR / f"{dataset}_{split}_lgbm.txt"
    txt_path = SUBMISSION_DIR / txt_name

    # Group scores back by impression
    ptr = 0
    n_written = 0
    with open(out_path, "w") as f:
        for g_size in groups:
            imp_id      = imp_ids[ptr]
            imp_arts    = art_ids[ptr:ptr + g_size]
            imp_scores  = scores[ptr:ptr + g_size]
            imp_orig    = behaviors.filter(
                pl.col("impression_id") == imp_id
            )["impressions"][0] or []

            # Sort candidates by LGBM score
            order   = np.argsort(-imp_scores)
            ranked  = [imp_arts[i] for i in order]
            rank_map = {aid: rk for rk, aid in enumerate(ranked, 1)}
            ranks   = [str(rank_map.get(aid, g_size + 1)) for aid in imp_orig]
            f.write(f"{imp_id} [{','.join(ranks)}]\n")
            n_written += 1
            ptr += g_size

    print(f"  Saved → {out_path} ({n_written:,} impressions)")
    shutil.copy(out_path, txt_path)
    zip_path = SUBMISSION_DIR / f"{dataset}_{split}_lgbm.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(txt_path, arcname=txt_name)
    print(f"  Zip → {zip_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="LightGBM Ranker (Q5)")
    parser.add_argument("--dataset", choices=["mind", "ebnerd"], required=True)
    parser.add_argument("--max-train-rows", type=int, default=500_000)
    parser.add_argument("--infer", action="store_true",
                        help="Run inference instead of training")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    model_path = MODELS_DIR / f"lgbm_{args.dataset}.pkl"

    if args.infer:
        ranker = LGBMRanker()
        ranker.load(model_path)
        inference(ranker, args.dataset, args.split)
    else:
        train_pipeline(args.dataset, args.max_train_rows)
        # Run inference on test set immediately after training
        ranker = LGBMRanker()
        ranker.load(model_path)
        inference(ranker, args.dataset, "test")


if __name__ == "__main__":
    main()
