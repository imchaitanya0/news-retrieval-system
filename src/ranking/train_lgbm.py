"""
LightGBM Learning-to-Rank Trainer & Inference (A1 Q5 / A2 Q2)
==============================================================
Trains a LambdaMART ranker on the training behaviors, then uses it
to rerank test impression candidates.

Training flow:
  1. Load training behaviors (article_ids_inview + click labels)
  2. Build 8 features per (user, candidate) pair via feature_store.py
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

    L1/L2 Architecture:
      This is the L2 Ranker. It expects the L1 stage (bm25.py / semantic.py)
      to have already selected candidates per impression. Here we train LightGBM
      on those candidates using 8 in-memory features — no global index lookups.
    """
    from src.features.feature_store import build_features, FEATURE_NAMES
    from src.retrieval.semantic import load_or_compute_embeddings
    from src.retrieval.bm25 import _article_text

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
        behaviors_val = behaviors_val.filter(
            pl.col("labels").list.sum() > 0
        )
        behaviors_val = behaviors_val.head(50_000)
        print(f"  Val impressions:   {len(behaviors_val):,}")

    # --- Embeddings ---
    print("  Loading embeddings...")
    embs, ids = load_or_compute_embeddings(articles, dataset)
    embedding_map = dict(zip(ids, embs))

    # --- Article text map for Lexical Overlap feature ---
    art_rows = articles.select(["article_id", "title", "subtitle"]).to_dicts()
    article_text_map = {r["article_id"]: _article_text(r) for r in art_rows}

    # --- Article publish time map for Freshness feature ---
    pub_col = "published_time" if "published_time" in articles.columns else None
    if pub_col:
        article_publish_map = {
            r["article_id"]: r[pub_col]
            for r in articles.select(["article_id", pub_col]).to_dicts()
        }
    else:
        article_publish_map = {}

    # --- Popularity from train ---
    print("  Computing popularity...")
    pop_map = build_popularity_map(behaviors_train)
    print(f"  {len(pop_map):,} articles with click counts.")

    # --- Build features (GPU-accelerated, 8 features) ---
    print("  Building train features...")
    X_tr, y_tr, g_tr, _, _ = build_features(
        behaviors_train, articles, embedding_map,
        article_text_map=article_text_map,
        popularity_map=pop_map,
        article_publish_map=article_publish_map,
    )
    mask = y_tr >= 0
    X_tr, y_tr = X_tr[mask], y_tr[mask]

    X_vl = y_vl = g_vl = None
    if behaviors_val is not None:
        print("  Building val features...")
        X_vl, y_vl, g_vl, _, _ = build_features(
            behaviors_val, articles, embedding_map,
            article_text_map=article_text_map,
            popularity_map=pop_map,
            article_publish_map=article_publish_map,
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
    chunk_rows: int = 50_000,
) -> Path:
    """
    Run LGBM inference on behaviors and write Codabench submission.
    Processes behaviors in chunks of `chunk_rows` to prevent OOM on large splits.

    MIND test set has 2.37M impressions — building all features at once requires
    ~30 GB RAM. Chunked processing keeps peak RAM under ~8 GB.
    """
    import shutil, zipfile, gc
    from src.features.feature_store import build_features, FEATURE_NAMES
    from src.retrieval.semantic import load_or_compute_embeddings
    from src.retrieval.bm25 import _article_text

    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        def _tqdm(it, **kw): return it

    articles_path = PROCESSED_DIR / f"articles_{dataset}.parquet"
    beh_path      = PROCESSED_DIR / f"behaviors_{dataset}_{split}.parquet"

    articles  = pl.read_parquet(articles_path)
    behaviors = pl.read_parquet(beh_path)
    n_total   = len(behaviors)
    print(f"  {split} set: {n_total:,} impressions — chunk_rows={chunk_rows:,}")

    embs, ids = load_or_compute_embeddings(articles, dataset)
    embedding_map = dict(zip(ids, embs))
    del embs  # free the large numpy array

    art_rows = articles.select(["article_id", "title", "subtitle"]).to_dicts()
    article_text_map = {r["article_id"]: _article_text(r) for r in art_rows}

    pub_col = "published_time" if "published_time" in articles.columns else None
    article_publish_map = (
        {r["article_id"]: r[pub_col]
         for r in articles.select(["article_id", pub_col]).to_dicts()}
        if pub_col else {}
    )

    pop_map = {}
    train_path = PROCESSED_DIR / f"behaviors_{dataset}_train.parquet"
    if train_path.exists():
        pop_map = build_popularity_map(pl.read_parquet(train_path).head(200_000))

    # Build impression_id -> original candidate list map for rank formatting
    imp_to_orig = {}
    for row in behaviors.select(["impression_id", "impressions"]).iter_rows(named=True):
        imp_to_orig[row["impression_id"]] = row["impressions"] or []

    SUBMISSION_DIR = Path("data/submissions")
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    txt_name = "predictions.txt" if dataset == "ebnerd" else "prediction.txt"
    if out_path is None:
        out_path = SUBMISSION_DIR / f"{dataset}_{split}_lgbm.txt"
    txt_path = SUBMISSION_DIR / txt_name

    n_written = 0
    n_chunks  = (n_total + chunk_rows - 1) // chunk_rows

    with open(out_path, "w") as f_out:
        for chunk_idx in _tqdm(range(n_chunks), desc=f"Inference chunks ({split})"):
            chunk_start = chunk_idx * chunk_rows
            chunk_end   = min(chunk_start + chunk_rows, n_total)
            beh_chunk   = behaviors.slice(chunk_start, chunk_end - chunk_start)

            # Build features for this chunk only
            X, y, groups, imp_ids, art_ids = build_features(
                beh_chunk, articles, embedding_map,
                article_text_map=article_text_map,
                popularity_map=pop_map,
                article_publish_map=article_publish_map,
            )

            scores = ranker.predict(X)
            del X   # free immediately

            # Write ranked predictions
            ptr = 0
            for g_size in groups:
                imp_id     = imp_ids[ptr]
                imp_id_key = int(imp_id) if not isinstance(imp_id, str) else imp_id
                imp_arts   = art_ids[ptr:ptr + g_size]
                imp_scores = scores[ptr:ptr + g_size]
                imp_orig   = imp_to_orig.get(imp_id_key, imp_arts)

                order    = np.argsort(-imp_scores)
                ranked   = [imp_arts[i] for i in order]
                rank_map = {aid: rk for rk, aid in enumerate(ranked, 1)}
                ranks    = [str(rank_map.get(aid, g_size + 1)) for aid in imp_orig]
                f_out.write(f"{imp_id} [{','.join(ranks)}]\n")
                n_written += 1
                ptr += g_size

            del beh_chunk, y, groups, imp_ids, art_ids, scores
            gc.collect()

    print(f"  Saved → {out_path} ({n_written:,} impressions)")
    shutil.copy(out_path, txt_path)
    zip_path = SUBMISSION_DIR / f"{dataset}_{split}_lgbm.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(txt_path, arcname=txt_name)
    print(f"  Zip  → {zip_path}")
    return out_path




def main():
    parser = argparse.ArgumentParser(description="LightGBM Ranker (A1 Q5 / A2 Q2)")
    parser.add_argument("--dataset",        choices=["mind", "ebnerd"], required=True)
    parser.add_argument("--max-train-rows", type=int, default=500_000)
    parser.add_argument("--infer",          action="store_true",
                        help="Run inference only (skip training)")
    parser.add_argument("--split",          default="test",
                        help="Split to run inference on (test, val)")
    args = parser.parse_args()

    model_path = MODELS_DIR / f"lgbm_{args.dataset}.pkl"

    if args.infer:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}. Train first (remove --infer)."
            )
        ranker = LGBMRanker()
        ranker.load(model_path)
        inference(ranker, args.dataset, args.split)
    else:
        train_pipeline(args.dataset, args.max_train_rows)
        ranker = LGBMRanker()
        ranker.load(model_path)
        # Generate val predictions (for evaluate.py)
        print("\n--- Generating VAL predictions for evaluation ---")
        inference(ranker, args.dataset, "val",
                  out_path=Path("data/submissions") / f"{args.dataset}_val_lgbm.txt")
        # Generate test predictions (for Codabench submission)
        print("\n--- Generating TEST predictions for Codabench ---")
        inference(ranker, args.dataset, "test")



if __name__ == "__main__":
    main()
