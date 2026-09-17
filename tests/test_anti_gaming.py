"""
Anti-gaming & data integrity tests (Q9 requirement)
=====================================================
Run with:
    pytest tests/test_anti_gaming.py -v
"""

import polars as pl
from pathlib import Path
import pytest

PROCESSED_DIR = Path("data/processed")


def _load_if_exists(path: Path) -> pl.DataFrame | None:
    if path.exists():
        return pl.read_parquet(path)
    return None


# ------------------------------------------------------------------ #
#  Temporal split integrity                                           #
# ------------------------------------------------------------------ #

class TestSplitIntegrity:
    """Verify train/val/test splits are non-overlapping in time."""

    @pytest.mark.parametrize("dataset", ["mind", "ebnerd"])
    def test_splits_non_overlapping(self, dataset: str):
        """No time range of one split should overlap another."""
        splits = {}
        for split in ["train", "val", "test"]:
            path = PROCESSED_DIR / f"behaviors_{dataset}_{split}.parquet"
            df = _load_if_exists(path)
            if df is None or len(df) == 0:
                pytest.skip(f"{path} not found — run build_pipeline.py first")
            splits[split] = (
                df["impression_time"].min(),
                df["impression_time"].max(),
            )

        train_min, train_max = splits["train"]
        val_min, val_max     = splits["val"]
        test_min, test_max   = splits["test"]

        # train.max <= val.min (no overlap between train and val)
        assert train_max <= val_min, (
            f"[{dataset}] Train bleeds into val: train_max={train_max}, val_min={val_min}"
        )
        # val.max <= test.min (no overlap between val and test)
        assert val_max <= test_min, (
            f"[{dataset}] Val bleeds into test: val_max={val_max}, test_min={test_min}"
        )

    @pytest.mark.parametrize("dataset", ["mind", "ebnerd"])
    def test_split_sizes_reasonable(self, dataset: str):
        """Each split must have at least 1 row."""
        for split in ["train", "val", "test"]:
            path = PROCESSED_DIR / f"behaviors_{dataset}_{split}.parquet"
            df = _load_if_exists(path)
            if df is None:
                pytest.skip(f"{path} not found")
            assert len(df) > 0, f"[{dataset}] {split} split is empty!"


# ------------------------------------------------------------------ #
#  Future-click leakage — STRONG test                                #
# ------------------------------------------------------------------ #

class TestNoFutureLeakage:
    """
    History used for retrieval must contain ONLY articles published
    BEFORE the impression_time.

    We validate this via two complementary checks:
      1. Impression time is never null (pre-condition for temporal ordering).
      2. For a sample of rows, article published_time in history is strictly
         < impression_time. This is the actual leakage proof.
    """

    @pytest.mark.parametrize("dataset,split", [
        ("mind", "train"), ("mind", "val"),
    ])
    def test_impression_time_not_null(self, dataset: str, split: str):
        """All impression_time values must be non-null."""
        path = PROCESSED_DIR / f"behaviors_{dataset}_{split}.parquet"
        df = _load_if_exists(path)
        if df is None:
            pytest.skip(f"{path} not found — run build_pipeline.py first")

        null_count = df["impression_time"].is_null().sum()
        assert null_count == 0, (
            f"[{dataset}/{split}] {null_count} rows have null impression_time — "
            "temporal ordering is impossible"
        )

    @pytest.mark.parametrize("dataset,split", [
        ("mind", "train"), ("mind", "val"),
    ])
    def test_no_future_article_in_history(self, dataset: str, split: str):
        """
        STRONG leakage check:
        For a random sample of 500 impressions, verify that no article
        in the user's history was published AFTER the impression_time.

        We join behaviors.history with articles.published_time and assert
        that max(published_time in history) < impression_time for every row.

        This is the gold-standard anti-gaming proof.
        """
        beh_path = PROCESSED_DIR / f"behaviors_{dataset}_{split}.parquet"
        art_path = PROCESSED_DIR / f"articles_{dataset}.parquet"
        behaviors = _load_if_exists(beh_path)
        articles  = _load_if_exists(art_path)

        if behaviors is None or articles is None:
            pytest.skip("Data not available — run build_pipeline.py first")

        if "published_time" not in articles.columns:
            pytest.skip("articles parquet missing published_time column")

        if "history" not in behaviors.columns:
            pytest.skip("behaviors parquet missing history column")

        # Build article → published_time lookup
        pub_map = {
            r["article_id"]: r["published_time"]
            for r in articles.select(["article_id", "published_time"]).iter_rows(named=True)
            if r["published_time"] is not None
        }

        # Sample 500 rows for efficiency
        sample_size = min(500, len(behaviors))
        sample = behaviors.sample(sample_size, seed=42)

        leakage_rows = 0
        total_checked = 0

        for row in sample.iter_rows(named=True):
            imp_time = row.get("impression_time")
            history  = row.get("history") or []

            if imp_time is None or not history:
                continue

            for aid in history:
                pub_time = pub_map.get(aid)
                if pub_time is None:
                    continue
                total_checked += 1
                # published_time should be <= impression_time
                if pub_time > imp_time:
                    leakage_rows += 1

        assert leakage_rows == 0, (
            f"[{dataset}/{split}] LEAKAGE DETECTED! {leakage_rows} history articles "
            f"published AFTER the impression_time (checked {total_checked} history entries "
            f"in {sample_size} sampled impressions). "
            f"This violates the behaviour-window boundary."
        )


# ------------------------------------------------------------------ #
#  Schema integrity                                                   #
# ------------------------------------------------------------------ #

class TestSchemaIntegrity:
    """Articles and behaviors must conform to the unified schema."""

    ARTICLE_REQUIRED_COLS = [
        "dataset", "article_id", "title", "subtitle", "body",
        "category", "subcategory", "published_time", "popularity",
        "entities", "abstract_entities",
    ]
    BEHAVIOR_REQUIRED_COLS = [
        "dataset", "impression_id", "user_id", "impression_time",
        "impressions", "labels",
    ]

    @pytest.mark.parametrize("dataset", ["mind", "ebnerd"])
    def test_articles_schema(self, dataset: str):
        path = PROCESSED_DIR / f"articles_{dataset}.parquet"
        df = _load_if_exists(path)
        if df is None:
            pytest.skip(f"{path} not found")
        missing = [c for c in self.ARTICLE_REQUIRED_COLS if c not in df.columns]
        assert not missing, f"[{dataset}] articles missing columns: {missing}"

    @pytest.mark.parametrize("dataset,split", [
        ("mind", "train"), ("mind", "val"), ("mind", "test"),
        ("ebnerd", "train"), ("ebnerd", "val"), ("ebnerd", "test"),
    ])
    def test_behaviors_schema(self, dataset: str, split: str):
        path = PROCESSED_DIR / f"behaviors_{dataset}_{split}.parquet"
        df = _load_if_exists(path)
        if df is None:
            pytest.skip(f"{path} not found")
        missing = [c for c in self.BEHAVIOR_REQUIRED_COLS if c not in df.columns]
        assert not missing, f"[{dataset}/{split}] behaviors missing columns: {missing}"

    @pytest.mark.parametrize("dataset,split", [
        ("mind", "train"), ("mind", "val"),
    ])
    def test_labels_match_impressions_length(self, dataset: str, split: str):
        """Every row must have len(labels) == len(impressions)."""
        path = PROCESSED_DIR / f"behaviors_{dataset}_{split}.parquet"
        df = _load_if_exists(path)
        if df is None:
            pytest.skip(f"{path} not found")

        mismatch = df.filter(
            pl.col("impressions").list.len() != pl.col("labels").list.len()
        )
        assert len(mismatch) == 0, (
            f"[{dataset}/{split}] {len(mismatch)} rows have mismatched "
            "impressions/labels lengths"
        )


# ------------------------------------------------------------------ #
#  Behaviour-window boundary enforcement                              #
# ------------------------------------------------------------------ #

class TestBehaviourWindowBoundary:
    """
    The behaviour-window boundary rule: features used at serving time
    must only use information available BEFORE the impression time.

    This checks that our feature store doesn't use 'labels' from the test set
    (which are absent at serving time) during inference.
    """

    @pytest.mark.parametrize("dataset", ["mind", "ebnerd"])
    def test_test_split_has_no_labels(self, dataset: str):
        """
        Test split impressions must have empty or all-zero labels.
        If labels are present, they must all be -1 (unknown) or missing.
        This ensures our features are computed without future click info.
        """
        path = PROCESSED_DIR / f"behaviors_{dataset}_test.parquet"
        df = _load_if_exists(path)
        if df is None:
            pytest.skip(f"{path} not found")

        if "labels" not in df.columns:
            return  # No labels column at all → good

        # Either all null or all empty lists
        non_null = df.filter(pl.col("labels").is_not_null())
        if len(non_null) == 0:
            return  # All null → good

        # If labels exist, they should all be empty lists or [-1, -1, ...] placeholders
        has_positive = non_null.filter(
            pl.col("labels").list.sum() > 0
        )
        assert len(has_positive) == 0, (
            f"[{dataset}/test] {len(has_positive)} test impressions have positive labels — "
            "this means ground-truth clicks leaked into the test split!"
        )
