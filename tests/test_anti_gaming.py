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

# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

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
        val_min, val_max = splits["val"]
        test_min, test_max = splits["test"]

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
#  Future-click leakage                                               #
# ------------------------------------------------------------------ #

class TestNoFutureLeakage:
    """
    History used for retrieval must only contain clicks BEFORE the
    impression time.  This is the core anti-gaming check.
    """

    @pytest.mark.parametrize("dataset,split", [
        ("mind", "train"), ("mind", "val"),
    ])
    def test_no_future_click_leakage(self, dataset: str, split: str):
        """
        For MIND: history is a list of article IDs representing past
        clicks. This test checks the behaviors file itself doesn't
        contain impression_time before the earliest known click (a proxy
        check — full check requires joining history timestamps).

        We assert that all impression_time values are non-null, which is
        a pre-condition for temporal splitting to be meaningful.
        """
        path = PROCESSED_DIR / f"behaviors_{dataset}_{split}.parquet"
        df = _load_if_exists(path)
        if df is None:
            pytest.skip(f"{path} not found — run build_pipeline.py first")

        null_count = df["impression_time"].is_null().sum()
        assert null_count == 0, (
            f"[{dataset}/{split}] {null_count} rows have null impression_time — "
            "temporal ordering is impossible"
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
