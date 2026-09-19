"""
build_pipeline.py — ETL: raw files → unified parquet (A1 Q1)
=============================================================
Reads raw MIND TSV and EB-NeRD parquet files from data/raw/,
produces unified parquets in data/processed/.

Usage:
    python -m src.data.build_pipeline           # both datasets
    python -m src.data.build_pipeline --mind    # MIND only
    python -m src.data.build_pipeline --ebnerd  # EB-NeRD only

Raw file layout expected:
    data/raw/mind/MINDlarge_train/  (behaviors.tsv + news.tsv)
    data/raw/mind/MINDlarge_dev/    (behaviors.tsv + news.tsv)
    data/raw/mind/MINDlarge_test/   (behaviors.tsv — no labels)
    data/raw/ebnerd/train/          (behaviors.parquet, history.parquet)
    data/raw/ebnerd/validation/     (behaviors.parquet, history.parquet)
    data/raw/ebnerd/test/           (behaviors.parquet — no labels)
    data/raw/ebnerd/articles.parquet  OR  data/raw/ebnerd/*/articles.parquet

Kaggle dataset mount point:
    /kaggle/input/<dataset-name>/   (symlinked or copied to data/raw/)
"""

import argparse
import csv
import os
from pathlib import Path
from datetime import datetime, timezone

import polars as pl

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


# ------------------------------------------------------------------ #
#  MIND helpers                                                       #
# ------------------------------------------------------------------ #

def safe_read_mind_tsv(file_path: Path, col_names: list) -> pl.DataFrame:
    """
    Read MIND TSV with potential embedded newlines/quotes.
    More robust than pl.read_csv for this format.
    """
    print(f"  Reading {file_path.name} ...")
    rows = []
    n_cols = len(col_names)
    with open(file_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            if len(row) >= n_cols:
                rows.append(row[:n_cols])
            else:
                rows.append(row + [""] * (n_cols - len(row)))
    print(f"  Read {len(rows):,} rows from {file_path.name}")
    return pl.DataFrame(rows, schema=col_names, orient="row")


def parse_mind_behaviors(df: pl.DataFrame) -> pl.DataFrame:
    """
    Convert MIND behaviors TSV → unified schema.
    Input cols:  impression_id, user_id, time, history, impressions
    Output cols: dataset, impression_id, user_id, impression_time,
                 history (List[str]), impressions (List[str]), labels (List[int])
    """
    # Parse "N1-1 N2-0 N3-1" into ids and labels
    df = df.with_columns(
        pl.col("impressions").str.split(" ").alias("imp_pairs")
    )
    df = df.with_columns([
        pl.col("imp_pairs").map_elements(
            lambda pairs: [p.split("-")[0] for p in pairs if p.strip()],
            return_dtype=pl.List(pl.Utf8),
        ).alias("impressions"),
        pl.col("imp_pairs").map_elements(
            lambda pairs: [
                int(p.split("-")[1]) if "-" in p else -1
                for p in pairs if p.strip()
            ],
            return_dtype=pl.List(pl.Int64),
        ).alias("labels"),
    ])
    df = df.drop("imp_pairs")

    # Parse history: "N1 N2 N3" → list[str], handle null/empty
    df = df.with_columns(
        pl.col("history")
        .fill_null("")
        .str.split(" ")
        .list.eval(pl.element().filter(pl.element().str.len_chars() > 0))
        .alias("history_list")
    ).drop("history").rename({"history_list": "history"})

    # Parse time string → datetime (MIND format: "11/15/2019 10:22:32 AM")
    df = df.with_columns(
        pl.col("time")
        .str.to_datetime("%m/%d/%Y %I:%M:%S %p", strict=False)
        .alias("impression_time")
    ).drop("time")

    # Add dataset label
    df = df.with_columns(pl.lit("MIND").alias("dataset"))

    # Cast impression_id to Int64
    df = df.with_columns(
        pl.col("impression_id").cast(pl.Int64, strict=False)
    )

    return df.select([
        "dataset", "impression_id", "user_id", "impression_time",
        "history", "impressions", "labels",
    ])


def parse_mind_news(df: pl.DataFrame) -> pl.DataFrame:
    """
    Rename MIND news TSV columns → unified article schema.
    MIND columns: news_id, category, subcategory, title, abstract,
                  url, title_entities, abstract_entities
    """
    df = df.rename({
        "news_id":        "article_id",
        "abstract":       "subtitle",
        "title_entities": "entities",
    })
    df = df.with_columns([
        pl.lit("MIND").alias("dataset"),
        pl.lit("").alias("body"),
        pl.lit(0).cast(pl.Int64).alias("popularity"),
        # MIND has no published_time — use null (freshness feature will default to 0.5)
        pl.lit(None).cast(pl.Datetime("us")).alias("published_time"),
    ])
    # Ensure abstract_entities column exists
    if "abstract_entities" not in df.columns:
        df = df.with_columns(pl.lit("").alias("abstract_entities"))

    return df.select([
        "dataset", "article_id", "title", "subtitle", "body",
        "category", "subcategory", "published_time", "popularity",
        "entities", "abstract_entities",
    ])


# ------------------------------------------------------------------ #
#  EB-NeRD helpers                                                    #
# ------------------------------------------------------------------ #

def parse_ebnerd_articles(df: pl.DataFrame) -> pl.DataFrame:
    """
    Normalise EB-NeRD articles.parquet → unified schema.
    Key EB-NeRD columns: article_id, title, subtitle, body,
                         category_str, subcategory, published_time,
                         total_inviews, entity_groups, topics
    """
    rename_map = {}
    if "category_str" in df.columns:
        rename_map["category_str"] = "category"
    if "total_inviews" in df.columns:
        rename_map["total_inviews"] = "popularity"
    if rename_map:
        df = df.rename(rename_map)

    # Subcategory may be a list
    if "subcategory" in df.columns:
        if df["subcategory"].dtype == pl.List(pl.Utf8):
            df = df.with_columns(
                pl.col("subcategory").list.join(",")
            )
        else:
            df = df.with_columns(pl.col("subcategory").cast(pl.Utf8))
    else:
        df = df.with_columns(pl.lit("").alias("subcategory"))

    # Entities columns
    for col, alias in [("entity_groups", "entities"), ("topics", "abstract_entities")]:
        if col in df.columns:
            df = df.rename({col: alias}) if alias not in df.columns else df
        elif alias not in df.columns:
            df = df.with_columns(pl.lit("").alias(alias))

    if "entities" not in df.columns:
        df = df.with_columns(pl.lit("").alias("entities"))
    if "abstract_entities" not in df.columns:
        df = df.with_columns(pl.lit("").alias("abstract_entities"))
    if "body" not in df.columns:
        df = df.with_columns(pl.lit("").alias("body"))
    if "subtitle" not in df.columns:
        df = df.with_columns(pl.lit("").alias("subtitle"))
    if "popularity" not in df.columns:
        df = df.with_columns(pl.lit(0).cast(pl.Int64).alias("popularity"))

    df = df.with_columns(pl.lit("EB-NeRD").alias("dataset"))

    return df.select([
        "dataset", "article_id", "title", "subtitle", "body",
        "category", "subcategory", "published_time", "popularity",
        "entities", "abstract_entities",
    ])


def parse_ebnerd_behaviors(
    behaviors_df: pl.DataFrame,
    history_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Combine EB-NeRD behaviors + history → unified schema.
    Output: dataset, impression_id, user_id, impression_time,
            history (List[int]), impressions (List[int]), labels (List[int])
    """
    df = behaviors_df.with_columns(pl.lit("EB-NeRD").alias("dataset"))

    # Rename inview to impressions
    if "article_ids_inview" in df.columns:
        df = df.rename({"article_ids_inview": "impressions"})

    # clicked_ids → used to create labels
    if "article_ids_clicked" in df.columns:
        df = df.rename({"article_ids_clicked": "clicked_ids"})
    else:
        df = df.with_columns(pl.lit(None).alias("clicked_ids"))

    # Join history — history_df has user_id + article_id_fixed (list)
    hist_col = "article_id_fixed" if "article_id_fixed" in history_df.columns else history_df.columns[-1]
    df = df.join(
        history_df.select(["user_id", hist_col]).rename({hist_col: "history"}),
        on="user_id",
        how="left",
    )

    # Build labels from clicked_ids
    df = df.with_columns(
        pl.struct(["impressions", "clicked_ids"]).map_elements(
            lambda x: (
                [-1] * len(x["impressions"] or [])
                if not x["clicked_ids"]
                else [
                    1 if aid in (x["clicked_ids"] or []) else 0
                    for aid in (x["impressions"] or [])
                ]
            ),
            return_dtype=pl.List(pl.Int64),
        ).alias("labels")
    ).drop("clicked_ids")

    return df.select([
        "dataset", "impression_id", "user_id", "impression_time",
        "history", "impressions", "labels",
    ])


# ------------------------------------------------------------------ #
#  Pipeline runners                                                   #
# ------------------------------------------------------------------ #

def build_mind_pipeline():
    """
    Process all MIND splits found under data/raw/mind/.
    Looks for subdirectories containing behaviors.tsv and news.tsv.
    Subdirectory name determines split: train/dev/val → val, test → test, else train.
    """
    mind_dir = RAW_DIR / "mind"
    if not mind_dir.exists():
        print(f"[MIND] No directory found at {mind_dir} — skipping.")
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # --- Articles (dedup across all splits) ---
    articles_frames = []
    for news_path in sorted(mind_dir.rglob("news.tsv")):
        print(f"[MIND] Articles from: {news_path.parent.name}")
        news_df = safe_read_mind_tsv(news_path, [
            "news_id", "category", "subcategory", "title", "abstract",
            "url", "title_entities", "abstract_entities",
        ])
        articles_frames.append(parse_mind_news(news_df))

    if not articles_frames:
        print("[MIND] No news.tsv files found!")
        return

    articles = pl.concat(articles_frames).unique(subset=["article_id"])
    out = PROCESSED_DIR / "articles_mind.parquet"
    articles.write_parquet(out)
    print(f"[MIND] Articles → {out}  ({len(articles):,} rows)")

    # --- Behaviors ---
    for behav_path in sorted(mind_dir.rglob("behaviors.tsv")):
        folder = behav_path.parent.name.lower()
        if "test" in folder:
            split_name = "test"
        elif "dev" in folder or "val" in folder:
            split_name = "val"
        else:
            split_name = "train"

        print(f"[MIND] Behaviors {behav_path.parent.name} → {split_name}")
        behav_df = safe_read_mind_tsv(behav_path, [
            "impression_id", "user_id", "time", "history", "impressions",
        ])
        unified = parse_mind_behaviors(behav_df)
        out = PROCESSED_DIR / f"behaviors_mind_{split_name}.parquet"
        unified.write_parquet(out)
        print(f"[MIND] {split_name} → {out}  ({len(unified):,} rows)")


def build_ebnerd_pipeline():
    """
    Process all EB-NeRD splits found under data/raw/ebnerd/.
    Subdirectory name determines split: test → test, validation/dev → val, else train.
    """
    ebnerd_dir = RAW_DIR / "ebnerd"
    if not ebnerd_dir.exists():
        print(f"[EB-NeRD] No directory found at {ebnerd_dir} — skipping.")
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # --- Articles ---
    # Search multiple possible locations
    art_candidates = list(ebnerd_dir.rglob("articles.parquet"))
    articles_frames = []
    seen_ids = set()
    for art_path in sorted(art_candidates):
        print(f"[EB-NeRD] Articles from: {art_path.parent.name}")
        df = pl.read_parquet(art_path)
        df = parse_ebnerd_articles(df)
        # Dedup by source to avoid double-counting articles.parquet at multiple levels
        new_ids = set(df["article_id"].to_list()) - seen_ids
        if new_ids:
            articles_frames.append(df.filter(pl.col("article_id").is_in(new_ids)))
            seen_ids |= new_ids

    if not articles_frames:
        print("[EB-NeRD] No articles.parquet found!")
        return

    articles = pl.concat(articles_frames).unique(subset=["article_id"])
    out = PROCESSED_DIR / "articles_ebnerd.parquet"
    articles.write_parquet(out)
    print(f"[EB-NeRD] Articles → {out}  ({len(articles):,} rows)")

    # --- Behaviors ---
    for behav_path in sorted(ebnerd_dir.rglob("behaviors.parquet")):
        history_path = behav_path.parent / "history.parquet"
        if not history_path.exists():
            print(f"[EB-NeRD] No history.parquet next to {behav_path} — skipping")
            continue

        folder = behav_path.parent.name.lower()
        if "test" in folder:
            split_name = "test"
        elif "val" in folder or "dev" in folder:
            split_name = "val"
        else:
            split_name = "train"

        print(f"[EB-NeRD] Behaviors {behav_path.parent.name} → {split_name}")
        behaviors = pl.read_parquet(behav_path)
        history   = pl.read_parquet(history_path)
        unified   = parse_ebnerd_behaviors(behaviors, history)
        out       = PROCESSED_DIR / f"behaviors_ebnerd_{split_name}.parquet"
        unified.write_parquet(out)
        print(f"[EB-NeRD] {split_name} → {out}  ({len(unified):,} rows)")


# ------------------------------------------------------------------ #
#  Kaggle helper: symlink datasets into data/raw/                    #
# ------------------------------------------------------------------ #

def setup_kaggle_raw_dirs(
    mind_input_dir: str = None,
    ebnerd_input_dir: str = None,
):
    """
    Helper to symlink or copy Kaggle input dataset folders into data/raw/.
    Call this BEFORE build_mind_pipeline() / build_ebnerd_pipeline().

    Parameters
    ----------
    mind_input_dir   : path to Kaggle MIND dataset dir (contains MINDlarge_*/  or MINDsmall_*/)
    ebnerd_input_dir : path to Kaggle EB-NeRD dataset dir (contains train/, validation/, test/)
    """
    import shutil

    if mind_input_dir and os.path.exists(mind_input_dir):
        target = RAW_DIR / "mind"
        if not target.exists():
            print(f"Linking {mind_input_dir} → {target}")
            os.makedirs(RAW_DIR, exist_ok=True)
            os.symlink(mind_input_dir, target)
        else:
            print(f"{target} already exists, skipping symlink")

    if ebnerd_input_dir and os.path.exists(ebnerd_input_dir):
        target = RAW_DIR / "ebnerd"
        if not target.exists():
            print(f"Linking {ebnerd_input_dir} → {target}")
            os.makedirs(RAW_DIR, exist_ok=True)
            os.symlink(ebnerd_input_dir, target)
        else:
            print(f"{target} already exists, skipping symlink")


def main():
    parser = argparse.ArgumentParser(description="Build unified parquets from raw data (A1 Q1)")
    parser.add_argument("--mind",   action="store_true", help="Process MIND only")
    parser.add_argument("--ebnerd", action="store_true", help="Process EB-NeRD only")
    args = parser.parse_args()

    run_mind   = args.mind   or (not args.mind and not args.ebnerd)
    run_ebnerd = args.ebnerd or (not args.mind and not args.ebnerd)

    if run_mind:
        print("\n=== Building MIND pipeline ===")
        build_mind_pipeline()
    if run_ebnerd:
        print("\n=== Building EB-NeRD pipeline ===")
        build_ebnerd_pipeline()

    print("\n✓ Pipeline complete. Files in data/processed/:")
    for f in sorted(PROCESSED_DIR.glob("*.parquet")):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name:50s}  {size_mb:6.1f} MB")


if __name__ == "__main__":
    main()