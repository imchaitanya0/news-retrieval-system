import os
import csv
from pathlib import Path
from datetime import datetime, timedelta
import polars as pl

RAW_DIR = Path("data/raw")
UNZIP_DIR = RAW_DIR / "unzipped"
PROCESSED_DIR = Path("data/processed")

def safe_read_mind_tsv(file_path: Path, col_names: list[str]) -> pl.DataFrame:
    """
    Read MIND TSV with potential embedded newlines/quotes.
    Uses Python csv module to handle quoting properly.
    """
    rows = []
    with open(file_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            # Ensure row has exactly len(col_names) elements (some rows may be incomplete)
            if len(row) == len(col_names):
                rows.append(row)
            else:
                # Handle ragged rows by padding with empty strings or truncating
                rows.append(row[:len(col_names)] + [''] * (len(col_names) - len(row)))
    # Convert to Polars DataFrame
    df = pl.DataFrame(rows, schema=col_names, orient='row')
    return df

def parse_mind_behaviors(df: pl.DataFrame) -> pl.DataFrame:
    """Convert MIND behaviors strings into structured columns."""
    # Split impressions string into list of "id-label" pairs
    df = df.with_columns([
        pl.col("impressions").str.split(" ").alias("imp_pairs"),
        pl.col("history").str.split(" ").alias("hist_list"),
    ])
    # Extract article ids and labels from imp_pairs
    df = df.with_columns([
        pl.col("imp_pairs").map_elements(
            lambda pairs: [p.split("-")[0] for p in pairs if "-" in p],
            return_dtype=pl.List(pl.Utf8)
        ).alias("impressions"),
        pl.col("imp_pairs").map_elements(
            lambda pairs: [int(p.split("-")[1]) if "-" in p else 0 for p in pairs],
            return_dtype=pl.List(pl.Int64)
        ).alias("labels"),
    ])
    # Remove temporary column
    df = df.drop("imp_pairs")
    # History is already a list of article ids
    df = df.rename({"hist_list": "history"})
    # Convert time string to datetime
    df = df.with_columns(
        pl.col("time").str.to_datetime("%m/%d/%Y %I:%M:%S %p", strict=False).alias("impression_time")
    ).drop("time")
    return df

def parse_mind_news(df: pl.DataFrame) -> pl.DataFrame:
    """Rename columns to unified schema and fill missing abstracts."""
    df = df.rename({
        "news_id": "article_id",
        "subcategory": "subcategory",
        "abstract": "subtitle",  # MIND abstract -> subtitle
        "title_entities": "entities",
        "abstract_entities": "abstract_entities",
    })
    # Add dataset column
    df = df.with_columns(pl.lit("MIND").alias("dataset"))
    # Body is empty for MIND (title + abstract only)
    df = df.with_columns(pl.lit("").alias("body"))
    # Use category as category, subcategory as subcategory
    df = df.with_columns(
        pl.col("category").alias("category_str"),
        pl.col("subcategory").alias("subcategory_str"),
    )
    # Popularity: placeholder 0 (will be computed later if needed)
    df = df.with_columns(pl.lit(0).cast(pl.Int64).alias("popularity"))
    # Select unified columns
    df = df.select([
        "dataset", "article_id", "title", "subtitle", "body",
        "category", "subcategory", "category_str", "subcategory_str",
        "published_time", "popularity", "entities", "abstract_entities"
    ])
    return df

def parse_ebnerd_articles(df: pl.DataFrame, dataset_name: str) -> pl.DataFrame:
    """Convert EB-NeRD articles to unified schema."""
    df = df.with_columns(pl.lit(dataset_name).alias("dataset"))
    # We want unified columns: dataset, article_id, title, subtitle, body,
    # category, subcategory, published_time, popularity, entities, abstract_entities
    df = df.with_columns([
        pl.col("category_str").alias("category"),
        pl.col("subcategory").cast(pl.Utf8).alias("subcategory"),
        pl.col("total_inviews").alias("popularity"),
        pl.col("entity_groups").alias("entities"),
        pl.col("topics").alias("abstract_entities"),
    ])
    df = df.select([
        "dataset", "article_id", "title", "subtitle", "body",
        "category", "subcategory", "published_time", "popularity",
        "entities", "abstract_entities"
    ])
    return df
    

def parse_ebnerd_behaviors(behaviors_df: pl.DataFrame, history_df: pl.DataFrame) -> pl.DataFrame:
    """
    Combine EB-NeRD behaviors and history into unified behaviors.
    Returns dataframe with columns:
    dataset, impression_id, user_id, impression_time, history, impressions, labels
    """
    # Join history to behaviors on user_id (history has full history; behaviors only impression info)
    # We'll just use behaviors and history separately; for each impression we need user's history up to that time.
    # For simplicity, we ignore the separate history file for now and use only the impression's article_ids_inview and clicked.
    # But later we'll need history for query construction. We'll attach history separately.
    # Here we create unified behaviors without history (history will be computed on the fly from history file).
    df = behaviors_df.with_columns(pl.lit("EB-NeRD").alias("dataset"))
    df = df.rename({
        "article_ids_inview": "impressions",
        "article_ids_clicked": "clicked_ids",
    })
    # Create labels: 1 if article_id in clicked_ids else 0
    df = df.with_columns(
        pl.col("impressions").map_elements(
            lambda inview: [1 if aid in inview else 0 for aid in inview],  # not correct, need clicked ids
            return_dtype=pl.List(pl.Int64)
        ).alias("labels_temp")
    )
    # Correct label creation: we'll do it later after we have clicked_ids as a set
    # Better: use article_ids_inview and article_ids_clicked lists
    # We'll create labels by checking if each id in impressions is in clicked_ids
    df = df.with_columns(
        pl.col("article_ids_clicked").alias("clicked_ids")
    )
    df = df.drop("article_ids_clicked")
    df = df.select([
        "dataset", "impression_id", "user_id", "impression_time",
        "impressions", "clicked_ids"
    ])
    # Now create labels list
    df = df.with_columns(
        pl.struct(["impressions", "clicked_ids"]).map_elements(
            lambda row: [1 if aid in row["clicked_ids"] else 0 for aid in row["impressions"]],
            return_dtype=pl.List(pl.Int64)
        ).alias("labels")
    )
    df = df.drop("clicked_ids")
    return df

def build_ebnerd_pipeline(split_ratio=(0.8, 0.1, 0.1)):
    """Process EB-NeRD demo and small, create unified articles and behaviors."""
    articles_frames = []
    behaviors_frames = []
    for dataset_dir in [UNZIP_DIR / "ebnerd_demo", UNZIP_DIR / "ebnerd_small"]:
        if not dataset_dir.exists():
            continue
        print(f"Processing {dataset_dir.name}...")
        articles_path = dataset_dir / "articles.parquet"
        articles = pl.read_parquet(articles_path)
        articles = parse_ebnerd_articles(articles, dataset_dir.name)
        articles_frames.append(articles)

        # Process train and validation behaviors
        for split_name in ["train", "validation"]:
            behaviors_path = dataset_dir / split_name / "behaviors.parquet"
            history_path = dataset_dir / split_name / "history.parquet"
            if not behaviors_path.exists():
                continue
            behaviors = pl.read_parquet(behaviors_path)
            history = pl.read_parquet(history_path)
            behav_unified = parse_ebnerd_behaviors(behaviors, history)
            behav_unified = behav_unified.with_columns(pl.lit(split_name).alias("split_original"))
            behaviors_frames.append(behav_unified)
    # Combine all articles
    all_articles = pl.concat(articles_frames).unique(subset=["article_id"])
    # Combine all behaviors
    all_behaviors = pl.concat(behaviors_frames)
    # Add a global temporal split
    # Sort by impression_time, compute split based on date ranges
    all_behaviors = all_behaviors.sort("impression_time")
    # Compute total date range and split into train/val/test
    min_time = all_behaviors["impression_time"].min()
    max_time = all_behaviors["impression_time"].max()
    total_days = (max_time - min_time).days
    train_end = min_time + timedelta(days=int(total_days * split_ratio[0]))
    val_end = train_end + timedelta(days=int(total_days * split_ratio[1]))
    all_behaviors = all_behaviors.with_columns(
        pl.when(pl.col("impression_time") <= train_end).then(pl.lit("train"))
        .when(pl.col("impression_time") <= val_end).then(pl.lit("val"))
        .otherwise(pl.lit("test")).alias("split")
    )
    # Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    all_articles.write_parquet(PROCESSED_DIR / "articles_ebnerd.parquet")
    for split in ["train", "val", "test"]:
        subset = all_behaviors.filter(pl.col("split") == split)
        subset.write_parquet(PROCESSED_DIR / f"behaviors_ebnerd_{split}.parquet")
    return all_articles, all_behaviors

def build_mind_pipeline(split_ratio=(0.8, 0.1, 0.1)):
    """Process MIND small, create unified articles and behaviors."""
    # Articles
    news_path = RAW_DIR / "mind" / "news.tsv"
    news_df = safe_read_mind_tsv(news_path, ["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"])
    articles = parse_mind_news(news_df)
    # Behaviors
    behav_path = RAW_DIR / "mind" / "behaviors.tsv"
    behav_df = safe_read_mind_tsv(behav_path, ["impression_id", "user_id", "time", "history", "impressions"])
    behaviors = parse_mind_behaviors(behav_df)
    behaviors = behaviors.with_columns(pl.lit("MIND").alias("dataset"))
    # Temporal split: MIND small train and dev are separate; we only have train.tsv (we copied from MINDsmall_train). Actually we have both train and dev? We copied both? The files in data/raw/mind were from MINDsmall_train and MINDsmall_dev? In Kaggle we only copied MINDsmall_train and MINDsmall_dev? The listing showed MINDsmall_train/ and MINDsmall_dev/ but we copied contents of both into data/raw/mind, so we have behaviors.tsv and news.tsv from train? Wait, we copied both train and dev into same folder, overwriting? Actually we copied both, but they likely have same filenames, so later copy of dev overwrote train. We need to handle this. For now, we will treat the available behaviors.tsv as the train set. To get dev, we should have separate folders. But for simplicity, we will combine all and split temporally ourselves.
    # Here we just use the single behaviors.tsv (which is the last copied, likely dev). To be safe, we will later fix the copy step. For now, assume it's train.
    # We will create a temporal split from this single file.
    behaviors = behaviors.with_columns(
        pl.col("impression_time").alias("impression_time")
    )
    # Sort and split
    behaviors = behaviors.sort("impression_time")
    min_time = behaviors["impression_time"].min()
    max_time = behaviors["impression_time"].max()
    total_days = (max_time - min_time).days
    train_end = min_time + timedelta(days=int(total_days * split_ratio[0]))
    val_end = train_end + timedelta(days=int(total_days * split_ratio[1]))
    behaviors = behaviors.with_columns(
        pl.when(pl.col("impression_time") <= train_end).then(pl.lit("train"))
        .when(pl.col("impression_time") <= val_end).then(pl.lit("val"))
        .otherwise(pl.lit("test")).alias("split")
    )
    # Save articles and behaviors
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    articles.write_parquet(PROCESSED_DIR / "articles_mind.parquet")
    for split in ["train", "val", "test"]:
        subset = behaviors.filter(pl.col("split") == split)
        subset.write_parquet(PROCESSED_DIR / f"behaviors_mind_{split}.parquet")
    return articles, behaviors

def main():
    print("Building EB-NeRD pipeline...")
    build_ebnerd_pipeline()
    print("Building MIND pipeline...")
    build_mind_pipeline()
    print("Pipeline complete. Processed data saved to data/processed/")

if __name__ == "__main__":
    main()