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
    """Convert MIND behaviors strings into structured columns.

    Input columns (from TSV): impression_id, user_id, time, history, impressions
    Output columns: impression_id, user_id, impression_time, history (List[Utf8]),
                    impressions (List[Utf8]), labels (List[Int64])

    Fix: split 'history' string into a list in-place by dropping the original
    string column then renaming the parsed list column, avoiding DuplicateError.
    """
    # Parse impressions: "N1-1 N2-0 N3-1" → separate id and label lists
    df = df.with_columns(
        pl.col("impressions").str.split(" ").alias("imp_pairs"),
    )
    df = df.with_columns([
        pl.col("imp_pairs").map_elements(
            # Keep ALL article IDs — strip the '-label' suffix if present
            lambda pairs: [p.split("-")[0] for p in pairs if p.strip()],
            return_dtype=pl.List(pl.Utf8)
        ).alias("impressions"),
        pl.col("imp_pairs").map_elements(
            # Label = 1/0 if suffix present, -1 if unknown (test/dev set)
            lambda pairs: [
                int(p.split("-")[1]) if "-" in p else -1
                for p in pairs if p.strip()
            ],
            return_dtype=pl.List(pl.Int64)
        ).alias("labels"),
    ])
    df = df.drop("imp_pairs")

    df = df.with_columns(
        pl.col("history").fill_null("").str.split(" ").list.eval(
            pl.element().filter(pl.element().str.len_chars() > 0)
        ).alias("history_list"),
    )
    df = df.drop("history")
    df = df.rename({"history_list": "history"})

    # Convert time string to datetime
    df = df.with_columns(
        pl.col("time").str.to_datetime("%m/%d/%Y %I:%M:%S %p", strict=False).alias("impression_time")
    ).drop("time")
    return df

def parse_mind_news(df: pl.DataFrame) -> pl.DataFrame:
    """Rename columns to unified schema and fill missing abstracts.
    
    MIND TSV columns: news_id, category, subcategory, title, abstract,
                      url, title_entities, abstract_entities
    NOTE: MIND has no published_time — we add a null placeholder so the
    schema stays unified with EB-NeRD.
    """
    df = df.rename({
        "news_id": "article_id",
        "abstract": "subtitle",   # MIND 'abstract' -> unified 'subtitle'
        "title_entities": "entities",
    })
    # Add dataset column
    df = df.with_columns(pl.lit("MIND").alias("dataset"))
    # Body is empty for MIND (title + abstract only)
    df = df.with_columns(pl.lit("").alias("body"))
    # category_str / subcategory_str mirrors EB-NeRD naming
    df = df.with_columns(
        pl.col("category").alias("category_str"),
        pl.col("subcategory").alias("subcategory_str"),
    )
    # Popularity: placeholder 0 (will be computed from click counts later)
    df = df.with_columns(pl.lit(0).cast(pl.Int64).alias("popularity"))
    # MIND has no publish date — add null so unified schema is consistent
    df = df.with_columns(pl.lit(None).cast(pl.Datetime).alias("published_time"))
    # Select unified columns
    df = df.select([
        "dataset", "article_id", "title", "subtitle", "body",
        "category", "subcategory", "category_str", "subcategory_str",
        "published_time", "popularity", "entities", "abstract_entities"
    ])
    return df

def parse_ebnerd_articles(df: pl.DataFrame, dataset_name: str) -> pl.DataFrame:
    df = df.with_columns(pl.lit(dataset_name).alias("dataset"))
    df = df.with_columns([
        pl.col("category_str").alias("category"),
        pl.col("subcategory").cast(pl.List(pl.Utf8)).list.join(",").alias("subcategory"),
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
    # Rename inview/clicked columns to unified schema names
    # Note: test set does not have article_ids_clicked!
    df = df.rename({"article_ids_inview": "impressions"})
    if "article_ids_clicked" in df.columns:
        df = df.rename({"article_ids_clicked": "clicked_ids"})
    else:
        df = df.with_columns(pl.lit(None).alias("clicked_ids"))

    # Select only columns we need (drops everything else safely)
    df = df.select([
        "dataset", "impression_id", "user_id", "impression_time",
        "impressions", "clicked_ids"
    ])
    # Create labels:    # Labels: 1 if in clicked_ids, else 0 (or -1 if clicked_ids is null for test set)
    df = df.with_columns(
        pl.struct(["impressions", "clicked_ids"]).map_elements(
            lambda x: (
                [-1] * len(x["impressions"]) if x["clicked_ids"] is None
                else [1 if aid in x["clicked_ids"] else 0 for aid in x["impressions"]]
            ),
            return_dtype=pl.List(pl.Int64)
        ).alias("labels")
    )
    df = df.drop("clicked_ids")
    return df

def build_ebnerd_pipeline():
    """Process EB-NeRD data recursively, creating unified articles and behaviors."""
    ebnerd_dir = RAW_DIR / "ebnerd"
    if not ebnerd_dir.exists():
        print("No EB-NeRD directory found.")
        return

    articles_frames = []
    # Find all articles.parquet
    for articles_path in ebnerd_dir.rglob("articles.parquet"):
        print(f"Processing EB-NeRD articles from {articles_path.parent.name}...")
        articles = pl.read_parquet(articles_path)
        articles = parse_ebnerd_articles(articles, articles_path.parent.name)
        articles_frames.append(articles)

    if articles_frames:
        all_articles = pl.concat(articles_frames).unique(subset=["article_id"])
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        all_articles.write_parquet(PROCESSED_DIR / "articles_ebnerd.parquet")
    else:
        print("No EB-NeRD articles found.")

    # Find all behaviors.parquet and save them independently to avoid OOM and preserve order
    for behaviors_path in ebnerd_dir.rglob("behaviors.parquet"):
        history_path = behaviors_path.parent / "history.parquet"
        if not history_path.exists():
            continue
        folder = behaviors_path.parent.name.lower()
        if "test" in folder:
            split_name = "test"
        elif "dev" in folder or "validation" in folder:
            split_name = "val"
        else:
            split_name = "train"
        
        print(f"Processing EB-NeRD behaviors from {folder} (mapped to {split_name})...")
        behaviors = pl.read_parquet(behaviors_path)
        history = pl.read_parquet(history_path)
        behav_unified = parse_ebnerd_behaviors(behaviors, history)
        
        # Save directly without concat to preserve exact row order and save RAM
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        behav_unified.write_parquet(PROCESSED_DIR / f"behaviors_ebnerd_{split_name}.parquet")
        
    return

def build_mind_pipeline(split_ratio=(0.8, 0.1, 0.1)):
    """Process MIND-small, create unified articles and behaviors parquets.
    
    Reads from data/raw/mind/news.tsv and data/raw/mind/behaviors.tsv.
    If train and dev TSVs were merged into the same folder, the later copy
    (usually dev) wins. We treat the single file as the full available set
    and apply a temporal split ourselves.
    """
    mind_dir = RAW_DIR / "mind"
    if not mind_dir.exists():
        print("No MIND directory found.")
        return

    # Articles
    articles_frames = []
    for news_path in mind_dir.rglob("news.tsv"):
        print(f"Processing MIND articles from {news_path.parent.name}...")
        news_df = safe_read_mind_tsv(news_path, [
            "news_id", "category", "subcategory", "title", "abstract",
            "url", "title_entities", "abstract_entities"
        ])
        articles_frames.append(parse_mind_news(news_df))
    
    if not articles_frames:
        print("No MIND articles found.")
        return
    articles = pl.concat(articles_frames).unique(subset=["article_id"])

    # Behaviors
    for behav_path in mind_dir.rglob("behaviors.tsv"):
        folder = behav_path.parent.name.lower()
        if "test" in folder:
            split_name = "test"
        elif "dev" in folder or "val" in folder:
            split_name = "val"
        else:
            split_name = "train"
        
        print(f"Processing MIND behaviors from {behav_path.parent.name} (mapped to {split_name})...")
        behav_df = safe_read_mind_tsv(behav_path, [
            "impression_id", "user_id", "time", "history", "impressions"
        ])
        behav_unified = parse_mind_behaviors(behav_df)
        behav_unified = behav_unified.with_columns(pl.lit("MIND").alias("dataset"))
        
        # Save directly to preserve exact row order (critical for Codabench)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        behav_unified.write_parquet(PROCESSED_DIR / f"behaviors_mind_{split_name}.parquet")
        
    return

def main():
    print("Building EB-NeRD pipeline...")
    build_ebnerd_pipeline()
    print("Building MIND pipeline...")
    build_mind_pipeline()
    print("Pipeline complete. Processed data saved to data/processed/")

if __name__ == "__main__":
    main()