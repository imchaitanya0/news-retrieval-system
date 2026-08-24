import os
import zipfile
from pathlib import Path
import polars as pl

RAW_DIR = Path("data/raw")
EBNERD_DIR = RAW_DIR / "ebnerd"
MIND_DIR = RAW_DIR / "mind"
UNZIP_DIR = RAW_DIR / "unzipped"

def is_valid_file(path: Path) -> bool:
    """Filter out hidden files and macOS metadata."""
    if "__MACOSX" in path.parts:
        return False
    if path.name.startswith("._"):
        return False
    return True

def unzip_file(zip_path):
    """Unzip a zip file into UNZIP_DIR/<stem>."""
    extract_to = UNZIP_DIR / zip_path.stem
    extract_to.mkdir(parents=True, exist_ok=True)
    print(f"Unzipping {zip_path.name} -> {extract_to}")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_to)
    return extract_to

def inspect_parquet(file_path):
    print(f"\n===== {file_path.relative_to(RAW_DIR)} =====")
    df = pl.read_parquet(file_path, n_rows=5)
    print("Columns:", df.columns)
    print(df.head(5))
    print(f"Total columns: {len(df.columns)}")

def inspect_mind_tsv(file_path):
    print(f"\n===== {file_path.relative_to(RAW_DIR)} =====")
    if file_path.name == "behaviors.tsv":
        col_names = ["impression_id", "user_id", "time", "history", "impressions"]
    elif file_path.name == "news.tsv":
        col_names = ["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
    else:
        print(f"Skipping {file_path.name} (not recognized)")
        return
    # Read all columns as strings, no header, infer_schema_length=0 to avoid parse errors
    df = pl.read_csv(
        file_path,
        separator="\t",
        has_header=False,
        new_columns=col_names,
        n_rows=5,
        infer_schema_length=0,
        dtypes=[pl.Utf8] * len(col_names),
        ignore_errors=True,
    )
    print("Columns:", df.columns)
    print(df.head(5))
    print(f"Total columns: {len(df.columns)}")

def inspect_directory(dir_path):
    """Recursively inspect all .parquet, .tsv, .csv, .txt files in dir."""
    for file in sorted(dir_path.rglob("*")):
        if file.is_file() and is_valid_file(file):
            if file.suffix == ".parquet":
                try:
                    inspect_parquet(file)
                except Exception as e:
                    print(f"Error reading {file}: {e}")
            elif file.suffix in [".tsv", ".csv", ".txt"]:
                print(f"\n===== {file.relative_to(RAW_DIR)} =====")
                print("Skipping non-parquet file in EB-NeRD unzipped folder.")
            # else skip

def main():
    # Process EB-NeRD zips
    if EBNERD_DIR.exists():
        for zip_path in EBNERD_DIR.glob("*.zip"):
            if zip_path.stat().st_size > 0:
                try:
                    extract_to = unzip_file(zip_path)
                    inspect_directory(extract_to)
                except Exception as e:
                    print(f"Failed to unzip {zip_path}: {e}")

    # Process MIND files (already extracted)
    if MIND_DIR.exists():
        print(f"\n\n########## MIND files ##########")
        for file in sorted(MIND_DIR.iterdir()):
            if file.is_file() and is_valid_file(file):
                if file.suffix == ".tsv":
                    inspect_mind_tsv(file)
                elif file.suffix == ".parquet":
                    inspect_parquet(file)
                else:
                    print(f"Skipping {file.name}")

if __name__ == "__main__":
    main()