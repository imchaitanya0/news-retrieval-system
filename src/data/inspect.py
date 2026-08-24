import os
import zipfile
from pathlib import Path
import polars as pl

RAW_DIR = Path("data/raw")
EBNERD_DIR = RAW_DIR / "ebnerd"
MIND_DIR = RAW_DIR / "mind"
UNZIP_DIR = RAW_DIR / "unzipped"

def is_valid_file(path: Path) -> bool:
    if "__MACOSX" in path.parts:
        return False
    if path.name.startswith("._"):
        return False
    return True

def unzip_file(zip_path):
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
    # Known column names for MIND files
    if file_path.name == "behaviors.tsv":
        col_names = ["impression_id", "user_id", "time", "history", "impressions"]
    elif file_path.name == "news.tsv":
        col_names = ["news_id", "category", "subcategory", "title", "abstract",
                     "url", "title_entities", "abstract_entities"]
    else:
        print(f"Skipping {file_path.name} (unrecognized)")
        return

    print("Columns:", col_names)
    # Read first 5 lines as raw text to avoid parsing issues
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [f.readline().strip() for _ in range(5)]
    for i, line in enumerate(lines):
        fields = line.split('\t')
        print(f"Row {i+1} (first 5 fields): {fields[:5]}")
    print(f"Total columns: {len(col_names)}")

def inspect_directory(dir_path):
    for file in sorted(dir_path.rglob("*")):
        if file.is_file() and is_valid_file(file):
            if file.suffix == ".parquet":
                try:
                    inspect_parquet(file)
                except Exception as e:
                    print(f"Error reading {file}: {e}")
            # We ignore other file types in EB-NeRD unzipped folder

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

    # Process MIND files
    if MIND_DIR.exists():
        print(f"\n\n########## MIND files ##########")
        for file in sorted(MIND_DIR.iterdir()):
            if file.is_file() and is_valid_file(file):
                if file.suffix == ".tsv":
                    inspect_mind_tsv(file)
                # .vec files skipped

if __name__ == "__main__":
    main()