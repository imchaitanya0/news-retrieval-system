import os
import zipfile
from pathlib import Path
import polars as pl

RAW_DIR = Path("data/raw")
EBNERD_DIR = RAW_DIR / "ebnerd"
MIND_DIR = RAW_DIR / "mind"
UNZIP_DIR = RAW_DIR / "unzipped"

def unzip_file(zip_path):
    """Unzip a zip file into UNZIP_DIR/<stem>."""
    extract_to = UNZIP_DIR / zip_path.stem
    extract_to.mkdir(parents=True, exist_ok=True)
    print(f"Unzipping {zip_path.name} -> {extract_to}")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_to)
    return extract_to

def inspect_file(file_path):
    """Print first 5 rows and schema for parquet, tsv, or csv."""
    print(f"\n===== {file_path.relative_to(RAW_DIR)} =====")
    try:
        if file_path.suffix == ".parquet":
            df = pl.read_parquet(file_path, n_rows=5)
        elif file_path.suffix in [".tsv", ".txt"]:
            df = pl.read_csv(file_path, separator="\t", n_rows=5)
        elif file_path.suffix == ".csv":
            df = pl.read_csv(file_path, n_rows=5)
        else:
            print("Skipping unsupported file type.")
            return
        print("Columns:", df.columns)
        print(df.head(5))
        print(f"Total columns: {len(df.columns)}")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

def inspect_directory(dir_path):
    """Recursively inspect all .parquet, .tsv, .csv, .txt files in dir."""
    for file in sorted(dir_path.rglob("*")):
        if file.is_file() and file.suffix in [".parquet", ".tsv", ".csv", ".txt"]:
            inspect_file(file)

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
            if file.is_file() and file.suffix in [".tsv", ".csv", ".parquet"]:
                inspect_file(file)

if __name__ == "__main__":
    main()