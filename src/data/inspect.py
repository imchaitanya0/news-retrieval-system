import os
import zipfile
from pathlib import Path
import polars as pl

RAW_DIR = Path("data/raw")
EBNERD_DIR = RAW_DIR / "ebnerd"
MIND_DIR = RAW_DIR / "mind"
UNZIP_DIR = RAW_DIR / "unzipped"

def unzip_all():
    """Unzip all zip files in raw directories into a common unzipped folder."""
    UNZIP_DIR.mkdir(parents=True, exist_ok=True)
    for dataset_dir in [EBNERD_DIR, MIND_DIR]:
        if not dataset_dir.exists():
            continue
        for zip_path in dataset_dir.glob("*.zip"):
            print(f"Unzipping {zip_path.name} ...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as z:
                    z.extractall(UNZIP_DIR / zip_path.stem)
                print(f"Unzipped to {UNZIP_DIR / zip_path.stem}")
            except Exception as e:
                print(f"Failed to unzip {zip_path}: {e}")

def inspect_tsv_files(folder: Path):
    """Print first 5 rows and schema of each tsv/csv in folder."""
    for file in sorted(folder.iterdir()):
        if file.suffix in [".tsv", ".csv", ".txt"]:
            print(f"\n===== {file.name} =====")
            try:
                # Try reading as TSV first, fallback to CSV
                df = pl.read_csv(file, separator="\t", n_rows=5)
                sep = "tab"
            except Exception:
                try:
                    df = pl.read_csv(file, n_rows=5)
                    sep = "comma"
                except Exception as e:
                    print(f"Could not read {file.name}: {e}")
                    continue
            print(f"Separator: {sep}")
            print("Columns:", df.columns)
            print(df.head(5))
            print(f"Total columns: {len(df.columns)}")
        else:
            print(f"Skipping non-tabular file: {file.name}")

def main():
    if not any([(EBNERD_DIR.glob("*.zip")), (MIND_DIR.glob("*.zip"))]):
        print("No zip files found. Run download first.")
        return

    unzip_all()

    for folder in UNZIP_DIR.iterdir():
        print(f"\n\n########## Dataset: {folder.name} ##########")
        inspect_tsv_files(folder)

if __name__ == "__main__":
    main()