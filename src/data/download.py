import os
import subprocess
import shutil
from pathlib import Path

RAW_DIR = Path("data/raw")
EBNERD_DIR = RAW_DIR / "ebnerd"
MIND_DIR = RAW_DIR / "mind"

EBNERD_FILES = {
    "ebnerd_demo.zip": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_demo.zip",
    "ebnerd_small.zip": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_small.zip",
}

# Direct Microsoft Azure blob URLs for MIND (public)
MIND_BASE_URL = "https://mind201910small.blob.core.windows.net/release"
MIND_FILES = {
    "MINDsmall_train.zip": f"{MIND_BASE_URL}/MINDsmall_train.zip",
    "MINDsmall_dev.zip": f"{MIND_BASE_URL}/MINDsmall_dev.zip",
}


def download_file(url, dest_path):
    """Download a file using wget with resume support."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"File already exists and non-empty: {dest_path} (skipping)")
        return
    print(f"Downloading {url} -> {dest_path}")
    cmd = ["wget", "-c", "-O", str(dest_path), url]
    subprocess.run(cmd, check=True)
    print(f"Downloaded {dest_path}")


def download_ebnerd():
    for filename, url in EBNERD_FILES.items():
        download_file(url, EBNERD_DIR / filename)


def download_mind():
    for filename, url in MIND_FILES.items():
        download_file(url, MIND_DIR / filename)


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    print("Downloading EB-NeRD demo/small...")
    download_ebnerd()
    print("Downloading MIND-small...")
    download_mind()
    print("All downloads complete.")


if __name__ == "__main__":
    main()