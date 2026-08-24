import os
import subprocess
from pathlib import Path

RAW_DIR = Path("data/raw")
EBNERD_DIR = RAW_DIR / "ebnerd"
MIND_DIR = RAW_DIR / "mind"

EBNERD_FILES = {
    "ebnerd_demo.zip": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_demo.zip",
    "ebnerd_small.zip": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_small.zip",
    # Optional: large files if you have enough disk/time later
    # "ebnerd_large.zip": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_large.zip",
    # "articles_large_only.zip": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/articles_large_only.zip",
    # "ebnerd_testset.zip": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_testset.zip",
}

# For MIND, we will download via Hugging Face datasets or direct wget from the repo.
# The assignment says: https://huggingface.co/datasets/yjw1029/MIND
# We'll use `hf download` but that requires `huggingface_hub`. We'll include it in requirements.
# Alternatively, download directly from the GitHub repo? The dataset is large, HF is best.
# We'll use hf command if available, else wget individual files.

MIND_BASE_URL = "https://huggingface.co/datasets/yjw1029/MIND/resolve/main"

MIND_FILES = {
    "MINDsmall_train.zip": f"{MIND_BASE_URL}/MINDsmall_train.zip",
    "MINDsmall_dev.zip": f"{MIND_BASE_URL}/MINDsmall_dev.zip",
    # Optional:
    # "MINDlarge_train.zip": f"{MIND_BASE_URL}/MINDlarge_train.zip",
    # "MINDlarge_dev.zip": f"{MIND_BASE_URL}/MINDlarge_dev.zip",
    # "MINDlarge_test.zip": f"{MIND_BASE_URL}/MINDlarge_test.zip",
}


def download_file(url, dest_path):
    """Download a file using wget with resume support."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        print(f"File already exists: {dest_path} (skipping)")
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