import os
import subprocess
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

RAW_DIR = Path("data/raw")
EBNERD_DIR = RAW_DIR / "ebnerd"
MIND_DIR = RAW_DIR / "mind"

EBNERD_FILES = {
    "ebnerd_demo.zip": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_demo.zip",
    "ebnerd_small.zip": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_small.zip",
}

# MIND dataset repository
MIND_REPO_ID = "yjw1029/MIND"
MIND_FILES = [
    "MINDsmall_train.zip",
    "MINDsmall_dev.zip",
]


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
    for filename in MIND_FILES:
        local_path = MIND_DIR / filename
        if local_path.exists():
            print(f"File already exists: {local_path} (skipping)")
            continue
        print(f"Downloading {filename} from Hugging Face dataset {MIND_REPO_ID}...")
        try:
            # Download to cache and copy to our data/raw/mind/ folder
            cached_path = hf_hub_download(
                repo_id=MIND_REPO_ID,
                filename=filename,
                repo_type="dataset",
            )
            shutil.copy(cached_path, local_path)
            print(f"Downloaded {filename} -> {local_path}")
        except Exception as e:
            print(f"Failed to download {filename}: {e}")


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    print("Downloading EB-NeRD demo/small...")
    download_ebnerd()
    print("Downloading MIND-small...")
    download_mind()
    print("All downloads complete.")


if __name__ == "__main__":
    main()