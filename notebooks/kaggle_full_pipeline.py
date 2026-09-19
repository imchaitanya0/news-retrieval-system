# ============================================================
#  COMPLETE A1 + A2 KAGGLE NOTEBOOK
#  Copy each cell block into a Kaggle notebook cell
# ============================================================
#
# DATASETS TO ADD IN KAGGLE (Settings → Add Data):
#
#   MIND (pick ONE based on what you have access to):
#     Option A (recommended - already on Kaggle):
#       Search for: "mind-news-dataset" by  "arashnic"
#       This gives: MINDlarge_train/ MINDlarge_dev/ MINDlarge_test/
#       Mount path will be: /kaggle/input/mind-news-dataset/
#
#     Option B: Upload your zip as a Kaggle Dataset
#       Create dataset from MINDlarge_train.zip + MINDlarge_dev.zip + MINDlarge_test.zip
#
#   EB-NeRD:
#     Search for: "eb-nerd-benchmark-news-recommendation"
#     Mount path will be: /kaggle/input/eb-nerd-benchmark-news-recommendation/
#     This gives: train/ validation/ test/ articles.parquet
#
# GPU: Enable T4 GPU in Settings → Accelerator
# RAM: Enable internet in Settings → Internet (needed for git clone + pip install)
# ============================================================

# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 1: Install dependencies (~3 min, run once)        ║
# ╚══════════════════════════════════════════════════════════╝
import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

# Core ML
install("lightgbm")
install("sentence-transformers")
install("faiss-cpu")
install("bm25s")
install("rank-bm25")    # fallback if bm25s fails
install("polars")
install("tqdm")

# Check GPU
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 2: Clone repo and set paths                       ║
# ╚══════════════════════════════════════════════════════════╝
import os, sys

REPO_DIR = "/kaggle/working/news-retrieval-system"

# Clone or update
if not os.path.exists(REPO_DIR):
    os.system(f"git clone https://github.com/imchaitanya0/news-retrieval-system.git {REPO_DIR}")
else:
    os.system(f"cd {REPO_DIR} && git pull origin main")

os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)
print(f"Working dir: {os.getcwd()}")

# Create required dirs
for d in ["data/raw/mind", "data/raw/ebnerd", "data/processed",
          "data/feature_store/embeddings/mind", "data/feature_store/embeddings/ebnerd",
          "data/feature_store/bm25", "data/feature_store/semantic",
          "data/models", "data/results", "data/submissions"]:
    os.makedirs(d, exist_ok=True)

print("✓ Directory structure created")

# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 3: Link Kaggle datasets into data/raw/            ║
# ╚══════════════════════════════════════════════════════════╝
import os

# ── MIND ──────────────────────────────────────────────────
# Check what Kaggle input dirs look like
print("Kaggle input dirs:")
for d in sorted(os.listdir("/kaggle/input/")):
    print(f"  /kaggle/input/{d}/")
    for sub in sorted(os.listdir(f"/kaggle/input/{d}/"))[:5]:
        print(f"    {sub}/")

# ── EDIT THESE TWO LINES to match your actual Kaggle dataset name ──
MIND_KAGGLE_DIR   = "/kaggle/input/mind-news-dataset"    # <── change if different
EBNERD_KAGGLE_DIR = "/kaggle/input/eb-nerd-benchmark-news-recommendation"  # <── change if different

def safe_symlink(src, dst):
    """Create symlink dst → src if dst doesn't exist."""
    if os.path.exists(src):
        if not os.path.exists(dst):
            # Remove any broken symlink first
            if os.path.islink(dst):
                os.unlink(dst)
            os.symlink(src, dst)
            print(f"  Linked: {src} → {dst}")
        else:
            print(f"  Already exists: {dst}")
    else:
        print(f"  NOT FOUND: {src}  ← fix MIND_KAGGLE_DIR / EBNERD_KAGGLE_DIR above!")

# Remove old symlinks if they exist
for link in ["data/raw/mind", "data/raw/ebnerd"]:
    if os.path.islink(link):
        os.unlink(link)

safe_symlink(MIND_KAGGLE_DIR,   "data/raw/mind")
safe_symlink(EBNERD_KAGGLE_DIR, "data/raw/ebnerd")

# Verify
print("\nContents of data/raw/mind/:")
try:
    for f in sorted(os.listdir("data/raw/mind"))[:10]:
        print(f"  {f}")
except Exception as e:
    print(f"  ERROR: {e}")

print("\nContents of data/raw/ebnerd/:")
try:
    for f in sorted(os.listdir("data/raw/ebnerd"))[:10]:
        print(f"  {f}")
except Exception as e:
    print(f"  ERROR: {e}")

# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 4: Build processed parquets (~10-20 min)          ║
# ╚══════════════════════════════════════════════════════════╝
from src.data.build_pipeline import build_mind_pipeline, build_ebnerd_pipeline
from pathlib import Path

# Build MIND
print("=" * 60)
print("Building MIND pipeline...")
build_mind_pipeline()

# Build EB-NeRD
print("\n" + "=" * 60)
print("Building EB-NeRD pipeline...")
build_ebnerd_pipeline()

# Verify outputs
print("\n✓ Processed parquets:")
for f in sorted(Path("data/processed").glob("*.parquet")):
    size_mb = f.stat().st_size / 1e6
    import polars as pl
    df = pl.read_parquet(f)
    print(f"  {f.name:50s}  {size_mb:6.1f} MB  ({len(df):,} rows)  cols={df.columns}")

# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 5: Compute embeddings (~20-40 min on T4)          ║
# ╠══════════════════════════════════════════════════════════╣
# ║  This is the longest step. Run once and cache.          ║
# ╚══════════════════════════════════════════════════════════╝
import polars as pl
from src.retrieval.semantic import load_or_compute_embeddings

print("Computing MIND embeddings...")
articles_mind = pl.read_parquet("data/processed/articles_mind.parquet")
embs_mind, ids_mind = load_or_compute_embeddings(articles_mind, "mind")
print(f"✓ MIND embeddings: {embs_mind.shape}")

print("\nComputing EB-NeRD embeddings...")
articles_ebnerd = pl.read_parquet("data/processed/articles_ebnerd.parquet")
embs_ebnerd, ids_ebnerd = load_or_compute_embeddings(articles_ebnerd, "ebnerd")
print(f"✓ EB-NeRD embeddings: {embs_ebnerd.shape}")

# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 6a: BM25 Recall@K — MIND (~10 min)               ║
# ╚══════════════════════════════════════════════════════════╝
from src.retrieval.bm25 import evaluate_bm25
import json

print("=" * 60)
print("BM25 Recall@K — MIND/val")
results_bm25_mind = evaluate_bm25("mind", split="val", k_values=[50, 100, 200])
print("\nResults:", json.dumps(results_bm25_mind, indent=2))

# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 6b: BM25 Recall@K — EB-NeRD (~10 min)            ║
# ╚══════════════════════════════════════════════════════════╝
print("=" * 60)
print("BM25 Recall@K — EB-NeRD/val")
results_bm25_ebnerd = evaluate_bm25("ebnerd", split="val", k_values=[50, 100, 200])
print("\nResults:", json.dumps(results_bm25_ebnerd, indent=2))

# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 7a: Semantic Recall@K — MIND (~5 min)             ║
# ╚══════════════════════════════════════════════════════════╝
from src.retrieval.semantic import evaluate_semantic

print("=" * 60)
print("Semantic Recall@K — MIND/val")
results_sem_mind = evaluate_semantic("mind", split="val", k_values=[50, 100, 200])
print("\nResults:", json.dumps(results_sem_mind, indent=2))

# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 7b: Semantic Recall@K — EB-NeRD (~5 min)          ║
# ╚══════════════════════════════════════════════════════════╝
print("=" * 60)
print("Semantic Recall@K — EB-NeRD/val")
results_sem_ebnerd = evaluate_semantic("ebnerd", split="val", k_values=[50, 100, 200])
print("\nResults:", json.dumps(results_sem_ebnerd, indent=2))

# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 8: Train LightGBM — MIND (~30-60 min)            ║
# ╚══════════════════════════════════════════════════════════╝
from src.ranking.train_lgbm import train_pipeline, inference, LGBMRanker
from pathlib import Path

print("=" * 60)
print("Training LightGBM — MIND")
train_pipeline("mind", max_train_rows=500_000)

# Run inference on val (for evaluation)
ranker = LGBMRanker()
ranker.load(Path("data/models/lgbm_mind.pkl"))
inference(ranker, "mind", split="val",
          out_path=Path("data/submissions/mind_val_lgbm.txt"))
print("✓ MIND LightGBM training + val inference done")

# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 9: Train LightGBM — EB-NeRD (~30-60 min)         ║
# ╚══════════════════════════════════════════════════════════╝
print("=" * 60)
print("Training LightGBM — EB-NeRD")
train_pipeline("ebnerd", max_train_rows=500_000)

ranker_ebnerd = LGBMRanker()
ranker_ebnerd.load(Path("data/models/lgbm_ebnerd.pkl"))
inference(ranker_ebnerd, "ebnerd", split="val",
          out_path=Path("data/submissions/ebnerd_val_lgbm.txt"))
print("✓ EB-NeRD LightGBM training + val inference done")

# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 10: Full evaluation — MIND + EB-NeRD (~5 min ea) ║
# ╚══════════════════════════════════════════════════════════╝
from src.evaluation.evaluate import evaluate_submission
import json

print("=" * 60)
print("Evaluating MIND / val / lgbm")
results_eval_mind = evaluate_submission("mind", "lgbm", "val")
print("\nMIND Eval Results:")
print(json.dumps(results_eval_mind, indent=2))

print("\n" + "=" * 60)
print("Evaluating EB-NeRD / val / lgbm")
results_eval_ebnerd = evaluate_submission("ebnerd", "lgbm", "val")
print("\nEB-NeRD Eval Results:")
print(json.dumps(results_eval_ebnerd, indent=2))

# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 11: NRMS baseline — MIND (~45 min on T4)         ║
# ╠══════════════════════════════════════════════════════════╣
# ║  Run in a SEPARATE NOTEBOOK to avoid OOM               ║
# ╚══════════════════════════════════════════════════════════╝
from src.models.train_nrms import train_nrms

print("=" * 60)
print("Training NRMS baseline — MIND (3 epochs)")
# Use smaller training set to fit in time/RAM
results_nrms_mind = train_nrms(
    dataset="mind",
    epochs=3,
    max_train_rows=150_000,  # reduce if OOM
    max_val_rows=10_000,
)
print("\nNRMS Results per epoch:")
for r in results_nrms_mind:
    print(f"  Epoch {r['epoch']}: AUC={r['AUC']:.4f} MRR={r['MRR']:.4f} nDCG@5={r['nDCG@5']:.4f}")

# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 12: NRMS baseline — EB-NeRD (~45 min on T4)      ║
# ╚══════════════════════════════════════════════════════════╝
from src.models.train_nrms import train_nrms

print("Training NRMS baseline — EB-NeRD (3 epochs)")
results_nrms_ebnerd = train_nrms(
    dataset="ebnerd",
    epochs=3,
    max_train_rows=150_000,
    max_val_rows=10_000,
)
print("\nNRMS Results per epoch:")
for r in results_nrms_ebnerd:
    print(f"  Epoch {r['epoch']}: AUC={r['AUC']:.4f} MRR={r['MRR']:.4f} nDCG@5={r['nDCG@5']:.4f}")

# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 13: Ablation Study (~30 min)                      ║
# ╚══════════════════════════════════════════════════════════╝
from src.ranking.ablation import run_ablation
import json

print("=" * 60)
print("Ablation Study — MIND")
ablation_mind = run_ablation("mind", max_val_rows=20_000)
print("\nAblation Results — MIND:")
for variant, res in ablation_mind.items():
    print(f"  {variant}: AUC={res['AUC']['mean']:.4f} nDCG@5={res['nDCG@5']['mean']:.4f}")

print("\nAblation Study — EB-NeRD")
ablation_ebnerd = run_ablation("ebnerd", max_val_rows=20_000)
print("\nAblation Results — EB-NeRD:")
for variant, res in ablation_ebnerd.items():
    print(f"  {variant}: AUC={res['AUC']['mean']:.4f} nDCG@5={res['nDCG@5']['mean']:.4f}")

# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 14: Serving Benchmark (~2 min)                    ║
# ╚══════════════════════════════════════════════════════════╝
from src.evaluation.serving_benchmark import run_benchmark

print("=" * 60)
bench_mind = run_benchmark("mind", n_requests=500, k_candidates=50)
print("\nMIND Benchmark:")
for k, v in bench_mind.items():
    if k != "scale_10x":
        print(f"  {k}: {v}")

# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 15: Generate Codabench submissions (test split)   ║
# ╚══════════════════════════════════════════════════════════╝
from src.ranking.train_lgbm import inference, LGBMRanker
from pathlib import Path

# MIND test
print("Generating MIND test submission...")
ranker = LGBMRanker()
ranker.load(Path("data/models/lgbm_mind.pkl"))
inference(ranker, "mind", split="test")
print("✓ MIND test submission generated")

# EB-NeRD test
print("\nGenerating EB-NeRD test submission...")
ranker_eb = LGBMRanker()
ranker_eb.load(Path("data/models/lgbm_ebnerd.pkl"))
inference(ranker_eb, "ebnerd", split="test")
print("✓ EB-NeRD test submission generated")

# List all submission files
print("\nSubmission files:")
for f in sorted(Path("data/submissions").glob("*.zip")):
    print(f"  {f.name}  ({f.stat().st_size/1e6:.1f} MB)")

# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 16: Print ALL results summary (paste to chat)     ║
# ╚══════════════════════════════════════════════════════════╝
import json
from pathlib import Path

print("=" * 70)
print("COMPLETE RESULTS SUMMARY — PASTE THIS TO CHAT")
print("=" * 70)

for f in sorted(Path("data/results").glob("*.json")):
    print(f"\n{'=' * 50}")
    print(f"FILE: {f.name}")
    print('=' * 50)
    with open(f) as fp:
        print(json.dumps(json.load(fp), indent=2))
