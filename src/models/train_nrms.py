"""
NRMS Training Script (A2 Q3 — Baseline Reproduced)
====================================================
Trains and evaluates the NRMS baseline on MIND or EB-NeRD.

This script:
  1. Builds a vocabulary from article titles in the training set.
  2. Creates training samples (1 pos + 4 neg per impression).
  3. Trains NRMS for N epochs with negative sampling.
  4. Evaluates on val split with AUC, MRR, nDCG@5, nDCG@10.
  5. Saves model and results JSON.

Usage (Kaggle / Colab):
    python -m src.models.train_nrms --dataset mind --epochs 3
    python -m src.models.train_nrms --dataset ebnerd --epochs 3

Expected runtime on Kaggle T4:
    MIND (small train):  ~25-40 min/epoch
    EB-NeRD (small):     ~20-30 min/epoch

Inputs:
    data/processed/articles_{dataset}.parquet   — article id, title, subtitle
    data/processed/behaviors_{dataset}_train.parquet
    data/processed/behaviors_{dataset}_val.parquet

Outputs:
    data/models/nrms_{dataset}.pt          — saved model weights
    data/results/nrms_{dataset}_val.json   — val metrics (AUC, MRR, nDCG@5, nDCG@10)
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.models.nrms import NRMSModel

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("data/models")
RESULTS_DIR   = Path("data/results")

# ------------------------------------------------------------------ #
#  Hyper-parameters                                                   #
# ------------------------------------------------------------------ #
MAX_TITLE_LEN  = 30   # words per title
MAX_HISTORY    = 50   # recent clicked articles per user
WORD_EMB_DIM   = 300  # word embedding dimension
D_MODEL        = 200  # attention output dimension
N_HEADS        = 4    # attention heads
DROPOUT        = 0.2
NEG_SAMPLES    = 4    # negatives per positive at training
BATCH_SIZE     = 32
LEARNING_RATE  = 1e-3
WEIGHT_DECAY   = 1e-5


# ------------------------------------------------------------------ #
#  Vocabulary                                                         #
# ------------------------------------------------------------------ #

def build_vocab(
    articles: pl.DataFrame,
    min_freq: int = 2,
    max_vocab: int = 50_000,
) -> dict:
    """
    Build word → index vocabulary from article titles.
    Returns dict: word → int index (1-indexed, 0 reserved for PAD).
    """
    counter = Counter()
    for row in articles.select(["title"]).iter_rows():
        title = row[0] or ""
        counter.update(title.lower().split())

    # Sort by frequency, keep top max_vocab
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in counter.most_common(max_vocab - 2):
        if freq < min_freq:
            break
        vocab[word] = len(vocab)

    print(f"  Vocabulary: {len(vocab):,} tokens")
    return vocab


def tokenize_title(title: str, vocab: dict, max_len: int = MAX_TITLE_LEN) -> list:
    """Tokenize title to fixed-length int list (padded / truncated)."""
    tokens = (title or "").lower().split()[:max_len]
    ids    = [vocab.get(w, 1) for w in tokens]  # 1 = UNK
    ids   += [0] * (max_len - len(ids))          # 0 = PAD
    return ids


# ------------------------------------------------------------------ #
#  Dataset                                                            #
# ------------------------------------------------------------------ #

class NRMSDataset(Dataset):
    """
    Training dataset. For each impression, yields:
      (hist_ids, hist_mask, cand_ids, label_idx)
    where label_idx = 0 (positive is always first among n_cands).

    Negative sampling: 1 positive + NEG_SAMPLES random negatives.
    """

    def __init__(
        self,
        behaviors: pl.DataFrame,
        article_title_ids: dict,     # article_id -> list[int] of token ids
        neg_samples: int = NEG_SAMPLES,
        max_history: int = MAX_HISTORY,
        max_title_len: int = MAX_TITLE_LEN,
    ):
        self.article_title_ids = article_title_ids
        self.neg_samples  = neg_samples
        self.max_history  = max_history
        self.max_title_len = max_title_len
        self._pad_title   = [0] * max_title_len

        # Build samples: only impressions that have at least one click
        self.samples = []
        for row in behaviors.iter_rows(named=True):
            impressions = row.get("impressions") or []
            labels      = row.get("labels")      or []
            history     = row.get("history")     or []

            positives = [aid for aid, lbl in zip(impressions, labels) if lbl == 1]
            negatives = [aid for aid, lbl in zip(impressions, labels) if lbl == 0]

            for pos in positives:
                self.samples.append((history, pos, negatives))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        history, pos_id, neg_pool = self.samples[idx]

        # --- History ---
        hist_recent = (history or [])[-self.max_history:]
        hist_titles = [
            self.article_title_ids.get(aid, self._pad_title)
            for aid in hist_recent
        ]
        # Pad history to max_history
        hist_pad   = [self._pad_title] * (self.max_history - len(hist_titles))
        hist_ids   = torch.tensor(hist_titles + hist_pad, dtype=torch.long)  # (H, T)
        hist_mask  = torch.zeros(self.max_history, dtype=torch.long)
        hist_mask[:len(hist_titles)] = 1

        # --- Candidates: pos + sampled negatives ---
        neg_sample = random.sample(neg_pool, min(self.neg_samples, len(neg_pool)))
        # Pad if not enough negatives
        while len(neg_sample) < self.neg_samples:
            neg_sample.append(neg_pool[0] if neg_pool else pos_id)

        candidates = [pos_id] + neg_sample  # pos first → label index = 0
        cand_titles = [
            self.article_title_ids.get(aid, self._pad_title)
            for aid in candidates
        ]
        cand_ids = torch.tensor(cand_titles, dtype=torch.long)  # (1+K, T)
        label    = torch.tensor(0, dtype=torch.long)             # pos is always at index 0

        return hist_ids, hist_mask, cand_ids, label


# ------------------------------------------------------------------ #
#  Evaluation                                                         #
# ------------------------------------------------------------------ #

def evaluate_nrms(
    model: NRMSModel,
    behaviors: pl.DataFrame,
    article_title_ids: dict,
    device: torch.device,
    max_history: int = MAX_HISTORY,
    max_title_len: int = MAX_TITLE_LEN,
    batch_size: int = 64,
) -> dict:
    """
    Evaluate NRMS on a val split.
    Returns dict: AUC, MRR, nDCG@5, nDCG@10 (mean over impressions).
    """
    from sklearn.metrics import roc_auc_score

    model.eval()
    pad_title = [0] * max_title_len

    auc_vals, mrr_vals, ndcg5_vals, ndcg10_vals = [], [], [], []

    with torch.no_grad():
        for row in tqdm(behaviors.iter_rows(named=True), total=len(behaviors), desc="Eval"):
            impressions = row.get("impressions") or []
            labels      = row.get("labels")      or []
            history     = row.get("history")     or []

            if not impressions or len(set(labels)) < 2:
                continue

            # Build history
            hist_recent  = (history or [])[-max_history:]
            hist_titles  = [article_title_ids.get(aid, pad_title) for aid in hist_recent]
            hist_pad_    = [pad_title] * (max_history - len(hist_titles))
            hist_ids     = torch.tensor(hist_titles + hist_pad_, dtype=torch.long).unsqueeze(0).to(device)
            hist_mask_   = torch.zeros(1, max_history, dtype=torch.long, device=device)
            hist_mask_[0, :len(hist_titles)] = 1

            # Build candidates in batches of 32 (for memory)
            all_scores = []
            for c_start in range(0, len(impressions), 32):
                c_chunk = impressions[c_start:c_start + 32]
                cand_titles = [article_title_ids.get(aid, pad_title) for aid in c_chunk]
                cand_ids_t  = torch.tensor(cand_titles, dtype=torch.long).unsqueeze(0).to(device)  # (1,K,T)
                # Expand history to batch
                h_ids  = hist_ids.expand(1, -1, -1)
                h_mask = hist_mask_.expand(1, -1)
                scores = model(h_ids, h_mask, cand_ids_t).squeeze(0).cpu().numpy()
                all_scores.extend(scores.tolist())

            scores_arr = np.array(all_scores)
            labels_arr = np.array([int(l) for l in labels])

            # AUC
            try:
                auc = roc_auc_score(labels_arr, scores_arr)
                auc_vals.append(auc)
            except Exception:
                pass

            # MRR
            order = np.argsort(scores_arr)[::-1]
            for rank, idx in enumerate(order, 1):
                if labels_arr[idx] == 1:
                    mrr_vals.append(1.0 / rank)
                    break
            else:
                mrr_vals.append(0.0)

            # nDCG@5 and @10
            for k, store in [(5, ndcg5_vals), (10, ndcg10_vals)]:
                top_k  = order[:k]
                dcg    = sum(labels_arr[i] / np.log2(r + 2) for r, i in enumerate(top_k))
                ideal  = sorted(labels_arr, reverse=True)[:k]
                idcg   = sum(v / np.log2(r + 2) for r, v in enumerate(ideal))
                store.append(dcg / idcg if idcg > 0 else 0.0)

    results = {
        "AUC":     float(np.mean(auc_vals))   if auc_vals   else 0.0,
        "MRR":     float(np.mean(mrr_vals))   if mrr_vals   else 0.0,
        "nDCG@5":  float(np.mean(ndcg5_vals)) if ndcg5_vals else 0.0,
        "nDCG@10": float(np.mean(ndcg10_vals)) if ndcg10_vals else 0.0,
        "n_impressions": len(auc_vals),
    }
    return results


# ------------------------------------------------------------------ #
#  Training loop                                                      #
# ------------------------------------------------------------------ #

def train_nrms(
    dataset: str,
    epochs: int = 3,
    max_train_rows: int = 200_000,
    max_val_rows: int = 20_000,
):
    """
    Full NRMS training pipeline.

    Parameters
    ----------
    dataset       : "mind" or "ebnerd"
    epochs        : number of training epochs
    max_train_rows: cap training impressions for memory/speed
    max_val_rows  : cap val impressions for evaluation speed
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[NRMS] Training on {dataset} | device: {device}")

    # --- Load data ---
    articles_path = PROCESSED_DIR / f"articles_{dataset}.parquet"
    train_path    = PROCESSED_DIR / f"behaviors_{dataset}_train.parquet"
    val_path      = PROCESSED_DIR / f"behaviors_{dataset}_val.parquet"

    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}\nRun build_pipeline.py first.")

    articles = pl.read_parquet(articles_path)
    behaviors_train = pl.read_parquet(train_path)
    if len(behaviors_train) > max_train_rows:
        behaviors_train = behaviors_train.sample(max_train_rows, seed=42)
    print(f"  Train impressions: {len(behaviors_train):,}")

    behaviors_val = None
    if val_path.exists():
        behaviors_val = pl.read_parquet(val_path)
        behaviors_val = behaviors_val.filter(pl.col("labels").list.sum() > 0).head(max_val_rows)
        print(f"  Val impressions:   {len(behaviors_val):,}")

    # --- Vocabulary ---
    print("  Building vocabulary...")
    vocab = build_vocab(articles)

    # --- Pre-tokenize all articles ---
    print("  Pre-tokenizing article titles...")
    article_title_ids = {}
    for row in articles.select(["article_id", "title"]).iter_rows(named=True):
        article_title_ids[row["article_id"]] = tokenize_title(row["title"] or "", vocab)

    # --- Dataset & DataLoader ---
    print("  Building training dataset (negative sampling)...")
    train_ds = NRMSDataset(behaviors_train, article_title_ids, neg_samples=NEG_SAMPLES)
    print(f"  Training samples: {len(train_ds):,}")
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    # --- Model ---
    model = NRMSModel(
        vocab_size    = len(vocab),
        word_emb_dim  = WORD_EMB_DIM,
        d_model       = D_MODEL,
        n_heads       = N_HEADS,
        max_title_len = MAX_TITLE_LEN,
        max_history   = MAX_HISTORY,
        dropout       = DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  NRMS parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
    )
    criterion = nn.CrossEntropyLoss()

    # --- Training loop ---
    best_ndcg = 0.0
    all_epoch_results = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for hist_ids, hist_mask, cand_ids, labels in tqdm(
            train_loader, desc=f"Epoch {epoch}/{epochs}"
        ):
            hist_ids  = hist_ids.to(device)
            hist_mask = hist_mask.to(device)
            cand_ids  = cand_ids.to(device)
            labels    = labels.to(device)

            logits = model(hist_ids, hist_mask, cand_ids)   # (B, 1+K)
            loss   = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches  += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"  Epoch {epoch} | Loss: {avg_loss:.4f}")

        # --- Validation ---
        if behaviors_val is not None:
            metrics = evaluate_nrms(model, behaviors_val, article_title_ids, device)
            print(
                f"  Val → AUC: {metrics['AUC']:.4f} | MRR: {metrics['MRR']:.4f} "
                f"| nDCG@5: {metrics['nDCG@5']:.4f} | nDCG@10: {metrics['nDCG@10']:.4f}"
            )
            metrics["epoch"] = epoch
            metrics["loss"]  = avg_loss
            all_epoch_results.append(metrics)

            if metrics["nDCG@5"] > best_ndcg:
                best_ndcg = metrics["nDCG@5"]
                MODELS_DIR.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), MODELS_DIR / f"nrms_{dataset}.pt")
                # Also save vocab for inference
                import pickle
                with open(MODELS_DIR / f"nrms_{dataset}_vocab.pkl", "wb") as f:
                    pickle.dump(vocab, f)
                print(f"  ★ New best nDCG@5={best_ndcg:.4f} → model saved")

    # --- Save results ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"nrms_{dataset}_val.json"
    with open(out_path, "w") as f:
        json.dump(all_epoch_results, f, indent=2)
    print(f"\n[NRMS] Done. Best nDCG@5 = {best_ndcg:.4f}")
    print(f"Results saved → {out_path}")
    return all_epoch_results


# ------------------------------------------------------------------ #
#  CLI                                                                #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Train NRMS baseline (A2 Q3)")
    parser.add_argument("--dataset",        choices=["mind", "ebnerd"], required=True)
    parser.add_argument("--epochs",         type=int, default=3)
    parser.add_argument("--max-train-rows", type=int, default=200_000)
    parser.add_argument("--max-val-rows",   type=int, default=20_000)
    args = parser.parse_args()

    train_nrms(
        dataset       = args.dataset,
        epochs        = args.epochs,
        max_train_rows = args.max_train_rows,
        max_val_rows  = args.max_val_rows,
    )


if __name__ == "__main__":
    main()
