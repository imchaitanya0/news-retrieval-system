# News Retrieval System — IRE Assignment 1

A hybrid neural news recommendation system built for the **MIND** and **EB-NeRD RecSys 2024** competitions.

---

## What We Built and Why

### Architecture Overview

```
Raw Data (MIND TSV / EB-NeRD Parquet)
        ↓
src/data/build_pipeline.py     ← Unified schema, temporal split
        ↓
data/processed/                ← articles_*.parquet, behaviors_*.parquet
        ↓
src/retrieval/
  bm25.py       ← Lexical retrieval (BM25 keyword matching)
  semantic.py   ← Neural retrieval (sentence-transformer embeddings + FAISS)
        ↓
src/submission/generate.py     ← GPU-accelerated hybrid scorer → Codabench ZIP
        ↓
data/submissions/*.zip         ← prediction.txt / predictions.txt
```

---

## Key Design Decisions

### 1. Why Two Retrievers (BM25 + Semantic)?

News recommendation has two very different kinds of queries:

| Signal | BM25 | Semantic |
|--------|------|----------|
| Keyword match (e.g. "Manchester United") | ✅ Exact | ❌ Misses if paraphrase |
| Topic similarity (sports → sports news) | ❌ Misses | ✅ Understands meaning |
| Cold-start users (no history) | ❌ No query | ❌ No vector |
| Speed | Very fast | Fast (GPU) |

Neither alone is sufficient. We combine them.

### 2. Why Direct Candidate Scoring (not Global Retrieval)?

The test impressions already contain **20–100 pre-selected candidate articles** from the platform. Our job is only to **rerank** these candidates.

Early versions retrieved top-200 from 120,000 articles globally, then filtered. The overlap with the 20 pre-selected candidates was nearly zero, so 90%+ of impressions fell back to random order → AUC ≈ 0.51.

**Fix:** We directly score each candidate in the impression:
- **Semantic:** `dot(user_embedding, candidate_embedding)` — GPU batch matmul across all impressions
- **BM25:** Global BM25 retrieval of top-500, check if each candidate is in that set
- **Popularity:** Frequency of article appearing as a candidate globally (cold-start signal)

### 3. Why GPU Batch Matrix Multiply?

For 2.3M MIND impressions × 20 candidates × 384-dim embeddings:

| Method | Time |
|--------|------|
| Python for loop, one-by-one | ~9 hours |
| Numpy CPU batch (10K × 120K) | ~30 minutes |
| GPU CUDA batch (10K × 120K) | ~10 minutes |

We pre-load all 120K article embeddings into a single CUDA tensor, then compute `torch.mm(user_vectors, embedding_matrix.T)` — one matrix multiply for 10,000 users at once.

### 4. Why `paraphrase-multilingual-MiniLM-L12-v2`?

- **Multilingual:** Works for both English (MIND) and Danish (EB-NeRD) without separate models
- **Small (384-dim):** Fast encoding, small FAISS index, fits in Kaggle RAM
- **Pre-trained:** Strong semantic understanding without any fine-tuning on our data

### 5. Why Not Fine-Tune the Embeddings?

Fine-tuning (e.g., NRMS, NAML) requires the training dataset (~50GB) and GPU training time. For the zero-shot baseline, pre-trained multilingual embeddings give competitive performance quickly. The LightGBM reranker (next phase) provides additional improvement using handcrafted features.

### 6. Score Fusion Weights

`score = 0.7 × semantic + 0.2 × bm25_overlap + 0.1 × popularity`

- **0.7 semantic:** Neural embeddings capture topic similarity reliably
- **0.2 BM25:** Keyword match prevents semantic drift for specific queries
- **0.1 popularity:** Tiebreaker for cold-start users with no history

### 7. Why Polars (not Pandas)?

Polars processes data in parallel using Rust under the hood. Reading 2.3M behavior rows takes ~2 seconds in Polars vs ~30 seconds in Pandas. This was critical given Kaggle's time and RAM constraints.

---

## Current Status

| Task | Status |
|------|--------|
| MIND data processing | ✅ Done |
| EB-NeRD data processing | ✅ Done |
| BM25 retrieval | ✅ Done |
| Semantic embeddings (FAISS) | ✅ Done |
| Hybrid scoring | ✅ Done |
| MIND baseline submission | ✅ Submitted (AUC: 0.5131 — random baseline) |
| EB-NeRD baseline submission | ✅ Submitted |
| Direct candidate scoring fix | ✅ Done (expected AUC: 0.60+) |
| GPU-accelerated generation | ✅ Done |
| LightGBM reranker | 🔵 In progress |
| Evaluation metrics (AUC/MRR/nDCG) | ⬜ Pending |
| Design note | ⬜ Pending |

---

## What Went Wrong and How We Fixed It

### Problem 1: `ValueError: cannot concat empty list`
**Cause:** `build_pipeline.py` had hardcoded folder names (`ebnerd_demo`, `ebnerd_small`) that didn't match the downloaded `ebnerd_testset` structure.
**Fix:** Replaced hardcoded paths with `rglob("behaviors.parquet")` to dynamically discover all data regardless of folder names.

### Problem 2: AUC = 0.51 (barely above random)
**Cause:** Global BM25/FAISS retrieval from 120K articles had near-zero overlap with the 20 pre-selected test candidates. Fell back to original (random) order.
**Fix:** Switched to **direct candidate scoring** — score each candidate in the impression directly using dot products, not global retrieval.

### Problem 3: 9–34 hour runtime estimates
**Cause 1:** Per-impression BM25 mini-index rebuild (2.3M × BM25 index construction = hours of CPU).
**Cause 2:** Re-reading the parquet file 1186 times inside the chunk loop.
**Fix:** Load parquet once → load embeddings into GPU tensor once → use `torch.mm()` for batch matrix multiply → BM25 as global overlap check (not per-impression index).

### Problem 4: EB-NeRD OOM crash
**Cause:** `behaviors.to_dicts()` on 6M rows = ~6GB of Python dicts.
**Fix:** Stream 10K rows at a time using `behaviors.slice(offset, chunk_size).to_dicts()`.

### Problem 5: Codabench validation failure
**Cause:** Wrong file name (`predictions.txt` vs `prediction.txt`) and wrong format (space-separated vs comma-separated ranks).
**Fix:** MIND → `prediction.txt`, EB-NeRD → `predictions.txt`. Format: `{imp_id} [{r1},{r2},...}]` with no spaces inside brackets.

---

## One-Command Reproduction (Kaggle)

```bash
# 1. Install
!pip install -q bm25s rank_bm25 faiss-gpu sentence-transformers polars

# 2. Download raw data
!mkdir -p data/raw/mind
!wget -q https://mind201910small.blob.core.windows.net/release/MINDlarge_test.zip
!unzip -q MINDlarge_test.zip -d data/raw/mind/ && rm MINDlarge_test.zip
!huggingface-cli download Ekstra-Bladet/ebnerd_testset --repo-type dataset --local-dir data/raw/ebnerd/

# 3. Build processed parquets
!python -m src.data.build_pipeline

# 4. Generate submission ZIPs (GPU-accelerated, ~10 minutes)
!python -m src.submission.generate --dataset mind --split test --strategy hybrid
!python -m src.submission.generate --dataset ebnerd --split test --strategy hybrid

# ZIPs are at data/submissions/mind_test_hybrid.zip and ebnerd_test_hybrid.zip
```

---

## Next: LightGBM Reranker

The LightGBM ranker will train on the training behaviors using features:
- BM25 score between user query and candidate
- Semantic similarity score
- Article recency (hours since publication)
- Article popularity (global click count)
- User history length (warm vs cold start)
- Category match between user history and candidate

Expected AUC improvement: `0.60 → 0.65+`
