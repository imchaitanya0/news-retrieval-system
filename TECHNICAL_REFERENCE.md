# News Retrieval System — Complete Technical Reference
## CS4.406: Information Retrieval & Extraction | Assignment 1 & 2

---

## 1. Repository Structure & File Map

```
news-retrieval-system/
├── src/
│   ├── data/
│   │   ├── build_pipeline.py    — ETL: raw → unified parquet (A1 Q1)
│   │   ├── download.py          — Dataset download helpers
│   │   └── inspect.py           — Data inspection utilities
│   ├── retrieval/
│   │   ├── bm25.py              — BM25 inverted index (A1 Q2)
│   │   ├── semantic.py          — Sentence-transformer + FAISS (A1 Q3)
│   │   └── hybrid.py            — Weighted score fusion
│   ├── features/
│   │   └── feature_store.py     — 8-feature GPU-batched builder (A2 Q1)
│   ├── ranking/
│   │   ├── train_lgbm.py        — LightGBM LambdaMART (A2 Q2)
│   │   └── ablation.py          — 4-variant ablation study (A2 Q3)
│   ├── models/
│   │   ├── nrms.py              — NRMS PyTorch architecture (A2 Q3)
│   │   └── train_nrms.py        — NRMS training script (A2 Q3)
│   ├── evaluation/
│   │   ├── metrics.py           — AUC/MRR/nDCG/diversity/novelty/CI (A1 Q4)
│   │   ├── evaluate.py          — Full eval harness + cold/warm + head/tail (A1 Q4)
│   │   └── serving_benchmark.py — Latency/memory/cost benchmark (A2 Q4)
│   └── submission/
│       └── generate.py          — Codabench submission file builder (A1 Q5)
├── tests/
│   ├── test_anti_gaming.py      — Temporal leakage + schema + boundary tests (Q9)
│   ├── test_bm25.py             — BM25 unit tests
│   └── test_semantic.py         — Semantic retrieval unit tests
├── screenshots/
│   ├── MIND_submission.png      — MIND Codabench leaderboard
│   └── EbNERD.png               — EB-NeRD Codabench leaderboard
├── design_note.pdf              — A1 design note (≤4 pages)
├── requirements.txt
└── README.md
```

---

## 2. Data Pipeline

### Input Files
| Dataset | File | Description |
|---------|------|-------------|
| MIND    | `MINDlarge_train/behaviors.tsv` | 15M impression logs (TSV) |
| MIND    | `MINDlarge_train/news.tsv`      | 130K article metadata |
| EB-NeRD | `ebnerd_large/train/behaviors.parquet` | Behavior logs |
| EB-NeRD | `articles.parquet`                     | 120K+ articles |

### Unified Schema (output parquets)
**`data/processed/articles_{dataset}.parquet`**
| Column | Type | Description |
|--------|------|-------------|
| dataset | str | "mind" or "ebnerd" |
| article_id | str/int | Unique article identifier |
| title | str | Article headline |
| subtitle | str | Subtitle/abstract |
| body | str | Full body text |
| category | str | News category (e.g., "sports") |
| subcategory | str | Sub-category |
| published_time | datetime | Publication timestamp (used for freshness) |
| popularity | float | Global click count |
| entities | list[str] | Named entities in title |
| abstract_entities | list[str] | Named entities in abstract |

**`data/processed/behaviors_{dataset}_{split}.parquet`** (split ∈ {train, val, test})
| Column | Type | Description |
|--------|------|-------------|
| dataset | str | "mind" or "ebnerd" |
| impression_id | int | Unique impression identifier |
| user_id | str | Anonymized user ID |
| impression_time | datetime | When the impression was shown |
| history | list[str] | Ordered list of previously clicked article IDs (oldest → newest) |
| impressions | list[str] | Candidate article IDs shown in this impression |
| labels | list[int] | 1=clicked, 0=not-clicked (empty in test split) |

---

## 3. Models & Components

### 3.1 BM25 Retriever (`src/retrieval/bm25.py`)

**What it does:** Builds a full-corpus inverted index over article `title + subtitle`. Given a user's click history, concatenates the titles of the most-recently-clicked articles into a query string, retrieves top-K candidates.

**Algorithm:** BM25 Okapi (k1=1.5, b=0.75)

**Input:**
- Articles DataFrame with `article_id`, `title`, `subtitle`
- User click history (list of article IDs)

**Output:** Ranked list of article IDs (top-K)

**Key functions:**
- `BM25Retriever.build(articles)` — builds and caches the index
- `BM25Retriever.retrieve(query, k=100)` — returns top-K article IDs
- `evaluate_bm25(dataset, split, k_values=[50,100,200])` — computes Recall@K

**CLI:**
```bash
python -m src.retrieval.bm25 --dataset mind --split val --k 50 100 200
```

**Evaluation metric:** Recall@K — fraction of ground-truth clicked articles that appear in the top-K retrieved set.

---

### 3.2 Semantic Retriever (`src/retrieval/semantic.py`)

**What it does:** Encodes articles with `sentence-transformers`, stores in FAISS index, retrieves top-K by cosine similarity to user vector (mean of clicked article embeddings).

**Model:** `paraphrase-multilingual-MiniLM-L12-v2`
- Languages: multilingual (covers both English MIND and Danish EB-NeRD)
- Embedding dim: **384**
- Model size: ~471 MB

**FAISS index:** `FlatIP` (exact inner product search = cosine similarity for L2-normalized vectors)

**User representation:** Mean pool of the last 10 clicked article embeddings → 384-dim vector, L2-normalized.

**Dimensions:**
```
Article embedding: (N_articles, 384)   e.g., (130379, 384) for MIND
User vector:       (384,)
FAISS index size:  N_articles × 384 × 4 bytes = ~200 MB for MIND
```

**Input:**
- Articles DataFrame
- User click history (list of article IDs)

**Output:** Ranked list of top-K article IDs

**CLI:**
```bash
python -m src.retrieval.semantic --dataset mind --split val --k 50 100 200
```

---

### 3.3 Feature Store (`src/features/feature_store.py`)

**What it does:** Computes 8 features for every (user, candidate article) pair in a batch.

#### Feature Descriptions

| # | Name | Formula | Why it matters |
|---|------|---------|----------------|
| 1 | `sem_score` | `dot(user_vec_decayed, cand_embedding)` | Topical similarity between user's current interest and candidate |
| 2 | `lex_overlap` | `|user_tokens ∩ cand_tokens| / |user_tokens ∪ cand_tokens|` | Exact keyword match (Messi → Messi); catches specifics semantic models miss |
| 3 | `log_popularity` | `log(1 + click_count_from_train)` | Cold-start fallback; popular articles are safe bets for new users |
| 4 | `hist_len` | `min(len(history), 50)` | Distinguishes cold users (0) from power users (50); model weights features accordingly |
| 5 | `cat_match` | `1 if cand_category == user_top_category` | User's dominant interest category alignment |
| 6 | `inv_position` | `1 / editorial_position` | Captures editorial placement bias (position 1 = most visible) |
| 7 | `freshness` | `exp(-hours_since_publish / 24)` | News quality degrades rapidly; day-old article gets 0.37, week-old gets <0.01 |
| 8 | `recency_sem_score` | `dot(user_vec_last_3_clicks, cand_embedding)` | Short-term interest shift; captures sudden topic change (e.g., user just started reading tech after sports) |

#### Recency-Weighted User Vector (Feature 1)
```
Standard mean pool:  user_vec = (1/n) Σ embedding_i          [outdated]
Recency-weighted:    user_vec = Σ w_i × embedding_i          [ours]
  where w_i = exp(-0.15 × (n-1-i)) / Σ exp(-0.15 × (n-1-j))
  (i=0 oldest, i=n-1 newest; decay_rate=0.15)
```

#### GPU Batch Matrix Multiplication
```python
# For chunk of B=1500 users, N_articles=130K:
sc_all = torch.mm(user_vectors_batch,    # (1500, 384)
                  article_embeddings.T)  # (384, 130379)
# → sc_all: (1500, 130379) — all dot products in one BLAS call
# Immediately extract only the K candidate scores per impression
# and delete sc_all to free 780MB of VRAM
```

**Input:**
- behaviors: pl.DataFrame with history, impressions, labels, impression_time
- articles: pl.DataFrame with article_id, category, published_time
- embedding_map: dict {article_id → np.ndarray(384,)}
- article_text_map: dict {article_id → "title subtitle" string}
- popularity_map: dict {article_id → int click count}
- article_publish_map: dict {article_id → datetime}

**Output:** `(X, y, groups, imp_ids, art_ids)` where X has shape `(N_pairs, 8)`.

---

### 3.4 LightGBM Ranker (`src/ranking/train_lgbm.py`)

**Algorithm:** LambdaMART (gradient boosted trees optimized for NDCG ranking)

**Hyperparameters:**
| Parameter | Value | Reason |
|-----------|-------|--------|
| objective | lambdarank | Directly optimizes ranking quality |
| metric | ndcg | Evaluate at positions 5 and 10 |
| learning_rate | 0.05 | Conservative to avoid overfitting |
| num_leaves | 63 | Moderate complexity |
| n_estimators | 300 (max) | Early stopping with patience=30 |
| feature_fraction | 0.8 | Feature bagging for regularization |
| bagging_fraction | 0.8 | Row bagging |
| min_data_in_leaf | 20 | Prevents overfitting to tiny groups |

**Training objective:** Maximize nDCG@5 on the validation set.

**Label format:** Binary (0 = not clicked, 1 = clicked). `label_gain=[0, 1]`.

**Input features:** X of shape (N_pairs, 8) from feature_store.py

**Output:** Per-candidate relevance scores → argsort → ranked article list → Codabench submission file.

**CLI:**
```bash
python -m src.ranking.train_lgbm --dataset mind    # trains + infers on test
python -m src.ranking.train_lgbm --dataset ebnerd  # trains + infers on test
```

---

### 3.5 NRMS Baseline (`src/models/nrms.py` + `src/models/train_nrms.py`)

**Paper:** "Neural News Recommendation with Multi-head Self-Attention" (Wu et al., EMNLP 2019)

**Architecture:**

```
News Encoder (per article):
  1. Word Embedding:   title_ids (30 tokens) → word_embeddings (30, 300)
  2. Linear projection: (30, 300) → (30, 200)
  3. LayerNorm
  4. Multi-head Self-Attention: 4 heads × 50-dim = 200-dim
     Q=K=V: (30, 200) → (30, 200)  [attend each word to all others]
  5. Residual connection: input + attn_out
  6. Additive Attention Pool: W·tanh(q) → weight → weighted sum
     (30, 200) → (200,)   [one vector per article]

User Encoder:
  1. Encode all history articles via News Encoder → (50, 200)
  2. Multi-head Self-Attention over history news vectors
     4 heads × 50-dim: (50, 200) → (50, 200)  [attend each article to all others]
  3. Additive Attention Pool: (50, 200) → (200,)  [one user vector]

Scoring:
  score = dot(user_vector, candidate_news_vector)
  During training: softmax over [1 pos + 4 neg] candidates
  During inference: independent score for each candidate
```

**Dimensions:**
| Component | Dimension |
|-----------|-----------|
| Vocabulary | ~50K words |
| Word embedding | 300-dim |
| Attention output | 200-dim |
| News vector | 200-dim |
| User vector | 200-dim |
| Max title tokens | 30 |
| Max history articles | 50 |
| Negative samples/positive | 4 |
| Total parameters | ~17M |

**Training:**
- Loss: Cross-entropy (softmax over 1+4 candidates)
- Optimizer: Adam (lr=1e-3, weight_decay=1e-5)
- Scheduler: OneCycleLR (peaks at epoch 1, decays to 0)
- Gradient clipping: 1.0
- Batch size: 32

**CLI:**
```bash
python -m src.models.train_nrms --dataset mind --epochs 3
python -m src.models.train_nrms --dataset ebnerd --epochs 3
```

**Expected runtime on Kaggle T4:** ~30-45 minutes per epoch

---

### 3.6 Ablation Study (`src/ranking/ablation.py`)

Trains 4 LightGBM variants with progressively more features. Each is trained on the same data (200K train rows, 30K val rows), evaluated with 95% bootstrap CI.

| Variant | Features Used | Purpose |
|---------|---------------|---------|
| A | sem_score only | Pure semantic signal baseline |
| B | sem_score, lex_overlap | + lexical matching |
| C | Original 6 (sem, lex, pop, hist, cat, pos) | Pre-A2 model |
| D | All 8 (+ freshness, recency_sem_score) | Full A2 model |

**CLI:**
```bash
python -m src.ranking.ablation --dataset mind
python -m src.ranking.ablation --dataset ebnerd
```

---

### 3.7 Evaluation Harness (`src/evaluation/metrics.py` + `evaluate.py`)

#### Core Metrics
| Metric | Formula | Interpretation |
|--------|---------|----------------|
| AUC | Area under ROC curve | Pairwise ranking quality (0.5=random, 1.0=perfect) |
| MRR | 1/rank_of_first_clicked | How early does the first relevant item appear? |
| nDCG@5 | DCG@5 / IDCG@5 | Quality of top-5 ranked items (position-discounted) |
| nDCG@10 | DCG@10 / IDCG@10 | Quality of top-10 ranked items |

#### Beyond-Accuracy Metrics
| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Diversity | unique_categories / k | Variety of topics in top-K recommendations |
| Novelty | mean(-log2(pop_i/N)) | How unpopular are the recommended items? Higher = more novel |
| Coverage | |recommended| / |all_articles| | Fraction of catalog ever recommended |

#### Slices
| Slice | Definition | Purpose |
|-------|-----------|---------|
| Cold users | history_len ≤ 5 | No behavioral data; popularity-based fallback |
| Warm users | history_len > 5 | Rich behavioral data; semantic features useful |
| Head articles | top 20% by appearance count | Popular articles; easy to rank correctly |
| Tail articles | bottom 80% by appearance count | Niche articles; harder to rank; key for novelty |

#### Bootstrap CI
1000 resamples, 95% confidence interval (2.5th–97.5th percentile).

**CLI:**
```bash
python -m src.evaluation.evaluate --dataset mind --strategy lgbm --split val
python -m src.evaluation.evaluate --dataset ebnerd --strategy lgbm --split val
```

---

### 3.8 Serving Benchmark (`src/evaluation/serving_benchmark.py`)

**What it measures:**
1. FAISS index peak memory (MB) via `tracemalloc`
2. Embedding matrix size on disk and in memory
3. LightGBM model file size
4. p50/p95/p99 latency for 1000 random user requests
5. Max QPS at p99 SLA
6. Cost per 1000 queries (assuming T4-equivalent cloud)
7. 10× scale analysis (what breaks and how to fix it)

**CLI:**
```bash
python -m src.evaluation.serving_benchmark --dataset mind
python -m src.evaluation.serving_benchmark --dataset ebnerd
```

---

## 4. Evaluation Commands (Run on Kaggle)

### A1 Requirements
```bash
# BM25 Recall@K (Q2)
python -m src.retrieval.bm25 --dataset mind  --split val --k 50 100 200
python -m src.retrieval.bm25 --dataset ebnerd --split val --k 50 100 200

# Semantic Recall@K (Q3)
python -m src.retrieval.semantic --dataset mind  --split val --k 50 100 200
python -m src.retrieval.semantic --dataset ebnerd --split val --k 50 100 200

# Full eval harness on LightGBM val predictions (Q4)
python -m src.evaluation.evaluate --dataset mind  --strategy lgbm --split val
python -m src.evaluation.evaluate --dataset ebnerd --strategy lgbm --split val
```

### A2 Requirements
```bash
# NRMS baseline (Q3)
python -m src.models.train_nrms --dataset mind --epochs 3
python -m src.models.train_nrms --dataset ebnerd --epochs 3

# Retrain LightGBM with 8 features (Q2)
python -m src.ranking.train_lgbm --dataset mind
python -m src.ranking.train_lgbm --dataset ebnerd

# Ablation study (Q3)
python -m src.ranking.ablation --dataset mind
python -m src.ranking.ablation --dataset ebnerd

# Serving benchmark (Q4)
python -m src.evaluation.serving_benchmark --dataset mind
python -m src.evaluation.serving_benchmark --dataset ebnerd
```

---

## 5. Why Each Choice Was Made (For Viva)

### Why Polars instead of Pandas?
Polars uses Rust under the hood and processes data in parallel. Reading 15M MIND behavior rows takes ~2s in Polars vs ~30s in Pandas, and uses ~3× less memory. Critical for Kaggle's 16GB RAM limit.

### Why mean-pool → recency-weighted mean-pool?
News reading interests shift rapidly. A user who read sports for a month but just clicked 3 tech articles in the last hour is interested in tech right now. Flat mean-pool would still show them sports. Exponential decay (decay_rate=0.15) gives the most-recent click ~1.5× the weight of a click 5 positions earlier.

### Why lexical overlap instead of global BM25 query?
For L1 retrieval (candidate generation), BM25 is standard. But we already have the candidates from the platform. Computing a full BM25 query over 130K articles for each of 6M users takes >2 hours on CPU. Since we only need to score 50 pre-provided candidates, Jaccard token overlap achieves exactly the same keyword-match signal in microseconds (pure set operations).

### Why freshness feature?
News has a unique "temporal decay" property. A story about a sports match result from last week is irrelevant today. `exp(-hours/24)` means: 0 hours old → score 1.0; 24 hours old → score 0.37; 7 days old → score 0.0009. The LightGBM model learns to penalize stale content.

### Why LightGBM over neural ranker for L2?
- LightGBM is interpretable (feature importances prove what signals matter)
- Trains in minutes on CPU, not hours
- Achieves competitive accuracy with hand-crafted features on small datasets
- NRMS (our separate neural baseline) takes hours to train — having both satisfies the assignment requirement to reproduce AND improve

### Why FAISS FlatIP for ANN?
At 130K articles, brute-force exact search (FlatIP) takes <5ms per query. The overhead of approximation (IVF/HNSW) isn't justified below ~1M articles. At 10× scale, we would switch to IndexIVFFlat.

### Temporal split (never random for interaction data)
Random splitting would mean training on future clicks and evaluating on past clicks — data leakage. Instead: `train < val < test` in wall-clock time. The dataset provides this split natively (MIND: 6 weeks train + 1 week dev; EB-NeRD: similar).

---

## 6. Anti-Leakage Guarantees

Three independent mechanisms prevent future-click leakage:

1. **Temporal split:** `train.max_time ≤ val.min_time ≤ test.min_time` (asserted in `TestSplitIntegrity`)
2. **Strong leakage test:** Joins history article `published_time` with `impression_time`, asserts no history article was published after the impression (asserted in `TestNoFutureLeakage.test_no_future_article_in_history`)
3. **Test labels empty:** Test split has no positive labels — features cannot peek at click outcomes (asserted in `TestBehaviourWindowBoundary`)

---

## 7. Troubleshooting & Bug Fixes History (For Exam Prep)

Throughout the development and Kaggle deployment of this pipeline, we faced several technical roadblocks. Understanding *why* they happened and *how* they were fixed is critical for exam preparation.

### 7.1 FAISS Semantic Retrieval Too Slow (2 hours)
- **Error/Symptom:** Running Semantic Recall@K took over 2 hours on Kaggle T4 GPU.
- **Root Cause:** In `src/retrieval/semantic.py`, the code was unconditionally instantiating `faiss.IndexFlatIP(dim)` which runs exclusively on the CPU. The cell-level `os.environ['FAISS_GPU']` patch in the notebook was not propagating to the subprocess.
- **Solution:** Edited `semantic.py` to dynamically check for `faiss.StandardGpuResources` and `faiss.GpuIndexFlatIP`. If available, it loads the embeddings onto the GPU VRAM.
- **Numbers:** 
  - CPU FAISS: ~2 hours for 376K validation queries.
  - GPU FAISS: ~2 minutes. 
  - Index size: 130,379 vectors × 384 dimensions × 4 bytes = ~200 MB (easily fits in 15GB T4 VRAM).

### 7.2 EB-NeRD Pipeline: `DuplicateError: column 'category' is duplicate`
- **Error:** Polars threw `DuplicateError` when running `build_pipeline.py`.
- **Root Cause:** EB-NeRD's `articles.parquet` had an existing `category` column (from upstream updates), and our script tried to unconditionally rename `category_str` to `category`. Polars forbids duplicate column names.
- **Solution:** Updated the rename logic: if `category` already exists, we safely drop `category_str` instead of renaming it. 

### 7.3 EB-NeRD Pipeline: `InvalidOperationError: cannot cast List type (inner: 'Int16', to: 'String')`
- **Error:** Occurred during subcategory normalization in `build_pipeline.py`.
- **Root Cause:** EB-NeRD's `subcategory` column was stored as a list of integers (`List(Int16)`). We attempted to cast it directly to `Utf8` (String). Polars cannot cast a list of integers directly into a single string.
- **Solution:** Added a dtype check `if "List" in str(df["subcategory"].dtype):`. If it's a list, we first cast it to a list of strings `cast(pl.List(pl.Utf8))` and then use `.list.join(",")` to flatten it into a single comma-separated string.
- **Numbers:** 
  - Input: `[12, 45]` (Type: `List(Int16)`)
  - Output: `"12,45"` (Type: `Utf8`)

### 7.4 NumPy Array Indexing Error in Evaluation
- **Error:** `TypeError: only integer scalar arrays can be converted to a scalar index` in `metrics.py`.
- **Root Cause:** During evaluation, the ground truth labels array was being indexed using `i` where `i` was a numpy array (e.g., `labels[np.array([2])]`) instead of an integer (`labels[2]`). 
- **Solution:** Forced standard Python integer casting via `labels[int(i)]`.

### 7.5 Type Mismatch on `impression_id`
- **Error:** Impressions from the LightGBM prediction text file couldn't be joined with the validation Parquet file.
- **Root Cause:** The text file parser read `impression_id` as a string (`"12345"`), but the Parquet schema stored it as an integer (`12345`). Dictionary lookups failed silently resulting in zero metrics.
- **Solution:** Normalized all `impression_id` keys to integers using `int(imp_id)` when building the lookup maps in memory.

### 7.6 Serving Benchmark Signature Mismatch
- **Error:** `TypeError: SemanticRetriever.build() takes 2 positional arguments but 3 were given`
- **Root Cause:** The benchmark script was passing the entire DataFrame `articles` to `SemanticRetriever.build()`, but the updated signature expected `(embeddings: np.ndarray, article_ids: list)`.
- **Solution:** Updated the benchmark to call `load_or_compute_embeddings()` first, then pass the resulting raw `numpy` arrays directly to the builder.

### 7.7 FAISS GPU Index Serialization Bug
- **Error:** `RuntimeError: Error in virtual void faiss::Index::write_index(...)` when calling `faiss.write_index` on a GPU index.
- **Root Cause:** FAISS `GpuIndexFlatIP` cannot be written directly to disk via `write_index()`. Furthermore, loading an index with `faiss.read_index()` always yields a CPU index. When we used `faiss.index_cpu_to_gpu()`, it crashed because it doesn't natively support translating loaded CPU `IndexFlatIP` instances on some GPU architectures without the proper cloner.
- **Solution:**
  1. **Saving:** Modified `save()` to use `faiss.index_gpu_to_cpu()` before writing to disk. This costs a negligible ~0.2s VRAM-to-RAM transfer.
  2. **Loading:** Modified `load()` to bypass `index_cpu_to_gpu()`. Instead, we extract the raw vectors directly from the loaded CPU index using `cpu_index.reconstruct_n(0, n)` and explicitly reconstruct a new `GpuIndexFlatIP` object in memory. 
- **Numbers:** 
  - VRAM to RAM Transfer Size: `130,379 x 384 x 4 bytes` = ~200 MB.
  - Transfer Latency: ~0.2 seconds.

### 7.8 Feature Store PyTorch CUDA OOM
- **Error:** `torch.cuda.OutOfMemoryError: CUDA out of memory` during `feature_store.py` execution (Cell 18).
- **Root Cause:** 
  1. **Chunk Size:** The default chunk size tried to allocate ~1,500 queries simultaneously against 130K articles. `(1500 × 130379 × 4 bytes) ≈ 780 MB` per matrix, requiring ~2 GB per iteration when computing both main and recent semantic scores.
  2. **Fragmentation & Gradients:** PyTorch’s autograd graph implicitly tracked operations, and the caching allocator became highly fragmented because `torch.cuda.empty_cache()` was not called within the loop.
- **Solution:**
  1. Enforced a `_SAFE_CHUNK = 256` limit to cap per-matrix VRAM at ~133 MB.
  2. Wrapped the loop in `torch.no_grad()` to disable graph tracking.
  3. Replaced `torch.tensor().cuda()` with `torch.as_tensor(..., device='cuda')` to eliminate an intermediate CPU float64 → GPU float32 round-trip.
  4. Placed explicit `torch.cuda.empty_cache()` calls inside the loop after deleting intermediate tensors (`sc_all`, `sc_rec`) to release memory blocks back to the OS and prevent progressive fragmentation.

### 7.9 BM25 Evaluation Loop Extremely Slow
- **Error:** `evaluate_bm25` took 9+ hours to process EB-NeRD validation split (244,647 queries).
- **Root Cause:** The `evaluate_bm25` loop iteratively tokenized and queried the index one row at a time. This introduced massive Python loop overhead (244k distinct calls to the C++ backend), negating the performance benefits of the inverted index.
- **Solution:** 
  1. Added a `search_batch` method to `BM25Retriever` to process queries in chunks of 10,000.
  2. Extracted query construction from the loop, precomputing all history texts.
  3. Executed a single batched `.search_batch(queries, k=max_k)` call, reducing C-level calls from 244,647 to just 25.
- **Numbers:**
  - Before: ~35 it/s (~9+ hours runtime).
  - After: ~400+ it/s (~10 minutes runtime).

### 7.10 BM25 Batched Retrieval OOM/Thrashing
- **Error:** System hung/crashed during the initial BM25 batched search call with `batch_size=10000`.
- **Root Cause:** `bm25s.retrieve()` allocates a score buffer matrix for the entire batch. A batch of 10,000 queries against a 130,379 document corpus requires `(10000 × 130379 × 4 bytes)` = ~5.2 GB of continuous RAM for dense float32 operations. This triggered severe memory thrashing and eventual OOM kills on limited memory environments.
- **Solution:** Reduced the default `batch_size` from 10,000 to 500, limiting per-batch RAM usage to ~260 MB. Additionally, added batch progress logging to explicitly monitor retrieval throughput.

### 7.11 BM25 `bm25s` JAX/Numba Overhead Bypass
- **Error:** Even with batched queries, `bm25s.retrieve()` caused Kaggle kernels to stall or fail due to background JAX/numba compilation and runtime overhead.
- **Root Cause:** The `bm25s.retrieve` API wraps its core sparse operations with JAX/numba for parallel top-k selection. In restricted container environments like Kaggle, the JIT compiler can deadlock or exhaust resources, causing silent hangs or taking ~9 hours to execute queries that should take seconds.
- **Solution:** Rewrote `search_batch` to completely bypass the `.retrieve()` method.
  - Accessed the internal `self._index.scores` (CSR sparse matrix) and `self._index.idf` directly.
  - Implemented a pure `scipy.sparse` batched matrix multiplication (`Q @ weighted`) followed by a standard `numpy.argpartition` for top-k selection.
- **Numbers:**
  - Before (JAX/numba overhead): System hangs or takes 5+ minutes just for JIT warmup.
  - After (Pure scipy.sparse): Stable execution, processing 244,000 queries in ~3-5 minutes with predictable memory footprint.

### 7.12 Notebook-Level Bugs and OOM Issues in Kaggle

The following issues were identified across the full notebook execution and are now resolved in `notebooks/final.ipynb`:

**7.12.1 Notebook Cell Labels Mismatched with Content**
- The old `final.ipynb` had cells labeled "Compute MIND embeddings" but actually executing EB-NeRD download code, making it impossible to debug failures.
- **Fix:** Rewrote the entire notebook with 29 sequentially labeled cells, each with a clear comment block explaining inputs, outputs, timing, and OOM risks.

**7.12.2 JAX CUDA Memory Reservation**
- Even after `pip uninstall jax`, if JAX was imported earlier in the session, it retains a CUDA device context that prevents other CUDA consumers (FAISS, PyTorch) from allocating contiguous blocks.
- **Fix:** JAX is now uninstalled as the very first step in Cell 4, before any other package install. Cell 4 also verifies JAX is absent via `importlib.util.find_spec('jax')`.

**7.12.3 `PYTORCH_CUDA_ALLOC_CONF` Not Set**
- Without `expandable_segments:True`, PyTorch cannot release partial VRAM blocks back to the OS, causing progressive VRAM fragmentation across cells in a long session.
- **Fix:** Cell 3 now sets `os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'` before any GPU work starts.

**7.12.4 Feature Store VRAM OOM Due to Large Chunks**
- The GPU feature scoring loop allocated `chunk_size x 130,379 x 4 bytes = 780 MB` per matrix. Default chunk_size was ~1500, requiring ~2 GB per iteration.
- **Fix:** Hardcoded `_SAFE_CHUNK = 256` inside `feature_store.py`. Wrapped loop in `torch.no_grad()`. Used `torch.as_tensor(..., device='cuda')` to avoid intermediate CPU roundtrips. Added `torch.cuda.empty_cache()` inside the loop.
- **Numbers:** `256 x 130379 x 4 bytes = 133 MB` per matrix call. Safe on T4 (15 GB VRAM).

**7.12.5 NRMS OOM Due to Full Training Set**
- Training NRMS on the full 2.2M MIND behaviors required building a vocabulary of all seen words and batching 2.2M impressions through the GPU. This hits T4 VRAM and/or kills the Kaggle CPU RAM budget.
- **Fix:** Pass `--max-train-rows 50000` and `--max-val-rows 5000`. This limits training to 50K impressions (enough to converge) and validation to 5K (fast feedback per epoch). Full model still trains all 3 epochs.

**7.12.6 No `gc.collect()` Between Cells**
- Python's garbage collector does not immediately free large numpy arrays when they go out of scope. Without explicit `gc.collect()`, stale arrays from Cell N occupied RAM during Cell N+1, causing silent OOM crashes.
- **Fix:** Every cell that loads large data (embeddings, behaviors, feature matrices) now explicitly calls `del <large_var>; gc.collect(); torch.cuda.empty_cache()` before the cell exits.

### 7.13 LightGBM Test Inference OOM (MIND: 2.37M Impressions)
- **Error:** `inference()` in `train_lgbm.py` loaded the full MIND test set (2.37M impressions) at once, built all features in a single `build_features()` call, causing RAM to climb past 30 GB and OOM-killing the Kaggle session.
- **Root Cause:** No chunking existed in the `inference()` function. `--max-train-rows` only caps training rows. Test inference ran unconditionally on the full split.
- **Solution:** Rewrote `inference()` to process `chunk_rows=50_000` rows at a time using `behaviors.slice(start, size)`. Each iteration builds features, runs `ranker.predict()`, writes results to the output file, and immediately frees the chunk via `del` + `gc.collect()`. Peak RAM stays under ~8 GB.
- **Numbers:**
  - MIND test: 2,370,727 impressions / 50,000 per chunk = 48 chunks.
  - Peak RAM per chunk: ~3-4 GB (feature matrix for 50K impressions).
  - Total time: ~45 minutes (same as before, just OOM-safe).

**7.13.1 Issues Audited and Confirmed Status**

| Issue | Status | Action Taken |
|---|---|---|
| **Issue 1**: `inference()` has no chunked test path — OOM at 30 GB | **REAL** | Rewrote `inference()` with `chunk_rows=50_000` in `train_lgbm.py` |
| **Issue 2**: `bm25.py search_batch` unknown — might still use old path | **FALSE** | Confirmed `from scipy import sparse` is active at line 139 |
| **Issue 3**: EB-NeRD skip-check used hardcoded path, misses nested layouts | **PARTIAL** | Fixed Cell 8 to use `rglob('behaviors.parquet')` instead of hardcoded path |
