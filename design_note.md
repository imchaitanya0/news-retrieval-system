# Design Note: Hybrid Neural News Retrieval System

## 1. System Architecture

We built a hybrid news recommendation system combining zero-shot retrieval methods with a learning-to-rank (LTR) reranker, designed to evaluate on the MIND and EB-NeRD RecSys 2024 benchmarks.

### Pipeline Overview
1. **Data Ingestion & Unification:** We load raw TSV (MIND) and Parquet (EB-NeRD) data into Polars dataframes, standardizing schemas so the downstream pipeline is dataset-agnostic.
2. **Feature Store:** We compute embeddings for all articles using `sentence-transformers` and index them using FAISS. Simultaneously, we build a global BM25 index over article texts (title + abstract).
3. **Candidate Scoring (Zero-Shot):** For a given user impression, we compute semantic similarity using batched GPU matrix multiplication (user vector dot candidate vectors) and lexical similarity (BM25 overlap).
4. **Learning-to-Rank (LightGBM):** We train a LambdaMART model on historical behaviors to fuse semantic scores, BM25 signals, popularity, and context features (e.g., category match, history length).
5. **Inference & Submission:** The LightGBM model reranks the predefined candidates for each impression in the test set, outputting Codabench-compliant formatted predictions.

---

## 2. Key Design Choices and Rationale

### Multilingual Sentence-Transformers
**Choice:** `paraphrase-multilingual-MiniLM-L12-v2`
**Rationale:** The system needed to support both English (MIND) and Danish (EB-NeRD). Instead of managing separate language models or translating Danish to English, we opted for a multilingual model. The `MiniLM` variant provides a compact 384-dimensional embedding, ensuring fast inference and low memory footprint (fitting easily on Kaggle's T4 GPUs) without requiring domain-specific fine-tuning.

### Direct Candidate Scoring vs. Global Retrieval
**Choice:** Scoring predefined candidates rather than full-catalog retrieval.
**Rationale:** In both MIND and EB-NeRD evaluation formats, each impression comes with a predefined set of candidate articles (typically 20-100). Initial attempts used global retrieval (fetching top-200 from 120k+ articles), which yielded near-zero overlap with the predefined candidates, dropping performance to a random baseline (AUC ~0.51). Shifting to direct dot-product scoring of *only* the candidates in the impression restored performance.

### GPU Batch Matrix Multiplication
**Choice:** Storing the full embedding matrix on the GPU and using `torch.mm`.
**Rationale:** Looping over 2.3M impressions to compute per-user-candidate similarities in Python took >9 hours. By batching 8,000 users at a time and performing a single dense matrix multiplication (`[8000, 384] x [384, 120000]`), we reduced the semantic scoring phase to ~15 minutes on a single T4 GPU.

### LightGBM (LambdaMART) Reranker
**Choice:** LightGBM over complex neural rerankers or simple weighted fusion.
**Rationale:** While zero-shot semantic similarity is a strong baseline, tree-based LTR models excel at combining heterogeneous features (dense scores, sparse categorical matches, log-scaled popularity). LightGBM is highly optimized, handles missing values gracefully, and trains in minutes on CPUs.

---

## 3. Experimental Observations

### Lexical vs. Semantic
Semantic embeddings (MiniLM) heavily outperformed pure BM25 on this task. BM25 suffers when users' historical clicks use different vocabulary than the target article (e.g., "football match" vs "soccer game"). However, BM25 proved valuable as an auxiliary feature in the LightGBM model, anchoring recommendations to exact keyword matches when applicable.

### MIND vs. EB-NeRD Differences
- **Scale:** EB-NeRD is substantially larger (6M test impressions vs MIND's 2.3M), exposing memory bottlenecks in Python's standard `to_dicts()` conversions, which we mitigated using Polars chunking.
- **Data Contamination:** EB-NeRD's raw distribution contained train/val overlap in the test sets.
- **Language:** The multilingual model handled Danish seamlessly, validating the unified architecture.

### Expected Performance (AUC)
| Model | Strategy | MIND | EB-NeRD |
| :--- | :--- | :--- | :--- |
| Random | Baseline | 0.500 | 0.500 |
| Semantic | Zero-Shot | ~0.62 | ~0.60 |
| LightGBM | LambdaMART | ~0.65+ | ~0.64+ |

---

## 4. Scaling Limits (10x Scale)

If the dataset size increased by 10x (e.g., 1.2M articles, 23M impressions):

1. **Memory (OOM):** The current system loads the full 120k x 384 embedding matrix onto the GPU (requires ~180MB). At 1.2M articles, this becomes ~1.8GB, which still fits on a T4. However, Polars loading a 10x behaviors dataset (~60GB) would OOM standard Kaggle instances (30GB RAM). We would need to implement disk-backed lazy streaming or Spark.
2. **Latency:** The GPU matrix multiplication scales well, but the current batching logic creates a `[B, N_articles]` dense array in memory before indexing candidates. At 1.2M articles, an 8K batch output takes `8000 x 1.2M x 4 bytes = ~38GB` of VRAM, causing an immediate CUDA OOM. We would need to switch to sparse operations or a localized `gather` operation (e.g., `F.embedding` lookups) instead of full dense matmuls.
3. **Index Size:** The BM25 global index would become prohibitively slow to build and query at inference time. We would need to move to an optimized, sharded search backend like Elasticsearch or Vespa.
