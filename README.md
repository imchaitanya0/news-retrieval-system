# News Retrieval & Recommendation System

Welcome to our Hybrid Neural News Recommendation System! This project was built to rank news articles for users based on their reading history, session context, and article content. It was developed for the **MIND (Microsoft News Dataset)** and **EB-NeRD (Ekstra Bladet RecSys 2024)** competitions.

---

## 🚀 One-Command Reproduction (Kaggle Environment)

If you are running this in a Kaggle notebook (or any environment with a GPU), you can reproduce our entire pipeline from scratch with these commands:

```bash
# 1. Install required dependencies
pip install -q bm25s rank_bm25 faiss-gpu sentence-transformers polars lightgbm

# 2. Download the required raw data files
mkdir -p data/raw/mind
wget -q https://mind201910small.blob.core.windows.net/release/MINDlarge_test.zip
unzip -q MINDlarge_test.zip -d data/raw/mind/ && rm MINDlarge_test.zip
huggingface-cli download Ekstra-Bladet/ebnerd_testset --repo-type dataset --local-dir data/raw/ebnerd/

# 3. Build the data pipeline (Cleans data, enforces temporal splits, builds unified feature store)
python -m src.data.build_pipeline

# 4. Train LightGBM & Generate Final Submission ZIPs
python -m src.ranking.train_lgbm --dataset mind
python -m src.ranking.train_lgbm --dataset ebnerd
```
*The final Codabench-ready zip files will be generated at `data/submissions/mind_test_lgbm.zip` and `ebnerd_test_lgbm.zip`.*

---

## 🏗️ Architecture & Design Choices (The "Why")

This project implements an **Industry-Standard Multi-Stage Architecture**. If you look at how Netflix, YouTube, or Google News recommend content, they do not score every single item in their database for every user. Instead, they use a two-step process: **L1 (Retrieval)** and **L2 (Ranking)**.

### 1. L1 Retrieval (Candidate Generation)
**The Problem:** We have over 120,000 articles. Running a heavy Machine Learning model on all 120,000 articles for 6 million users is computationally impossible in real-time.
**The Solution:** An L1 Retriever uses fast, approximate methods (like FAISS for embeddings or Elasticsearch for keyword matching) to narrow down the 120,000 articles to a small list of ~50 highly relevant "candidates" per user.
**How we used it:** In the Codabench dataset, the L1 retrieval has *already been done for us*. The dataset provides an `impressions` column, which contains the ~50 candidate articles selected by the platform. Our job is to take those candidates and perform L2 Ranking.

### 2. L2 Ranking (LightGBM)
**The Problem:** Now that we have 50 candidates, how do we order them from best to worst? 
**The Solution:** We extract dense features for each candidate and pass them into a heavy Machine Learning model (LightGBM LambdaMART) trained specifically to optimize ranking metrics (like nDCG).
**How we used it:** We built a Feature Store (`src/features/feature_store.py`) that extracts 6 powerful features for every single candidate, which our LightGBM model (`src/ranking/train_lgbm.py`) uses to make its final ranking decision.

---

## ⚙️ The Features (What they solve)

To rank the articles effectively, our LightGBM model relies on the following features:

### Feature 1: Semantic Score (GPU Batched)
- **Why we used it:** We need the system to understand "meaning". If a user reads about "Lionel Messi", they might also like an article about "Cristiano Ronaldo" because they are semantically similar (both soccer).
- **How we built it:** We use `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`) to turn article text into math vectors.
- **The Speed Problem:** Doing vector math (dot products) for 6 million users one-by-one in Python takes 9+ hours.
- **The Fix:** We pre-load all 120,000 article vectors into the GPU (VRAM) and use PyTorch Batch Matrix Multiplication (`torch.mm`) to score 1,500 users at exactly the same time. This drops the processing time from 9 hours to ~2 minutes.

### Feature 2: Lexical Overlap (Replaces BM25)
- **Why we used it:** Semantic models sometimes drift. They might recommend "Basketball" when the user only wants "Soccer". We need an "exact keyword match" signal.
- **The Speed Problem:** Usually, this is done using BM25. However, doing a global BM25 search for 6 million users takes over 2 hours on a CPU. In the real world, you would use a giant Elasticsearch server cluster, which we don't have on Kaggle.
- **The Fix:** Since we already know the 50 candidates (thanks to L1), we do not need to do a global search! Instead, we compute an in-memory **Lexical Overlap** (Jaccard token intersection) between the words in the user's history and the words in the candidate article. It gives the exact same keyword-matching signal as BM25, but computes locally in microseconds.

### Feature 3: Log Popularity
- **Why we used it:** What if a user has no history (Cold-Start)? We have no semantic vector and no lexical tokens for them. 
- **The Fix:** We calculate how many times an article was clicked globally in the training set. If we know nothing about a user, recommending the most generally popular article is the mathematically safest bet.

### Additional Features:
- **History Length:** Tells the model if the user is a cold-start (0 clicks) or a power user (50+ clicks), allowing it to dynamically weight Popularity vs Semantic scores.
- **Category Match:** A simple binary check if the candidate article matches the user's most frequently clicked news category.
- **Inverse Position:** Articles placed higher on the screen by the original publisher naturally get more clicks. This captures that editorial placement bias.

---

## 📁 Repository Structure

- `src/data/`: Scripts to download, clean, and temporally split the datasets.
- `src/retrieval/`: Semantic embedding generators and BM25 indexers.
- `src/features/`: GPU-accelerated Feature Store for the L2 Ranker.
- `src/ranking/`: LightGBM training, inference, and ZIP packaging scripts.
- `src/evaluation/`: Offline evaluation harness (AUC, MRR, nDCG, Diversity, Novelty).
- `tests/`: Anti-gaming tests to mathematically prove no future-click leakage occurs.
