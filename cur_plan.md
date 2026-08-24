
CS4.406 Assignment 1 — News Retrieval System: Full Plan
What the Assignment Requires (A1.pdf Summary)
Build a news retrieval system on MIND-small and EB-NeRD demo/small datasets with 6 deliverables:

Q#	Deliverable	Weight
Q1	Reproducible Data Pipeline	Foundation
Q2	Lexical Retrieval (BM25)	Core
Q3	Semantic Retrieval (Embeddings)	Core
Q4	Offline Evaluation Harness	Critical
Q5	Codabench Submission (both leaderboards)	Leaderboard
Q6	Design Note ≤4 pages	Report
Deadline: August 27, 2026 (at Quiz-1)

IMPORTANT

TA explicitly said leaderboard standings affect grading. The leaderboard uses the large datasets (ebnerd_large + MINDlarge), not the demo/small sets. You must download them and generate predictions on the large test sets.

🐛 Immediate Bug Fix Required
Error: ColumnNotFoundError: unable to find column "article_ids_clicked"

Root cause: In
parse_ebnerd_behaviors
, the column article_ids_clicked is renamed to clicked_ids at line 117, but then referenced again by its original name at line 130 — after the rename already happened. This is a double-rename bug.

Fix: Remove lines 129–132 (the redundant re-reference). The rename on line 117 already handles this. See fix below.

Current State vs. What You Have Done ✅
What's Done
 Project scaffolding (src/data/, src/retrieval/, etc.)
 Data download script (download.py)
 Data inspection script (inspect.py)
 Build pipeline skeleton (build_pipeline.py) — partially working
 MIND TSV parser
 EB-NeRD articles parser
 Temporal splitting logic
 Parquet output
 Bug in parse_ebnerd_behaviors — see above
What's Empty/Missing
 src/retrieval/ — BM25 retriever (Q2)
 src/retrieval/ — Semantic/embedding retriever (Q3)
 src/evaluation/ — Metrics harness (Q4)
 src/submission/ — Prediction file generator (Q5)
 src/models/ — Any ranking model
 src/features/ — Feature store
 docs/ — Design note
Full To-Do Checklist
🔥 Priority 1: Fix & Finish Data Pipeline (TODAY)
 Fix article_ids_clicked bug in parse_ebnerd_behaviors (lines 129–132)
 Fix history integration — currently history_df is passed but never used in parse_ebnerd_behaviors. User click history must be joined so you can form queries.
 Verify temporal split produces non-overlapping windows (the test in test_anti_gaming.py is a placeholder — add real assertions)
 Download ebnerd_large + ebnerd_testset + MINDlarge_train/dev/test (needed for final submission)
 Run python src/data/build_pipeline.py successfully end-to-end
🔥 Priority 2: BM25 Retrieval — src/retrieval/bm25.py (Q2)

For each user's impression:

1. Take their click history (article IDs)
2. Fetch titles of clicked articles → concatenate into a query string
3. Score all candidate articles with BM25
4. Return top-K (K=50, 100, 200)
   Build inverted index over title + abstract (or title + subtitle in EB-NeRD)
   Implement BM25 scoring (use rank_bm25 library or implement from scratch)
   Construct query from user history (concatenate recent article titles)
   Evaluate Recall@K for K ∈ {50, 100, 200} on both datasets
   🔥 Priority 3: Semantic Retrieval — src/retrieval/semantic.py (Q3)
   Load pre-trained embeddings from EB-NeRD (Ekstra_Bladet_word2vec.zip) OR compute with sentence-transformers
   For MIND: use paraphrase-multilingual-MiniLM-L12-v2 or similar
   Build FAISS flat index (brute-force cosine for small scale)
   Compute user vector = mean of clicked article embeddings
   Retrieve top-K candidates
   Evaluate Recall@K for K ∈ {50, 100, 200}
   🔥 Priority 4: Evaluation Harness — src/evaluation/metrics.py (Q4)
   Required metrics:

 AUC (per-impression, then macro-averaged)
 MRR
 nDCG@5, nDCG@10
 Beyond-accuracy: Intra-list diversity, Novelty, Coverage
 Bootstrap 95% CI for all metrics
 Slicing: cold-start (≤5 clicks) vs. warm users; head (top-10% popular) vs. tail articles
🔥 Priority 5: Codabench Submission — src/submission/generate.py (Q5)
MIND submission format:

{impression_id} [{article_id}-{rank} ...]
EB-NeRD submission format: Check RecSys 2024 challenge for exact format.

 Generate predictions on large test set (MINDlarge_test, ebnerd_testset)
 Validate prediction format
 Submit to both Codabench competitions
 Screenshot leaderboard scores
Priority 6: Design Note — docs/design_note.pdf (Q6, ≤4 pages)
 What you built and key choices (BM25 vs. semantic, FAISS type, query construction)
 Alternatives considered
 Experimental observations (lexical vs. semantic, MIND vs. EB-NeRD differences)
 Where it breaks at 10× scale
Leaderboard Strategy (to maximize rank)
TIP

Since the TA said leaderboard rank affects grading, here are high-impact improvements in priority order:

Quick Wins (implement before first submission)
Query construction matters: Instead of just concatenating all history titles, use only the last 5–10 clicked articles (recency matters in news). Older clicks are noisy.
BM25 tuning: Set k1=1.5, b=0.75 as starting point. Tune on val set.
Hybrid retrieval: Combine BM25 + semantic scores with a simple linear blend α * bm25 + (1-α) * semantic. This almost always beats either alone.
Category/subcategory filtering: Before scoring, filter to articles in the same category as user's recent history. This improves precision dramatically.
Medium-effort boosts
Reciprocal Rank Fusion (RRF): Instead of score blending, use RRF to combine BM25 and semantic ranked lists — more robust.
Popularity bias correction: Boost slightly fresh articles (< 3 days old) and penalize already-seen ones.
Entity overlap: Add entity overlap between user history entities and candidate article entities as a signal.
High-effort (if time allows)
LightGBM ranker (plan.md Day 3): Feature engineer and train a LambdaMART ranker using the candidates from BM25+semantic as input. This is the biggest quality jump.
Architecture Overview

data/raw/
  ├── ebnerd/{demo,small,large,testset}.zip
  └── mind/{behaviors,news}.tsv + large test
         ↓
src/data/build_pipeline.py
         ↓
data/processed/
  ├── articles_{ebnerd,mind}.parquet
  ├── behaviors_{ebnerd,mind}_{train,val,test}.parquet
  └── feature_store/ (embeddings, BM25 index)
         ↓
src/retrieval/
  ├── bm25.py       → top-K candidates (lexical)
  └── semantic.py   → top-K candidates (embedding)
         ↓
src/evaluation/metrics.py  → AUC, MRR, nDCG, diversity
         ↓
src/submission/generate.py → prediction files → Codabench
Key Constraints (Anti-Gaming — Q9)
WARNING

The pipeline must strictly enforce the behavior-window boundary:

History used for query construction must only include clicks before the impression time
No article features from after the impression time
Add a pytest assertion in tests/test_anti_gaming.py to verify this
Files to Create
File	Status	Priority
src/data/build_pipeline.py	Fix bug	🔥 Now
src/retrieval/bm25.py	Create	🔥 Today
src/retrieval/semantic.py	Create	🔥 Today
src/evaluation/metrics.py	Create	🔥 Today
src/submission/generate.py	Create	Before deadline
src/features/feature_store.py	Create	Today
tests/test_anti_gaming.py	Expand	Today
docs/design_note.md	Write	Aug 26
Makefile	Expand	Aug 26
README.md	Update	Aug 26
