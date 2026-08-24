
1. MASTER TRACKER

# Task	Status	Test

1	Understand assignment	✅	Assignment requirements mapped
2	Baseline architecture	✅	Defined
3	Leaderboard architecture	✅	Defined
4	Project structure	✅	Starter repo created
5	Environment	🔵	Import smoke test
6	MIND inspection	⬜	Schema/time/row-count checks
7	EB-NeRD inspection	⬜	Schema/time/row-count checks
8	MIND adapter	⬜	Parse + integrity tests
9	EB-NeRD adapter	⬜	Parse + integrity tests
10	Unified schema	⬜	Schema assertions
11	Temporal split	⬜	No-future assertion
12	Leakage tests	⬜	Automated test
13	BM25	⬜	Retrieval sanity + Recall@K
14	Semantic embeddings	⬜	Vector shape/nearest-neighbor
15	FAISS	⬜	Self-neighbor test
16	Candidate union	⬜	Coverage/dedup test
17	Baseline scorer	⬜	All candidates scored
18	First MIND submission	⬜	Submission validator
19	First EB-NeRD submission	⬜	Submission validator
20	AUC/MRR/nDCG	⬜	Metric sanity tests
21	Diversity/novelty/coverage	⬜	Metric tests
22	Cold/warm slice	⬜	Slice comparison
23	Bootstrap CI	⬜	Reproducibility test
24	Recency representation	⬜	Ablation
25	Popularity/freshness	⬜	Ablation
26	BM25 + semantic fusion	⬜	Ablation
27	RRF	⬜	Ablation
28	Candidate features	⬜	Feature completeness
29	Behavioral features	⬜	Feature ablation
30	Session/context features	⬜	Feature ablation
31	LightGBM ranker	⬜	Validation improvement
32	CatBoost	⬜	Only if better
33	Ensemble	⬜	Only if better
34	Large-scale processing	⬜	Memory + runtime
35	Leaderboard iteration	⬜	Champion/challenger
36	Final MIND model	⬜	Final validation
37	Final EB-NeRD model	⬜	Final validation
38	Final submissions	⬜	Codabench
39	Design note	⬜	≤4 pages
40	README	⬜	One-command reproduction
41	Git/AI log	⬜	Assignment compliance
42	Quiz prep	⬜	Explain every design choice
Current position

🔵 #5 Environment

2. What is already inside the repository

The starter repository contains:

news_retrieval_assignment/
│
├── README.md
├── RUNBOOK.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   └── default.yaml
│
├── src/
│   ├── data/
│   │   ├── mind.py
│   │   └── ebnerd.py
│   │
│   ├── retrieval/
│   │   ├── bm25.py
│   │   └── semantic.py
│   │
│   ├── ranking/
│   │   ├── features.py
│   │   ├── utils.py
│   │   └── train_lgbm.py
│   │
│   └── evaluation/
│       ├── metrics.py
│       └── retrieval.py
│
├── scripts/
│   ├── inspect_mind.py
│   ├── inspect_ebnerd.py
│   ├── build_mind_features.py
│   ├── build_ebnerd_features.py
│   ├── train_ranker.py
│   └── score_validation.py
│
└── tests/
    ├── test_metrics.py
    ├── test_bm25.py
    └── test_semantic.py

This is intentionally a scaffold, not yet the final competition package. In particular, we are not guessing the exact Codabench output format; we'll validate it against the current competition sample/output specification before generating final files.

3. The project architecture we are actually implementing
   Phase A — assignment baseline
   MIND / EB-NeRD
   │
   ▼
   data adapter
   │
   ▼
   unified representation
   │
   ▼
   temporal validation
   │
   ├───────────────┐
   ▼               ▼
   BM25          embeddings
   │               │
   ▼               ▼
   lexical top-K     FAISS top-K
   │               │
   └───────┬───────┘
   ▼
   retrieval metrics

The assignment explicitly requires BM25 candidate generation, semantic candidate generation, recall@50/100/200, and comparison between lexical and semantic retrieval.

Phase B — competition scorer
user history
      │
      ▼
user representation
      │
      ▼
impression candidates
      │
      ├── BM25 similarity
      ├── semantic similarity
      ├── popularity
      ├── freshness
      ├── category affinity
      ├── entity affinity
      ├── session affinity
      └── context
              │
              ▼
         LightGBM ranker
              │
              ▼
       candidate ranking

That is where we expect the real leaderboard gains.

The EB-NeRD challenge is explicitly an impression-level ranking problem using user history, session information, personal metadata and candidate article information.

4. Phase C — eventual strongest architecture
   USER
   │
   ┌────────────┼────────────┐
   ▼            ▼            ▼
   long-term      short-term    session
   history        intent        context
   │            │            │
   └────────────┼────────────┘
   ▼
   user features
   │
   ▼
   ┌──────────────────────────┐
   │ Candidate article        │
   │ features                 │
   │                          │
   │ BM25                     │
   │ semantic similarity      │
   │ category/entity match    │
   │ popularity               │
   │ freshness                │
   │ recency                  │
   │ session affinity         │
   │ impression-relative rank │
   └────────────┬─────────────┘
   ▼
   LightGBM Ranker
   │
   ▼
   CatBoost challenger
   │
   ▼
   model ensemble
   │
   ▼
   final ranking

The reason we're targeting this is empirical rather than theoretical: published EB-NeRD challenge solutions that performed very strongly used combinations of behavioral, temporal, content and learned ranking components rather than relying on BM25 alone. The challenge repository also provides NRMS/LSTUR/NPA/NAML/NRMSDocVec implementations, showing that learned user-interest modeling is a major part of the benchmark ecosystem.

5. Why each technology was chosen
   Component	Chosen approach	Main alternatives	Why ours first
   Data	Polars/PyArrow + pandas where convenient	Spark, DuckDB	Faster to develop on free-tier machines
   MIND	Official train/dev structure	Re-split everything manually	Dataset already provides chronological recommendation splits
   EB-NeRD	Official parquet structure	Convert everything to CSV	Parquet is much more memory-efficient
   Lexical	BM25	TF-IDF, Lucene, Elasticsearch	Required by assignment + strong classical baseline
   Semantic MIND	Sentence-transformer	BERT/XLM-R manually	Much faster to deploy in 4 days
   Semantic EB	Provided article embeddings	Generate own embeddings	Saves huge compute; official dataset provides embeddings
   ANN	FAISS	ScaNN, HNSWlib	Easy, fast, well understood
   User vector	Mean first	attention, RNN, transformer	Fast baseline and clean ablation
   Fusion	RRF / weighted fusion	complicated neural fusion	Low-cost experiment
   Ranker	LightGBM	CatBoost, neural CTR, Transformer	Excellent speed/performance trade-off
   Deep ranker	Optional later	NRMS/LSTUR/etc.	Only after simpler ranker establishes a strong baseline
   Ensemble	Optional final	single model	Competition winners show ensembles can help

MIND contains roughly 160K English articles and 15M+ impression logs, with click history and impression labels.

EB-NeRD provides articles.parquet, behaviors.parquet, history.parquet, and article artifacts including embeddings.

6. Every task: what we're testing and how long it should take

These are the practical timing targets I want you to use.

Task	What proves it works	Approx. test/runtime
Environment	imports succeed	<1 min
MIND inspection	correct columns, counts, time range	10–60 sec
EB inspection	parquet schemas and counts print	10–90 sec
MIND parsing	no malformed rows	30 sec–3 min
EB parsing	list columns parse correctly	30 sec–5 min
Temporal split	every history event precedes prediction	<2 min
Leakage test	zero violations	seconds
BM25 smoke test	known query retrieves expected topic	seconds
BM25 recall	Recall@50/100/200 computed	1–10 min small data
Embedding generation	shape + no NaNs	5–60 min depending on corpus/GPU
FAISS	self-query returns self	seconds
Semantic recall	Recall@50/100/200	1–10 min
Candidate union	no duplicate candidates	seconds
Baseline prediction	every impression fully ranked	1–10 min
AUC/MRR/nDCG	sensible values and ranges	1–10 min
Diversity/novelty/coverage	nonzero sane metrics	<5 min
Slice analysis	cold/warm or head/tail reports	<5 min
Bootstrap	reproducible CI	1–15 min
Recency experiment	validation comparison	2–15 min
Fusion experiment	beats one baseline or not	2–15 min
Feature generation	no missing required features	1–15 min
LightGBM	validation improves or loses	2–30 min
CatBoost	challenger comparison	5–45 min
Ensemble	validation improvement	minutes
Large preprocessing	completes within RAM/storage	10 min–hours
Final prediction	every test impression gets ranking	potentially hours
Codabench	accepted submission	submission-dependent

These are deliberately ranges, because free-tier Kaggle/Colab hardware is variable.

7. What you should do RIGHT NOW

Do not touch the model yet.

Step 1 — download/open the repository

Download the starter repository

Extract it.

Then:

cd news_retrieval_assignment
Step 2 — create environment
Colab/Kaggle
pip install -r requirements.txt
Local
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
8. First test

Run:

python -c "import numpy, pandas, polars, pyarrow, faiss, lightgbm; print('ENVIRONMENT OK')"

Expected:

ENVIRONMENT OK

Runtime:

< 1 minute

Then:

pytest -q

Expected:

all tests pass

Runtime:

a few seconds

9. Then put the development datasets here
   data/
   ├── mind/
   │   ├── MINDsmall_train/
   │   │   ├── news.tsv
   │   │   └── behaviors.tsv
   │   │
   │   └── MINDsmall_dev/
   │       ├── news.tsv
   │       └── behaviors.tsv
   │
   └── ebnerd/
   ├── train/
   │   ├── articles.parquet
   │   ├── behaviors.parquet
   │   └── history.parquet
   │
   └── validation/
   ├── articles.parquet
   ├── behaviors.parquet
   └── history.parquet

For MIND, behaviors.tsv contains five fields: impression ID, user ID, time, history and impressions; news.tsv contains article ID, category/subcategory, title, abstract, URL and entity fields.

For EB-NeRD, behaviors.parquet contains impression/session/candidate/click/time/context fields, while history.parquet contains user click histories and timestamps.

10. Run the two inspection scripts
    MIND
    python scripts/inspect_mind.py 
    --path data/mind/MINDsmall_train
    EB-NeRD
    python scripts/inspect_ebnerd.py 
    --path data/ebnerd/train

The first result we care about is not model accuracy.

We need to know:

article count
behavior count
user count
time range
columns
history structure
impression structure
embedding availability

That will determine the exact next code we write.

11. Why this is the correct first coding step

Because MIND and EB-NeRD are not structurally identical.

MIND gives us a row-based TSV impression representation.

EB-NeRD gives us nested parquet lists for in-view articles, clicked articles and histories.

So if we skip inspection and immediately build the model, we risk designing the wrong data representation.

12. What we will do immediately after inspection

The sequence is now locked:

YOU RUN INSPECTION
        ↓
I READ THE ACTUAL OUTPUT
        ↓
BUILD UNIFIED SCHEMA
        ↓
TEST LEAKAGE
        ↓
BM25
        ↓
SEMANTIC
        ↓
FIRST SUBMISSIONS
        ↓
LEARNED RANKER
        ↓
LEADERBOARD OPTIMIZATION

No more jumping around.

13. One important limitation of the starter package

I don't want to pretend that the ZIP I generated is already the finished competition solution.

It currently gives you the project skeleton + data readers + BM25/FAISS primitives + evaluation + LightGBM skeleton + tests.

The next iterations need to add:

exact unified candidate table;
proper BM25 features rather than the temporary smoke-test lexical feature;
MIND article embeddings;
EB-NeRD embedding loading;
full candidate-level feature generation;
exact MIND Codabench submission writer;
exact EB-NeRD Codabench submission writer;
stronger ranker experiments.

I deliberately did not invent the final Codabench file format. The competition system should be matched against its actual sample/output specification rather than a guessed schema.

14. What we've accomplished with the implementation choice

We've now committed to:

                 ASSIGNMENT
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
      BM25 track            Semantic track
          │                     │
          └──────────┬──────────┘
                     ▼
             retrieval baseline
                     │
                     ▼
                FIRST SUBMIT
                     │
                     ▼
              hybrid retrieval
                     │
                     ▼
              feature engineering
                     │
                     ▼
              LightGBM ranker
                     │
                     ▼
          CatBoost / ensemble
                     │
                     ▼
             leaderboard loop

The critical strategic decision is:

Do not spend the four days perfecting retrieval before submitting.

Get a valid system online first. Then every remaining hour is spent asking:

What change actually improves validation, and does it improve Codabench?

That is the fastest route to a strong result.
