
DAY 1 — Make it work
Target

First valid MIND + EB-NeRD submission.

We build:

environment
↓
data inspection
↓
schema
↓
temporal split
↓
leakage protection
↓
BM25
↓
semantic retrieval
↓
candidate generation
↓
prediction generation
↓
submission

No unnecessary optimization before we can submit.

DAY 2 — Make retrieval strong

We attack:

BM25
semantic retrieval
recency
popularity
history representation
fusion
RRF
candidate count

Then:

evaluation
+
ablation table

And submit improved models.

DAY 3 — Build the actual leaderboard model

This is where we transition to:

candidate
+
user
+
article
+
history
+
session
+
time
+
popularity
+
similarity

→

LightGBM Ranker

Then:

CatBoost

Then possibly:

ensemble

We will not jump straight into a Transformer.

The published winner shows Transformer + GBDT can be powerful, but a Transformer adds considerably more compute and implementation complexity.

Our first objective is to get GBDT-level gains quickly.

DAY 4 — Optimization day

Only ideas that have evidence of helping survive.

Potential experiments:

history length
recency decay
candidate count
BM25/semantic fusion
popularity windows
freshness
category affinity
entity affinity
session features
LightGBM parameters
CatBoost
ensemble weights
recent-vs-full training window

Then final:

best MIND model
        ↓
MIND submission

best EB-NeRD model
        ↓
EB-NeRD submission
