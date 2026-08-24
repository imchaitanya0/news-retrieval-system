# One-command rebuild — run from repo root (same dir as this Makefile)
# Usage:
#   make data        → build all processed parquets
#   make bm25        → evaluate BM25 on MIND val
#   make semantic    → evaluate semantic on MIND val
#   make hybrid      → evaluate hybrid on MIND val
#   make eval        → run full evaluation harness on MIND val
#   make submit      → generate hybrid submission for MIND test
#   make test        → run all tests
#   make all         → data + bm25 + semantic + hybrid + eval + test

.PHONY: data bm25 semantic hybrid eval submit test all

data:
	python -m src.data.build_pipeline

bm25:
	python -m src.retrieval.bm25 --dataset mind --split val

semantic:
	python -m src.retrieval.semantic --dataset mind --split val

hybrid:
	python -m src.retrieval.hybrid --dataset mind --split val

eval:
	python -m src.evaluation.metrics --dataset mind --split val

submit:
	python -m src.submission.generate --dataset mind --strategy hybrid
	python -m src.submission.generate --dataset ebnerd --strategy hybrid

test:
	pytest tests/ -v

all: data bm25 semantic hybrid eval test
