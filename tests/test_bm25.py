"""
BM25 unit tests — run without large data (uses synthetic DataFrames)
"""
import polars as pl
import pytest
from src.retrieval.bm25 import (
    BM25Retriever, build_query_from_history, recall_at_k, _article_text
)


@pytest.fixture
def tiny_articles():
    return pl.DataFrame({
        "article_id": ["A1", "A2", "A3", "A4"],
        "title": ["football match winner", "basketball game score", "recipe pasta vegan", "technology AI news"],
        "subtitle": ["team wins cup final", "nba championship result", "healthy dinner idea", "machine learning update"],
    })


@pytest.fixture
def retriever(tiny_articles):
    r = BM25Retriever()
    r.build(tiny_articles)
    return r


class TestBM25Retriever:
    def test_build_indexes_all_articles(self, retriever, tiny_articles):
        assert len(retriever.article_ids) == len(tiny_articles)

    def test_retrieve_returns_list(self, retriever):
        result = retriever.retrieve("football winner", k=2)
        assert isinstance(result, list)
        assert len(result) <= 2

    def test_retrieve_sports_query(self, retriever):
        """A sports query should surface A1 or A2, not recipe article."""
        result = retriever.retrieve("football basketball sports", k=2)
        assert set(result).issubset({"A1", "A2", "A3", "A4"})
        # A3 (recipe) should NOT be top-1 for a sports query
        if result:
            assert result[0] != "A3", "Recipe article ranked first for sports query — suspicious"

    def test_retrieve_empty_query(self, retriever):
        result = retriever.retrieve("", k=5)
        assert result == []

    def test_retrieve_k_limits_results(self, retriever):
        result = retriever.retrieve("the", k=2)
        assert len(result) <= 2

    def test_article_text_concatenation(self):
        row = {"title": "hello world", "subtitle": "extra info"}
        assert "hello world" in _article_text(row)
        assert "extra info" in _article_text(row)

    def test_article_text_none_fields(self):
        row = {"title": None, "subtitle": None}
        assert _article_text(row) == ""


class TestQueryFromHistory:
    def test_basic_query(self):
        text_map = {"A1": "football match", "A2": "basketball game"}
        q = build_query_from_history(["A1", "A2"], text_map)
        assert "football" in q
        assert "basketball" in q

    def test_max_articles_truncates(self):
        text_map = {f"A{i}": f"topic {i}" for i in range(20)}
        history = [f"A{i}" for i in range(20)]
        q = build_query_from_history(history, text_map, max_articles=3)
        # Only last 3: A17, A18, A19
        assert "topic 19" in q
        assert "topic 17" in q
        # A0 should NOT be in query (too old)
        assert "topic 0" not in q

    def test_empty_history(self):
        q = build_query_from_history([], {})
        assert q == ""

    def test_missing_article_in_map(self):
        q = build_query_from_history(["MISSING"], {})
        assert q == ""


class TestRecallAtK:
    def test_perfect_recall(self):
        assert recall_at_k(["A", "B", "C"], {"A", "B"}, k=3) == 1.0

    def test_zero_recall(self):
        assert recall_at_k(["X", "Y", "Z"], {"A"}, k=3) == 0.0

    def test_partial_recall(self):
        r = recall_at_k(["A", "B", "C", "D"], {"A", "C"}, k=2)
        # retrieved[:2] = [A, B] → only A is in ground truth → 1/2
        assert r == 0.5

    def test_empty_ground_truth(self):
        assert recall_at_k(["A", "B"], set(), k=2) == 0.0
