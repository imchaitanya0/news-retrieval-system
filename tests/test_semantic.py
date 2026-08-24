"""
Semantic retriever unit tests — synthetic data, no GPU required.
"""
import numpy as np
import pytest
from src.retrieval.semantic import (
    SemanticRetriever, build_user_vector, recall_at_k
)


@pytest.fixture
def tiny_embeddings():
    """4 unit-normalised 8-dim vectors."""
    np.random.seed(42)
    vecs = np.random.randn(4, 8).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


@pytest.fixture
def article_ids():
    return ["A1", "A2", "A3", "A4"]


@pytest.fixture
def retriever(tiny_embeddings, article_ids):
    r = SemanticRetriever()
    r.build(tiny_embeddings, article_ids)
    return r


class TestSemanticRetriever:
    def test_self_neighbor(self, retriever, tiny_embeddings, article_ids):
        """Each article's embedding should retrieve itself as the top result."""
        for i, aid in enumerate(article_ids):
            result = retriever.retrieve_by_vector(tiny_embeddings[i], k=1)
            assert result[0] == aid, f"{aid} is not its own nearest neighbor"

    def test_retrieve_returns_list(self, retriever, tiny_embeddings):
        result = retriever.retrieve_by_vector(tiny_embeddings[0], k=2)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_retrieve_k_limit(self, retriever, tiny_embeddings):
        result = retriever.retrieve_by_vector(tiny_embeddings[0], k=2)
        assert len(result) <= 2

    def test_zero_vector_handled(self, retriever):
        zero = np.zeros(8, dtype=np.float32)
        # Should not crash (returns something)
        result = retriever.retrieve_by_vector(zero, k=2)
        assert isinstance(result, list)


class TestUserVector:
    def test_single_article(self):
        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        embedding_map = {"A1": emb}
        vec = build_user_vector(["A1"], embedding_map)
        assert vec is not None
        np.testing.assert_allclose(vec, emb, atol=1e-6)

    def test_mean_of_two(self):
        e1 = np.array([1.0, 0.0], dtype=np.float32)
        e2 = np.array([0.0, 1.0], dtype=np.float32)
        embedding_map = {"A1": e1, "A2": e2}
        vec = build_user_vector(["A1", "A2"], embedding_map)
        # Mean of unit vectors then normalised
        assert vec is not None
        assert vec.shape == (2,)

    def test_empty_history(self):
        vec = build_user_vector([], {})
        assert vec is None

    def test_no_matching_embeddings(self):
        vec = build_user_vector(["UNKNOWN"], {"A1": np.zeros(4, dtype=np.float32)})
        assert vec is None

    def test_max_articles_uses_recent(self):
        embs = {f"A{i}": np.eye(10, dtype=np.float32)[i % 10] for i in range(20)}
        history = [f"A{i}" for i in range(20)]
        # Only last 3
        vec = build_user_vector(history, embs, max_articles=3)
        assert vec is not None


class TestRecallAtK:
    def test_perfect(self):
        assert recall_at_k(["A", "B"], {"A", "B"}, k=2) == 1.0

    def test_zero(self):
        assert recall_at_k(["X"], {"A"}, k=1) == 0.0

    def test_empty_gt(self):
        assert recall_at_k(["A"], set(), k=1) == 0.0
