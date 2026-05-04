"""Unit tests for reranking_rag.implementation.reranker."""
from unittest.mock import patch, MagicMock
import numpy as np


def _mock_cross_encoder(scores):
    """Return a mock CrossEncoder whose predict() returns a fixed numpy array."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array(scores, dtype=float)
    return mock_model


def test_rerank_returns_list(initial_docs):
    mock_model = _mock_cross_encoder([float(i) for i in range(len(initial_docs))])
    with patch("reranking_rag.implementation.reranker._get_cross_encoder",
               return_value=mock_model):
        from reranking_rag.implementation.reranker import rerank
        result = rerank("what is the delivery time?", initial_docs)
    assert isinstance(result, list)


def test_rerank_returns_top_k_docs(initial_docs):
    mock_model = _mock_cross_encoder([float(i) for i in range(len(initial_docs))])
    with patch("reranking_rag.implementation.reranker._get_cross_encoder",
               return_value=mock_model):
        from reranking_rag.implementation.reranker import rerank
        result = rerank("query", initial_docs, top_k=5)
    assert len(result) == 5


def test_rerank_each_doc_has_rerank_score(initial_docs):
    mock_model = _mock_cross_encoder([float(i) for i in range(len(initial_docs))])
    with patch("reranking_rag.implementation.reranker._get_cross_encoder",
               return_value=mock_model):
        from reranking_rag.implementation.reranker import rerank
        result = rerank("query", initial_docs)
    for doc in result:
        assert "rerank_score" in doc


def test_rerank_rerank_score_is_float(initial_docs):
    mock_model = _mock_cross_encoder([float(i) for i in range(len(initial_docs))])
    with patch("reranking_rag.implementation.reranker._get_cross_encoder",
               return_value=mock_model):
        from reranking_rag.implementation.reranker import rerank
        result = rerank("query", initial_docs)
    for doc in result:
        assert isinstance(doc["rerank_score"], float)


def test_rerank_sorted_descending(initial_docs):
    scores     = [float(i) for i in range(len(initial_docs))]
    mock_model = _mock_cross_encoder(scores)
    with patch("reranking_rag.implementation.reranker._get_cross_encoder",
               return_value=mock_model):
        from reranking_rag.implementation.reranker import rerank
        result = rerank("query", initial_docs, top_k=5)
    scores_out = [d["rerank_score"] for d in result]
    assert scores_out == sorted(scores_out, reverse=True)


def test_rerank_highest_scored_doc_is_first(initial_docs):
    """The doc with the highest cross-encoder score must appear first."""
    scores     = list(range(len(initial_docs)))
    scores[7]  = 999.0   # give doc_007 a very high score
    mock_model = _mock_cross_encoder(scores)
    with patch("reranking_rag.implementation.reranker._get_cross_encoder",
               return_value=mock_model):
        from reranking_rag.implementation.reranker import rerank
        result = rerank("query", initial_docs, top_k=5)
    assert result[0]["id"] == initial_docs[7]["id"]


def test_rerank_preserves_original_doc_fields(initial_docs):
    """All original fields (id, text, metadata, distance) survive reranking."""
    mock_model = _mock_cross_encoder([float(i) for i in range(len(initial_docs))])
    with patch("reranking_rag.implementation.reranker._get_cross_encoder",
               return_value=mock_model):
        from reranking_rag.implementation.reranker import rerank
        result = rerank("query", initial_docs)
    for doc in result:
        assert "id"       in doc
        assert "text"     in doc
        assert "metadata" in doc
        assert "distance" in doc


def test_rerank_calls_predict_with_pairs(initial_docs):
    """predict() must be called with (query, doc_text) pairs for all candidates."""
    mock_model = _mock_cross_encoder([float(i) for i in range(len(initial_docs))])
    query      = "what is the average delivery time?"
    with patch("reranking_rag.implementation.reranker._get_cross_encoder",
               return_value=mock_model):
        from reranking_rag.implementation.reranker import rerank
        rerank(query, initial_docs)
    call_args = mock_model.predict.call_args[0][0]
    assert len(call_args) == len(initial_docs)
    for pair in call_args:
        assert pair[0] == query
        assert isinstance(pair[1], str)


def test_rerank_empty_docs_returns_empty():
    mock_model = _mock_cross_encoder([])
    with patch("reranking_rag.implementation.reranker._get_cross_encoder",
               return_value=mock_model):
        from reranking_rag.implementation.reranker import rerank
        result = rerank("query", [])
    assert result == []


def test_rerank_top_k_capped_by_input_size(initial_docs):
    """Requesting more docs than available should return all available docs."""
    small_docs = initial_docs[:3]
    mock_model = _mock_cross_encoder([1.0, 2.0, 3.0])
    with patch("reranking_rag.implementation.reranker._get_cross_encoder",
               return_value=mock_model):
        from reranking_rag.implementation.reranker import rerank
        result = rerank("query", small_docs, top_k=10)
    assert len(result) == 3
