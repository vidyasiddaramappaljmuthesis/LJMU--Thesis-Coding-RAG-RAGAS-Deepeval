"""Unit tests for multiquery_rag.implementation.pipeline (end-to-end with mocks)."""
import os
from unittest.mock import patch, MagicMock

import pytest

os.environ.setdefault("GROQ_API_KEYS", "test-key-1,test-key-2")

_MOCK_ANSWER   = "The average delivery time is 12 days based on SP orders."
_EXPANDED_Q    = [
    "What is the average delivery time?",
    "How long does it take for orders to arrive?",
    "What is the typical shipping duration?",
    "How many days until delivery on average?",
]
_PER_VARIANT_DOCS = [
    {"id": f"doc_{i:03d}", "text": f"Doc {i}.", "metadata": {}, "distance": round(0.05*i, 4)}
    for i in range(1, 11)
]
_FUSED_DOCS = [
    {"id": f"doc_{i:03d}", "text": f"Doc {i}.", "metadata": {}, "distance": round(0.05*i, 4),
     "rrf_score": round(1.0 / (60 + i), 6)}
    for i in range(1, 6)
]


def _mock_expand(query, n=4):
    return _EXPANDED_Q


def _mock_retrieve_multi(queries, top_n=10):
    return {q: _PER_VARIANT_DOCS for q in queries}


def _mock_fuse(ranked_lists, top_n=5):
    return _FUSED_DOCS


def test_run_multiquery_rag_returns_dict():
    with patch("multiquery_rag.implementation.pipeline.expand_query", side_effect=_mock_expand), \
         patch("multiquery_rag.implementation.pipeline.retrieve_multi", side_effect=_mock_retrieve_multi), \
         patch("multiquery_rag.implementation.pipeline.rrf_fuse", side_effect=_mock_fuse), \
         patch("multiquery_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER):
        from multiquery_rag.implementation.pipeline import run_multiquery_rag
        result = run_multiquery_rag("What is the average delivery time?")
    assert isinstance(result, dict)


def test_run_multiquery_rag_has_query_key():
    with patch("multiquery_rag.implementation.pipeline.expand_query", side_effect=_mock_expand), \
         patch("multiquery_rag.implementation.pipeline.retrieve_multi", side_effect=_mock_retrieve_multi), \
         patch("multiquery_rag.implementation.pipeline.rrf_fuse", side_effect=_mock_fuse), \
         patch("multiquery_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER):
        from multiquery_rag.implementation.pipeline import run_multiquery_rag
        result = run_multiquery_rag("question?")
    assert "query" in result


def test_run_multiquery_rag_has_answer_key():
    with patch("multiquery_rag.implementation.pipeline.expand_query", side_effect=_mock_expand), \
         patch("multiquery_rag.implementation.pipeline.retrieve_multi", side_effect=_mock_retrieve_multi), \
         patch("multiquery_rag.implementation.pipeline.rrf_fuse", side_effect=_mock_fuse), \
         patch("multiquery_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER):
        from multiquery_rag.implementation.pipeline import run_multiquery_rag
        result = run_multiquery_rag("question?")
    assert "answer" in result


def test_run_multiquery_rag_has_expanded_queries_key():
    with patch("multiquery_rag.implementation.pipeline.expand_query", side_effect=_mock_expand), \
         patch("multiquery_rag.implementation.pipeline.retrieve_multi", side_effect=_mock_retrieve_multi), \
         patch("multiquery_rag.implementation.pipeline.rrf_fuse", side_effect=_mock_fuse), \
         patch("multiquery_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER):
        from multiquery_rag.implementation.pipeline import run_multiquery_rag
        result = run_multiquery_rag("question?")
    assert "expanded_queries" in result


def test_run_multiquery_rag_has_query_results_key():
    with patch("multiquery_rag.implementation.pipeline.expand_query", side_effect=_mock_expand), \
         patch("multiquery_rag.implementation.pipeline.retrieve_multi", side_effect=_mock_retrieve_multi), \
         patch("multiquery_rag.implementation.pipeline.rrf_fuse", side_effect=_mock_fuse), \
         patch("multiquery_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER):
        from multiquery_rag.implementation.pipeline import run_multiquery_rag
        result = run_multiquery_rag("question?")
    assert "query_results" in result


def test_run_multiquery_rag_has_retrieved_docs_key():
    with patch("multiquery_rag.implementation.pipeline.expand_query", side_effect=_mock_expand), \
         patch("multiquery_rag.implementation.pipeline.retrieve_multi", side_effect=_mock_retrieve_multi), \
         patch("multiquery_rag.implementation.pipeline.rrf_fuse", side_effect=_mock_fuse), \
         patch("multiquery_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER):
        from multiquery_rag.implementation.pipeline import run_multiquery_rag
        result = run_multiquery_rag("question?")
    assert "retrieved_docs" in result


def test_run_multiquery_rag_query_matches_input():
    q = "What is the average delivery time?"
    with patch("multiquery_rag.implementation.pipeline.expand_query", side_effect=_mock_expand), \
         patch("multiquery_rag.implementation.pipeline.retrieve_multi", side_effect=_mock_retrieve_multi), \
         patch("multiquery_rag.implementation.pipeline.rrf_fuse", side_effect=_mock_fuse), \
         patch("multiquery_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER):
        from multiquery_rag.implementation.pipeline import run_multiquery_rag
        result = run_multiquery_rag(q)
    assert result["query"] == q


def test_run_multiquery_rag_answer_from_generator():
    with patch("multiquery_rag.implementation.pipeline.expand_query", side_effect=_mock_expand), \
         patch("multiquery_rag.implementation.pipeline.retrieve_multi", side_effect=_mock_retrieve_multi), \
         patch("multiquery_rag.implementation.pipeline.rrf_fuse", side_effect=_mock_fuse), \
         patch("multiquery_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER):
        from multiquery_rag.implementation.pipeline import run_multiquery_rag
        result = run_multiquery_rag("query")
    assert result["answer"] == _MOCK_ANSWER


def test_run_multiquery_rag_retrieved_docs_are_fused():
    with patch("multiquery_rag.implementation.pipeline.expand_query", side_effect=_mock_expand), \
         patch("multiquery_rag.implementation.pipeline.retrieve_multi", side_effect=_mock_retrieve_multi), \
         patch("multiquery_rag.implementation.pipeline.rrf_fuse", side_effect=_mock_fuse), \
         patch("multiquery_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER):
        from multiquery_rag.implementation.pipeline import run_multiquery_rag
        result = run_multiquery_rag("query")
    assert result["retrieved_docs"] == _FUSED_DOCS


def test_run_multiquery_rag_expanded_queries_is_list():
    with patch("multiquery_rag.implementation.pipeline.expand_query", side_effect=_mock_expand), \
         patch("multiquery_rag.implementation.pipeline.retrieve_multi", side_effect=_mock_retrieve_multi), \
         patch("multiquery_rag.implementation.pipeline.rrf_fuse", side_effect=_mock_fuse), \
         patch("multiquery_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER):
        from multiquery_rag.implementation.pipeline import run_multiquery_rag
        result = run_multiquery_rag("query")
    assert isinstance(result["expanded_queries"], list)


def test_run_multiquery_rag_passes_query_to_expander():
    with patch("multiquery_rag.implementation.pipeline.expand_query",
               side_effect=_mock_expand) as mock_exp, \
         patch("multiquery_rag.implementation.pipeline.retrieve_multi", side_effect=_mock_retrieve_multi), \
         patch("multiquery_rag.implementation.pipeline.rrf_fuse", side_effect=_mock_fuse), \
         patch("multiquery_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER):
        from multiquery_rag.implementation.pipeline import run_multiquery_rag
        run_multiquery_rag("specific question here")
    mock_exp.assert_called_once()
    assert mock_exp.call_args[0][0] == "specific question here"


def test_run_multiquery_rag_passes_expanded_queries_to_retriever():
    with patch("multiquery_rag.implementation.pipeline.expand_query", side_effect=_mock_expand), \
         patch("multiquery_rag.implementation.pipeline.retrieve_multi",
               side_effect=_mock_retrieve_multi) as mock_ret, \
         patch("multiquery_rag.implementation.pipeline.rrf_fuse", side_effect=_mock_fuse), \
         patch("multiquery_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER):
        from multiquery_rag.implementation.pipeline import run_multiquery_rag
        run_multiquery_rag("query")
    mock_ret.assert_called_once()
    called_queries = mock_ret.call_args[0][0]
    assert called_queries == _EXPANDED_Q


def test_run_multiquery_rag_passes_fused_docs_to_generator():
    with patch("multiquery_rag.implementation.pipeline.expand_query", side_effect=_mock_expand), \
         patch("multiquery_rag.implementation.pipeline.retrieve_multi", side_effect=_mock_retrieve_multi), \
         patch("multiquery_rag.implementation.pipeline.rrf_fuse", side_effect=_mock_fuse), \
         patch("multiquery_rag.implementation.pipeline.generate",
               return_value=_MOCK_ANSWER) as mock_gen:
        from multiquery_rag.implementation.pipeline import run_multiquery_rag
        run_multiquery_rag("query")
    mock_gen.assert_called_once()
    assert mock_gen.call_args[0][1] == _FUSED_DOCS


def test_run_multiquery_rag_query_results_keys_are_expanded_queries():
    with patch("multiquery_rag.implementation.pipeline.expand_query", side_effect=_mock_expand), \
         patch("multiquery_rag.implementation.pipeline.retrieve_multi", side_effect=_mock_retrieve_multi), \
         patch("multiquery_rag.implementation.pipeline.rrf_fuse", side_effect=_mock_fuse), \
         patch("multiquery_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER):
        from multiquery_rag.implementation.pipeline import run_multiquery_rag
        result = run_multiquery_rag("query")
    assert set(result["query_results"].keys()) == set(_EXPANDED_Q)


def test_run_multiquery_rag_retrieved_docs_at_most_final_top_k():
    with patch("multiquery_rag.implementation.pipeline.expand_query", side_effect=_mock_expand), \
         patch("multiquery_rag.implementation.pipeline.retrieve_multi", side_effect=_mock_retrieve_multi), \
         patch("multiquery_rag.implementation.pipeline.rrf_fuse", side_effect=_mock_fuse), \
         patch("multiquery_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER):
        from multiquery_rag.implementation.pipeline import run_multiquery_rag
        from multiquery_rag.implementation.config import FINAL_TOP_K
        result = run_multiquery_rag("query")
    assert len(result["retrieved_docs"]) <= FINAL_TOP_K
