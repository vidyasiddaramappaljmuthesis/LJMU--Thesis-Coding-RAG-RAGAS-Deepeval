"""Unit tests for reranking_rag.implementation.pipeline (end-to-end with mocks)."""
from unittest.mock import patch, MagicMock
import numpy as np

_MOCK_ANSWER  = "Average delivery time is 8 days based on SP orders."
_INITIAL_DOCS = [
    {"id": f"doc_{i:03d}", "text": f"Candidate doc {i}.",
     "metadata": {"document_type": "order"}, "distance": round(0.05 * i, 4)}
    for i in range(1, 21)
]
_RERANKED_DOCS = [
    {"id": "doc_019", "text": "Candidate doc 19.", "metadata": {"document_type": "order"},
     "distance": 0.95, "rerank_score": 9.5},
    {"id": "doc_018", "text": "Candidate doc 18.", "metadata": {"document_type": "order"},
     "distance": 0.90, "rerank_score": 8.8},
    {"id": "doc_017", "text": "Candidate doc 17.", "metadata": {"document_type": "order"},
     "distance": 0.85, "rerank_score": 7.2},
    {"id": "doc_016", "text": "Candidate doc 16.", "metadata": {"document_type": "order"},
     "distance": 0.80, "rerank_score": 6.1},
    {"id": "doc_015", "text": "Candidate doc 15.", "metadata": {"document_type": "order"},
     "distance": 0.75, "rerank_score": 5.0},
]


def _mock_retrieve(query, top_n=20):
    return _INITIAL_DOCS


def _mock_rerank(query, docs, top_k=5):
    return _RERANKED_DOCS


def test_run_reranking_rag_returns_dict():
    with patch("reranking_rag.implementation.pipeline.retrieve_initial", side_effect=_mock_retrieve), \
         patch("reranking_rag.implementation.pipeline.rerank", side_effect=_mock_rerank), \
         patch("reranking_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER):
        from reranking_rag.implementation.pipeline import run_reranking_rag
        result = run_reranking_rag("What is average delivery time?")
    assert isinstance(result, dict)


def test_run_reranking_rag_has_query_key():
    with patch("reranking_rag.implementation.pipeline.retrieve_initial", side_effect=_mock_retrieve), \
         patch("reranking_rag.implementation.pipeline.rerank", side_effect=_mock_rerank), \
         patch("reranking_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER):
        from reranking_rag.implementation.pipeline import run_reranking_rag
        result = run_reranking_rag("question?")
    assert "query" in result


def test_run_reranking_rag_has_answer_key():
    with patch("reranking_rag.implementation.pipeline.retrieve_initial", side_effect=_mock_retrieve), \
         patch("reranking_rag.implementation.pipeline.rerank", side_effect=_mock_rerank), \
         patch("reranking_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER):
        from reranking_rag.implementation.pipeline import run_reranking_rag
        result = run_reranking_rag("question?")
    assert "answer" in result


def test_run_reranking_rag_has_retrieved_docs_key():
    with patch("reranking_rag.implementation.pipeline.retrieve_initial", side_effect=_mock_retrieve), \
         patch("reranking_rag.implementation.pipeline.rerank", side_effect=_mock_rerank), \
         patch("reranking_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER):
        from reranking_rag.implementation.pipeline import run_reranking_rag
        result = run_reranking_rag("question?")
    assert "retrieved_docs" in result


def test_run_reranking_rag_has_initial_docs_key():
    with patch("reranking_rag.implementation.pipeline.retrieve_initial", side_effect=_mock_retrieve), \
         patch("reranking_rag.implementation.pipeline.rerank", side_effect=_mock_rerank), \
         patch("reranking_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER):
        from reranking_rag.implementation.pipeline import run_reranking_rag
        result = run_reranking_rag("question?")
    assert "initial_docs" in result


def test_run_reranking_rag_query_matches_input():
    q = "What is average delivery time?"
    with patch("reranking_rag.implementation.pipeline.retrieve_initial", side_effect=_mock_retrieve), \
         patch("reranking_rag.implementation.pipeline.rerank", side_effect=_mock_rerank), \
         patch("reranking_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER):
        from reranking_rag.implementation.pipeline import run_reranking_rag
        result = run_reranking_rag(q)
    assert result["query"] == q


def test_run_reranking_rag_answer_from_generator():
    with patch("reranking_rag.implementation.pipeline.retrieve_initial", side_effect=_mock_retrieve), \
         patch("reranking_rag.implementation.pipeline.rerank", side_effect=_mock_rerank), \
         patch("reranking_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER):
        from reranking_rag.implementation.pipeline import run_reranking_rag
        result = run_reranking_rag("query")
    assert result["answer"] == _MOCK_ANSWER


def test_run_reranking_rag_retrieved_docs_are_reranked():
    """retrieved_docs must be the reranked list, not the initial list."""
    with patch("reranking_rag.implementation.pipeline.retrieve_initial", side_effect=_mock_retrieve), \
         patch("reranking_rag.implementation.pipeline.rerank", side_effect=_mock_rerank), \
         patch("reranking_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER):
        from reranking_rag.implementation.pipeline import run_reranking_rag
        result = run_reranking_rag("query")
    assert result["retrieved_docs"] == _RERANKED_DOCS


def test_run_reranking_rag_initial_docs_are_full_candidate_set():
    """initial_docs must be the unfiltered candidate list from retrieve_initial."""
    with patch("reranking_rag.implementation.pipeline.retrieve_initial", side_effect=_mock_retrieve), \
         patch("reranking_rag.implementation.pipeline.rerank", side_effect=_mock_rerank), \
         patch("reranking_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER):
        from reranking_rag.implementation.pipeline import run_reranking_rag
        result = run_reranking_rag("query")
    assert result["initial_docs"] == _INITIAL_DOCS


def test_run_reranking_rag_initial_larger_than_retrieved():
    """initial_docs should have more candidates than retrieved_docs."""
    with patch("reranking_rag.implementation.pipeline.retrieve_initial", side_effect=_mock_retrieve), \
         patch("reranking_rag.implementation.pipeline.rerank", side_effect=_mock_rerank), \
         patch("reranking_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER):
        from reranking_rag.implementation.pipeline import run_reranking_rag
        result = run_reranking_rag("query")
    assert len(result["initial_docs"]) > len(result["retrieved_docs"])


def test_run_reranking_rag_passes_query_to_retrieve():
    with patch("reranking_rag.implementation.pipeline.retrieve_initial",
               side_effect=_mock_retrieve) as mock_ret, \
         patch("reranking_rag.implementation.pipeline.rerank", side_effect=_mock_rerank), \
         patch("reranking_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER):
        from reranking_rag.implementation.pipeline import run_reranking_rag
        run_reranking_rag("specific question here")
    mock_ret.assert_called_once_with("specific question here", top_n=20)


def test_run_reranking_rag_passes_initial_docs_to_reranker():
    with patch("reranking_rag.implementation.pipeline.retrieve_initial",
               side_effect=_mock_retrieve), \
         patch("reranking_rag.implementation.pipeline.rerank",
               side_effect=_mock_rerank) as mock_rerank, \
         patch("reranking_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER):
        from reranking_rag.implementation.pipeline import run_reranking_rag
        run_reranking_rag("query")
    call_args = mock_rerank.call_args
    assert call_args[0][1] == _INITIAL_DOCS


def test_run_reranking_rag_passes_reranked_docs_to_generator():
    with patch("reranking_rag.implementation.pipeline.retrieve_initial", side_effect=_mock_retrieve), \
         patch("reranking_rag.implementation.pipeline.rerank", side_effect=_mock_rerank), \
         patch("reranking_rag.implementation.pipeline.generate",
               return_value=_MOCK_ANSWER) as mock_gen:
        from reranking_rag.implementation.pipeline import run_reranking_rag
        run_reranking_rag("query")
    mock_gen.assert_called_once()
    assert mock_gen.call_args[0][1] == _RERANKED_DOCS
