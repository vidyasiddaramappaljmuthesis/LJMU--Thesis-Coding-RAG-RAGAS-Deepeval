"""Unit tests for reranking_rag.implementation.pipeline (end-to-end with mocks).

Verifies that ``run_reranking_rag`` wires all three stages correctly:
1. retrieve_initial is called with the query and top_n=20.
2. rerank receives the full initial candidate list.
3. generate receives the reranked list (not the initial list).
Result dict must contain query, answer, retrieved_docs (reranked), and initial_docs.
"""
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
    """Stub retrieve_initial returning the fixed 20-doc initial candidate list."""
    return _INITIAL_DOCS


def _mock_rerank(query, docs, top_k=5):
    """Stub rerank returning the fixed 5-doc reranked list."""
    return _RERANKED_DOCS


def test_run_reranking_rag_returns_dict():
    """run_reranking_rag must return a dict."""
    with patch("reranking_rag.implementation.pipeline.retrieve_initial", side_effect=_mock_retrieve), \
         patch("reranking_rag.implementation.pipeline.rerank", side_effect=_mock_rerank), \
         patch("reranking_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER):
        from reranking_rag.implementation.pipeline import run_reranking_rag
        result = run_reranking_rag("What is average delivery time?")
    assert isinstance(result, dict)


def test_run_reranking_rag_has_query_key():
    """Result dict must contain the 'query' key."""
    with patch("reranking_rag.implementation.pipeline.retrieve_initial", side_effect=_mock_retrieve), \
         patch("reranking_rag.implementation.pipeline.rerank", side_effect=_mock_rerank), \
         patch("reranking_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER):
        from reranking_rag.implementation.pipeline import run_reranking_rag
        result = run_reranking_rag("question?")
    assert "query" in result


def test_run_reranking_rag_has_answer_key():
    """Result dict must contain the 'answer' key."""
    with patch("reranking_rag.implementation.pipeline.retrieve_initial", side_effect=_mock_retrieve), \
         patch("reranking_rag.implementation.pipeline.rerank", side_effect=_mock_rerank), \
         patch("reranking_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER):
        from reranking_rag.implementation.pipeline import run_reranking_rag
        result = run_reranking_rag("question?")
    assert "answer" in result


def test_run_reranking_rag_has_retrieved_docs_key():
    """Result dict must contain the 'retrieved_docs' key (the reranked list)."""
    with patch("reranking_rag.implementation.pipeline.retrieve_initial", side_effect=_mock_retrieve), \
         patch("reranking_rag.implementation.pipeline.rerank", side_effect=_mock_rerank), \
         patch("reranking_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER):
        from reranking_rag.implementation.pipeline import run_reranking_rag
        result = run_reranking_rag("question?")
    assert "retrieved_docs" in result


def test_run_reranking_rag_has_initial_docs_key():
    """Result dict must contain the 'initial_docs' key (Stage 1 candidate set)."""
    with patch("reranking_rag.implementation.pipeline.retrieve_initial", side_effect=_mock_retrieve), \
         patch("reranking_rag.implementation.pipeline.rerank", side_effect=_mock_rerank), \
         patch("reranking_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER):
        from reranking_rag.implementation.pipeline import run_reranking_rag
        result = run_reranking_rag("question?")
    assert "initial_docs" in result


def test_run_reranking_rag_query_matches_input():
    """result['query'] must be the exact string passed to run_reranking_rag."""
    q = "What is average delivery time?"
    with patch("reranking_rag.implementation.pipeline.retrieve_initial", side_effect=_mock_retrieve), \
         patch("reranking_rag.implementation.pipeline.rerank", side_effect=_mock_rerank), \
         patch("reranking_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER):
        from reranking_rag.implementation.pipeline import run_reranking_rag
        result = run_reranking_rag(q)
    assert result["query"] == q


def test_run_reranking_rag_answer_from_generator():
    """result['answer'] must equal the value returned by the generate stub."""
    with patch("reranking_rag.implementation.pipeline.retrieve_initial", side_effect=_mock_retrieve), \
         patch("reranking_rag.implementation.pipeline.rerank", side_effect=_mock_rerank), \
         patch("reranking_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER):
        from reranking_rag.implementation.pipeline import run_reranking_rag
        result = run_reranking_rag("query")
    assert result["answer"] == _MOCK_ANSWER


def test_run_reranking_rag_retrieved_docs_are_reranked():
    """retrieved_docs must be the reranked list, not the initial Stage 1 list."""
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
    """initial_docs must contain more candidates than the final retrieved_docs."""
    with patch("reranking_rag.implementation.pipeline.retrieve_initial", side_effect=_mock_retrieve), \
         patch("reranking_rag.implementation.pipeline.rerank", side_effect=_mock_rerank), \
         patch("reranking_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER):
        from reranking_rag.implementation.pipeline import run_reranking_rag
        result = run_reranking_rag("query")
    assert len(result["initial_docs"]) > len(result["retrieved_docs"])


def test_run_reranking_rag_passes_query_to_retrieve():
    """run_reranking_rag must forward the query to retrieve_initial with top_n=20."""
    with patch("reranking_rag.implementation.pipeline.retrieve_initial",
               side_effect=_mock_retrieve) as mock_ret, \
         patch("reranking_rag.implementation.pipeline.rerank", side_effect=_mock_rerank), \
         patch("reranking_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER):
        from reranking_rag.implementation.pipeline import run_reranking_rag
        run_reranking_rag("specific question here")
    mock_ret.assert_called_once_with("specific question here", top_n=20)


def test_run_reranking_rag_passes_initial_docs_to_reranker():
    """rerank must receive the full initial candidate list as its second argument."""
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
    """generate must receive the reranked list (not the initial list) as context."""
    with patch("reranking_rag.implementation.pipeline.retrieve_initial", side_effect=_mock_retrieve), \
         patch("reranking_rag.implementation.pipeline.rerank", side_effect=_mock_rerank), \
         patch("reranking_rag.implementation.pipeline.generate",
               return_value=_MOCK_ANSWER) as mock_gen:
        from reranking_rag.implementation.pipeline import run_reranking_rag
        run_reranking_rag("query")
    mock_gen.assert_called_once()
    assert mock_gen.call_args[0][1] == _RERANKED_DOCS
