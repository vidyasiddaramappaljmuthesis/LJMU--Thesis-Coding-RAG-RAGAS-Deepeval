"""Unit tests for hybrid_rag.implementation.pipeline (end-to-end with mocks)."""
from unittest.mock import patch

_ANSWER = "The most popular product category is Health & Beauty."
_FUSED   = [{"id": "d1", "text": "t1", "metadata": {}, "rrf_score": 0.032}]
_KEYWORD = [{"id": "d1", "text": "t1", "metadata": {}, "bm25_score": 3.5}]
_SEMANTIC= [{"id": "d1", "text": "t1", "metadata": {}, "distance": 0.1}]
_RETRIEVE_RESULT = {"fused": _FUSED, "keyword": _KEYWORD, "semantic": _SEMANTIC}


def test_run_hybrid_rag_returns_dict():
    with patch("hybrid_rag.implementation.pipeline.retrieve", return_value=_RETRIEVE_RESULT), \
         patch("hybrid_rag.implementation.pipeline.generate", return_value=_ANSWER):
        from hybrid_rag.implementation.pipeline import run_hybrid_rag
        result = run_hybrid_rag("What is the top category?")
    assert isinstance(result, dict)


def test_run_hybrid_rag_has_query_key():
    with patch("hybrid_rag.implementation.pipeline.retrieve", return_value=_RETRIEVE_RESULT), \
         patch("hybrid_rag.implementation.pipeline.generate", return_value=_ANSWER):
        from hybrid_rag.implementation.pipeline import run_hybrid_rag
        result = run_hybrid_rag("query?")
    assert "query" in result


def test_run_hybrid_rag_has_answer_key():
    with patch("hybrid_rag.implementation.pipeline.retrieve", return_value=_RETRIEVE_RESULT), \
         patch("hybrid_rag.implementation.pipeline.generate", return_value=_ANSWER):
        from hybrid_rag.implementation.pipeline import run_hybrid_rag
        result = run_hybrid_rag("query?")
    assert "answer" in result


def test_run_hybrid_rag_has_retrieved_docs_key():
    with patch("hybrid_rag.implementation.pipeline.retrieve", return_value=_RETRIEVE_RESULT), \
         patch("hybrid_rag.implementation.pipeline.generate", return_value=_ANSWER):
        from hybrid_rag.implementation.pipeline import run_hybrid_rag
        result = run_hybrid_rag("query?")
    assert "retrieved_docs" in result


def test_run_hybrid_rag_has_keyword_docs_key():
    with patch("hybrid_rag.implementation.pipeline.retrieve", return_value=_RETRIEVE_RESULT), \
         patch("hybrid_rag.implementation.pipeline.generate", return_value=_ANSWER):
        from hybrid_rag.implementation.pipeline import run_hybrid_rag
        result = run_hybrid_rag("query?")
    assert "keyword_docs" in result


def test_run_hybrid_rag_has_semantic_docs_key():
    with patch("hybrid_rag.implementation.pipeline.retrieve", return_value=_RETRIEVE_RESULT), \
         patch("hybrid_rag.implementation.pipeline.generate", return_value=_ANSWER):
        from hybrid_rag.implementation.pipeline import run_hybrid_rag
        result = run_hybrid_rag("query?")
    assert "semantic_docs" in result


def test_run_hybrid_rag_query_in_result():
    q = "What is the top category?"
    with patch("hybrid_rag.implementation.pipeline.retrieve", return_value=_RETRIEVE_RESULT), \
         patch("hybrid_rag.implementation.pipeline.generate", return_value=_ANSWER):
        from hybrid_rag.implementation.pipeline import run_hybrid_rag
        result = run_hybrid_rag(q)
    assert result["query"] == q


def test_run_hybrid_rag_answer_from_generator():
    with patch("hybrid_rag.implementation.pipeline.retrieve", return_value=_RETRIEVE_RESULT), \
         patch("hybrid_rag.implementation.pipeline.generate", return_value=_ANSWER):
        from hybrid_rag.implementation.pipeline import run_hybrid_rag
        result = run_hybrid_rag("query?")
    assert result["answer"] == _ANSWER


def test_run_hybrid_rag_retrieved_docs_are_fused():
    with patch("hybrid_rag.implementation.pipeline.retrieve", return_value=_RETRIEVE_RESULT), \
         patch("hybrid_rag.implementation.pipeline.generate", return_value=_ANSWER):
        from hybrid_rag.implementation.pipeline import run_hybrid_rag
        result = run_hybrid_rag("query?")
    assert result["retrieved_docs"] == _FUSED


def test_run_hybrid_rag_keyword_docs_separate():
    with patch("hybrid_rag.implementation.pipeline.retrieve", return_value=_RETRIEVE_RESULT), \
         patch("hybrid_rag.implementation.pipeline.generate", return_value=_ANSWER):
        from hybrid_rag.implementation.pipeline import run_hybrid_rag
        result = run_hybrid_rag("query?")
    assert result["keyword_docs"] == _KEYWORD


def test_run_hybrid_rag_passes_query_to_retrieve():
    with patch("hybrid_rag.implementation.pipeline.retrieve", return_value=_RETRIEVE_RESULT) as mock_ret, \
         patch("hybrid_rag.implementation.pipeline.generate", return_value=_ANSWER):
        from hybrid_rag.implementation.pipeline import run_hybrid_rag
        run_hybrid_rag("unique question for retriever")
    mock_ret.assert_called_once_with("unique question for retriever", final_top_k=5)
