"""Unit tests for hybrid_rag.implementation.pipeline (end-to-end with mocks).

Verifies that ``run_hybrid_rag`` returns all five expected keys (query, answer,
retrieved_docs, keyword_docs, semantic_docs), routes values correctly from
retrieve and generate stubs, and passes the query through unchanged.
"""
from unittest.mock import patch

_ANSWER = "The most popular product category is Health & Beauty."
_FUSED   = [{"id": "d1", "text": "t1", "metadata": {}, "rrf_score": 0.032}]
_KEYWORD = [{"id": "d1", "text": "t1", "metadata": {}, "bm25_score": 3.5}]
_SEMANTIC= [{"id": "d1", "text": "t1", "metadata": {}, "distance": 0.1}]
_RETRIEVE_RESULT = {"fused": _FUSED, "keyword": _KEYWORD, "semantic": _SEMANTIC}


def test_run_hybrid_rag_returns_dict():
    """run_hybrid_rag must return a dict."""
    with patch("hybrid_rag.implementation.pipeline.retrieve", return_value=_RETRIEVE_RESULT), \
         patch("hybrid_rag.implementation.pipeline.generate", return_value=_ANSWER):
        from hybrid_rag.implementation.pipeline import run_hybrid_rag
        result = run_hybrid_rag("What is the top category?")
    assert isinstance(result, dict)


def test_run_hybrid_rag_has_query_key():
    """Result dict must contain the 'query' key."""
    with patch("hybrid_rag.implementation.pipeline.retrieve", return_value=_RETRIEVE_RESULT), \
         patch("hybrid_rag.implementation.pipeline.generate", return_value=_ANSWER):
        from hybrid_rag.implementation.pipeline import run_hybrid_rag
        result = run_hybrid_rag("query?")
    assert "query" in result


def test_run_hybrid_rag_has_answer_key():
    """Result dict must contain the 'answer' key."""
    with patch("hybrid_rag.implementation.pipeline.retrieve", return_value=_RETRIEVE_RESULT), \
         patch("hybrid_rag.implementation.pipeline.generate", return_value=_ANSWER):
        from hybrid_rag.implementation.pipeline import run_hybrid_rag
        result = run_hybrid_rag("query?")
    assert "answer" in result


def test_run_hybrid_rag_has_retrieved_docs_key():
    """Result dict must contain the 'retrieved_docs' key (the RRF-fused list)."""
    with patch("hybrid_rag.implementation.pipeline.retrieve", return_value=_RETRIEVE_RESULT), \
         patch("hybrid_rag.implementation.pipeline.generate", return_value=_ANSWER):
        from hybrid_rag.implementation.pipeline import run_hybrid_rag
        result = run_hybrid_rag("query?")
    assert "retrieved_docs" in result


def test_run_hybrid_rag_has_keyword_docs_key():
    """Result dict must contain the 'keyword_docs' key (BM25 candidates)."""
    with patch("hybrid_rag.implementation.pipeline.retrieve", return_value=_RETRIEVE_RESULT), \
         patch("hybrid_rag.implementation.pipeline.generate", return_value=_ANSWER):
        from hybrid_rag.implementation.pipeline import run_hybrid_rag
        result = run_hybrid_rag("query?")
    assert "keyword_docs" in result


def test_run_hybrid_rag_has_semantic_docs_key():
    """Result dict must contain the 'semantic_docs' key (ChromaDB candidates)."""
    with patch("hybrid_rag.implementation.pipeline.retrieve", return_value=_RETRIEVE_RESULT), \
         patch("hybrid_rag.implementation.pipeline.generate", return_value=_ANSWER):
        from hybrid_rag.implementation.pipeline import run_hybrid_rag
        result = run_hybrid_rag("query?")
    assert "semantic_docs" in result


def test_run_hybrid_rag_query_in_result():
    """result['query'] must be the exact string passed to run_hybrid_rag."""
    q = "What is the top category?"
    with patch("hybrid_rag.implementation.pipeline.retrieve", return_value=_RETRIEVE_RESULT), \
         patch("hybrid_rag.implementation.pipeline.generate", return_value=_ANSWER):
        from hybrid_rag.implementation.pipeline import run_hybrid_rag
        result = run_hybrid_rag(q)
    assert result["query"] == q


def test_run_hybrid_rag_answer_from_generator():
    """result['answer'] must equal the value returned by the generate stub."""
    with patch("hybrid_rag.implementation.pipeline.retrieve", return_value=_RETRIEVE_RESULT), \
         patch("hybrid_rag.implementation.pipeline.generate", return_value=_ANSWER):
        from hybrid_rag.implementation.pipeline import run_hybrid_rag
        result = run_hybrid_rag("query?")
    assert result["answer"] == _ANSWER


def test_run_hybrid_rag_retrieved_docs_are_fused():
    """result['retrieved_docs'] must equal the 'fused' list from the retrieve stub."""
    with patch("hybrid_rag.implementation.pipeline.retrieve", return_value=_RETRIEVE_RESULT), \
         patch("hybrid_rag.implementation.pipeline.generate", return_value=_ANSWER):
        from hybrid_rag.implementation.pipeline import run_hybrid_rag
        result = run_hybrid_rag("query?")
    assert result["retrieved_docs"] == _FUSED


def test_run_hybrid_rag_keyword_docs_separate():
    """result['keyword_docs'] must equal the 'keyword' list from the retrieve stub."""
    with patch("hybrid_rag.implementation.pipeline.retrieve", return_value=_RETRIEVE_RESULT), \
         patch("hybrid_rag.implementation.pipeline.generate", return_value=_ANSWER):
        from hybrid_rag.implementation.pipeline import run_hybrid_rag
        result = run_hybrid_rag("query?")
    assert result["keyword_docs"] == _KEYWORD


def test_run_hybrid_rag_passes_query_to_retrieve():
    """run_hybrid_rag must forward the query to retrieve with final_top_k=5."""
    with patch("hybrid_rag.implementation.pipeline.retrieve", return_value=_RETRIEVE_RESULT) as mock_ret, \
         patch("hybrid_rag.implementation.pipeline.generate", return_value=_ANSWER):
        from hybrid_rag.implementation.pipeline import run_hybrid_rag
        run_hybrid_rag("unique question for retriever")
    mock_ret.assert_called_once_with("unique question for retriever", final_top_k=5)
