"""Unit tests for naive_rag.implementation.pipeline (end-to-end with mocks).

Verifies the contract of ``run_rag``: correct return shape, key presence,
value passthrough, and that retrieve/generate are wired together properly.
All Groq and ChromaDB calls are replaced with lightweight stubs.
"""
from unittest.mock import patch, MagicMock

_MOCK_ANSWER = "Average delivery time is 8 days based on SP orders."


def _mock_retrieve(query, top_k=5):
    """Stub retriever returning two fixed documents regardless of the query."""
    return [
        {"id": "doc_001", "text": "Order delivered in 8 days.", "metadata": {"document_type": "order"}, "distance": 0.10},
        {"id": "doc_002", "text": "SP region data.", "metadata": {"document_type": "region"}, "distance": 0.22},
    ]


def test_run_rag_returns_dict():
    """run_rag must return a dict (not a list, string, or other type)."""
    with patch("naive_rag.implementation.pipeline.retrieve", side_effect=_mock_retrieve), \
         patch("naive_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER):
        from naive_rag.implementation.pipeline import run_rag
        result = run_rag("What is average delivery time?")
    assert isinstance(result, dict)


def test_run_rag_has_query_key():
    """Result dict must contain the 'query' key."""
    with patch("naive_rag.implementation.pipeline.retrieve", side_effect=_mock_retrieve), \
         patch("naive_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER):
        from naive_rag.implementation.pipeline import run_rag
        result = run_rag("What is average delivery time?")
    assert "query" in result


def test_run_rag_has_answer_key():
    """Result dict must contain the 'answer' key."""
    with patch("naive_rag.implementation.pipeline.retrieve", side_effect=_mock_retrieve), \
         patch("naive_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER):
        from naive_rag.implementation.pipeline import run_rag
        result = run_rag("What is average delivery time?")
    assert "answer" in result


def test_run_rag_has_retrieved_docs_key():
    """Result dict must contain the 'retrieved_docs' key."""
    with patch("naive_rag.implementation.pipeline.retrieve", side_effect=_mock_retrieve), \
         patch("naive_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER):
        from naive_rag.implementation.pipeline import run_rag
        result = run_rag("What is average delivery time?")
    assert "retrieved_docs" in result


def test_run_rag_query_matches_input():
    """result['query'] must be the exact string passed to run_rag."""
    query = "What is average delivery time?"
    with patch("naive_rag.implementation.pipeline.retrieve", side_effect=_mock_retrieve), \
         patch("naive_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER):
        from naive_rag.implementation.pipeline import run_rag
        result = run_rag(query)
    assert result["query"] == query


def test_run_rag_answer_from_generator():
    """result['answer'] must be the value returned by the generate stub."""
    with patch("naive_rag.implementation.pipeline.retrieve", side_effect=_mock_retrieve), \
         patch("naive_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER):
        from naive_rag.implementation.pipeline import run_rag
        result = run_rag("query")
    assert result["answer"] == _MOCK_ANSWER


def test_run_rag_docs_from_retriever():
    """result['retrieved_docs'] must equal the list returned by the retrieve stub."""
    with patch("naive_rag.implementation.pipeline.retrieve", side_effect=_mock_retrieve), \
         patch("naive_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER):
        from naive_rag.implementation.pipeline import run_rag
        result = run_rag("query")
    assert result["retrieved_docs"] == _mock_retrieve("query")


def test_run_rag_passes_query_to_retrieve():
    """run_rag must forward the query string to retrieve with top_k=5."""
    with patch("naive_rag.implementation.pipeline.retrieve", side_effect=_mock_retrieve) as mock_ret, \
         patch("naive_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER):
        from naive_rag.implementation.pipeline import run_rag
        run_rag("specific question here")
    mock_ret.assert_called_once_with("specific question here", top_k=5)


def test_run_rag_passes_docs_to_generate():
    """run_rag must pass the retrieved docs list as the second positional arg to generate."""
    with patch("naive_rag.implementation.pipeline.retrieve", side_effect=_mock_retrieve), \
         patch("naive_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER) as mock_gen:
        from naive_rag.implementation.pipeline import run_rag
        run_rag("query")
    mock_gen.assert_called_once()
    args = mock_gen.call_args[0]
    assert args[1] == _mock_retrieve("query")
