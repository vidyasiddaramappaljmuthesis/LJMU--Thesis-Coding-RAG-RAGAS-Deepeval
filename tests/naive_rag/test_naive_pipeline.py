"""Unit tests for naive_rag.implementation.pipeline (end-to-end with mocks)."""
from unittest.mock import patch, MagicMock

_MOCK_ANSWER = "Average delivery time is 8 days based on SP orders."


def _mock_retrieve(query, top_k=5):
    return [
        {"id": "doc_001", "text": "Order delivered in 8 days.", "metadata": {"document_type": "order"}, "distance": 0.10},
        {"id": "doc_002", "text": "SP region data.", "metadata": {"document_type": "region"}, "distance": 0.22},
    ]


def test_run_rag_returns_dict():
    with patch("naive_rag.implementation.pipeline.retrieve", side_effect=_mock_retrieve), \
         patch("naive_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER):
        from naive_rag.implementation.pipeline import run_rag
        result = run_rag("What is average delivery time?")
    assert isinstance(result, dict)


def test_run_rag_has_query_key():
    with patch("naive_rag.implementation.pipeline.retrieve", side_effect=_mock_retrieve), \
         patch("naive_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER):
        from naive_rag.implementation.pipeline import run_rag
        result = run_rag("What is average delivery time?")
    assert "query" in result


def test_run_rag_has_answer_key():
    with patch("naive_rag.implementation.pipeline.retrieve", side_effect=_mock_retrieve), \
         patch("naive_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER):
        from naive_rag.implementation.pipeline import run_rag
        result = run_rag("What is average delivery time?")
    assert "answer" in result


def test_run_rag_has_retrieved_docs_key():
    with patch("naive_rag.implementation.pipeline.retrieve", side_effect=_mock_retrieve), \
         patch("naive_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER):
        from naive_rag.implementation.pipeline import run_rag
        result = run_rag("What is average delivery time?")
    assert "retrieved_docs" in result


def test_run_rag_query_matches_input():
    query = "What is average delivery time?"
    with patch("naive_rag.implementation.pipeline.retrieve", side_effect=_mock_retrieve), \
         patch("naive_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER):
        from naive_rag.implementation.pipeline import run_rag
        result = run_rag(query)
    assert result["query"] == query


def test_run_rag_answer_from_generator():
    with patch("naive_rag.implementation.pipeline.retrieve", side_effect=_mock_retrieve), \
         patch("naive_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER):
        from naive_rag.implementation.pipeline import run_rag
        result = run_rag("query")
    assert result["answer"] == _MOCK_ANSWER


def test_run_rag_docs_from_retriever():
    with patch("naive_rag.implementation.pipeline.retrieve", side_effect=_mock_retrieve), \
         patch("naive_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER):
        from naive_rag.implementation.pipeline import run_rag
        result = run_rag("query")
    assert result["retrieved_docs"] == _mock_retrieve("query")


def test_run_rag_passes_query_to_retrieve():
    with patch("naive_rag.implementation.pipeline.retrieve", side_effect=_mock_retrieve) as mock_ret, \
         patch("naive_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER):
        from naive_rag.implementation.pipeline import run_rag
        run_rag("specific question here")
    mock_ret.assert_called_once_with("specific question here", top_k=5)


def test_run_rag_passes_docs_to_generate():
    with patch("naive_rag.implementation.pipeline.retrieve", side_effect=_mock_retrieve), \
         patch("naive_rag.implementation.pipeline.generate", return_value=_MOCK_ANSWER) as mock_gen:
        from naive_rag.implementation.pipeline import run_rag
        run_rag("query")
    mock_gen.assert_called_once()
    args = mock_gen.call_args[0]
    assert args[1] == _mock_retrieve("query")  # docs passed to generate
