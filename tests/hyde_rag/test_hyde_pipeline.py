"""Unit tests for hyde_rag.implementation.pipeline (end-to-end with mocks)."""
from unittest.mock import patch

_HYPO_DOC = "Electronics in SP averaged 8.3 days delivery."
_ANSWER = "The average delivery time for electronics in SP is 8.3 days."
_DOCS = [
    {"id": "doc_001", "text": "Order data.", "metadata": {"document_type": "order"}, "distance": 0.1},
    {"id": "doc_002", "text": "Product data.", "metadata": {"document_type": "product"}, "distance": 0.2},
]

_RETRIEVE_RESULT = {"retrieved_docs": _DOCS, "hypothetical_doc": _HYPO_DOC}


def test_run_hyde_rag_returns_dict():
    with patch("hyde_rag.implementation.pipeline.retrieve", return_value=_RETRIEVE_RESULT), \
         patch("hyde_rag.implementation.pipeline.generate", return_value=_ANSWER):
        from hyde_rag.implementation.pipeline import run_hyde_rag
        result = run_hyde_rag("What is delivery time?")
    assert isinstance(result, dict)


def test_run_hyde_rag_has_query_key():
    with patch("hyde_rag.implementation.pipeline.retrieve", return_value=_RETRIEVE_RESULT), \
         patch("hyde_rag.implementation.pipeline.generate", return_value=_ANSWER):
        from hyde_rag.implementation.pipeline import run_hyde_rag
        result = run_hyde_rag("question?")
    assert "query" in result


def test_run_hyde_rag_has_answer_key():
    with patch("hyde_rag.implementation.pipeline.retrieve", return_value=_RETRIEVE_RESULT), \
         patch("hyde_rag.implementation.pipeline.generate", return_value=_ANSWER):
        from hyde_rag.implementation.pipeline import run_hyde_rag
        result = run_hyde_rag("question?")
    assert "answer" in result


def test_run_hyde_rag_has_retrieved_docs_key():
    with patch("hyde_rag.implementation.pipeline.retrieve", return_value=_RETRIEVE_RESULT), \
         patch("hyde_rag.implementation.pipeline.generate", return_value=_ANSWER):
        from hyde_rag.implementation.pipeline import run_hyde_rag
        result = run_hyde_rag("question?")
    assert "retrieved_docs" in result


def test_run_hyde_rag_has_hypothetical_doc_key():
    with patch("hyde_rag.implementation.pipeline.retrieve", return_value=_RETRIEVE_RESULT), \
         patch("hyde_rag.implementation.pipeline.generate", return_value=_ANSWER):
        from hyde_rag.implementation.pipeline import run_hyde_rag
        result = run_hyde_rag("question?")
    assert "hypothetical_doc" in result


def test_run_hyde_rag_query_in_result():
    q = "What is the average delivery time?"
    with patch("hyde_rag.implementation.pipeline.retrieve", return_value=_RETRIEVE_RESULT), \
         patch("hyde_rag.implementation.pipeline.generate", return_value=_ANSWER):
        from hyde_rag.implementation.pipeline import run_hyde_rag
        result = run_hyde_rag(q)
    assert result["query"] == q


def test_run_hyde_rag_answer_from_generator():
    with patch("hyde_rag.implementation.pipeline.retrieve", return_value=_RETRIEVE_RESULT), \
         patch("hyde_rag.implementation.pipeline.generate", return_value=_ANSWER):
        from hyde_rag.implementation.pipeline import run_hyde_rag
        result = run_hyde_rag("question?")
    assert result["answer"] == _ANSWER


def test_run_hyde_rag_hypothetical_doc_from_retriever():
    with patch("hyde_rag.implementation.pipeline.retrieve", return_value=_RETRIEVE_RESULT), \
         patch("hyde_rag.implementation.pipeline.generate", return_value=_ANSWER):
        from hyde_rag.implementation.pipeline import run_hyde_rag
        result = run_hyde_rag("question?")
    assert result["hypothetical_doc"] == _HYPO_DOC


def test_run_hyde_rag_passes_query_to_retrieve():
    with patch("hyde_rag.implementation.pipeline.retrieve", return_value=_RETRIEVE_RESULT) as mock_ret, \
         patch("hyde_rag.implementation.pipeline.generate", return_value=_ANSWER):
        from hyde_rag.implementation.pipeline import run_hyde_rag
        run_hyde_rag("specific question")
    mock_ret.assert_called_once_with("specific question")
