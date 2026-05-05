"""Unit tests for hyde_rag.implementation.pipeline (end-to-end with mocks).

Verifies that ``run_hyde_rag`` wires retrieve and generate correctly, returns
all four expected keys (query, answer, retrieved_docs, hypothetical_doc), and
passes arguments through without mutation.
"""
from unittest.mock import patch

_HYPO_DOC = "Electronics in SP averaged 8.3 days delivery."
_ANSWER = "The average delivery time for electronics in SP is 8.3 days."
_DOCS = [
    {"id": "doc_001", "text": "Order data.", "metadata": {"document_type": "order"}, "distance": 0.1},
    {"id": "doc_002", "text": "Product data.", "metadata": {"document_type": "product"}, "distance": 0.2},
]

_RETRIEVE_RESULT = {"retrieved_docs": _DOCS, "hypothetical_doc": _HYPO_DOC}


def test_run_hyde_rag_returns_dict():
    """run_hyde_rag must return a dict."""
    with patch("hyde_rag.implementation.pipeline.retrieve", return_value=_RETRIEVE_RESULT), \
         patch("hyde_rag.implementation.pipeline.generate", return_value=_ANSWER):
        from hyde_rag.implementation.pipeline import run_hyde_rag
        result = run_hyde_rag("What is delivery time?")
    assert isinstance(result, dict)


def test_run_hyde_rag_has_query_key():
    """Result dict must contain the 'query' key."""
    with patch("hyde_rag.implementation.pipeline.retrieve", return_value=_RETRIEVE_RESULT), \
         patch("hyde_rag.implementation.pipeline.generate", return_value=_ANSWER):
        from hyde_rag.implementation.pipeline import run_hyde_rag
        result = run_hyde_rag("question?")
    assert "query" in result


def test_run_hyde_rag_has_answer_key():
    """Result dict must contain the 'answer' key."""
    with patch("hyde_rag.implementation.pipeline.retrieve", return_value=_RETRIEVE_RESULT), \
         patch("hyde_rag.implementation.pipeline.generate", return_value=_ANSWER):
        from hyde_rag.implementation.pipeline import run_hyde_rag
        result = run_hyde_rag("question?")
    assert "answer" in result


def test_run_hyde_rag_has_retrieved_docs_key():
    """Result dict must contain the 'retrieved_docs' key."""
    with patch("hyde_rag.implementation.pipeline.retrieve", return_value=_RETRIEVE_RESULT), \
         patch("hyde_rag.implementation.pipeline.generate", return_value=_ANSWER):
        from hyde_rag.implementation.pipeline import run_hyde_rag
        result = run_hyde_rag("question?")
    assert "retrieved_docs" in result


def test_run_hyde_rag_has_hypothetical_doc_key():
    """Result dict must contain the 'hypothetical_doc' key unique to HyDE."""
    with patch("hyde_rag.implementation.pipeline.retrieve", return_value=_RETRIEVE_RESULT), \
         patch("hyde_rag.implementation.pipeline.generate", return_value=_ANSWER):
        from hyde_rag.implementation.pipeline import run_hyde_rag
        result = run_hyde_rag("question?")
    assert "hypothetical_doc" in result


def test_run_hyde_rag_query_in_result():
    """result['query'] must be the exact string passed to run_hyde_rag."""
    q = "What is the average delivery time?"
    with patch("hyde_rag.implementation.pipeline.retrieve", return_value=_RETRIEVE_RESULT), \
         patch("hyde_rag.implementation.pipeline.generate", return_value=_ANSWER):
        from hyde_rag.implementation.pipeline import run_hyde_rag
        result = run_hyde_rag(q)
    assert result["query"] == q


def test_run_hyde_rag_answer_from_generator():
    """result['answer'] must equal the value returned by the generate stub."""
    with patch("hyde_rag.implementation.pipeline.retrieve", return_value=_RETRIEVE_RESULT), \
         patch("hyde_rag.implementation.pipeline.generate", return_value=_ANSWER):
        from hyde_rag.implementation.pipeline import run_hyde_rag
        result = run_hyde_rag("question?")
    assert result["answer"] == _ANSWER


def test_run_hyde_rag_hypothetical_doc_from_retriever():
    """result['hypothetical_doc'] must equal the value from the retrieve stub."""
    with patch("hyde_rag.implementation.pipeline.retrieve", return_value=_RETRIEVE_RESULT), \
         patch("hyde_rag.implementation.pipeline.generate", return_value=_ANSWER):
        from hyde_rag.implementation.pipeline import run_hyde_rag
        result = run_hyde_rag("question?")
    assert result["hypothetical_doc"] == _HYPO_DOC


def test_run_hyde_rag_passes_query_to_retrieve():
    """run_hyde_rag must forward the query to retrieve with top_k=5."""
    with patch("hyde_rag.implementation.pipeline.retrieve", return_value=_RETRIEVE_RESULT) as mock_ret, \
         patch("hyde_rag.implementation.pipeline.generate", return_value=_ANSWER):
        from hyde_rag.implementation.pipeline import run_hyde_rag
        run_hyde_rag("specific question")
    mock_ret.assert_called_once_with("specific question", top_k=5)
