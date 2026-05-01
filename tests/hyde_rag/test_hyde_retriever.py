"""Unit tests for hyde_rag.implementation.retriever."""
from unittest.mock import patch, MagicMock

_HYPO_DOC = "Electronics orders in SP had an average delivery of 8.3 days."


def test_retrieve_calls_generate_hypothetical_doc(mock_chroma_collection):
    with patch("hyde_rag.implementation.generator.generate_hypothetical_doc", return_value=_HYPO_DOC) as mock_hypo, \
         patch("hyde_rag.implementation.ingestion.get_collection", return_value=mock_chroma_collection):
        from hyde_rag.implementation.retriever import retrieve
        retrieve("What is delivery time in SP?")
    mock_hypo.assert_called_once_with("What is delivery time in SP?")


def test_retrieve_queries_chromadb_with_hypothetical_doc(mock_chroma_collection):
    with patch("hyde_rag.implementation.generator.generate_hypothetical_doc", return_value=_HYPO_DOC), \
         patch("hyde_rag.implementation.ingestion.get_collection", return_value=mock_chroma_collection):
        from hyde_rag.implementation.retriever import retrieve
        retrieve("query")
    mock_chroma_collection.query.assert_called_once_with(
        query_texts=[_HYPO_DOC],
        n_results=5,
    )


def test_retrieve_returns_dict():
    mock_col = MagicMock()
    mock_col.query.return_value = {
        "ids": [["doc_001"]], "documents": [["Some text."]],
        "metadatas": [[{"document_type": "order"}]], "distances": [[0.1]],
    }
    with patch("hyde_rag.implementation.generator.generate_hypothetical_doc", return_value=_HYPO_DOC), \
         patch("hyde_rag.implementation.ingestion.get_collection", return_value=mock_col):
        from hyde_rag.implementation.retriever import retrieve
        result = retrieve("question")
    assert isinstance(result, dict)


def test_retrieve_result_has_retrieved_docs_key(mock_chroma_collection):
    with patch("hyde_rag.implementation.generator.generate_hypothetical_doc", return_value=_HYPO_DOC), \
         patch("hyde_rag.implementation.ingestion.get_collection", return_value=mock_chroma_collection):
        from hyde_rag.implementation.retriever import retrieve
        result = retrieve("question")
    assert "retrieved_docs" in result


def test_retrieve_result_has_hypothetical_doc_key(mock_chroma_collection):
    with patch("hyde_rag.implementation.generator.generate_hypothetical_doc", return_value=_HYPO_DOC), \
         patch("hyde_rag.implementation.ingestion.get_collection", return_value=mock_chroma_collection):
        from hyde_rag.implementation.retriever import retrieve
        result = retrieve("question")
    assert "hypothetical_doc" in result


def test_retrieve_hypothetical_doc_matches_generated(mock_chroma_collection):
    with patch("hyde_rag.implementation.generator.generate_hypothetical_doc", return_value=_HYPO_DOC), \
         patch("hyde_rag.implementation.ingestion.get_collection", return_value=mock_chroma_collection):
        from hyde_rag.implementation.retriever import retrieve
        result = retrieve("question")
    assert result["hypothetical_doc"] == _HYPO_DOC


def test_retrieve_docs_have_required_keys(mock_chroma_collection):
    with patch("hyde_rag.implementation.generator.generate_hypothetical_doc", return_value=_HYPO_DOC), \
         patch("hyde_rag.implementation.ingestion.get_collection", return_value=mock_chroma_collection):
        from hyde_rag.implementation.retriever import retrieve
        result = retrieve("question")
    for doc in result["retrieved_docs"]:
        assert "id" in doc
        assert "text" in doc
        assert "metadata" in doc
        assert "distance" in doc


def test_retrieve_respects_top_k(mock_chroma_collection):
    with patch("hyde_rag.implementation.generator.generate_hypothetical_doc", return_value=_HYPO_DOC), \
         patch("hyde_rag.implementation.ingestion.get_collection", return_value=mock_chroma_collection):
        from hyde_rag.implementation.retriever import retrieve
        retrieve("question", top_k=3)
    mock_chroma_collection.query.assert_called_once_with(
        query_texts=[_HYPO_DOC], n_results=3
    )
