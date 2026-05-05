"""Unit tests for naive_rag.implementation.retriever.

Verifies that ``retrieve`` queries ChromaDB correctly, maps the raw response
into the expected dict shape (id, text, metadata, distance), respects the
``top_k`` parameter, and preserves document order from ChromaDB output.
"""
from unittest.mock import patch


def _make_chroma_result(n=3):
    """Build a synthetic ChromaDB query result with *n* documents."""
    return {
        "ids":       [[f"doc_{i:03d}" for i in range(n)]],
        "documents": [[f"Text {i}." for i in range(n)]],
        "metadatas": [[{"document_type": "order"} for _ in range(n)]],
        "distances": [[round(0.1 * (i + 1), 2) for i in range(n)]],
    }


def test_retrieve_returns_list(mock_chroma_collection):
    """retrieve must return a list (not a dict or generator)."""
    with patch("naive_rag.implementation.retriever.get_collection", return_value=mock_chroma_collection):
        from naive_rag.implementation.retriever import retrieve
        result = retrieve("what is the average delivery time?")
    assert isinstance(result, list)


def test_retrieve_returns_non_empty(mock_chroma_collection):
    """retrieve must return at least one document for a valid query."""
    with patch("naive_rag.implementation.retriever.get_collection", return_value=mock_chroma_collection):
        from naive_rag.implementation.retriever import retrieve
        result = retrieve("average delivery time?")
    assert len(result) > 0


def test_retrieve_each_doc_has_required_keys(mock_chroma_collection):
    """Every document in the result must have 'id', 'text', 'metadata', and 'distance' keys."""
    with patch("naive_rag.implementation.retriever.get_collection", return_value=mock_chroma_collection):
        from naive_rag.implementation.retriever import retrieve
        result = retrieve("delivery time?")
    for doc in result:
        assert "id" in doc
        assert "text" in doc
        assert "metadata" in doc
        assert "distance" in doc


def test_retrieve_respects_top_k(mock_chroma_collection):
    """retrieve must pass n_results=top_k to the ChromaDB query call."""
    mock_chroma_collection.query.return_value = _make_chroma_result(2)
    with patch("naive_rag.implementation.retriever.get_collection", return_value=mock_chroma_collection):
        from naive_rag.implementation.retriever import retrieve
        retrieve("query", top_k=2)
    mock_chroma_collection.query.assert_called_once_with(query_texts=["query"], n_results=2)


def test_retrieve_passes_query_text(mock_chroma_collection):
    """retrieve must pass the query string as a single-element list to ChromaDB."""
    with patch("naive_rag.implementation.retriever.get_collection", return_value=mock_chroma_collection):
        from naive_rag.implementation.retriever import retrieve
        retrieve("what is the best seller city?")
    call_kwargs = mock_chroma_collection.query.call_args[1]
    assert call_kwargs["query_texts"] == ["what is the best seller city?"]


def test_retrieve_distance_is_float(mock_chroma_collection):
    """The 'distance' value in each result doc must be a Python float."""
    with patch("naive_rag.implementation.retriever.get_collection", return_value=mock_chroma_collection):
        from naive_rag.implementation.retriever import retrieve
        result = retrieve("test query")
    for doc in result:
        assert isinstance(doc["distance"], float)


def test_retrieve_ids_match_chroma_output(mock_chroma_collection):
    """Document IDs must be returned in the same order as ChromaDB's output."""
    with patch("naive_rag.implementation.retriever.get_collection", return_value=mock_chroma_collection):
        from naive_rag.implementation.retriever import retrieve
        result = retrieve("test")
    returned_ids = [d["id"] for d in result]
    assert returned_ids == ["doc_001", "doc_002", "doc_003"]
