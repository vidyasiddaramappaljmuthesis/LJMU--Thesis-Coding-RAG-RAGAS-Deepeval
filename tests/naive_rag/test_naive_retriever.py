"""Unit tests for naive_rag.implementation.retriever."""
from unittest.mock import patch


def _make_chroma_result(n=3):
    return {
        "ids":       [[f"doc_{i:03d}" for i in range(n)]],
        "documents": [[f"Text {i}." for i in range(n)]],
        "metadatas": [[{"document_type": "order"} for _ in range(n)]],
        "distances": [[round(0.1 * (i + 1), 2) for i in range(n)]],
    }


def test_retrieve_returns_list(mock_chroma_collection):
    with patch("naive_rag.implementation.retriever.get_collection", return_value=mock_chroma_collection):
        from naive_rag.implementation.retriever import retrieve
        result = retrieve("what is the average delivery time?")
    assert isinstance(result, list)


def test_retrieve_returns_non_empty(mock_chroma_collection):
    with patch("naive_rag.implementation.retriever.get_collection", return_value=mock_chroma_collection):
        from naive_rag.implementation.retriever import retrieve
        result = retrieve("average delivery time?")
    assert len(result) > 0


def test_retrieve_each_doc_has_required_keys(mock_chroma_collection):
    with patch("naive_rag.implementation.retriever.get_collection", return_value=mock_chroma_collection):
        from naive_rag.implementation.retriever import retrieve
        result = retrieve("delivery time?")
    for doc in result:
        assert "id" in doc
        assert "text" in doc
        assert "metadata" in doc
        assert "distance" in doc


def test_retrieve_respects_top_k(mock_chroma_collection):
    mock_chroma_collection.query.return_value = _make_chroma_result(2)
    with patch("naive_rag.implementation.retriever.get_collection", return_value=mock_chroma_collection):
        from naive_rag.implementation.retriever import retrieve
        result = retrieve("query", top_k=2)
    mock_chroma_collection.query.assert_called_once_with(query_texts=["query"], n_results=2)


def test_retrieve_passes_query_text(mock_chroma_collection):
    with patch("naive_rag.implementation.retriever.get_collection", return_value=mock_chroma_collection):
        from naive_rag.implementation.retriever import retrieve
        retrieve("what is the best seller city?")
    call_kwargs = mock_chroma_collection.query.call_args[1]
    assert call_kwargs["query_texts"] == ["what is the best seller city?"]


def test_retrieve_distance_is_float(mock_chroma_collection):
    with patch("naive_rag.implementation.retriever.get_collection", return_value=mock_chroma_collection):
        from naive_rag.implementation.retriever import retrieve
        result = retrieve("test query")
    for doc in result:
        assert isinstance(doc["distance"], float)


def test_retrieve_ids_match_chroma_output(mock_chroma_collection):
    with patch("naive_rag.implementation.retriever.get_collection", return_value=mock_chroma_collection):
        from naive_rag.implementation.retriever import retrieve
        result = retrieve("test")
    returned_ids = [d["id"] for d in result]
    assert returned_ids == ["doc_001", "doc_002", "doc_003"]
