"""Unit tests for reranking_rag.implementation.retriever."""
from unittest.mock import patch


def _make_chroma_result(n=20):
    return {
        "ids":       [[f"doc_{i:03d}" for i in range(n)]],
        "documents": [[f"Text {i}." for i in range(n)]],
        "metadatas": [[{"document_type": "order"} for _ in range(n)]],
        "distances": [[round(0.05 * (i + 1), 4) for i in range(n)]],
    }


def test_retrieve_initial_returns_list(mock_chroma_collection):
    with patch("reranking_rag.implementation.retriever.get_collection",
               return_value=mock_chroma_collection):
        from reranking_rag.implementation.retriever import retrieve_initial
        result = retrieve_initial("what is the average delivery time?")
    assert isinstance(result, list)


def test_retrieve_initial_returns_non_empty(mock_chroma_collection):
    with patch("reranking_rag.implementation.retriever.get_collection",
               return_value=mock_chroma_collection):
        from reranking_rag.implementation.retriever import retrieve_initial
        result = retrieve_initial("average delivery time?")
    assert len(result) > 0


def test_retrieve_initial_each_doc_has_required_keys(mock_chroma_collection):
    with patch("reranking_rag.implementation.retriever.get_collection",
               return_value=mock_chroma_collection):
        from reranking_rag.implementation.retriever import retrieve_initial
        result = retrieve_initial("delivery time?")
    for doc in result:
        assert "id"       in doc
        assert "text"     in doc
        assert "metadata" in doc
        assert "distance" in doc


def test_retrieve_initial_respects_top_n(mock_chroma_collection):
    mock_chroma_collection.query.return_value = _make_chroma_result(10)
    with patch("reranking_rag.implementation.retriever.get_collection",
               return_value=mock_chroma_collection):
        from reranking_rag.implementation.retriever import retrieve_initial
        retrieve_initial("query", top_n=10)
    mock_chroma_collection.query.assert_called_once_with(
        query_texts=["query"], n_results=10)


def test_retrieve_initial_default_top_n_is_twenty(mock_chroma_collection):
    mock_chroma_collection.query.return_value = _make_chroma_result(20)
    with patch("reranking_rag.implementation.retriever.get_collection",
               return_value=mock_chroma_collection):
        from reranking_rag.implementation.retriever import retrieve_initial
        retrieve_initial("some query")
    call_kwargs = mock_chroma_collection.query.call_args[1]
    assert call_kwargs["n_results"] == 20


def test_retrieve_initial_passes_query_text(mock_chroma_collection):
    with patch("reranking_rag.implementation.retriever.get_collection",
               return_value=mock_chroma_collection):
        from reranking_rag.implementation.retriever import retrieve_initial
        retrieve_initial("what is the best seller city?")
    call_kwargs = mock_chroma_collection.query.call_args[1]
    assert call_kwargs["query_texts"] == ["what is the best seller city?"]


def test_retrieve_initial_distance_is_float(mock_chroma_collection):
    with patch("reranking_rag.implementation.retriever.get_collection",
               return_value=mock_chroma_collection):
        from reranking_rag.implementation.retriever import retrieve_initial
        result = retrieve_initial("test query")
    for doc in result:
        assert isinstance(doc["distance"], float)


def test_retrieve_initial_ids_match_chroma_output(mock_chroma_collection):
    with patch("reranking_rag.implementation.retriever.get_collection",
               return_value=mock_chroma_collection):
        from reranking_rag.implementation.retriever import retrieve_initial
        result = retrieve_initial("test")
    returned_ids = [d["id"] for d in result]
    assert returned_ids == ["doc_001", "doc_002", "doc_003", "doc_004", "doc_005"]
