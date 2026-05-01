"""Unit tests for naive_rag.implementation.ingestion."""
import json
from unittest.mock import MagicMock, patch, mock_open


# ── build_vector_store ────────────────────────────────────────────────────────

def test_build_vector_store_creates_collection(sample_docs):
    mock_col = MagicMock()
    mock_client = MagicMock()
    mock_client.create_collection.return_value = mock_col

    with patch("naive_rag.implementation.ingestion._get_client", return_value=mock_client), \
         patch("naive_rag.implementation.ingestion._embedding_fn", return_value=MagicMock()), \
         patch("builtins.open", mock_open(read_data=json.dumps(sample_docs))):
        import naive_rag.implementation.ingestion as ing
        ing._collection = None
        ing._client = None
        result = ing.build_vector_store()

    mock_client.create_collection.assert_called_once()
    call_kwargs = mock_client.create_collection.call_args[1]
    assert call_kwargs["metadata"] == {"hnsw:space": "cosine"}


def test_build_vector_store_returns_collection(sample_docs):
    mock_col = MagicMock()
    mock_client = MagicMock()
    mock_client.create_collection.return_value = mock_col

    with patch("naive_rag.implementation.ingestion._get_client", return_value=mock_client), \
         patch("naive_rag.implementation.ingestion._embedding_fn", return_value=MagicMock()), \
         patch("builtins.open", mock_open(read_data=json.dumps(sample_docs))):
        import naive_rag.implementation.ingestion as ing
        ing._collection = None
        result = ing.build_vector_store()

    assert result is mock_col


def test_build_vector_store_adds_all_docs(sample_docs):
    mock_col = MagicMock()
    mock_client = MagicMock()
    mock_client.create_collection.return_value = mock_col

    with patch("naive_rag.implementation.ingestion._get_client", return_value=mock_client), \
         patch("naive_rag.implementation.ingestion._embedding_fn", return_value=MagicMock()), \
         patch("builtins.open", mock_open(read_data=json.dumps(sample_docs))):
        import naive_rag.implementation.ingestion as ing
        ing._collection = None
        ing.build_vector_store()

    mock_col.add.assert_called_once()
    call_kwargs = mock_col.add.call_args[1]
    assert set(call_kwargs["ids"]) == {"doc_001", "doc_002", "doc_003"}


def test_build_vector_store_batches_large_docs():
    """1 100-doc KB with batch_size=40 must call col.add() three times."""
    big_docs = [{"id": f"d_{i}", "text": f"doc {i}", "metadata": {}} for i in range(100)]
    mock_col = MagicMock()
    mock_client = MagicMock()
    mock_client.create_collection.return_value = mock_col

    with patch("naive_rag.implementation.ingestion._get_client", return_value=mock_client), \
         patch("naive_rag.implementation.ingestion._embedding_fn", return_value=MagicMock()), \
         patch("builtins.open", mock_open(read_data=json.dumps(big_docs))):
        import naive_rag.implementation.ingestion as ing
        ing._collection = None
        ing.build_vector_store(batch_size=40)

    assert mock_col.add.call_count == 3  # 40 + 40 + 20


def test_build_vector_store_drops_existing_collection(sample_docs):
    mock_col = MagicMock()
    mock_client = MagicMock()
    mock_client.create_collection.return_value = mock_col

    with patch("naive_rag.implementation.ingestion._get_client", return_value=mock_client), \
         patch("naive_rag.implementation.ingestion._embedding_fn", return_value=MagicMock()), \
         patch("builtins.open", mock_open(read_data=json.dumps(sample_docs))):
        import naive_rag.implementation.ingestion as ing
        ing._collection = None
        ing.build_vector_store()

    mock_client.delete_collection.assert_called_once()


# ── get_collection ────────────────────────────────────────────────────────────

def test_get_collection_returns_cached_if_set(mock_chroma_collection):
    import naive_rag.implementation.ingestion as ing
    ing._collection = mock_chroma_collection
    result = ing.get_collection()
    assert result is mock_chroma_collection


def test_get_collection_loads_from_client_if_not_cached(mock_chroma_collection):
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_chroma_collection

    import naive_rag.implementation.ingestion as ing
    ing._collection = None

    with patch("naive_rag.implementation.ingestion._get_client", return_value=mock_client), \
         patch("naive_rag.implementation.ingestion._embedding_fn", return_value=MagicMock()):
        result = ing.get_collection()

    assert result is mock_chroma_collection
    mock_client.get_collection.assert_called_once()


def test_get_collection_caches_after_first_load(mock_chroma_collection):
    """Calling get_collection() twice must only hit the client once."""
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_chroma_collection

    import naive_rag.implementation.ingestion as ing
    ing._collection = None

    with patch("naive_rag.implementation.ingestion._get_client", return_value=mock_client), \
         patch("naive_rag.implementation.ingestion._embedding_fn", return_value=MagicMock()):
        ing.get_collection()
        ing.get_collection()

    assert mock_client.get_collection.call_count == 1
