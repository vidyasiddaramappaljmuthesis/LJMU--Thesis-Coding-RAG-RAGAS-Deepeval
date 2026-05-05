"""Unit tests for hyde_rag.implementation.ingestion.

Covers ``build_vector_store`` (cosine-distance collection creation, full-doc
upsert, batched large-doc ingestion), the embedding-function singleton, and
``get_collection`` cache behaviour.  All ChromaDB and file I/O are mocked.
"""
import json
from unittest.mock import MagicMock, patch, mock_open


def test_build_vector_store_creates_collection(sample_docs):
    """build_vector_store must create a ChromaDB collection with cosine-distance metadata."""
    mock_col = MagicMock()
    mock_client = MagicMock()
    mock_client.create_collection.return_value = mock_col

    with patch("hyde_rag.implementation.ingestion._get_client", return_value=mock_client), \
         patch("hyde_rag.implementation.ingestion._embedding_fn", return_value=MagicMock()), \
         patch("builtins.open", mock_open(read_data=json.dumps(sample_docs))):
        import hyde_rag.implementation.ingestion as ing
        ing._collection = None
        ing.build_vector_store()

    mock_client.create_collection.assert_called_once()
    kwargs = mock_client.create_collection.call_args[1]
    assert kwargs["metadata"] == {"hnsw:space": "cosine"}


def test_build_vector_store_adds_all_docs(sample_docs):
    """build_vector_store must upsert all three fixture document IDs."""
    mock_col = MagicMock()
    mock_client = MagicMock()
    mock_client.create_collection.return_value = mock_col

    with patch("hyde_rag.implementation.ingestion._get_client", return_value=mock_client), \
         patch("hyde_rag.implementation.ingestion._embedding_fn", return_value=MagicMock()), \
         patch("builtins.open", mock_open(read_data=json.dumps(sample_docs))):
        import hyde_rag.implementation.ingestion as ing
        ing._collection = None
        ing.build_vector_store()

    added_ids = mock_col.add.call_args[1]["ids"]
    assert set(added_ids) == {"doc_001", "doc_002", "doc_003"}


def test_build_vector_store_batches_large_docs():
    """100-doc KB with batch_size=40 must call col.add() exactly three times (40+40+20)."""
    big_docs = [{"id": f"d_{i}", "text": f"doc {i}", "metadata": {}} for i in range(100)]
    mock_col = MagicMock()
    mock_client = MagicMock()
    mock_client.create_collection.return_value = mock_col

    with patch("hyde_rag.implementation.ingestion._get_client", return_value=mock_client), \
         patch("hyde_rag.implementation.ingestion._embedding_fn", return_value=MagicMock()), \
         patch("builtins.open", mock_open(read_data=json.dumps(big_docs))):
        import hyde_rag.implementation.ingestion as ing
        ing._collection = None
        ing.build_vector_store(batch_size=40)

    assert mock_col.add.call_count == 3


def test_ef_singleton_reused():
    """_embedding_fn() must return the same object on repeated calls (singleton)."""
    import hyde_rag.implementation.ingestion as ing
    ing._ef_singleton = None

    with patch("hyde_rag.implementation.ingestion.SentenceTransformerEmbeddingFunction") as MockEF:
        MockEF.return_value = MagicMock()
        ef1 = ing._embedding_fn()
        ef2 = ing._embedding_fn()

    assert ef1 is ef2
    assert MockEF.call_count == 1


def test_get_collection_returns_cached(mock_chroma_collection):
    """get_collection must return the cached collection without contacting ChromaDB."""
    import hyde_rag.implementation.ingestion as ing
    ing._collection = mock_chroma_collection
    assert ing.get_collection() is mock_chroma_collection


def test_get_collection_loads_from_client_when_none(mock_chroma_collection):
    """When the cache is empty, get_collection must fetch from the ChromaDB client."""
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_chroma_collection

    import hyde_rag.implementation.ingestion as ing
    ing._collection = None

    with patch("hyde_rag.implementation.ingestion._get_client", return_value=mock_client), \
         patch("hyde_rag.implementation.ingestion._embedding_fn", return_value=MagicMock()):
        result = ing.get_collection()

    assert result is mock_chroma_collection


def test_get_collection_caches_after_first_call(mock_chroma_collection):
    """Calling get_collection() twice must only hit the ChromaDB client once."""
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_chroma_collection

    import hyde_rag.implementation.ingestion as ing
    ing._collection = None

    with patch("hyde_rag.implementation.ingestion._get_client", return_value=mock_client), \
         patch("hyde_rag.implementation.ingestion._embedding_fn", return_value=MagicMock()):
        ing.get_collection()
        ing.get_collection()

    assert mock_client.get_collection.call_count == 1
