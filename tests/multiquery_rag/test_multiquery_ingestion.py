"""Unit tests for multiquery_rag.implementation.ingestion.

Covers ``get_collection`` (returns a non-None collection, uses the correct
collection name, and returns the same cached object on repeated calls) and
``build_vector_store`` (upserts documents and returns the collection object).
All ChromaDB and file I/O calls are replaced with mocks.
"""
import os
from unittest.mock import patch, MagicMock

import pytest

os.environ.setdefault("GROQ_API_KEYS", "test-key-1,test-key-2")


def test_get_collection_returns_collection(mock_chroma_collection):
    """get_collection must return a non-None ChromaDB collection."""
    with patch("multiquery_rag.implementation.ingestion._get_client") as mock_client, \
         patch("multiquery_rag.implementation.ingestion._collection", None):
        mock_client.return_value.get_collection.return_value = mock_chroma_collection
        from multiquery_rag.implementation.ingestion import get_collection
        import multiquery_rag.implementation.ingestion as ing
        ing._collection = None
        col = get_collection()
    assert col is not None


def test_get_collection_uses_correct_name(mock_chroma_collection):
    """get_collection must fetch the collection using COLLECTION_NAME."""
    with patch("multiquery_rag.implementation.ingestion._get_client") as mock_client:
        mock_client.return_value.get_collection.return_value = mock_chroma_collection
        from multiquery_rag.implementation.ingestion import get_collection, COLLECTION_NAME
        import multiquery_rag.implementation.ingestion as ing
        ing._collection = None
        get_collection()
    call_kwargs = mock_client.return_value.get_collection.call_args
    assert call_kwargs[1]["name"] == COLLECTION_NAME or \
           call_kwargs[0][0] == COLLECTION_NAME


def test_build_vector_store_adds_documents(sample_docs):
    """build_vector_store must call col.add() at least once to upsert documents."""
    fake_col = MagicMock()
    with patch("multiquery_rag.implementation.ingestion._get_client") as mock_client, \
         patch("builtins.open", MagicMock()), \
         patch("json.load", return_value=sample_docs):
        mock_client.return_value.create_collection.return_value = fake_col
        import multiquery_rag.implementation.ingestion as ing
        ing._collection = None
        from multiquery_rag.implementation.ingestion import build_vector_store
        build_vector_store()
    assert fake_col.add.called


def test_build_vector_store_returns_collection(sample_docs):
    """build_vector_store must return the ChromaDB collection object."""
    fake_col = MagicMock()
    with patch("multiquery_rag.implementation.ingestion._get_client") as mock_client, \
         patch("builtins.open", MagicMock()), \
         patch("json.load", return_value=sample_docs):
        mock_client.return_value.create_collection.return_value = fake_col
        import multiquery_rag.implementation.ingestion as ing
        ing._collection = None
        from multiquery_rag.implementation.ingestion import build_vector_store
        result = build_vector_store()
    assert result is fake_col


def test_get_collection_singleton_returns_same_object(mock_chroma_collection):
    """Calling get_collection() twice must return the exact same cached object."""
    import multiquery_rag.implementation.ingestion as ing
    ing._collection = mock_chroma_collection
    from multiquery_rag.implementation.ingestion import get_collection
    col1 = get_collection()
    col2 = get_collection()
    assert col1 is col2
