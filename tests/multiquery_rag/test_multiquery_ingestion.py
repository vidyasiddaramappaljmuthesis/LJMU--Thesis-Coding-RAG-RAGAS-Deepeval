"""Unit tests for multiquery_rag.implementation.ingestion."""
import os
from unittest.mock import patch, MagicMock

import pytest

os.environ.setdefault("GROQ_API_KEYS", "test-key-1,test-key-2")


def test_get_collection_returns_collection(mock_chroma_collection):
    with patch("multiquery_rag.implementation.ingestion._get_client") as mock_client, \
         patch("multiquery_rag.implementation.ingestion._collection", None):
        mock_client.return_value.get_collection.return_value = mock_chroma_collection
        from multiquery_rag.implementation.ingestion import get_collection
        import multiquery_rag.implementation.ingestion as ing
        ing._collection = None
        col = get_collection()
    assert col is not None


def test_get_collection_uses_correct_name(mock_chroma_collection):
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
    import multiquery_rag.implementation.ingestion as ing
    ing._collection = mock_chroma_collection
    from multiquery_rag.implementation.ingestion import get_collection
    col1 = get_collection()
    col2 = get_collection()
    assert col1 is col2
