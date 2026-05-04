"""Unit tests for multiquery_rag.implementation.retriever."""
import os
from unittest.mock import patch

import pytest

os.environ.setdefault("GROQ_API_KEYS", "test-key-1,test-key-2")


def test_retrieve_for_query_returns_list(mock_chroma_collection):
    with patch("multiquery_rag.implementation.retriever.get_collection",
               return_value=mock_chroma_collection):
        from multiquery_rag.implementation.retriever import retrieve_for_query
        result = retrieve_for_query("test query")
    assert isinstance(result, list)


def test_retrieve_for_query_returns_top_n_docs(mock_chroma_collection):
    with patch("multiquery_rag.implementation.retriever.get_collection",
               return_value=mock_chroma_collection):
        from multiquery_rag.implementation.retriever import retrieve_for_query
        result = retrieve_for_query("test query", top_n=10)
    assert len(result) == 10


def test_retrieve_for_query_each_doc_has_id(mock_chroma_collection):
    with patch("multiquery_rag.implementation.retriever.get_collection",
               return_value=mock_chroma_collection):
        from multiquery_rag.implementation.retriever import retrieve_for_query
        result = retrieve_for_query("test query")
    assert all("id" in d for d in result)


def test_retrieve_for_query_each_doc_has_text(mock_chroma_collection):
    with patch("multiquery_rag.implementation.retriever.get_collection",
               return_value=mock_chroma_collection):
        from multiquery_rag.implementation.retriever import retrieve_for_query
        result = retrieve_for_query("test query")
    assert all("text" in d for d in result)


def test_retrieve_for_query_each_doc_has_metadata(mock_chroma_collection):
    with patch("multiquery_rag.implementation.retriever.get_collection",
               return_value=mock_chroma_collection):
        from multiquery_rag.implementation.retriever import retrieve_for_query
        result = retrieve_for_query("test query")
    assert all("metadata" in d for d in result)


def test_retrieve_for_query_each_doc_has_distance(mock_chroma_collection):
    with patch("multiquery_rag.implementation.retriever.get_collection",
               return_value=mock_chroma_collection):
        from multiquery_rag.implementation.retriever import retrieve_for_query
        result = retrieve_for_query("test query")
    assert all("distance" in d for d in result)


def test_retrieve_for_query_distance_is_float(mock_chroma_collection):
    with patch("multiquery_rag.implementation.retriever.get_collection",
               return_value=mock_chroma_collection):
        from multiquery_rag.implementation.retriever import retrieve_for_query
        result = retrieve_for_query("test query")
    assert all(isinstance(d["distance"], float) for d in result)


def test_retrieve_multi_returns_dict(mock_chroma_collection):
    queries = ["query 1", "query 2", "query 3"]
    with patch("multiquery_rag.implementation.retriever.get_collection",
               return_value=mock_chroma_collection):
        from multiquery_rag.implementation.retriever import retrieve_multi
        result = retrieve_multi(queries)
    assert isinstance(result, dict)


def test_retrieve_multi_keys_are_queries(mock_chroma_collection):
    queries = ["query 1", "query 2", "query 3"]
    with patch("multiquery_rag.implementation.retriever.get_collection",
               return_value=mock_chroma_collection):
        from multiquery_rag.implementation.retriever import retrieve_multi
        result = retrieve_multi(queries)
    assert set(result.keys()) == set(queries)


def test_retrieve_multi_each_value_is_list(mock_chroma_collection):
    queries = ["query 1", "query 2"]
    with patch("multiquery_rag.implementation.retriever.get_collection",
               return_value=mock_chroma_collection):
        from multiquery_rag.implementation.retriever import retrieve_multi
        result = retrieve_multi(queries)
    assert all(isinstance(v, list) for v in result.values())


def test_retrieve_multi_calls_collection_once_per_query(mock_chroma_collection):
    queries = ["query A", "query B", "query C"]
    with patch("multiquery_rag.implementation.retriever.get_collection",
               return_value=mock_chroma_collection):
        from multiquery_rag.implementation.retriever import retrieve_multi
        retrieve_multi(queries)
    assert mock_chroma_collection.query.call_count == len(queries)


def test_retrieve_multi_empty_queries_returns_empty_dict(mock_chroma_collection):
    with patch("multiquery_rag.implementation.retriever.get_collection",
               return_value=mock_chroma_collection):
        from multiquery_rag.implementation.retriever import retrieve_multi
        result = retrieve_multi([])
    assert result == {}
