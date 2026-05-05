"""Unit tests for hybrid_rag.implementation.config.

Validates all path and scalar constants specific to the Hybrid RAG pipeline,
including the BM25 index path, RRF parameters, per-method candidate pool sizes,
and the guard that raises ``EnvironmentError`` when ``GROQ_API_KEYS`` is absent.
"""
import importlib
from pathlib import Path

import pytest


def test_base_dir_is_path():
    """BASE_DIR must be a pathlib.Path instance."""
    from hybrid_rag.implementation.config import BASE_DIR
    assert isinstance(BASE_DIR, Path)


def test_kb_all_docs_filename():
    """KB_ALL_DOCS must point to 'kb_all_documents.json'."""
    from hybrid_rag.implementation.config import KB_ALL_DOCS
    assert KB_ALL_DOCS.name == "kb_all_documents.json"


def test_bm25_index_path_is_path():
    """BM25_INDEX_PATH must be a pathlib.Path instance."""
    from hybrid_rag.implementation.config import BM25_INDEX_PATH
    assert isinstance(BM25_INDEX_PATH, Path)


def test_bm25_index_filename():
    """BM25_INDEX_PATH must point to the 'bm25_index.pkl' file."""
    from hybrid_rag.implementation.config import BM25_INDEX_PATH
    assert BM25_INDEX_PATH.name == "bm25_index.pkl"


def test_collection_name():
    """COLLECTION_NAME must equal 'ecommerce_kb'."""
    from hybrid_rag.implementation.config import COLLECTION_NAME
    assert COLLECTION_NAME == "ecommerce_kb"


def test_embedding_model():
    """EMBEDDING_MODEL must reference the all-MiniLM-L6-v2 sentence-transformer."""
    from hybrid_rag.implementation.config import EMBEDDING_MODEL
    assert EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"


def test_semantic_top_k():
    """SEMANTIC_TOP_K must equal 10 (candidate pool from ChromaDB)."""
    from hybrid_rag.implementation.config import SEMANTIC_TOP_K
    assert SEMANTIC_TOP_K == 10


def test_keyword_top_k():
    """KEYWORD_TOP_K must equal 10 (candidate pool from BM25)."""
    from hybrid_rag.implementation.config import KEYWORD_TOP_K
    assert KEYWORD_TOP_K == 10


def test_final_top_k():
    """FINAL_TOP_K must equal 5 (documents surfaced after RRF fusion)."""
    from hybrid_rag.implementation.config import FINAL_TOP_K
    assert FINAL_TOP_K == 5


def test_rrf_k():
    """RRF_K must equal 60 (standard reciprocal-rank fusion smoothing constant)."""
    from hybrid_rag.implementation.config import RRF_K
    assert RRF_K == 60


def test_final_top_k_less_than_candidate_pools():
    """Final top-k must be smaller than the per-method candidate pools."""
    from hybrid_rag.implementation.config import FINAL_TOP_K, SEMANTIC_TOP_K, KEYWORD_TOP_K
    assert FINAL_TOP_K < SEMANTIC_TOP_K
    assert FINAL_TOP_K < KEYWORD_TOP_K


def test_groq_api_keys_non_empty():
    """GROQ_API_KEYS must contain at least one key."""
    from hybrid_rag.implementation.config import GROQ_API_KEYS
    assert len(GROQ_API_KEYS) >= 1


def test_missing_groq_keys_raises(monkeypatch):
    """Config reload with empty GROQ_API_KEYS must raise EnvironmentError."""
    import hybrid_rag.implementation.config as cfg
    monkeypatch.setenv("GROQ_API_KEYS", "")
    with pytest.raises(EnvironmentError, match="GROQ_API_KEYS"):
        importlib.reload(cfg)
    monkeypatch.setenv("GROQ_API_KEYS", "test-key-1,test-key-2")
    importlib.reload(cfg)
