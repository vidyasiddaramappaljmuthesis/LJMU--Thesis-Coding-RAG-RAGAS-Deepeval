"""Unit tests for reranking_rag.implementation.config."""
import importlib
from pathlib import Path

import pytest


def test_base_dir_is_path():
    from reranking_rag.implementation.config import BASE_DIR
    assert isinstance(BASE_DIR, Path)


def test_kb_all_docs_filename():
    from reranking_rag.implementation.config import KB_ALL_DOCS
    assert KB_ALL_DOCS.name == "kb_all_documents.json"


def test_kb_all_docs_under_base():
    from reranking_rag.implementation.config import KB_ALL_DOCS, BASE_DIR
    assert str(BASE_DIR) in str(KB_ALL_DOCS)


def test_chroma_db_path_name():
    from reranking_rag.implementation.config import CHROMA_DB_PATH
    assert CHROMA_DB_PATH.name == "chroma_db"


def test_collection_name():
    from reranking_rag.implementation.config import COLLECTION_NAME
    assert COLLECTION_NAME == "ecommerce_kb"


def test_embedding_model():
    from reranking_rag.implementation.config import EMBEDDING_MODEL
    assert EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"


def test_reranker_model():
    from reranking_rag.implementation.config import RERANKER_MODEL
    assert RERANKER_MODEL == "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_groq_api_keys_is_list():
    from reranking_rag.implementation.config import GROQ_API_KEYS
    assert isinstance(GROQ_API_KEYS, list)


def test_groq_api_keys_non_empty():
    from reranking_rag.implementation.config import GROQ_API_KEYS
    assert len(GROQ_API_KEYS) >= 1


def test_groq_api_keys_no_blank_entries():
    from reranking_rag.implementation.config import GROQ_API_KEYS
    assert all(k.strip() != "" for k in GROQ_API_KEYS)


def test_groq_model():
    from reranking_rag.implementation.config import GROQ_MODEL
    assert GROQ_MODEL == "llama-3.3-70b-versatile"


def test_top_k_is_five():
    from reranking_rag.implementation.config import TOP_K
    assert TOP_K == 5


def test_initial_retrieval_k_greater_than_top_k():
    from reranking_rag.implementation.config import INITIAL_RETRIEVAL_K, TOP_K
    assert INITIAL_RETRIEVAL_K > TOP_K


def test_initial_retrieval_k_is_twenty():
    from reranking_rag.implementation.config import INITIAL_RETRIEVAL_K
    assert INITIAL_RETRIEVAL_K == 20


def test_missing_groq_keys_raises(monkeypatch):
    """Config reload with empty GROQ_API_KEYS must raise EnvironmentError."""
    import reranking_rag.implementation.config as cfg
    monkeypatch.setenv("GROQ_API_KEYS", "")
    with pytest.raises(EnvironmentError, match="GROQ_API_KEYS"):
        importlib.reload(cfg)
    monkeypatch.setenv("GROQ_API_KEYS", "test-key-1,test-key-2")
    importlib.reload(cfg)
