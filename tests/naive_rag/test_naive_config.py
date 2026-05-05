"""Unit tests for naive_rag.implementation.config.

Validates that all path constants are correct Path objects, string constants
match expected values, GROQ_API_KEYS is a non-empty list of valid strings,
and that the module raises ``EnvironmentError`` when the env var is missing.
"""
import importlib
import os
from pathlib import Path

import pytest


def test_base_dir_is_path():
    """BASE_DIR must be a pathlib.Path instance."""
    from naive_rag.implementation.config import BASE_DIR
    assert isinstance(BASE_DIR, Path)


def test_kb_all_docs_filename():
    """KB_ALL_DOCS must point to the 'kb_all_documents.json' filename."""
    from naive_rag.implementation.config import KB_ALL_DOCS
    assert KB_ALL_DOCS.name == "kb_all_documents.json"


def test_kb_all_docs_under_base():
    """KB_ALL_DOCS must be a descendant of BASE_DIR."""
    from naive_rag.implementation.config import KB_ALL_DOCS, BASE_DIR
    assert str(BASE_DIR) in str(KB_ALL_DOCS)


def test_chroma_db_path_name():
    """CHROMA_DB_PATH directory must be named 'chroma_db'."""
    from naive_rag.implementation.config import CHROMA_DB_PATH
    assert CHROMA_DB_PATH.name == "chroma_db"


def test_collection_name():
    """COLLECTION_NAME must equal 'ecommerce_kb'."""
    from naive_rag.implementation.config import COLLECTION_NAME
    assert COLLECTION_NAME == "ecommerce_kb"


def test_embedding_model():
    """EMBEDDING_MODEL must reference the all-MiniLM-L6-v2 sentence-transformer."""
    from naive_rag.implementation.config import EMBEDDING_MODEL
    assert EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"


def test_groq_api_keys_is_list():
    """GROQ_API_KEYS must be a list (not a string or other iterable)."""
    from naive_rag.implementation.config import GROQ_API_KEYS
    assert isinstance(GROQ_API_KEYS, list)


def test_groq_api_keys_non_empty():
    """GROQ_API_KEYS must contain at least one key."""
    from naive_rag.implementation.config import GROQ_API_KEYS
    assert len(GROQ_API_KEYS) >= 1


def test_groq_api_keys_no_blank_entries():
    """No entry in GROQ_API_KEYS may be an empty or whitespace-only string."""
    from naive_rag.implementation.config import GROQ_API_KEYS
    assert all(k.strip() != "" for k in GROQ_API_KEYS)


def test_groq_model():
    """GROQ_MODEL must reference the LLaMA 3.3 70B versatile model."""
    from naive_rag.implementation.config import GROQ_MODEL
    assert GROQ_MODEL == "llama-3.3-70b-versatile"


def test_top_k_is_five():
    """TOP_K must equal 5 (default retrieval depth)."""
    from naive_rag.implementation.config import TOP_K
    assert TOP_K == 5


def test_missing_groq_keys_raises(monkeypatch):
    """Config reload with empty GROQ_API_KEYS must raise EnvironmentError."""
    import naive_rag.implementation.config as cfg
    monkeypatch.setenv("GROQ_API_KEYS", "")
    with pytest.raises(EnvironmentError, match="GROQ_API_KEYS"):
        importlib.reload(cfg)
    # Restore so subsequent tests see valid keys.
    monkeypatch.setenv("GROQ_API_KEYS", "test-key-1,test-key-2")
    importlib.reload(cfg)
