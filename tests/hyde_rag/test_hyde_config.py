"""Unit tests for hyde_rag.implementation.config."""
import importlib
from pathlib import Path

import pytest


def test_base_dir_is_path():
    from hyde_rag.implementation.config import BASE_DIR
    assert isinstance(BASE_DIR, Path)


def test_kb_all_docs_filename():
    from hyde_rag.implementation.config import KB_ALL_DOCS
    assert KB_ALL_DOCS.name == "kb_all_documents.json"


def test_chroma_db_path_name():
    from hyde_rag.implementation.config import CHROMA_DB_PATH
    assert CHROMA_DB_PATH.name == "chroma_db"


def test_collection_name():
    from hyde_rag.implementation.config import COLLECTION_NAME
    assert COLLECTION_NAME == "ecommerce_kb"


def test_collection_name_matches_naive():
    """HyDE and Naive RAG must share the same ChromaDB collection."""
    from hyde_rag.implementation.config import COLLECTION_NAME as hyde_col
    from naive_rag.implementation.config import COLLECTION_NAME as naive_col
    assert hyde_col == naive_col


def test_embedding_model():
    from hyde_rag.implementation.config import EMBEDDING_MODEL
    assert EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"


def test_groq_api_keys_is_list():
    from hyde_rag.implementation.config import GROQ_API_KEYS
    assert isinstance(GROQ_API_KEYS, list)
    assert len(GROQ_API_KEYS) >= 1


def test_groq_model():
    from hyde_rag.implementation.config import GROQ_MODEL
    assert GROQ_MODEL == "llama-3.3-70b-versatile"


def test_top_k():
    from hyde_rag.implementation.config import TOP_K
    assert TOP_K == 5


def test_hyde_temperature():
    from hyde_rag.implementation.config import HYDE_TEMPERATURE
    assert HYDE_TEMPERATURE == 0.7


def test_answer_temperature():
    from hyde_rag.implementation.config import ANSWER_TEMPERATURE
    assert ANSWER_TEMPERATURE == 0.1


def test_hyde_temp_higher_than_answer_temp():
    """Hypothetical doc generation must use higher temperature than final answer."""
    from hyde_rag.implementation.config import HYDE_TEMPERATURE, ANSWER_TEMPERATURE
    assert HYDE_TEMPERATURE > ANSWER_TEMPERATURE


def test_missing_groq_keys_raises(monkeypatch):
    import hyde_rag.implementation.config as cfg
    monkeypatch.setenv("GROQ_API_KEYS", "")
    with pytest.raises(EnvironmentError, match="GROQ_API_KEYS"):
        importlib.reload(cfg)
    monkeypatch.setenv("GROQ_API_KEYS", "test-key-1,test-key-2")
    importlib.reload(cfg)
