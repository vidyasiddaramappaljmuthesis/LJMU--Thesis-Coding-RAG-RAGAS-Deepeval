"""Unit tests for multiquery_rag.implementation.config."""
import os
import pytest

os.environ.setdefault("GROQ_API_KEYS", "test-key-1,test-key-2")


def test_config_has_num_query_variants():
    from multiquery_rag.implementation.config import NUM_QUERY_VARIANTS
    assert NUM_QUERY_VARIANTS >= 2


def test_config_num_query_variants_is_int():
    from multiquery_rag.implementation.config import NUM_QUERY_VARIANTS
    assert isinstance(NUM_QUERY_VARIANTS, int)


def test_config_has_per_query_top_k():
    from multiquery_rag.implementation.config import PER_QUERY_TOP_K
    assert PER_QUERY_TOP_K >= 1


def test_config_has_final_top_k():
    from multiquery_rag.implementation.config import FINAL_TOP_K
    assert FINAL_TOP_K >= 1


def test_config_final_top_k_less_than_per_query():
    from multiquery_rag.implementation.config import FINAL_TOP_K, PER_QUERY_TOP_K
    assert FINAL_TOP_K <= PER_QUERY_TOP_K


def test_config_has_rrf_k():
    from multiquery_rag.implementation.config import RRF_K
    assert RRF_K > 0


def test_config_has_expander_temperature():
    from multiquery_rag.implementation.config import EXPANDER_TEMPERATURE
    assert 0.0 <= EXPANDER_TEMPERATURE <= 2.0


def test_config_groq_api_keys_is_list():
    from multiquery_rag.implementation.config import GROQ_API_KEYS
    assert isinstance(GROQ_API_KEYS, list)


def test_config_groq_api_keys_not_empty():
    from multiquery_rag.implementation.config import GROQ_API_KEYS
    assert len(GROQ_API_KEYS) >= 1


def test_config_groq_model_is_string():
    from multiquery_rag.implementation.config import GROQ_MODEL
    assert isinstance(GROQ_MODEL, str) and len(GROQ_MODEL) > 0


def test_config_collection_name_is_string():
    from multiquery_rag.implementation.config import COLLECTION_NAME
    assert isinstance(COLLECTION_NAME, str) and len(COLLECTION_NAME) > 0


def test_config_embedding_model_is_string():
    from multiquery_rag.implementation.config import EMBEDDING_MODEL
    assert isinstance(EMBEDDING_MODEL, str) and "MiniLM" in EMBEDDING_MODEL
