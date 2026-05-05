"""Unit tests for multiquery_rag.implementation.config.

Validates Multi-Query RAG specific constants: query expansion count
(NUM_QUERY_VARIANTS), per-query and final retrieval depths, RRF smoothing
constant, expander temperature, and basic API-key / model identifier checks.
"""
import os
import pytest

os.environ.setdefault("GROQ_API_KEYS", "test-key-1,test-key-2")


def test_config_has_num_query_variants():
    """NUM_QUERY_VARIANTS must be at least 2 to justify multi-query expansion."""
    from multiquery_rag.implementation.config import NUM_QUERY_VARIANTS
    assert NUM_QUERY_VARIANTS >= 2


def test_config_num_query_variants_is_int():
    """NUM_QUERY_VARIANTS must be an integer."""
    from multiquery_rag.implementation.config import NUM_QUERY_VARIANTS
    assert isinstance(NUM_QUERY_VARIANTS, int)


def test_config_has_per_query_top_k():
    """PER_QUERY_TOP_K must be at least 1 (candidates retrieved per query variant)."""
    from multiquery_rag.implementation.config import PER_QUERY_TOP_K
    assert PER_QUERY_TOP_K >= 1


def test_config_has_final_top_k():
    """FINAL_TOP_K must be at least 1 (documents passed to the generator)."""
    from multiquery_rag.implementation.config import FINAL_TOP_K
    assert FINAL_TOP_K >= 1


def test_config_final_top_k_less_than_per_query():
    """FINAL_TOP_K must not exceed PER_QUERY_TOP_K (fusion narrows, not expands)."""
    from multiquery_rag.implementation.config import FINAL_TOP_K, PER_QUERY_TOP_K
    assert FINAL_TOP_K <= PER_QUERY_TOP_K


def test_config_has_rrf_k():
    """RRF_K must be positive (standard smoothing constant for reciprocal rank fusion)."""
    from multiquery_rag.implementation.config import RRF_K
    assert RRF_K > 0


def test_config_has_expander_temperature():
    """EXPANDER_TEMPERATURE must be in [0.0, 2.0] for valid Groq sampling."""
    from multiquery_rag.implementation.config import EXPANDER_TEMPERATURE
    assert 0.0 <= EXPANDER_TEMPERATURE <= 2.0


def test_config_groq_api_keys_is_list():
    """GROQ_API_KEYS must be a list."""
    from multiquery_rag.implementation.config import GROQ_API_KEYS
    assert isinstance(GROQ_API_KEYS, list)


def test_config_groq_api_keys_not_empty():
    """GROQ_API_KEYS must contain at least one key."""
    from multiquery_rag.implementation.config import GROQ_API_KEYS
    assert len(GROQ_API_KEYS) >= 1


def test_config_groq_model_is_string():
    """GROQ_MODEL must be a non-empty string."""
    from multiquery_rag.implementation.config import GROQ_MODEL
    assert isinstance(GROQ_MODEL, str) and len(GROQ_MODEL) > 0


def test_config_collection_name_is_string():
    """COLLECTION_NAME must be a non-empty string."""
    from multiquery_rag.implementation.config import COLLECTION_NAME
    assert isinstance(COLLECTION_NAME, str) and len(COLLECTION_NAME) > 0


def test_config_embedding_model_is_string():
    """EMBEDDING_MODEL must reference a MiniLM sentence-transformer."""
    from multiquery_rag.implementation.config import EMBEDDING_MODEL
    assert isinstance(EMBEDDING_MODEL, str) and "MiniLM" in EMBEDDING_MODEL
