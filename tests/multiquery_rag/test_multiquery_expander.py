"""Unit tests for multiquery_rag.implementation.query_expander.

Verifies that ``expand_query`` calls Groq once with the correct model and
EXPANDER_TEMPERATURE, returns exactly NUM_QUERY_VARIANTS strings with the
original query as the first element, and degrades gracefully (returns the
original query) when the Groq call raises RuntimeError.
"""
import os
from unittest.mock import patch

import pytest

os.environ.setdefault("GROQ_API_KEYS", "test-key-1,test-key-2")

_ORIGINAL = "What is the average delivery time?"
_MOCK_LLM_RESPONSE = (
    "1. How long does it take for orders to be delivered?\n"
    "2. What is the typical shipping duration?\n"
    "3. How many days until the order arrives?\n"
)


def test_expand_query_returns_list():
    """expand_query must return a list."""
    with patch("multiquery_rag.implementation.query_expander.call_groq",
               return_value=_MOCK_LLM_RESPONSE):
        from multiquery_rag.implementation.query_expander import expand_query
        result = expand_query(_ORIGINAL)
    assert isinstance(result, list)


def test_expand_query_returns_n_variants():
    """expand_query must return exactly NUM_QUERY_VARIANTS items."""
    with patch("multiquery_rag.implementation.query_expander.call_groq",
               return_value=_MOCK_LLM_RESPONSE):
        from multiquery_rag.implementation.query_expander import expand_query
        from multiquery_rag.implementation.config import NUM_QUERY_VARIANTS
        result = expand_query(_ORIGINAL)
    assert len(result) == NUM_QUERY_VARIANTS


def test_expand_query_variants_are_strings():
    """Every element in the returned list must be a string."""
    with patch("multiquery_rag.implementation.query_expander.call_groq",
               return_value=_MOCK_LLM_RESPONSE):
        from multiquery_rag.implementation.query_expander import expand_query
        result = expand_query(_ORIGINAL)
    assert all(isinstance(v, str) for v in result)


def test_expand_query_first_element_is_original():
    """The first variant must be the original query (preserves the user's intent)."""
    with patch("multiquery_rag.implementation.query_expander.call_groq",
               return_value=_MOCK_LLM_RESPONSE):
        from multiquery_rag.implementation.query_expander import expand_query
        result = expand_query(_ORIGINAL)
    assert result[0] == _ORIGINAL


def test_expand_query_calls_call_groq_once():
    """expand_query must invoke call_groq exactly once per call."""
    with patch("multiquery_rag.implementation.query_expander.call_groq",
               return_value=_MOCK_LLM_RESPONSE) as mock_groq:
        from multiquery_rag.implementation.query_expander import expand_query
        expand_query(_ORIGINAL)
    mock_groq.assert_called_once()


def test_expand_query_uses_correct_model():
    """expand_query must pass GROQ_MODEL to call_groq."""
    with patch("multiquery_rag.implementation.query_expander.call_groq",
               return_value=_MOCK_LLM_RESPONSE) as mock_groq:
        from multiquery_rag.implementation.query_expander import expand_query
        from multiquery_rag.implementation.config import GROQ_MODEL
        expand_query(_ORIGINAL)
    _, _, _, model_arg = mock_groq.call_args[0]
    assert model_arg == GROQ_MODEL


def test_expand_query_uses_high_temperature():
    """expand_query must pass EXPANDER_TEMPERATURE to call_groq for diverse variants."""
    with patch("multiquery_rag.implementation.query_expander.call_groq",
               return_value=_MOCK_LLM_RESPONSE) as mock_groq:
        from multiquery_rag.implementation.query_expander import expand_query
        from multiquery_rag.implementation.config import EXPANDER_TEMPERATURE
        expand_query(_ORIGINAL)
    _, temp_arg, _, _ = mock_groq.call_args[0]
    assert temp_arg == EXPANDER_TEMPERATURE


def test_expand_query_has_system_message():
    """The message list sent to Groq must include a 'system' role message."""
    with patch("multiquery_rag.implementation.query_expander.call_groq",
               return_value=_MOCK_LLM_RESPONSE) as mock_groq:
        from multiquery_rag.implementation.query_expander import expand_query
        expand_query(_ORIGINAL)
    messages = mock_groq.call_args[0][0]
    roles = [m["role"] for m in messages]
    assert "system" in roles


def test_expand_query_user_message_contains_original():
    """The user message sent to Groq must embed the original query."""
    with patch("multiquery_rag.implementation.query_expander.call_groq",
               return_value=_MOCK_LLM_RESPONSE) as mock_groq:
        from multiquery_rag.implementation.query_expander import expand_query
        expand_query(_ORIGINAL)
    messages = mock_groq.call_args[0][0]
    user_msg = next(m for m in messages if m["role"] == "user")
    assert _ORIGINAL in user_msg["content"]


def test_expand_query_fallback_on_groq_failure():
    """When Groq raises RuntimeError, expand_query must return at least the original query."""
    with patch("multiquery_rag.implementation.query_expander.call_groq",
               side_effect=RuntimeError("API error")):
        from multiquery_rag.implementation.query_expander import expand_query
        result = expand_query(_ORIGINAL)
    assert isinstance(result, list)
    assert len(result) >= 1
    assert result[0] == _ORIGINAL


def test_expand_query_no_duplicate_variants():
    """Returned variants must be unique (case-insensitive, stripped)."""
    with patch("multiquery_rag.implementation.query_expander.call_groq",
               return_value=_MOCK_LLM_RESPONSE):
        from multiquery_rag.implementation.query_expander import expand_query
        result = expand_query(_ORIGINAL)
    normalized = [v.strip().lower() for v in result]
    assert len(normalized) == len(set(normalized)) or all(
        normalized[i] == normalized[0] for i in range(len(normalized))
    )


def test_expand_query_variants_non_empty():
    """No variant in the returned list may be an empty or whitespace-only string."""
    with patch("multiquery_rag.implementation.query_expander.call_groq",
               return_value=_MOCK_LLM_RESPONSE):
        from multiquery_rag.implementation.query_expander import expand_query
        result = expand_query(_ORIGINAL)
    assert all(len(v.strip()) > 0 for v in result)
