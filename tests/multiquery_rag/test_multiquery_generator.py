"""Unit tests for multiquery_rag.implementation.generator.

Verifies that ``generate`` calls Groq exactly once with the correct model,
temperature, and a prompt that embeds the user query and all RRF-fused context
documents. Also checks the 512-token cap, system-message presence, and that
an empty document list still triggers a Groq call. All Groq calls are stubbed.
"""
import os
from unittest.mock import patch

import pytest

os.environ.setdefault("GROQ_API_KEYS", "test-key-1,test-key-2")

_MOCK_ANSWER = "The average delivery time is 12 days."


def test_generate_returns_string(fused_docs):
    """generate must return a string."""
    with patch("multiquery_rag.implementation.generator.call_groq",
               return_value=_MOCK_ANSWER):
        from multiquery_rag.implementation.generator import generate
        result = generate("What is the delivery time?", fused_docs)
    assert isinstance(result, str)


def test_generate_returns_groq_response(fused_docs):
    """generate must return the exact string returned by call_groq."""
    with patch("multiquery_rag.implementation.generator.call_groq",
               return_value=_MOCK_ANSWER):
        from multiquery_rag.implementation.generator import generate
        result = generate("What is the delivery time?", fused_docs)
    assert result == _MOCK_ANSWER


def test_generate_calls_call_groq_once(fused_docs):
    """generate must invoke call_groq exactly once per call."""
    with patch("multiquery_rag.implementation.generator.call_groq",
               return_value=_MOCK_ANSWER) as mock_groq:
        from multiquery_rag.implementation.generator import generate
        generate("question?", fused_docs)
    mock_groq.assert_called_once()


def test_generate_uses_correct_model(fused_docs):
    """generate must pass GROQ_MODEL as the model argument to call_groq."""
    with patch("multiquery_rag.implementation.generator.call_groq",
               return_value=_MOCK_ANSWER) as mock_groq:
        from multiquery_rag.implementation.generator import generate
        from multiquery_rag.implementation.config import GROQ_MODEL
        generate("question?", fused_docs)
    _, _, _, model_arg = mock_groq.call_args[0]
    assert model_arg == GROQ_MODEL


def test_generate_uses_low_temperature(fused_docs):
    """generate must forward the temperature argument to call_groq."""
    with patch("multiquery_rag.implementation.generator.call_groq",
               return_value=_MOCK_ANSWER) as mock_groq:
        from multiquery_rag.implementation.generator import generate
        generate("question?", fused_docs, temperature=0.1)
    _, temp_arg, _, _ = mock_groq.call_args[0]
    assert temp_arg == 0.1


def test_generate_includes_query_in_messages(fused_docs):
    """The user message sent to Groq must embed the original query string."""
    with patch("multiquery_rag.implementation.generator.call_groq",
               return_value=_MOCK_ANSWER) as mock_groq:
        from multiquery_rag.implementation.generator import generate
        generate("What is the best product category?", fused_docs)
    messages = mock_groq.call_args[0][0]
    user_msg = next(m for m in messages if m["role"] == "user")
    assert "What is the best product category?" in user_msg["content"]


def test_generate_includes_context_docs_in_messages(fused_docs):
    """The user message must embed the text of every RRF-fused document."""
    with patch("multiquery_rag.implementation.generator.call_groq",
               return_value=_MOCK_ANSWER) as mock_groq:
        from multiquery_rag.implementation.generator import generate
        generate("question?", fused_docs)
    messages = mock_groq.call_args[0][0]
    user_content = next(m for m in messages if m["role"] == "user")["content"]
    for doc in fused_docs:
        assert doc["text"] in user_content


def test_generate_has_system_message(fused_docs):
    """The message list sent to Groq must include a 'system' role message."""
    with patch("multiquery_rag.implementation.generator.call_groq",
               return_value=_MOCK_ANSWER) as mock_groq:
        from multiquery_rag.implementation.generator import generate
        generate("question?", fused_docs)
    messages = mock_groq.call_args[0][0]
    roles = [m["role"] for m in messages]
    assert "system" in roles


def test_generate_max_tokens_512(fused_docs):
    """generate must cap the response at 512 tokens via the max_tokens kwarg."""
    with patch("multiquery_rag.implementation.generator.call_groq",
               return_value=_MOCK_ANSWER) as mock_groq:
        from multiquery_rag.implementation.generator import generate
        generate("question?", fused_docs)
    kwargs = mock_groq.call_args[1]
    assert kwargs.get("max_tokens") == 512


def test_generate_empty_docs_still_calls_groq():
    """generate must still call Groq even when the fused document list is empty."""
    with patch("multiquery_rag.implementation.generator.call_groq",
               return_value=_MOCK_ANSWER) as mock_groq:
        from multiquery_rag.implementation.generator import generate
        generate("question?", [])
    mock_groq.assert_called_once()
