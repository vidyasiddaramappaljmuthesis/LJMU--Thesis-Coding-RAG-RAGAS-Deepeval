"""Unit tests for hybrid_rag.implementation.generator."""
from unittest.mock import patch

_ANSWER = "The most popular product category is Health & Beauty with 12% of orders."


def test_generate_returns_string(fused_docs):
    with patch("hybrid_rag.implementation.generator.call_groq", return_value=_ANSWER):
        from hybrid_rag.implementation.generator import generate
        result = generate("What is the top category?", fused_docs)
    assert isinstance(result, str)


def test_generate_returns_groq_response(fused_docs):
    with patch("hybrid_rag.implementation.generator.call_groq", return_value=_ANSWER):
        from hybrid_rag.implementation.generator import generate
        result = generate("question?", fused_docs)
    assert result == _ANSWER


def test_generate_calls_call_groq_once(fused_docs):
    with patch("hybrid_rag.implementation.generator.call_groq", return_value=_ANSWER) as mock_groq:
        from hybrid_rag.implementation.generator import generate
        generate("question?", fused_docs)
    mock_groq.assert_called_once()


def test_generate_uses_correct_model(fused_docs):
    with patch("hybrid_rag.implementation.generator.call_groq", return_value=_ANSWER) as mock_groq:
        from hybrid_rag.implementation.generator import generate
        from hybrid_rag.implementation.config import GROQ_MODEL
        generate("question?", fused_docs)
    _, _, _, model = mock_groq.call_args[0]
    assert model == GROQ_MODEL


def test_generate_uses_low_temperature(fused_docs):
    with patch("hybrid_rag.implementation.generator.call_groq", return_value=_ANSWER) as mock_groq:
        from hybrid_rag.implementation.generator import generate
        generate("question?", fused_docs, temperature=0.1)
    _, temp, _, _ = mock_groq.call_args[0]
    assert temp == 0.1


def test_generate_includes_context_docs_in_messages(fused_docs):
    with patch("hybrid_rag.implementation.generator.call_groq", return_value=_ANSWER) as mock_groq:
        from hybrid_rag.implementation.generator import generate
        generate("question?", fused_docs)
    messages = mock_groq.call_args[0][0]
    user_content = next(m for m in messages if m["role"] == "user")["content"]
    for doc in fused_docs:
        assert doc["text"] in user_content


def test_generate_numbers_documents_in_context(fused_docs):
    with patch("hybrid_rag.implementation.generator.call_groq", return_value=_ANSWER) as mock_groq:
        from hybrid_rag.implementation.generator import generate
        generate("question?", fused_docs)
    messages = mock_groq.call_args[0][0]
    user_content = next(m for m in messages if m["role"] == "user")["content"]
    assert "[Document 1]" in user_content
    assert "[Document 2]" in user_content


def test_generate_includes_query_in_messages(fused_docs):
    with patch("hybrid_rag.implementation.generator.call_groq", return_value=_ANSWER) as mock_groq:
        from hybrid_rag.implementation.generator import generate
        generate("What is the best seller region?", fused_docs)
    messages = mock_groq.call_args[0][0]
    user_content = next(m for m in messages if m["role"] == "user")["content"]
    assert "What is the best seller region?" in user_content


def test_generate_has_system_message(fused_docs):
    with patch("hybrid_rag.implementation.generator.call_groq", return_value=_ANSWER) as mock_groq:
        from hybrid_rag.implementation.generator import generate
        generate("question?", fused_docs)
    messages = mock_groq.call_args[0][0]
    assert any(m["role"] == "system" for m in messages)


def test_generate_max_tokens_512(fused_docs):
    with patch("hybrid_rag.implementation.generator.call_groq", return_value=_ANSWER) as mock_groq:
        from hybrid_rag.implementation.generator import generate
        generate("question?", fused_docs)
    assert mock_groq.call_args[1]["max_tokens"] == 512
