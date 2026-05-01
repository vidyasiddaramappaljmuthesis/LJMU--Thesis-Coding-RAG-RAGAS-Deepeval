"""Unit tests for naive_rag.implementation.generator."""
from unittest.mock import patch, call


_MOCK_ANSWER = "The average delivery time is 8 days."


def test_generate_returns_string(retrieved_docs):
    with patch("naive_rag.implementation.generator.call_groq", return_value=_MOCK_ANSWER):
        from naive_rag.implementation.generator import generate
        result = generate("What is the delivery time?", retrieved_docs)
    assert isinstance(result, str)


def test_generate_returns_groq_response(retrieved_docs):
    with patch("naive_rag.implementation.generator.call_groq", return_value=_MOCK_ANSWER):
        from naive_rag.implementation.generator import generate
        result = generate("What is the delivery time?", retrieved_docs)
    assert result == _MOCK_ANSWER


def test_generate_calls_call_groq_once(retrieved_docs):
    with patch("naive_rag.implementation.generator.call_groq", return_value=_MOCK_ANSWER) as mock_groq:
        from naive_rag.implementation.generator import generate
        generate("question?", retrieved_docs)
    mock_groq.assert_called_once()


def test_generate_uses_correct_model(retrieved_docs):
    with patch("naive_rag.implementation.generator.call_groq", return_value=_MOCK_ANSWER) as mock_groq:
        from naive_rag.implementation.generator import generate
        from naive_rag.implementation.config import GROQ_MODEL
        generate("question?", retrieved_docs)
    _, _, _, model_arg = mock_groq.call_args[0]
    assert model_arg == GROQ_MODEL


def test_generate_uses_low_temperature(retrieved_docs):
    with patch("naive_rag.implementation.generator.call_groq", return_value=_MOCK_ANSWER) as mock_groq:
        from naive_rag.implementation.generator import generate
        generate("question?", retrieved_docs, temperature=0.1)
    _, temp_arg, _, _ = mock_groq.call_args[0]
    assert temp_arg == 0.1


def test_generate_includes_query_in_messages(retrieved_docs):
    with patch("naive_rag.implementation.generator.call_groq", return_value=_MOCK_ANSWER) as mock_groq:
        from naive_rag.implementation.generator import generate
        generate("What is the best product?", retrieved_docs)
    messages = mock_groq.call_args[0][0]
    user_message = next(m for m in messages if m["role"] == "user")
    assert "What is the best product?" in user_message["content"]


def test_generate_includes_context_docs_in_messages(retrieved_docs):
    with patch("naive_rag.implementation.generator.call_groq", return_value=_MOCK_ANSWER) as mock_groq:
        from naive_rag.implementation.generator import generate
        generate("question?", retrieved_docs)
    messages = mock_groq.call_args[0][0]
    user_content = next(m for m in messages if m["role"] == "user")["content"]
    for doc in retrieved_docs:
        assert doc["text"] in user_content


def test_generate_has_system_message(retrieved_docs):
    with patch("naive_rag.implementation.generator.call_groq", return_value=_MOCK_ANSWER) as mock_groq:
        from naive_rag.implementation.generator import generate
        generate("question?", retrieved_docs)
    messages = mock_groq.call_args[0][0]
    roles = [m["role"] for m in messages]
    assert "system" in roles


def test_generate_max_tokens_512(retrieved_docs):
    with patch("naive_rag.implementation.generator.call_groq", return_value=_MOCK_ANSWER) as mock_groq:
        from naive_rag.implementation.generator import generate
        generate("question?", retrieved_docs)
    kwargs = mock_groq.call_args[1]
    assert kwargs.get("max_tokens") == 512
