"""Unit tests for hyde_rag.implementation.generator."""
from unittest.mock import patch

_HYPO_DOC = "In Q4 2017, electronics orders in SP took 8.3 days to deliver."
_FINAL_ANSWER = "The average delivery time for electronics in SP is 8.3 days."


# ── generate_hypothetical_doc ─────────────────────────────────────────────────

def test_generate_hypothetical_doc_returns_string():
    with patch("hyde_rag.implementation.generator.call_groq", return_value=_HYPO_DOC):
        from hyde_rag.implementation.generator import generate_hypothetical_doc
        result = generate_hypothetical_doc("What is delivery time in SP?")
    assert isinstance(result, str)


def test_generate_hypothetical_doc_uses_hyde_temperature():
    with patch("hyde_rag.implementation.generator.call_groq", return_value=_HYPO_DOC) as mock_groq:
        from hyde_rag.implementation.generator import generate_hypothetical_doc
        from hyde_rag.implementation.config import HYDE_TEMPERATURE
        generate_hypothetical_doc("question?")
    _, temp, _, _ = mock_groq.call_args[0]
    assert temp == HYDE_TEMPERATURE


def test_generate_hypothetical_doc_max_tokens_256():
    with patch("hyde_rag.implementation.generator.call_groq", return_value=_HYPO_DOC) as mock_groq:
        from hyde_rag.implementation.generator import generate_hypothetical_doc
        generate_hypothetical_doc("question?")
    assert mock_groq.call_args[1]["max_tokens"] == 256


def test_generate_hypothetical_doc_contains_question_in_prompt():
    with patch("hyde_rag.implementation.generator.call_groq", return_value=_HYPO_DOC) as mock_groq:
        from hyde_rag.implementation.generator import generate_hypothetical_doc
        generate_hypothetical_doc("What is the top product category?")
    messages = mock_groq.call_args[0][0]
    user_msg = next(m for m in messages if m["role"] == "user")
    assert "What is the top product category?" in user_msg["content"]


def test_generate_hypothetical_doc_returns_groq_response():
    with patch("hyde_rag.implementation.generator.call_groq", return_value=_HYPO_DOC):
        from hyde_rag.implementation.generator import generate_hypothetical_doc
        result = generate_hypothetical_doc("any question")
    assert result == _HYPO_DOC


# ── generate (final answer) ───────────────────────────────────────────────────

def test_generate_answer_returns_string(retrieved_docs):
    with patch("hyde_rag.implementation.generator.call_groq", return_value=_FINAL_ANSWER):
        from hyde_rag.implementation.generator import generate
        result = generate("What is delivery time?", retrieved_docs)
    assert isinstance(result, str)


def test_generate_answer_uses_answer_temperature(retrieved_docs):
    with patch("hyde_rag.implementation.generator.call_groq", return_value=_FINAL_ANSWER) as mock_groq:
        from hyde_rag.implementation.generator import generate
        from hyde_rag.implementation.config import ANSWER_TEMPERATURE
        generate("question?", retrieved_docs)
    _, temp, _, _ = mock_groq.call_args[0]
    assert temp == ANSWER_TEMPERATURE


def test_generate_answer_max_tokens_512(retrieved_docs):
    with patch("hyde_rag.implementation.generator.call_groq", return_value=_FINAL_ANSWER) as mock_groq:
        from hyde_rag.implementation.generator import generate
        generate("question?", retrieved_docs)
    assert mock_groq.call_args[1]["max_tokens"] == 512


def test_generate_answer_includes_context_docs(retrieved_docs):
    with patch("hyde_rag.implementation.generator.call_groq", return_value=_FINAL_ANSWER) as mock_groq:
        from hyde_rag.implementation.generator import generate
        generate("question?", retrieved_docs)
    messages = mock_groq.call_args[0][0]
    user_content = next(m for m in messages if m["role"] == "user")["content"]
    for doc in retrieved_docs:
        assert doc["text"] in user_content


def test_generate_answer_includes_query(retrieved_docs):
    with patch("hyde_rag.implementation.generator.call_groq", return_value=_FINAL_ANSWER) as mock_groq:
        from hyde_rag.implementation.generator import generate
        generate("What is the top seller city?", retrieved_docs)
    messages = mock_groq.call_args[0][0]
    user_content = next(m for m in messages if m["role"] == "user")["content"]
    assert "What is the top seller city?" in user_content


def test_hyde_temp_higher_than_answer_temp():
    """Hypothetical doc temperature must be higher than answer temperature."""
    from hyde_rag.implementation.config import HYDE_TEMPERATURE, ANSWER_TEMPERATURE
    assert HYDE_TEMPERATURE > ANSWER_TEMPERATURE
