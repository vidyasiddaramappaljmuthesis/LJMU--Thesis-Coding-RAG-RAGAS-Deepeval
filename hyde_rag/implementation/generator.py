"""
Groq LLaMA 3.3 70B generator with two distinct roles for the HyDE pipeline.

``generate_hypothetical_doc``
    HyDE step — produces a plausible passage that would answer the query.
    Uses ``HYDE_TEMPERATURE`` (0.7) for creative diversity and caps output
    at 256 tokens so the embedding stays concise.

``generate``
    Final answer step — grounds the answer in real retrieved context.
    Uses ``ANSWER_TEMPERATURE`` (0.1) for determinism, capped at 512 tokens.
"""
import logging

from hyde_rag.implementation.config import (
    GROQ_API_KEYS,
    GROQ_MODEL,
    HYDE_TEMPERATURE,
    ANSWER_TEMPERATURE,
)
from shared.groq_client import call_groq

log = logging.getLogger(__name__)

_HYDE_SYSTEM_PROMPT = (
    "You are an e-commerce data expert. Given a question, write a concise, "
    "factual passage that would directly answer it. The passage should read "
    "as if extracted from an e-commerce analytics database or report. "
    "Write only the passage — no preamble, no labels."
)

_ANSWER_SYSTEM_PROMPT = (
    "You are a helpful e-commerce data assistant. "
    "Answer questions using only the provided context. "
    "If the answer cannot be found in the context, say so clearly."
)


def generate_hypothetical_doc(query: str) -> str:
    """HyDE step — generate a hypothetical passage that would answer *query*.

    The passage is written at a higher temperature to produce a diverse,
    embedding-rich representation of the expected answer.  It is never shown
    to the user; it is only used to retrieve relevant real documents.

    Args:
        query: The user's natural-language question.

    Returns:
        A short factual passage (≤256 tokens) that would answer *query*.
    """
    log.debug("Generating hypothetical document for query=%r", query[:80])
    messages = [
        {"role": "system", "content": _HYDE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {query}\n\nPassage:"},
    ]
    hypo_doc = call_groq(messages, HYDE_TEMPERATURE, GROQ_API_KEYS, GROQ_MODEL, max_tokens=256)
    log.debug("Hypothetical doc length=%d chars", len(hypo_doc))
    return hypo_doc


def generate(query: str, context_docs: list, temperature: float = ANSWER_TEMPERATURE) -> str:
    """Final answer step — answer *query* from real retrieved *context_docs*.

    Args:
        query:        The user's original question.
        context_docs: Real KB documents retrieved via the HyDE embedding.
                      Each must have a ``"text"`` key.
        temperature:  LLM sampling temperature (default ``ANSWER_TEMPERATURE`` = 0.1).

    Returns:
        The LLM-generated answer string grounded in *context_docs*.
    """
    log.debug(
        "Generating final answer for query=%r using %d context documents",
        query[:80],
        len(context_docs),
    )
    context_block = "\n\n".join(
        f"[Document {i + 1}]\n{doc['text']}" for i, doc in enumerate(context_docs)
    )
    messages = [
        {"role": "system", "content": _ANSWER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Context:\n{context_block}\n\n"
                f"Question: {query}\n\n"
                "Answer:"
            ),
        },
    ]
    answer = call_groq(messages, temperature, GROQ_API_KEYS, GROQ_MODEL, max_tokens=512)
    log.debug("Final answer length=%d chars", len(answer))
    return answer
