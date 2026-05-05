"""
Answer-generation module for the Reranking RAG pipeline.

Formats a RAG prompt from the cross-encoder reranked context documents and
calls the Groq LLaMA 3.3 70B model to produce a grounded answer.
"""
import logging

from reranking_rag.implementation.config import GROQ_API_KEYS, GROQ_MODEL
from shared.groq_client import call_groq

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a helpful e-commerce data assistant. "
    "Answer questions using only the provided context. "
    "If the answer cannot be found in the context, say so clearly."
)


def generate(query: str, context_docs: list, temperature: float = 0.1) -> str:
    """Build a RAG prompt from reranked docs and call the LLM.

    Args:
        query:        The user's natural-language question.
        context_docs: Cross-encoder reranked documents; each must have a
                      ``"text"`` key.  Documents are injected in rank order.
        temperature:  LLM sampling temperature (default 0.1 for determinism).

    Returns:
        The LLM-generated answer string.
    """
    log.debug(
        "Generating answer for query=%r using %d reranked context documents",
        query[:80],
        len(context_docs),
    )
    context_block = "\n\n".join(
        f"[Document {i + 1}]\n{doc['text']}" for i, doc in enumerate(context_docs)
    )
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
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
    log.debug("Generation complete; answer length=%d chars", len(answer))
    return answer
