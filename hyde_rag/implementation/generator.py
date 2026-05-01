"""
Groq LLaMA 3.3 70B generator with two roles:
  generate_hypothetical_doc  — HyDE step: produce a plausible passage that
                               would answer the query (higher temperature, capped at 256 tokens).
  generate                   — Final answer step: answer from real retrieved
                               context (low temperature, capped at 512 tokens).
"""
from hyde_rag.implementation.config import GROQ_API_KEYS, GROQ_MODEL, HYDE_TEMPERATURE, ANSWER_TEMPERATURE
from shared.groq_client import call_groq

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
    """HyDE step — generate a hypothetical passage that would answer *query*."""
    messages = [
        {"role": "system", "content": _HYDE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {query}\n\nPassage:"},
    ]
    return call_groq(messages, HYDE_TEMPERATURE, GROQ_API_KEYS, GROQ_MODEL, max_tokens=256)


def generate(query: str, context_docs: list, temperature: float = ANSWER_TEMPERATURE) -> str:
    """Final answer step — answer *query* from real retrieved *context_docs*."""
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
    return call_groq(messages, temperature, GROQ_API_KEYS, GROQ_MODEL, max_tokens=512)
