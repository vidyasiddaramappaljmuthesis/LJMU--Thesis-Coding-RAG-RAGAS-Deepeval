"""
Answer-generation module for the Naive RAG pipeline.

Formats a RAG prompt from retrieved context documents and calls the
Groq LLaMA 3.3 70B model to produce a grounded answer.
"""
from naive_rag.implementation.config import GROQ_API_KEYS, GROQ_MODEL
from shared.groq_client import call_groq

_SYSTEM_PROMPT = (
    "You are a helpful e-commerce data assistant. "
    "Answer questions using only the provided context. "
    "If the answer cannot be found in the context, say so clearly."
)


def generate(query: str, context_docs: list, temperature: float = 0.1) -> str:
    """Build a RAG prompt and call the LLM."""
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
    return call_groq(messages, temperature, GROQ_API_KEYS, GROQ_MODEL, max_tokens=512)
