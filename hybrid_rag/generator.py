from hybrid_rag.config import GROQ_API_KEYS, GROQ_MODEL
from shared.groq_client import call_groq

_SYSTEM_PROMPT = (
    "You are a helpful e-commerce data assistant. "
    "Answer questions using only the provided context. "
    "If the answer cannot be found in the context, say so clearly."
)


def generate(query: str, context_docs: list, temperature: float = 0.1) -> str:
    context_block = "\n\n".join(
        f"[Document {i + 1}]\n{doc['text']}" for i, doc in enumerate(context_docs)
    )
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{context_block}\n\nQuestion: {query}\n\nAnswer:",
        },
    ]
    return call_groq(messages, temperature, GROQ_API_KEYS, GROQ_MODEL, max_tokens=512)
