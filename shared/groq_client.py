"""
Shared Groq key-rotation client used by all three RAG pipelines.

One shared _current_key_idx means all pipelines draw from the same rotation,
so key 0 is not triple-hit when evaluations run across naive, hybrid, and HyDE.
Starting index is randomized once per process to spread load further.
"""
import logging
import random
from typing import Optional

import groq

_ROTATABLE = (
    groq.RateLimitError,
    groq.AuthenticationError,
    groq.APIConnectionError,
    groq.InternalServerError,
    groq.APITimeoutError,
)

log = logging.getLogger(__name__)

_current_key_idx: int = 0
_initialized: bool = False


def _ensure_init(api_keys: list) -> None:
    global _current_key_idx, _initialized
    if not _initialized:
        _current_key_idx = random.randrange(len(api_keys))
        _initialized = True


def call_groq(
    messages: list,
    temperature: float,
    api_keys: list,
    model: str,
    max_tokens: Optional[int] = None,
) -> str:
    """Rotate through api_keys on transient failures; raise when all exhausted."""
    global _current_key_idx
    _ensure_init(api_keys)

    for _ in range(len(api_keys)):
        key = api_keys[_current_key_idx]
        try:
            client = groq.Groq(api_key=key)
            kwargs: dict = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except _ROTATABLE as exc:
            log.warning(
                "Key index %d exhausted (%s). Rotating to next key.",
                _current_key_idx,
                type(exc).__name__,
            )
            _current_key_idx = (_current_key_idx + 1) % len(api_keys)

    raise RuntimeError(
        f"All {len(api_keys)} Groq API keys are exhausted. "
        "Please wait for quota to reset."
    )
