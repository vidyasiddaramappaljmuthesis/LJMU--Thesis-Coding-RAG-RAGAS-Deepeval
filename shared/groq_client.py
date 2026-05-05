"""
Shared Groq key-rotation client used by all RAG pipelines.

A single process-level ``_current_key_idx`` is shared across all pipelines
so that key 0 is not triple-hit when evaluations run across Naive, Hybrid,
and HyDE RAG in the same session.  The starting index is randomised once
per process to spread load further across keys.

Rotatable errors (rate-limit, auth, connection, timeout) trigger an
automatic rotation to the next key.  When every key has been exhausted a
``RuntimeError`` is raised to signal the caller.
"""
import logging
import random
from typing import Optional

import groq

# Error types that are considered transient and warrant a key rotation.
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
    """Randomise the starting key index once per process lifetime.

    Subsequent calls are no-ops.  Randomisation prevents every parallel
    worker from hammering key index 0 on startup.

    Args:
        api_keys: The full list of available Groq API keys.
    """
    global _current_key_idx, _initialized
    if not _initialized:
        _current_key_idx = random.randrange(len(api_keys))
        _initialized = True
        log.debug("Groq key rotation initialised at index %d.", _current_key_idx)


def call_groq(
    messages: list,
    temperature: float,
    api_keys: list,
    model: str,
    max_tokens: Optional[int] = None,
) -> str:
    """Call the Groq chat-completion API with automatic key rotation.

    On each rotatable error (rate-limit, auth, connection, timeout) the
    current key is skipped and the next key in the list is tried.  If every
    key is exhausted a ``RuntimeError`` is raised.

    Args:
        messages:    OpenAI-style message list, e.g.
                     ``[{"role": "user", "content": "..."}]``.
        temperature: Sampling temperature for the model.
        api_keys:    Ordered list of Groq API key strings.
        model:       Groq model identifier (e.g. ``"llama-3.3-70b-versatile"``).
        max_tokens:  Optional cap on output tokens.

    Returns:
        The content string from the first successful completion response.

    Raises:
        RuntimeError: When all API keys have been exhausted without a
                      successful response.
    """
    global _current_key_idx
    _ensure_init(api_keys)

    for _ in range(len(api_keys)):
        key = api_keys[_current_key_idx]
        try:
            client = groq.Groq(api_key=key)
            kwargs: dict = {
                "model":       model,
                "messages":    messages,
                "temperature": temperature,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            log.debug("Calling Groq model=%s key_idx=%d", model, _current_key_idx)
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
