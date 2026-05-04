"""
Query expansion module for the Multi-Query RAG pipeline.

Takes a single user query and uses Groq LLaMA 3.3 70B to generate
NUM_QUERY_VARIANTS - 1 paraphrased variants.  The original query is
always prepended as the first element so it is never dropped.

Why expand queries?
  A single embedding may miss relevant documents whose vocabulary differs
  from the query phrasing.  Paraphrases diversify the search surface and
  increase recall before RRF fusion.

Parsing strategy:
  The LLM is asked for a numbered list.  Lines are stripped of leading
  "1. ", "- ", "* " markers.  If the LLM returns fewer variants than
  requested (rare), the list is padded with the original query so the
  retrieval step always has something to work with.
"""
import re
import logging
from typing import List

from multiquery_rag.implementation.config import (
    GROQ_API_KEYS,
    GROQ_MODEL,
    NUM_QUERY_VARIANTS,
    EXPANDER_TEMPERATURE,
)
from shared.groq_client import call_groq

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a query rewriting assistant. "
    "Given a user question, produce exactly {n} different ways to ask the same question. "
    "Each variant must preserve the original intent but use different vocabulary or structure. "
    "Output ONLY a numbered list — one variant per line, no explanations, no preamble."
)


def _parse_variants(text: str) -> List[str]:
    """Extract non-empty lines from a numbered/bulleted LLM response."""
    variants = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        cleaned = re.sub(r"^\d+[.)]\s*|^[-*]\s*", "", line).strip()
        if cleaned:
            variants.append(cleaned)
    return variants


def expand_query(
    query: str,
    n: int = NUM_QUERY_VARIANTS,
    temperature: float = EXPANDER_TEMPERATURE,
) -> List[str]:
    """
    Generate *n* query variants for *query* (original included).

    The original query is always the first element so downstream retrieval
    always covers the user's exact phrasing even if the LLM fails to parse.

    Args:
        query:       The original user question.
        n:           Total number of variants to return (including original).
        temperature: Sampling temperature — higher → more diverse paraphrases.

    Returns:
        List of *n* query strings, starting with the original.
    """
    n_generated = max(1, n - 1)  # we add original ourselves, so ask for n-1
    messages = [
        {
            "role": "system",
            "content": _SYSTEM_PROMPT.format(n=n_generated),
        },
        {
            "role": "user",
            "content": f"Original question: {query}",
        },
    ]

    try:
        response = call_groq(
            messages, temperature, GROQ_API_KEYS, GROQ_MODEL, max_tokens=300
        )
        variants = _parse_variants(response)
    except Exception as exc:
        log.warning("Query expansion failed (%s) — using original only.", exc)
        variants = []

    # Always include the original as element [0]; de-duplicate without reordering
    seen: set = {query.strip().lower()}
    unique: List[str] = [query]
    for v in variants:
        key = v.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(v)
        if len(unique) >= n:
            break

    # Pad with original if fewer than n unique variants were generated
    while len(unique) < n:
        unique.append(query)

    return unique[:n]
