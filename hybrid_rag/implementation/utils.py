"""
Text-processing utilities shared across the Hybrid RAG pipeline.

Provides a consistent tokenisation function used by both the BM25 index
builder and the BM25 query scorer so the two operate on the same token
vocabulary.
"""
import re


def tokenize(text: str) -> list:
    """Lowercase, strip punctuation, split on whitespace."""
    return re.sub(r"[^\w\s]", " ", text.lower()).split()
