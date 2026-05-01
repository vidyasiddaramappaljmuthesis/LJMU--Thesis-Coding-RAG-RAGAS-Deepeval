"""
ChromaDB ingestion for the Naive RAG pipeline.

Embeds all 13,225 KB documents with all-MiniLM-L6-v2 and persists them
in a cosine-distance ChromaDB collection.  Module-level singletons keep
the client and collection alive across repeated calls within one process.
"""
import json
import logging
from typing import Optional

import chromadb
import chromadb.errors
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from naive_rag.implementation.config import (
    KB_ALL_DOCS,
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)

log = logging.getLogger(__name__)

_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[chromadb.Collection] = None


def _embedding_fn() -> SentenceTransformerEmbeddingFunction:
    return SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    return _client


def build_vector_store(batch_size: int = 500) -> chromadb.Collection:
    """Embed all KB documents and persist them in ChromaDB (cosine space)."""
    global _collection

    with open(KB_ALL_DOCS, "r", encoding="utf-8") as f:
        docs = json.load(f)

    client = _get_client()

    try:
        client.delete_collection(COLLECTION_NAME)
        log.info("Dropped existing collection '%s'.", COLLECTION_NAME)
    except Exception:
        pass  # collection did not pre-exist; nothing to drop

    col = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=_embedding_fn(),
        metadata={"hnsw:space": "cosine"},
    )
    _collection = col

    log.info("Ingesting %d documents into ChromaDB...", len(docs))
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        col.add(
            ids=[d["id"] for d in batch],
            documents=[d["text"] for d in batch],
            metadatas=[d["metadata"] for d in batch],
        )
        log.info("  Indexed %d/%d", min(i + batch_size, len(docs)), len(docs))

    log.info("Ingestion complete.")
    return col


def get_collection() -> chromadb.Collection:
    """Load an already-built collection (fast path — no re-embedding)."""
    global _collection
    if _collection is None:
        _collection = _get_client().get_collection(
            name=COLLECTION_NAME,
            embedding_function=_embedding_fn(),
        )
    return _collection
