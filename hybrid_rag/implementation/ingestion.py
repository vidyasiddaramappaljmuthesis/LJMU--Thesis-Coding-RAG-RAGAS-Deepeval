"""
Ingestion for the Hybrid RAG pipeline.

- ChromaDB  : shared with naive_rag (same collection, same embedding model).
              If already built by naive_rag, this step is skipped automatically.
- BM25 index: built fresh and persisted to dataset/bm25_index/bm25_index.pkl.
"""
import json
import logging
import pickle
from typing import Optional

import chromadb
import chromadb.errors
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from rank_bm25 import BM25Okapi

from hybrid_rag.implementation.config import (
    KB_ALL_DOCS,
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    BM25_INDEX_PATH,
)
from hybrid_rag.implementation.utils import tokenize

log = logging.getLogger(__name__)

_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[chromadb.Collection] = None
_bm25_cache: Optional[tuple] = None  # (BM25Okapi, list[dict])


def _ef() -> SentenceTransformerEmbeddingFunction:
    return SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    return _client


def _load_docs() -> list:
    with open(KB_ALL_DOCS, "r", encoding="utf-8") as f:
        return json.load(f)


# ── ChromaDB ──────────────────────────────────────────────────────────────────

def build_chroma(batch_size: int = 500, docs: Optional[list] = None) -> None:
    """Embed all KB documents and persist them in ChromaDB (cosine space)."""
    global _collection

    if docs is None:
        docs = _load_docs()

    client = _get_client()

    try:
        client.delete_collection(COLLECTION_NAME)
        log.info("Dropped existing ChromaDB collection.")
    except chromadb.errors.InvalidCollectionException:
        pass  # collection did not pre-exist; nothing to drop

    col = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=_ef(),
        metadata={"hnsw:space": "cosine"},
    )
    _collection = col

    log.info("Embedding %d docs into ChromaDB...", len(docs))
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        col.add(
            ids=[d["id"] for d in batch],
            documents=[d["text"] for d in batch],
            metadatas=[d["metadata"] for d in batch],
        )
        log.info("  %d/%d", min(i + batch_size, len(docs)), len(docs))


def get_chroma_collection() -> chromadb.Collection:
    """Return the cached ChromaDB collection, loading it from disk if needed."""
    global _collection
    if _collection is None:
        _collection = _get_client().get_collection(
            name=COLLECTION_NAME,
            embedding_function=_ef(),
        )
    return _collection


# ── BM25 ──────────────────────────────────────────────────────────────────────

def build_bm25(docs: Optional[list] = None) -> None:
    """Tokenize all KB documents, fit a BM25Okapi index, and pickle it to disk."""
    global _bm25_cache

    if docs is None:
        docs = _load_docs()

    corpus = [tokenize(d["text"]) for d in docs]
    bm25 = BM25Okapi(corpus)

    BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "docs": docs}, f)
    _bm25_cache = (bm25, docs)  # warm the cache immediately so next call skips disk I/O
    log.info("BM25 index saved — %d documents.", len(docs))


def get_bm25_index() -> tuple:
    """Return (BM25Okapi, docs) from the in-memory cache or from disk."""
    global _bm25_cache
    if _bm25_cache is None:
        with open(BM25_INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        _bm25_cache = (data["bm25"], data["docs"])
    return _bm25_cache


# ── Combined ──────────────────────────────────────────────────────────────────

def build_all(batch_size: int = 500) -> None:
    """Build both the ChromaDB and BM25 indexes in a single pass over the KB."""
    docs = _load_docs()  # load once and pass to both builders to avoid double I/O
    log.info("Building ChromaDB (semantic index)...")
    build_chroma(batch_size, docs=docs)
    log.info("Building BM25 (keyword index)...")
    build_bm25(docs=docs)
    log.info("Ingestion complete.")
