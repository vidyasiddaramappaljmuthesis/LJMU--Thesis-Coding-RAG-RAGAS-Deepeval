"""
Hybrid RAG — single entry point.

Usage::

    python -m hybrid_rag.run_hybrid_rag          # auto-setup + interactive Q&A
    python -m hybrid_rag.run_hybrid_rag --ingest # force re-index both stores

Both the ChromaDB semantic index and the BM25 keyword index are built on
first run and reused on subsequent runs.
"""
import logging
import sys

import chromadb

from .implementation.config import CHROMA_DB_PATH, COLLECTION_NAME, BM25_INDEX_PATH
from .implementation.ingestion import build_all, build_chroma, build_bm25
from .implementation.pipeline import run_hybrid_rag

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _chroma_exists() -> bool:
    """Return ``True`` if the ChromaDB collection is populated.

    Returns:
        ``True`` when the collection exists and contains at least one document.
    """
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        return client.get_collection(COLLECTION_NAME).count() > 0
    except Exception:
        return False


def _bm25_exists() -> bool:
    """Return ``True`` if the pickled BM25 index file is present on disk."""
    return BM25_INDEX_PATH.exists()


def _ensure_setup() -> None:
    """Build missing indexes (ChromaDB and/or BM25) if not already present.

    Checks each index independently so only the missing one is rebuilt
    when, for example, only the BM25 pickle has been deleted.
    """
    chroma_ok = _chroma_exists()
    bm25_ok   = _bm25_exists()

    if chroma_ok and bm25_ok:
        log.info("[Setup] ChromaDB + BM25 index found — skipping ingestion.")
        return

    if not chroma_ok:
        log.info("[Setup] ChromaDB not found. Building semantic index...")
        build_chroma()

    if not bm25_ok:
        log.info("[Setup] BM25 index not found. Building keyword index...")
        build_bm25()

    log.info("[Setup] Done.")


def _interactive() -> None:
    """Run an interactive question-answering loop using the Hybrid RAG pipeline.

    Reads queries from stdin, runs BM25 + semantic search with RRF fusion,
    generates an answer, and prints the result.  Type ``exit`` to quit.
    """
    print("=" * 60)
    print("  E-Commerce Hybrid RAG  |  LLaMA 3.3 70B via Groq")
    print("  Keyword  : BM25 (rank_bm25)")
    print("  Semantic : sentence-transformers/all-MiniLM-L6-v2")
    print("  Fusion   : Reciprocal Rank Fusion (RRF k=60)")
    print("  VectorDB : ChromaDB (cosine)")
    print("  Type 'exit' to quit.")
    print("=" * 60)
    print()

    while True:
        try:
            query = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        print()
        result = run_hybrid_rag(query)

        print(f"Answer:\n{result['answer']}")
        print()
        print(f"Top-{len(result['retrieved_docs'])} docs after RRF fusion:")
        for doc in result["retrieved_docs"]:
            print(
                f"  [{doc['id']}]  rrf={doc['rrf_score']:.5f}  "
                f"type={doc['metadata'].get('document_type', 'n/a')}"
            )
        print()


if __name__ == "__main__":
    if "--ingest" in sys.argv:
        log.info("[Setup] Force re-indexing both stores...")
        build_all()
    else:
        _ensure_setup()
        _interactive()
