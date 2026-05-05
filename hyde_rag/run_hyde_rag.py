"""
HyDE RAG — single entry point.

Usage::

    python -m hyde_rag.run_hyde_rag          # auto-setup + interactive Q&A
    python -m hyde_rag.run_hyde_rag --ingest # force re-index only (no Q&A)

On first run the ChromaDB vector store is built from the KB JSON file
(one-time, ~2 min).  Subsequent runs reuse the persisted index.
"""
import logging
import sys

import chromadb

from .implementation.config import CHROMA_DB_PATH, COLLECTION_NAME
from .implementation.ingestion import build_vector_store
from .implementation.pipeline import run_hyde_rag

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _vector_store_exists() -> bool:
    """Return ``True`` if the ChromaDB collection exists and is populated.

    Returns:
        ``True`` when the collection contains at least one document;
        ``False`` on any error or if the collection is empty.
    """
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        col = client.get_collection(COLLECTION_NAME)
        return col.count() > 0
    except Exception:
        return False


def _ensure_vector_store() -> None:
    """Build the ChromaDB vector store if it is not already populated.

    No-op when the store exists; triggers ``build_vector_store``
    (one-time, ~2 min) otherwise and logs setup progress.
    """
    if _vector_store_exists():
        log.info("[Setup] Vector store already exists — skipping ingestion.")
    else:
        log.info("[Setup] Vector store not found. Building now (this runs once)...")
        build_vector_store()
        log.info("[Setup] Done.")


def _interactive() -> None:
    """Run an interactive question-answering loop using the HyDE RAG pipeline.

    Reads queries from stdin, runs the full HyDE pipeline (hypothetical-doc
    generation → embedding retrieval → grounded answer generation), and prints
    the hypothetical document, answer, and retrieved document metadata.
    Type ``exit`` to quit.
    """
    print("=" * 60)
    print("  E-Commerce HyDE RAG  |  LLaMA 3.3 70B via Groq")
    print("  HyDE      : Hypothetical Document Embeddings")
    print("  Embedding : sentence-transformers/all-MiniLM-L6-v2")
    print("  Vector DB : ChromaDB (cosine)")
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
        result = run_hyde_rag(query)

        print("Hypothetical document (used for retrieval):")
        print(result["hypothetical_doc"])
        print()
        print(f"Answer:\n{result['answer']}")
        print()
        print("Retrieved documents:")
        for doc in result["retrieved_docs"]:
            print(
                f"  [{doc['id']}]  "
                f"distance={doc['distance']:.4f}  "
                f"type={doc['metadata'].get('document_type', 'n/a')}"
            )
        print()


if __name__ == "__main__":
    if "--ingest" in sys.argv:
        log.info("[Setup] Force re-indexing...")
        build_vector_store()
        log.info("[Setup] Done.")
    else:
        _ensure_vector_store()
        _interactive()
