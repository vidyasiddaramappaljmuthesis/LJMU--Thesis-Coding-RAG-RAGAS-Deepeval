"""
HyDE RAG — single entry point.

  python -m hyde_rag.run_hyde_rag          # auto-setup + interactive Q&A
  python -m hyde_rag.run_hyde_rag --ingest # force re-index only (no Q&A)
"""
import sys
import chromadb

from .implementation.config import CHROMA_DB_PATH, COLLECTION_NAME
from .implementation.ingestion import build_vector_store
from .implementation.pipeline import run_hyde_rag


def _vector_store_exists() -> bool:
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        col = client.get_collection(COLLECTION_NAME)
        return col.count() > 0
    except Exception:
        return False


def _ensure_vector_store() -> None:
    if _vector_store_exists():
        print("[Setup] Vector store already exists — skipping ingestion.")
    else:
        print("[Setup] Vector store not found. Building now (this runs once)...")
        build_vector_store()
        print("[Setup] Done.\n")


def _interactive() -> None:
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
        print("[Setup] Force re-indexing...")
        build_vector_store()
        print("[Setup] Done.")
    else:
        _ensure_vector_store()
        _interactive()
