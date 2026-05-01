"""
Configuration for the Naive RAG pipeline.

Loads environment variables, resolves filesystem paths, and exposes
constants consumed by ingestion, retrieval, and generation modules.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent.parent

KB_ALL_DOCS    = BASE_DIR / "dataset" / "knowledge_base" / "kb_all_documents.json"
CHROMA_DB_PATH = BASE_DIR / "chroma_db"
COLLECTION_NAME = "ecommerce_kb"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_raw = os.getenv("GROQ_API_KEYS", "")
GROQ_API_KEYS = [k.strip() for k in _raw.split(",") if k.strip()]
if not GROQ_API_KEYS:
    raise EnvironmentError(
        "GROQ_API_KEYS is not set. Copy .env.example to .env and add your keys."
    )

GROQ_MODEL = "llama-3.3-70b-versatile"

TOP_K = 5
