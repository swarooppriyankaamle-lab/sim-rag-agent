"""Central settings. Override any value with an environment variable."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "docs"
INDEX_PATH = ROOT / "index" / "index.pkl"
QUESTIONS_PATH = ROOT / "eval" / "questions.jsonl"
RESULTS_DIR = ROOT / "results"

# Claude models (see https://docs.claude.com for current model names)
MODEL = os.getenv("RAG_MODEL", "claude-sonnet-5-5")
GRADER_MODEL = os.getenv("RAG_GRADER_MODEL", MODEL)

# Embeddings: "auto" tries sentence-transformers, falls back to TF-IDF
EMBED_BACKEND = os.getenv("RAG_EMBED_BACKEND", "auto")
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Chunking and retrieval
CHUNK_CHARS = int(os.getenv("RAG_CHUNK_CHARS", "800"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("RAG_TOP_K", "4"))

# Model backend: "anthropic" (Claude, needs ANTHROPIC_API_KEY) or "local" (open-source model, no key)
PROVIDER = os.getenv("RAG_PROVIDER", "anthropic")

# Agent loop
MAX_AGENT_STEPS = int(os.getenv("RAG_MAX_STEPS", "8"))
