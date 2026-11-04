"""Chunk the documents in docs/, embed them and save the index.

    python ingest.py                  # auto: sentence-transformers, else TF-IDF
    python ingest.py --backend tfidf  # fully offline
"""
import argparse

import config
from store import VectorStore, chunk_documents


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["auto", "st", "tfidf"], default=config.EMBED_BACKEND)
    p.add_argument("--chunk-chars", type=int, default=config.CHUNK_CHARS)
    p.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)
    a = p.parse_args()

    chunks = chunk_documents(config.DOCS_DIR, a.chunk_chars, a.overlap)
    store = VectorStore.build(chunks, a.backend, config.EMBED_MODEL)
    store.save(config.INDEX_PATH)
    sources = sorted({c.source for c in chunks})
    print(f"Indexed {len(chunks)} chunks from {len(sources)} documents with '{store.embedder.backend}' embeddings.")
    for s in sources:
        print(f"  {s}: {sum(c.source == s for c in chunks)} chunks")


if __name__ == "__main__":
    main()
