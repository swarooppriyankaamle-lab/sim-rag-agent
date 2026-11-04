"""Retrieval-only evaluation: no API key needed.

For each answerable question, checks whether a chunk from an expected source
document appears in the top-k search results, across embedding backends and
chunk sizes.

    python retrieval_eval.py
    python retrieval_eval.py --backends tfidf --chunk-sizes 400 800

Metrics
  recall@k  share of answerable questions with an expected source in the top k
  MRR       mean of 1 / rank of the first expected-source chunk (0 if not in top 10)
"""
import argparse
import json
import time

import config
from store import VectorStore, chunk_documents


def load_questions(path):
    with open(path, encoding="utf-8") as f:
        return [q for q in (json.loads(l) for l in f if l.strip()) if q["answerable"] and q["expected_sources"]]


def evaluate(store, questions, ks, depth=10):
    hits = {k: 0 for k in ks}
    rr = 0.0
    for q in questions:
        ranked = [c.source for c, _ in store.search(q["question"], depth)]
        first = next((i for i, s in enumerate(ranked, 1) if s in q["expected_sources"]), None)
        rr += 1 / first if first else 0
        for k in ks:
            hits[k] += bool(first and first <= k)
    n = len(questions)
    return {**{f"recall@{k}": hits[k] / n for k in ks}, "MRR": rr / n}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backends", nargs="+", default=["tfidf", "st"], choices=["tfidf", "st"])
    p.add_argument("--chunk-sizes", nargs="+", type=int, default=[400, 800, 1600])
    p.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4])
    a = p.parse_args()

    questions = load_questions(config.QUESTIONS_PATH)
    print(f"{len(questions)} answerable questions\n")
    header = ["backend", "chunk_chars", "chunks"] + [f"recall@{k}" for k in a.ks] + ["MRR"]
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    for backend in a.backends:
        for size in a.chunk_sizes:
            chunks = chunk_documents(config.DOCS_DIR, size, min(config.CHUNK_OVERLAP, size // 4))
            store = VectorStore.build(chunks, backend, config.EMBED_MODEL)
            m = evaluate(store, questions, a.ks)
            cells = [backend, str(size), str(len(chunks))] + [f"{m[f'recall@{k}']:.0%}" for k in a.ks] + [f"{m['MRR']:.2f}"]
            lines.append("| " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    print(table)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    out = config.RESULTS_DIR / f"retrieval_{time.strftime('%Y%m%d_%H%M%S')}.md"
    out.write_text(f"# Retrieval evaluation\n\n{len(questions)} answerable questions\n\n{table}\n", encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
