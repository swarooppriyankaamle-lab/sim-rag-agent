"""Evaluation harness: run basic and/or agentic RAG over a ground-truth question set.

    python evaluate.py --mode both
    python evaluate.py --mode agentic --limit 5

Metrics (per mode)
  accuracy             mean grade score (correct 1, partial 0.5, incorrect 0) over all questions
  retrieval_recall     answerable questions where an expected source was retrieved
  citation_hit         answerable questions where the answer cites an expected source
  grounded_citations   answers whose cited chunk ids were all actually retrieved (no invented ids)
  refusal_accuracy     unanswerable questions the system correctly declined
  false_refusal_rate   answerable questions the system wrongly declined
  steps / latency / tokens   cost of each approach

Writes results/<mode>_<timestamp>.csv (with an empty human_label column for
grader-agreement checks), a .jsonl with full traces, and summary_<timestamp>.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import config
from grader import SCORE, grade
from llm import make_client, model_label
from rag import answer_agentic, answer_basic
from store import VectorStore

PIPELINES = {"basic": answer_basic, "agentic": answer_agentic}
CSV_FIELDS = [
    "id", "mode", "question", "answerable_expected", "answerable_pred", "grade", "grade_reason",
    "human_label", "retrieval_hit", "citation_hit", "citations_grounded", "refusal_correct",
    "steps", "latency_s", "input_tokens", "output_tokens", "answer", "citations", "retrieved", "error",
]


def load_questions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def source_of(chunk_id: str) -> str:
    return chunk_id.split("#", 1)[0]


def score_one(q: dict, out: dict, client) -> dict:
    expected = set(q.get("expected_sources", []))
    retrieved = set(out["retrieved"])
    row = {
        "answerable_pred": out["answerable"],
        "citations_grounded": all(c in retrieved for c in out["citations"]),
        "refusal_correct": out["answerable"] == q["answerable"],
        "retrieval_hit": "", "citation_hit": "",
    }
    if q["answerable"] and expected:
        row["retrieval_hit"] = bool(expected & {source_of(c) for c in retrieved})
        row["citation_hit"] = bool(expected & {source_of(c) for c in out["citations"]})

    if not q["answerable"]:
        ok = not out["answerable"]
        row["grade"] = "correct" if ok else "incorrect"
        row["grade_reason"] = "Correctly declined." if ok else "Answered a question the documents cannot answer."
    elif not out["answerable"]:
        row["grade"], row["grade_reason"] = "incorrect", "Declined an answerable question."
    else:
        g = grade(q["question"], q["reference_answer"], out["answer"], client)
        row["grade"], row["grade_reason"] = g["label"], g["reason"]
    return row


def mean(xs):
    xs = [float(x) for x in xs if x != "" and x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r.get("error")]
    ans = [r for r in ok if r["answerable_expected"]]
    unans = [r for r in ok if not r["answerable_expected"]]
    return {
        "n": len(rows),
        "errors": len(rows) - len(ok),
        "accuracy": mean(SCORE[r["grade"]] for r in ok),
        "retrieval_recall": mean(r["retrieval_hit"] for r in ans),
        "citation_hit": mean(r["citation_hit"] for r in ans),
        "grounded_citations": mean(r["citations_grounded"] for r in ok),
        "refusal_accuracy": mean(r["refusal_correct"] for r in unans),
        "false_refusal_rate": mean(not r["answerable_pred"] for r in ans),
        "mean_steps": mean(r["steps"] for r in ok),
        "mean_latency_s": mean(r["latency_s"] for r in ok),
        "mean_tokens": mean(r["input_tokens"] + r["output_tokens"] for r in ok),
    }


def summary_table(summaries: dict[str, dict]) -> str:
    modes = list(summaries)
    pct = {"accuracy", "retrieval_recall", "citation_hit", "grounded_citations", "refusal_accuracy", "false_refusal_rate"}
    lines = ["| Metric | " + " | ".join(modes) + " |", "| --- |" + " --- |" * len(modes)]
    for key in next(iter(summaries.values())):
        cells = []
        for m in modes:
            v = summaries[m][key]
            cells.append(f"{v:.0%}" if key in pct and v == v else (f"{v:.1f}" if isinstance(v, float) else str(v)))
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["basic", "agentic", "both"], default="both")
    p.add_argument("--questions", type=Path, default=config.QUESTIONS_PATH)
    p.add_argument("--limit", type=int, default=None, help="Only run the first N questions.")
    a = p.parse_args()

    store = VectorStore.load(config.INDEX_PATH)
    client = make_client()
    questions = load_questions(a.questions)[: a.limit]
    modes = ["basic", "agentic"] if a.mode == "both" else [a.mode]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    config.RESULTS_DIR.mkdir(exist_ok=True)
    summaries = {}

    for mode in modes:
        rows, traces = [], []
        for i, q in enumerate(questions, 1):
            row = {f: "" for f in CSV_FIELDS}
            row.update(id=q["id"], mode=mode, question=q["question"], answerable_expected=q["answerable"])
            try:
                out = PIPELINES[mode](q["question"], store, client)
                row.update(score_one(q, out, client))
                row.update(
                    steps=out["steps"], latency_s=out["latency_s"], input_tokens=out["input_tokens"],
                    output_tokens=out["output_tokens"], answer=out["answer"],
                    citations=";".join(out["citations"]), retrieved=";".join(out["retrieved"]),
                )
                traces.append({"id": q["id"], "mode": mode, **out, "grade": row["grade"]})
            except Exception as exc:  # keep going; the error is recorded in the CSV
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            print(f"[{mode}] {i}/{len(questions)} {q['id']}: {row['grade'] or 'ERROR ' + row['error']}")

        with open(config.RESULTS_DIR / f"{mode}_{stamp}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(rows)
        with open(config.RESULTS_DIR / f"{mode}_{stamp}.jsonl", "w", encoding="utf-8") as f:
            for t in traces:
                f.write(json.dumps(t, default=str) + "\n")
        summaries[mode] = summarize(rows)

    table = summary_table(summaries)
    (config.RESULTS_DIR / f"summary_{stamp}.md").write_text(
        f"# Evaluation summary {stamp}\n\nModel: {model_label()} | questions: {len(questions)}\n\n{table}\n",
        encoding="utf-8",
    )
    print("\n" + table + f"\n\nResults saved in {config.RESULTS_DIR}/ (timestamp {stamp})")


if __name__ == "__main__":
    main()
