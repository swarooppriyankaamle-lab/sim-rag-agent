"""Check how often the LLM grader agrees with your own labels.

1. Open a results CSV and fill the human_label column with correct / partial / incorrect.
2. python agreement.py results/agentic_20261103_220000.csv
"""
import csv
import sys
from collections import Counter

LABELS = ["correct", "partial", "incorrect"]


def main(path: str):
    with open(path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("human_label", "").strip().lower() in LABELS]
    if not rows:
        sys.exit("No rows with human_label filled in (use correct / partial / incorrect).")

    pairs = [(r["human_label"].strip().lower(), r["grade"]) for r in rows]
    agree = sum(h == g for h, g in pairs)
    print(f"Grader agreement: {agree}/{len(pairs)} = {agree / len(pairs):.0%}\n")

    counts = Counter(pairs)
    print("Rows = your label, columns = grader label")
    print(f"{'':>11}" + "".join(f"{l:>11}" for l in LABELS))
    for h in LABELS:
        print(f"{h:>11}" + "".join(f"{counts[(h, g)]:>11}" for g in LABELS))

    print("\nDisagreements:")
    for r in rows:
        if r["human_label"].strip().lower() != r["grade"]:
            print(f"- {r['id']}: you={r['human_label']} grader={r['grade']} | {r['grade_reason']}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
