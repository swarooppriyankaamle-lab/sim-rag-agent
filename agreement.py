"""Check how far the LLM grader agrees with your own labels, beyond chance.

1. Open a results CSV and fill the human_label column with correct / partial / incorrect.
2. python agreement.py results/basic_20261103_230801.csv results/agentic_20261103_230801.csv

Reports raw agreement and Cohen's kappa for each file and for all files pooled:
- kappa: unweighted; any disagreement counts the same.
- weighted kappa: linear weights on the ordered scale correct > partial > incorrect,
  so correct vs incorrect counts twice as much as correct vs partial.
- kappa on correct vs not correct, the pass/fail call.

Two row sets are scored. "All rows" includes the unanswerable and declined questions,
which evaluate.py grades by rule, not with the judge. "Judge-graded rows" keeps only
the answers the LLM judge actually graded, which is the fairer test of the judge.
"""
import csv
import sys
from collections import Counter

LABELS = ["correct", "partial", "incorrect"]


def cohen_kappa(pairs, labels, weights=None):
    """Cohen's kappa for (rater_a, rater_b) pairs. weights: None, or "linear" for an ordered scale."""
    n = len(pairs)
    k = len(labels)
    idx = {l: i for i, l in enumerate(labels)}
    if weights == "linear":
        w = [[abs(i - j) / (k - 1) for j in range(k)] for i in range(k)]
    else:
        w = [[0.0 if i == j else 1.0 for j in range(k)] for i in range(k)]
    obs = [[0] * k for _ in range(k)]
    for a, b in pairs:
        obs[idx[a]][idx[b]] += 1
    row = [sum(obs[i]) for i in range(k)]
    col = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    observed = sum(w[i][j] * obs[i][j] for i in range(k) for j in range(k)) / n
    expected = sum(w[i][j] * row[i] * col[j] for i in range(k) for j in range(k)) / n ** 2
    if expected == 0:
        return float("nan")      # both raters used one label only; kappa is undefined
    return 1 - observed / expected


def judge_graded(r):
    return r.get("answerable_expected") == "True" and r.get("answerable_pred") == "True"


def load(path):
    with open(path, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("human_label", "").strip().lower() in LABELS]


def report(name, rows):
    pairs = [(r["human_label"].strip().lower(), r["grade"]) for r in rows]
    if not pairs:
        print(f"{name}: no rows\n")
        return
    agree = sum(h == g for h, g in pairs)
    binary = [(h == "correct" and "correct" or "not", g == "correct" and "correct" or "not") for h, g in pairs]
    print(f"{name}: {len(pairs)} rows")
    print(f"  raw agreement        {agree}/{len(pairs)} = {agree / len(pairs):.0%}")
    print(f"  Cohen's kappa        {cohen_kappa(pairs, LABELS):.2f}")
    print(f"  weighted kappa       {cohen_kappa(pairs, LABELS, 'linear'):.2f}")
    print(f"  kappa, correct/not   {cohen_kappa(binary, ['correct', 'not']):.2f}")

    counts = Counter(pairs)
    print("  rows = your label, columns = grader label")
    print("  " + f"{'':>11}" + "".join(f"{l:>11}" for l in LABELS))
    for h in LABELS:
        print("  " + f"{h:>11}" + "".join(f"{counts[(h, g)]:>11}" for g in LABELS))
    print()


def main(paths):
    pooled = []
    for path in paths:
        rows = load(path)
        if not rows:
            print(f"{path}: no rows with human_label filled in (use correct / partial / incorrect).\n")
            continue
        for r in rows:
            r["_file"] = path
        pooled += rows
        report(f"{path}, all rows", rows)
        report(f"{path}, judge-graded rows", [r for r in rows if judge_graded(r)])
    if len(paths) > 1 and pooled:
        report("All files, all rows", pooled)
        report("All files, judge-graded rows", [r for r in pooled if judge_graded(r)])

    print("Disagreements:")
    for r in pooled:
        if r["human_label"].strip().lower() != r["grade"]:
            print(f"- {r['_file']} {r['id']}: you={r['human_label']} grader={r['grade']} | {r['grade_reason']}")
    print("\nKappa guide (Landis and Koch): 0.41-0.60 moderate, 0.61-0.80 substantial, above 0.80 almost perfect.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
