# Results: basic RAG vs agentic RAG

Run `20261103_230801`. Raw outputs are in `results/` (CSV per mode with grades and reviewed labels, JSONL with full agent traces).

## Setup

| | |
| --- | --- |
| Model | Qwen2.5-3B-Instruct, 4-bit GGUF (`qwen2.5-3b-instruct-q4_k_m.gguf`) via llama.cpp |
| Hardware | 2-core cloud CPU, no GPU |
| Corpus | 4 simulation-engineering documents, 21 chunks (800 characters), sentence-transformers embeddings |
| Questions | 12: 9 answerable, 3 not covered by the documents |
| Grading | LLM-as-judge (same model), then every answer reviewed by hand against the reference |

## Headline numbers

| Metric | Basic RAG | Agentic RAG |
| --- | --- | --- |
| Accuracy, hand-reviewed | 88% | 92% |
| Accuracy, LLM judge | 92% | 96% |
| Retrieval recall (expected source retrieved) | 100% | 100% |
| Citation hit (answer cites an expected source) | 89% | 100% |
| Grounded citations (cited ids were retrieved) | 100% | 100% |
| Refusal accuracy (3 unanswerable) | 100% | 100% |
| False refusals (9 answerable) | 0% | 0% |
| Model calls per answer | 1.0 | 2.4 |
| Latency per answer | 43 s | 107 s |
| Tokens per answer | 867 | 2,861 |

Accuracy scores correct as 1, partial as 0.5, incorrect as 0. With 12 questions, the 4-point accuracy gap is half a question: a signal, not proof.

## What the traces show

- **The calculator matters.** On the plasticity conversion (q07: 300 MPa at 0.05 engineering strain, E = 200,000 MPa), basic RAG answered 325 MPa, 0.0513 and 0.0013, all wrong. The agent called the calculator and gave 315 MPa, 0.0488 and 0.0472, all correct.
- **Agents cost more, most of all when the answer isn't there.** On q10 (license tokens, not in the documents) the agent searched 4 times and used 8,914 tokens before declining; basic RAG declined with 939.
- **The agent is not better on every question.** On q09 (12% hourglass energy) basic RAG gave the fuller checklist; the agent's answer missed the added-mass and energy-balance checks.
- **Structured output can break.** On q09 basic RAG returned its answer wrapped in raw JSON with no citations, which is why its citation hit is 89%.

## Grading the grader

The LLM judge agreed with the hand review on 22 of 24 answers (92%). Both disagreements were too generous:

| Question | Mode | LLM judge | Reviewed | Why |
| --- | --- | --- | --- | --- |
| q07 | basic | partial | incorrect | Judge said the true stress was correct; it was 325 MPa, not 315 MPa |
| q03 | agentic | correct | partial | Missed the main fix: removing rigid body motion before contact closes |

Raw agreement flatters the judge here, because most answers are correct and two raters who both say "correct" most of the time agree a lot by chance. Cohen's kappa corrects for that:

| Rows | n | Raw agreement | Cohen's kappa | Weighted kappa (linear) | Kappa, correct vs not |
| --- | --- | --- | --- | --- | --- |
| Basic, all | 12 | 92% | 0.71 | 0.77 | 1.00 |
| Agentic, all | 12 | 92% | 0.62 | 0.62 | 0.62 |
| Both, all | 24 | 92% | 0.67 | 0.71 | 0.83 |
| Both, judge-graded only | 18 | 89% | 0.66 | 0.70 | 0.82 |

Confusion matrix, both modes (rows = hand review, columns = LLM judge):

| | correct | partial | incorrect |
| --- | --- | --- | --- |
| correct | 20 | 0 | 0 |
| partial | 1 | 2 | 0 |
| incorrect | 0 | 1 | 0 |

- **92% agreement becomes kappa 0.67: substantial, not near-perfect** (Landis and Koch: 0.61 to 0.80). Chance agreement alone is 74% with these label counts.
- "Judge-graded only" drops the 6 unanswerable questions, which `evaluate.py` grades by rule, not with the judge. Kappa barely moves (0.66), so the rule-graded rows weren't inflating it.
- Both misses are one step too generous, and the judge never called anything incorrect. That is a bias, not noise: a judge that leans lenient will overstate accuracy, which is what happened here (92% and 96% by judge vs 88% and 92% by hand).
- With 24 labels and only one "incorrect", the kappa estimate is rough. A rubric that asks the judge to check each key number against the reference, and 50+ labelled answers, would test it properly.

Reproduce with `python agreement.py results/basic_20261103_230801.csv results/agentic_20261103_230801.csv`.

## Limitations and next steps

- 12 questions and 4 documents are a small test; retrieval is easy at this size (100% recall for both).
- The judge is the same 3B model being evaluated.
- Next: 50+ questions with overlapping documents, a run with Claude, and a stronger independent judge.
