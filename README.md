# Agentic RAG for Simulation Troubleshooting, with an Evaluation Harness

A Claude-based agent answers simulation-engineering questions (contact convergence, mesh convergence, explicit dynamics checks, units and plasticity input) from a document collection. It decides when and what to search, reads surrounding context, uses a calculator for arithmetic, and submits a cited answer, or declines when the documents don't cover the question.

The main point of the project is the **evaluation harness**: a ground-truth question set and metrics that compare a basic RAG baseline with the agentic version.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...                    # Windows: set ANTHROPIC_API_KEY=...
```

The model defaults to `claude-sonnet-5-5`. Change it with `RAG_MODEL` (see `config.py` for every setting).

**No API key? Run it on an open-source model.** The same pipeline runs locally on CPU with Qwen2.5-3B-Instruct through llama.cpp. Slower and less capable than Claude, but free:

```bash
pip install llama-cpp-python huggingface_hub
export RAG_PROVIDER=local          # downloads the ~2 GB model on first use
python evaluate.py --mode both     # about an hour on a 2-core CPU
```

## Run

```bash
python ingest.py                                       # chunk + embed docs/ into index/
python rag.py "Why does my contact model keep cutting back?"            # agentic, one question
python rag.py --mode basic "What density should I use for steel in mm-N-tonne-s?"
python evaluate.py --mode both                         # full comparison
python evaluate.py --mode agentic --limit 3            # quick, cheap check
python retrieval_eval.py                               # retrieval only, no API key needed
```

`ingest.py` uses sentence-transformers embeddings and falls back to TF-IDF if the model can't be downloaded (`--backend tfidf` forces it). Run `python ingest.py` again whenever you change `docs/`.

## How it works

| File | Purpose |
| --- | --- |
| `store.py` | Loads .md/.txt/.pdf, chunks by heading with overlapping windows, embeds, cosine search |
| `rag.py` | `answer_basic` (retrieve once, answer once) and `answer_agentic` (tool-use loop) |
| `tools.py` | Safe calculator the agent uses for arithmetic |
| `grader.py` | LLM-as-judge: correct / partial / incorrect against an expert reference |
| `evaluate.py` | Runs the question set, scores both pipelines, saves CSV, traces and a summary table |
| `agreement.py` | Measures how often the LLM grader agrees with your own labels |
| `retrieval_eval.py` | Retrieval-only benchmark across embedding methods and chunk sizes (no model needed) |
| `llm.py`, `local_llm.py` | Backend switch: Claude via the Anthropic API, or a local open-source model |
| `eval/questions.jsonl` | Ground-truth questions, including unanswerable ones |

Agent tools: `search_docs`, `read_chunk`, `calculator`, `submit_answer` (structured answer, chunk-id citations and an `answerable` flag).

## Metrics

| Metric | What it measures |
| --- | --- |
| accuracy | Mean grade (correct 1, partial 0.5, incorrect 0) over all questions |
| retrieval_recall | Answerable questions where an expected source was retrieved |
| citation_hit | Answerable questions where the answer cites an expected source |
| grounded_citations | Answers whose cited chunk ids were actually retrieved (no invented citations) |
| refusal_accuracy | Unanswerable questions correctly declined |
| false_refusal_rate | Answerable questions wrongly declined |
| mean_steps, latency, tokens | Cost of each approach |

## Checking the grader

An LLM grader is itself a model to validate. Open a results CSV, fill the `human_label` column for at least 20 rows with your own judgment, then:

```bash
python agreement.py results/agentic_<timestamp>.csv
```

## Make it yours

1. **Grow the corpus.** Add 30–50 documents you're allowed to use: public user-guide pages, public papers, your own notes. Don't add employer or customer material.
2. **Grow the question set** to 40–60, with about 20% unanswerable and a few that need multi-step reasoning or a calculation. Write the reference answers yourself; your domain expertise is the value here.
3. **Run one experiment** and report it: basic vs agentic, chunk size 400 vs 800 vs 1600 (`--chunk-chars`), or top-k 2 vs 4 vs 8.
4. **Write up the failure modes.** Save interesting wrong answers from the `.jsonl` traces: where did retrieval miss, where did the agent answer from general knowledge instead of the documents?
5. **Stretch: feedback loop.** Turn the failures into prompt or retrieval changes, re-run the evaluation and show the improvement.

## Results

First run on Qwen2.5-3B-Instruct (open source, 4-bit, CPU only), 12 questions. Full write-up in [RESULTS.md](RESULTS.md).

| Metric | Basic RAG | Agentic RAG |
| --- | --- | --- |
| Accuracy, hand-reviewed | 88% | 92% |
| Citation hit | 89% | 100% |
| Refusal accuracy | 100% | 100% |
| Tokens per answer | 867 | 2,861 |
| Latency per answer | 43 s | 107 s |

The LLM judge agreed with hand review on 22 of 24 answers; both misses were too generous.
