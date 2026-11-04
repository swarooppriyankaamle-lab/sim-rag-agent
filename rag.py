"""Two pipelines over the same store:

- answer_basic:   retrieve top-k once, then one Claude call (classic RAG baseline)
- answer_agentic: Claude decides when and what to search, reads context, uses a
                  calculator, and submits a cited answer (agentic RAG)

Both return the same result dict, so the evaluation harness can compare them.
"""
from __future__ import annotations

import time

import config
from llm import make_client
from store import Chunk, VectorStore
from tools import calculate

SUBMIT_TOOL = {
    "name": "submit_answer",
    "description": "Submit the final answer. Call exactly once, at the end.",
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "description": "The answer, grounded only in retrieved text."},
            "citations": {
                "type": "array", "items": {"type": "string"},
                "description": "Chunk ids (e.g. 'mesh_convergence.md#3') that support the answer.",
            },
            "answerable": {
                "type": "boolean",
                "description": "False if the documents do not contain the answer.",
            },
        },
        "required": ["answer", "citations", "answerable"],
    },
}

AGENT_TOOLS = [
    {
        "name": "search_docs",
        "description": "Semantic search over the simulation document collection. Returns chunk ids and text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "description": "Number of results (1-8).", "default": config.TOP_K},
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_chunk",
        "description": "Read a chunk plus its neighbouring chunks from the same document, for fuller context.",
        "input_schema": {
            "type": "object",
            "properties": {"chunk_id": {"type": "string"}},
            "required": ["chunk_id"],
        },
    },
    {
        "name": "calculator",
        "description": "Evaluate arithmetic. Supports + - * / ** and sqrt, ln, log10, exp, sin, cos, tan, pi.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
    SUBMIT_TOOL,
]

BASIC_SYSTEM = (
    "You answer engineering simulation questions using ONLY the document excerpts provided. "
    "Cite the chunk ids you relied on. If the excerpts do not contain the answer, set answerable "
    "to false and say so instead of guessing. Respond by calling submit_answer."
)

AGENT_SYSTEM = """You are a simulation-engineering assistant with tools over a document collection.
1. Search the documents with search_docs. If results are off-target, reformulate and search again.
2. Use read_chunk when an excerpt is cut off or you need surrounding detail.
3. Use calculator for any arithmetic instead of computing in your head.
4. Answer ONLY from text you retrieved, and cite every chunk id you used.
5. If, after searching, the documents do not contain the answer, submit with answerable=false.
   Do not fill gaps from general knowledge.
Finish by calling submit_answer exactly once."""


def format_chunk(c: Chunk, score: float | None = None) -> str:
    s = f"; score {score:.3f}" if score is not None else ""
    return f"[{c.id}] (source: {c.source}; section: {c.section or '-'}{s})\n{c.text}"


def _add_usage(totals: dict, resp) -> None:
    totals["input_tokens"] += resp.usage.input_tokens
    totals["output_tokens"] += resp.usage.output_tokens


def _result(submission: dict, retrieved: list[str], trace: list, steps: int, usage: dict, t0: float) -> dict:
    return {
        "answer": submission.get("answer", ""),
        "citations": list(submission.get("citations", []) or []),
        "answerable": bool(submission.get("answerable", True)),
        "retrieved": retrieved,
        "steps": steps,
        "trace": trace,
        "latency_s": round(time.time() - t0, 2),
        **usage,
    }


# ---------------------------------------------------------------- basic RAG

def answer_basic(question: str, store: VectorStore, client=None,
                 k: int = config.TOP_K) -> dict:
    client = client or make_client()
    t0, usage = time.time(), {"input_tokens": 0, "output_tokens": 0}
    hits = store.search(question, k)
    context = "\n\n---\n\n".join(format_chunk(c, s) for c, s in hits)
    resp = client.messages.create(
        model=config.MODEL, max_tokens=1024, system=BASIC_SYSTEM,
        tools=[SUBMIT_TOOL], tool_choice={"type": "tool", "name": "submit_answer"},
        messages=[{"role": "user", "content": f"Document excerpts:\n\n{context}\n\nQuestion: {question}"}],
    )
    _add_usage(usage, resp)
    submission = next(b.input for b in resp.content if b.type == "tool_use")
    trace = [{"tool": "search_docs", "input": {"query": question, "k": k}, "output_ids": [c.id for c, _ in hits]}]
    return _result(submission, [c.id for c, _ in hits], trace, 1, usage, t0)


# ---------------------------------------------------------------- agentic RAG

def _run_tool(name: str, args: dict, store: VectorStore, retrieved: list[str]) -> str:
    def remember(ids):
        for i in ids:
            if i not in retrieved:
                retrieved.append(i)

    if name == "search_docs":
        k = max(1, min(int(args.get("k", config.TOP_K)), 8))
        hits = store.search(str(args.get("query", "")), k)
        remember(c.id for c, _ in hits)
        return "\n\n---\n\n".join(format_chunk(c, s) for c, s in hits) or "No results."
    if name == "read_chunk":
        chunks = store.neighbors(str(args.get("chunk_id", "")), radius=1)
        if not chunks:
            return f"Unknown chunk id: {args.get('chunk_id')}. Use ids returned by search_docs."
        remember(c.id for c in chunks)
        return "\n\n---\n\n".join(format_chunk(c) for c in chunks)
    if name == "calculator":
        try:
            return repr(calculate(str(args.get("expression", ""))))
        except Exception as exc:
            return f"Error: {exc}"
    return f"Unknown tool: {name}"


def answer_agentic(question: str, store: VectorStore, client=None,
                   max_steps: int = config.MAX_AGENT_STEPS) -> dict:
    client = client or make_client()
    t0, usage = time.time(), {"input_tokens": 0, "output_tokens": 0}
    messages: list[dict] = [{"role": "user", "content": question}]
    retrieved: list[str] = []
    trace: list[dict] = []

    for step in range(1, max_steps + 1):
        resp = client.messages.create(
            model=config.MODEL, max_tokens=2048, system=AGENT_SYSTEM, tools=AGENT_TOOLS, messages=messages,
        )
        _add_usage(usage, resp)
        messages.append({"role": "assistant", "content": resp.content})
        tool_uses = [b for b in resp.content if b.type == "tool_use"]

        if not tool_uses:  # answered in prose: ask for the structured submission
            messages.append({"role": "user", "content": "Call submit_answer with your final answer and citations."})
            continue

        results = []
        for tu in tool_uses:
            if tu.name == "submit_answer":
                trace.append({"tool": "submit_answer", "input": tu.input})
                return _result(tu.input, retrieved, trace, step, usage, t0)
            output = _run_tool(tu.name, tu.input, store, retrieved)
            trace.append({"tool": tu.name, "input": tu.input, "output_preview": output[:300]})
            results.append({"type": "tool_result", "tool_use_id": tu.id, "content": output})
        messages.append({"role": "user", "content": results})

    # Step budget used up: force a final answer from what it has.
    if messages[-1]["role"] == "assistant":
        messages.append({"role": "user", "content": "Step limit reached. Submit your best answer now."})
    resp = client.messages.create(
        model=config.MODEL, max_tokens=1024, system=AGENT_SYSTEM, tools=AGENT_TOOLS,
        tool_choice={"type": "tool", "name": "submit_answer"}, messages=messages,
    )
    _add_usage(usage, resp)
    submission = next(b.input for b in resp.content if b.type == "tool_use")
    trace.append({"tool": "submit_answer", "input": submission, "forced": True})
    return _result(submission, retrieved, trace, max_steps + 1, usage, t0)


if __name__ == "__main__":
    import argparse
    import json

    p = argparse.ArgumentParser(description="Ask one question.")
    p.add_argument("question")
    p.add_argument("--mode", choices=["basic", "agentic"], default="agentic")
    a = p.parse_args()
    s = VectorStore.load(config.INDEX_PATH)
    out = (answer_agentic if a.mode == "agentic" else answer_basic)(a.question, s)
    print(out["answer"])
    print("\nCitations:", ", ".join(out["citations"]) or "-")
    print(f"Answerable: {out['answerable']} | steps: {out['steps']} | {out['latency_s']}s")
    print("\nTrace:\n" + json.dumps([{k: v for k, v in t.items() if k != 'output_preview'} for t in out["trace"]], indent=2))
