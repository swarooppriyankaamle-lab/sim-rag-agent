"""Local open-source model backend (no API key) that mimics the subset of the
Anthropic Messages API this project uses, so rag.py, grader.py and evaluate.py
run unchanged.

Uses llama.cpp with a GGUF model such as Qwen2.5-3B-Instruct, and Qwen's
<tool_call> format for tool use. Forced tool_choice is done by prefilling the
start of the tool call.

    pip install llama-cpp-python huggingface_hub
    RAG_PROVIDER=local python evaluate.py --mode both
"""
from __future__ import annotations

import itertools
import json
import os
import re
from types import SimpleNamespace

DEFAULT_REPO = "Qwen/Qwen2.5-3B-Instruct-GGUF"
DEFAULT_FILE = "qwen2.5-3b-instruct-q4_k_m.gguf"
_ids = itertools.count()


def _model_path() -> str:
    path = os.getenv("RAG_LOCAL_MODEL_PATH")
    if path and os.path.exists(path):
        return path
    from huggingface_hub import hf_hub_download
    return hf_hub_download(os.getenv("RAG_LOCAL_REPO", DEFAULT_REPO), os.getenv("RAG_LOCAL_FILE", DEFAULT_FILE))


def _block(type_, **kw):
    return SimpleNamespace(type=type_, **kw)


def _tool_use(name, args):
    return _block("tool_use", id=f"local_{next(_ids)}", name=name, input=args)


def _first_json_object(text: str):
    """Return (obj, end_index) for the first balanced {...} in text, or (None, -1)."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                esc = (ch == "\\") and not esc
                if ch == '"' and not esc:
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1]), i + 1
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None, -1


class _Messages:
    def __init__(self, llm, max_ctx):
        self.llm = llm
        self.max_ctx = max_ctx

    # ------------------------------------------------ prompt building
    @staticmethod
    def _tools_block(tools):
        lines = [json.dumps({"type": "function", "function": {
            "name": t["name"], "description": t.get("description", ""), "parameters": t["input_schema"]}})
            for t in tools]
        return (
            "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
            + "\n".join(lines) + "\n</tools>\n\n"
            "For each function call, return a json object with function name and arguments within "
            "<tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"
        )

    def _render(self, system, tools, messages):
        out = ["<|im_start|>system\n" + (system or "You are a helpful assistant.") + (self._tools_block(tools) if tools else "") + "<|im_end|>\n"]
        for m in messages:
            content = m["content"]
            if m["role"] == "user":
                if isinstance(content, str):
                    out.append(f"<|im_start|>user\n{content}<|im_end|>\n")
                else:  # list of tool_result blocks
                    parts = [f"<tool_response>\n{b['content']}\n</tool_response>" for b in content if b.get("type") == "tool_result"]
                    out.append("<|im_start|>user\n" + "\n".join(parts) + "<|im_end|>\n")
            else:
                pieces = []
                for b in content if not isinstance(content, str) else [_block("text", text=content)]:
                    if b.type == "text" and b.text:
                        pieces.append(b.text)
                    elif b.type == "tool_use":
                        pieces.append("<tool_call>\n" + json.dumps({"name": b.name, "arguments": b.input}) + "\n</tool_call>")
                out.append("<|im_start|>assistant\n" + "\n".join(pieces) + "<|im_end|>\n")
        out.append("<|im_start|>assistant\n")
        return "".join(out)

    def _generate(self, prompt, max_tokens):
        n_prompt = len(self.llm.tokenize(prompt.encode(), special=True))
        budget = max(64, min(max_tokens, self.max_ctx - n_prompt - 8))
        res = self.llm.create_completion(prompt, max_tokens=budget, temperature=0.0, stop=["<|im_end|>", "<|endoftext|>"])
        return res["choices"][0]["text"], n_prompt, res["usage"]["completion_tokens"]

    # ------------------------------------------------ public API
    def create(self, model=None, max_tokens=1024, system=None, tools=None, tool_choice=None, messages=None, **_):
        tools = tools or []
        forced = tool_choice and tool_choice.get("type") == "tool" and tool_choice.get("name")
        prompt = self._render(system, tools, messages)

        if forced:
            prefix = '<tool_call>\n{"name": "' + forced + '", "arguments": '
            text, n_in, n_out = self._generate(prompt + prefix, max_tokens)
            args, _ = _first_json_object(text)
            if not isinstance(args, dict):
                # Fall back: treat whatever was produced as the answer text.
                args = {"answer": text.strip(), "citations": [], "answerable": True} if forced == "submit_answer" \
                    else {"label": "incorrect", "reason": "Grader output could not be parsed."}
            content = [_tool_use(forced, args)]
        else:
            text, n_in, n_out = self._generate(prompt, max_tokens)
            content, names = [], {t["name"] for t in tools}
            calls = re.findall(r"<tool_call>(.*?)(?:</tool_call>|$)", text, flags=re.S)
            lead = text.split("<tool_call>")[0].strip()
            if lead:
                content.append(_block("text", text=lead))
            for raw in calls:
                obj, _ = _first_json_object(raw)
                if isinstance(obj, dict) and obj.get("name") in names:
                    args = obj.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    content.append(_tool_use(obj["name"], args if isinstance(args, dict) else {}))
            if not content:
                content.append(_block("text", text=text.strip()))
        return SimpleNamespace(content=content, usage=SimpleNamespace(input_tokens=n_in, output_tokens=n_out))


class LocalClient:
    def __init__(self, n_ctx: int | None = None):
        from llama_cpp import Llama
        n_ctx = n_ctx or int(os.getenv("RAG_LOCAL_CTX", "8192"))
        self.llm = Llama(model_path=_model_path(), n_ctx=n_ctx, n_threads=os.cpu_count(), verbose=False, seed=0)
        self.messages = _Messages(self.llm, n_ctx)
