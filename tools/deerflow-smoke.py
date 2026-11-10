#!/usr/bin/env python3
"""One DeerFlow run against a local OpenAI-compatible chat backend.

DeerFlow (~/Projects/deer-flow) is the agent harness; the model under test is
whichever `config.yaml` entry MODEL names (default: the llama.cpp reference arm
from tools/llama-chat-27b.sh; later baro-serve). Prints every AI/tool event,
then a receipt: tool calls made, final text, wall time. Exit 1 if the run
produced no final AI text.

usage: cd ~/Projects/deer-flow/backend && uv run python ~/Projects/mojo-baro/tools/deerflow-smoke.py [prompt]
env: MODEL (config model name), THINKING=0|1 (default 0), OUT (receipt path),
     NONINTERACTIVE=1 drops ask_clarification (DeerFlow non_interactive) so the run cannot stop on a question
"""
import json, os, sys, time
from deerflow.client import DeerFlowClient

model = os.environ.get("MODEL", "qwen3.8-27b-obliterated")
thinking = os.environ.get("THINKING", "0") == "1"
prompt = sys.argv[1] if len(sys.argv) > 1 else (
    "Create a file named fib.py containing a function fib(n) that returns the n-th "
    "Fibonacci number iteratively, then read the file back and show me its contents."
)
out = os.environ.get("OUT", "")
client = DeerFlowClient(model_name=model, thinking_enabled=thinking)
if os.environ.get("NONINTERACTIVE", "0") == "1":
    # The embedded client never sets configurable.non_interactive (gateway-only),
    # so drop the tool at the source: DeerFlow's own non-interactive filter.
    _orig_tools = client._get_tools
    client._get_tools = lambda **kw: [t for t in _orig_tools(**kw) if t.name != "ask_clarification"]
t0 = time.time()
tool_calls, final, n_ai, last_id = [], "", 0, None
raw = open(out + ".events.jsonl", "w") if out else None
for ev in client.stream(prompt, thread_id=f"smoke-{int(t0)}"):
    if ev.type != "messages-tuple":
        continue
    d = ev.data
    if raw:
        raw.write(json.dumps(d, default=str, ensure_ascii=False) + "\n"); raw.flush()
    typ = d.get("type")
    if typ == "ai":
        n_ai += 1
        for tc in d.get("tool_calls") or []:
            tool_calls.append(tc.get("name"))
            print(f"[tool_call] {tc.get('name')} {json.dumps(tc.get('args'))[:200]}", flush=True)
        c = d.get("content")
        if isinstance(c, list):
            c = "".join(x.get("text", "") for x in c if isinstance(x, dict))
        if c:
            # stream deltas arrive per token; the last AI message's text is the answer
            final = (final + c) if d.get("id") == last_id else c
            last_id = d.get("id")
    elif typ == "tool":
        print(f"[tool] {d.get('name')}: {str(d.get('content'))[:200]}", flush=True)
dt = time.time() - t0
if final:
    print(f"[ai] {final[:800]}", flush=True)
receipt = {"model": model, "thinking": thinking, "prompt": prompt, "tool_calls": tool_calls,
           "ai_messages": n_ai, "final_chars": len(final), "final": final[:2000], "wall_s": round(dt, 1)}
print("RECEIPT " + json.dumps(receipt, ensure_ascii=False), flush=True)
if out:
    with open(out, "w") as f:
        json.dump(receipt, f, ensure_ascii=False, indent=1)
sys.exit(0 if final else 1)
