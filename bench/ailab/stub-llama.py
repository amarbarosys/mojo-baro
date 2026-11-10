#!/usr/bin/env python3
"""CPU stand-in for llama-server, for the dry run of parity-llama.sh (LLAMA=bench/ailab/stub-llama.py bench/ailab/llama.sh up).
Speaks /health, /props, /apply-template, /tokenize, /v1/chat/completions, /completion; ignores every other flag.
The model answers `state` first and `hold` once a tool result is in the history: the runner's 2-round ask. Ids are a
deterministic function of the prompt, so two stub servers (control and reference) agree and the pipeline runs end to end."""
import json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

a, port, par, ctx, ctk = sys.argv[1:], 18462, 1, 8192, "f16"
for i, x in enumerate(a):
    if x == "--port":
        port = int(a[i + 1])
    elif x == "--parallel":
        par = int(a[i + 1])
    elif x == "-ctk":
        ctk = a[i + 1]
    elif x in ("-c", "--ctx-size"):
        ctx = int(a[i + 1])


def ids_of(text):
    # first id carries the number of tool messages in the prompt (the stub's /apply-template writes "T<n>|"), so /completion can answer per round
    n = int(text[1:text.index("|")]) if text[:1] == "T" and "|" in text else 0
    return [500 + n] + [b % 250 + 1 for b in text.encode()[::7]]


class H(BaseHTTPRequestHandler):
    def log_message(self, *x):
        pass

    def send(self, d, code=200):
        b = json.dumps(d).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/health":
            self.send({"status": "ok"})
        elif self.path == "/props":
            self.send({"total_slots": par, "default_generation_settings": {"n_ctx": ctx}})
        else:
            self.send({"error": "stub"}, 404)

    def do_POST(self):
        d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/apply-template":
            self.send({"prompt": f'T{sum(1 for m in d["messages"] if m["role"] == "tool")}|' + json.dumps(d["messages"])})
        elif self.path == "/tokenize":
            self.send({"tokens": ids_of(d["content"])})
        elif self.path == "/completion":
            p = d["prompt"] if isinstance(d["prompt"], list) else ids_of(d["prompt"])
            text = ("<tool_call>\n<function=state>\n</function>\n</tool_call>" if p[0] == 500
                    else "<tool_call>\n<function=hold>\n<parameter=action>\nmove_forward\n</parameter>\n<parameter=seconds>\n2.0\n</parameter>\n</function>\n</tool_call>")
            self.send({"tokens": [(x * 3) % 250 + 1 for x in p[:min(12, d.get("n_predict", 12))]], "content": text})
        elif self.path == "/v1/chat/completions":
            n_tool = sum(1 for m in d["messages"] if m["role"] == "tool")
            name, args = ("state", {}) if n_tool == 0 else ("hold", {"action": "move_forward", "seconds": 2.0})
            msg = {"role": "assistant", "content": "", "tool_calls": [{"id": f"call_{n_tool + 1}", "type": "function",
                   "function": {"name": name, "arguments": json.dumps(args)}}]}
            p = ids_of(json.dumps(d["messages"]))
            self.send({"choices": [{"message": msg, "finish_reason": "tool_calls"}], "usage": {"completion_tokens": len(p[:12])}})
        else:
            self.send({"error": "stub"}, 404)


print(f"0.00.000.001 I llama_kv_cache: size =  512.00 MiB (  8192 cells,   8 layers,  1/1 seqs), K ({ctk}):  256.00 MiB, V ({ctk}):  256.00 MiB", flush=True)  # the real server's line, at -lv 4
ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
