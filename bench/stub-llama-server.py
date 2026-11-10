#!/usr/bin/env python3
"""CPU stand-in for llama-server, for dry-running the llama arm of a GPU job.

    LLAMA=bench/stub-llama-server.py bench/regescore-blender-ab.sh llama OUT

Takes llama-server's flags (only --port, --parallel, -c, --temp, --top-p, --top-k, --min-p are read,
last one wins as in llama.cpp), answers /health, /props with the parsed sampling
defaults, and every /v1/chat/completions with empty content plus usage and timings.
STUB_PREDICTED=N reports N generated tokens per request (default 0), so a dry run can
exercise the token-gap flag against bench/stub-engine.py, which reports 0.
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

args, flags = sys.argv[1:], {"--port": "8080", "--temp": "0.8", "--top-p": "0.95", "--top-k": "40", "--min-p": "0.05", "--parallel": "1", "-c": "32768"}
for i, a in enumerate(args[:-1]):
    if a in flags:
        flags[a] = args[i + 1]
params = {"temperature": float(flags["--temp"]), "top_p": float(flags["--top-p"]),
          "top_k": int(flags["--top-k"]), "min_p": float(flags["--min-p"])}
predicted = int(os.environ.get("STUB_PREDICTED", "0"))


class H(BaseHTTPRequestHandler):
    def send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self.send({"status": "ok"})
        elif self.path == "/props":
            self.send({"total_slots": int(flags["--parallel"]), "default_generation_settings": {"params": params, "n_ctx": int(flags["-c"])}})
        else:
            self.send_error(404)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        n = sum(len(m.get("content") or "") for m in req.get("messages", [])) // 4
        self.send({"choices": [{"message": {"role": "assistant", "content": ""}}],
                   "usage": {"prompt_tokens": n, "completion_tokens": predicted},
                   "timings": {"cache_n": 0, "prompt_n": n, "prompt_ms": 0.0, "predicted_n": predicted,
                               "predicted_ms": 0.0, "predicted_per_second": 0.0}})

    def log_message(self, *a):
        pass


print(f"stub llama-server listening on 127.0.0.1:{flags['--port']} sampling {params}", flush=True)
ThreadingHTTPServer(("127.0.0.1", int(flags["--port"])), H).serve_forever()
