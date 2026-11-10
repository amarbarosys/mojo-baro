#!/usr/bin/env python3
"""Capture the Rust front's answers for the rows serve/front/test_http.mojo replays.

usage: capture.py BARO_SERVE_BINARY OUT.jsonl
Starts the binary once per engine setting with fake_engine.py behind it (tokenizer from make_tokenizer.py),
sends each request, records status, content type and the PARSED body (SSE: the parsed data payloads).
Rows: {name, script, mode, method, path, request, status, content_type, kind, body|sse|text}.
"""
import http.client
import json
import os
import socket
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BIN, OUT = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
WORK = tempfile.mkdtemp(prefix="front-oracle-", dir=os.environ.get("ORACLE_TMP", os.getcwd()))
subprocess.run([sys.executable, os.path.join(HERE, "make_tokenizer.py"), WORK], check=True)


def ids(s):
    return [b + 3 for b in s.encode()]


def chat(extra=None, stream=False, content="hi"):
    d = {"messages": [{"role": "user", "content": content}]}
    if stream:
        d["stream"] = True
    d.update(extra or {})
    return json.dumps(d)


OK = ids("ok")
TOOL_TEXT = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
TOOLS = {"tools": [{"type": "function", "function": {"name": "get_weather"}}]}
NAMED = {"type": "function", "function": {"name": "get_weather"}}
# (name, script, mode, method, path, request body)
G = []
for p in ("/health", "/v1/models", "/models", "/props"):
    G.append(("get" + p.replace("/", "-"), OK, "normal", "GET", p, ""))
G += [
    ("tokenize-special", OK, "normal", "POST", "/tokenize", '{"content":"hi","add_special":true}'),
    ("tokenize-plain", OK, "normal", "POST", "/tokenize", '{"content":"hi"}'),
    ("tokenize-chatml", OK, "normal", "POST", "/tokenize", '{"content":"<|im_start|>user\\nhi<|im_end|>"}'),
    ("detokenize", OK, "normal", "POST", "/detokenize", '{"tokens":[107,108]}'),
    ("detokenize-special", OK, "normal", "POST", "/detokenize", '{"tokens":[0,1,2,259,260,107]}'),
    ("comp-ids", OK, "normal", "POST", "/v1/completions", '{"prompt":[1,107,108],"max_tokens":8,"model":"anything"}'),
    ("comp-ids-alias", OK, "normal", "POST", "/completions", '{"prompt":[1,107,108],"max_tokens":8}'),
    ("comp-string", OK, "normal", "POST", "/v1/completions", '{"prompt":"hi","max_tokens":4}'),
    ("comp-omitted-max", OK, "normal", "POST", "/v1/completions", '{"prompt":[1,2,3]}'),
    ("comp-stream", OK, "normal", "POST", "/v1/completions", '{"prompt":[1,107],"max_tokens":8,"stream":true}'),
    ("chat", OK, "normal", "POST", "/v1/chat/completions", chat()),
    ("chat-alias", OK, "normal", "POST", "/chat/completions", chat({"model": "other"})),
    ("chat-stream", OK, "normal", "POST", "/v1/chat/completions", chat(stream=True)),
    ("chat-system", OK, "normal", "POST", "/v1/chat/completions", json.dumps({"messages": [{"role": "system", "content": "be brief"}, {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]})),
    ("err-bad-json", OK, "normal", "POST", "/v1/completions", "{not json"),
    ("err-no-prompt", OK, "normal", "POST", "/v1/completions", "{}"),
    ("err-empty-prompt", OK, "normal", "POST", "/v1/completions", '{"prompt":[],"max_tokens":4}'),
    ("err-max-zero", OK, "normal", "POST", "/v1/completions", '{"prompt":[1],"max_tokens":0}'),
    ("err-over-window", OK, "normal", "POST", "/v1/completions", '{"prompt":[1,2,3],"max_tokens":20000}'),
    ("err-empty-messages", OK, "normal", "POST", "/v1/chat/completions", '{"messages":[]}'),
    ("err-prompt-type", OK, "normal", "POST", "/v1/completions", '{"prompt":{"a":1}}'),
    ("err-tokenize-no-content", OK, "normal", "POST", "/tokenize", "{}"),
    ("cancel-unknown", OK, "normal", "POST", "/v1/cancel", '{"id":"cmpl-99"}'),
    ("err-cancel-bad-id", OK, "normal", "POST", "/v1/cancel", '{"id":"nonumber"}'),
    ("comp-stop-id", ids("o") + [2] + ids("k"), "normal", "POST", "/v1/completions", '{"prompt":[1],"max_tokens":8}'),
    ("comp-stop-im-end", ids("o") + [260] + ids("k"), "normal", "POST", "/v1/completions", '{"prompt":[1],"max_tokens":8}'),
    ("chat-tools-json", ids(TOOL_TEXT), "normal", "POST", "/v1/chat/completions", chat(TOOLS)),
    ("chat-tools-json-stream", ids(TOOL_TEXT), "normal", "POST", "/v1/chat/completions", chat(TOOLS, stream=True)),
    ("chat-tools-text-first", ids("Checking.\n" + TOOL_TEXT), "normal", "POST", "/v1/chat/completions", chat(TOOLS)),
    ("chat-tool-required", ids('get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'), "normal", "POST", "/v1/chat/completions", chat(dict(TOOLS, tool_choice="required"))),
    ("chat-tool-named-stream", ids('{"city": "Paris"}}\n</tool_call>'), "normal", "POST", "/v1/chat/completions", chat(dict(TOOLS, tool_choice=NAMED), stream=True)),
    ("err-tool-choice-unknown", OK, "normal", "POST", "/v1/chat/completions", chat(dict(TOOLS, tool_choice={"type": "function", "function": {"name": "nope"}}))),
    ("comp-err-before", OK, "err_before", "POST", "/v1/completions", '{"prompt":[1],"max_tokens":4}'),
    ("comp-stream-err-before", OK, "err_before", "POST", "/v1/completions", '{"prompt":[1],"max_tokens":4,"stream":true}'),
    ("comp-err-after", OK, "err_after", "POST", "/v1/completions", '{"prompt":[1],"max_tokens":4}'),
    ("comp-stream-err-after", OK, "err_after", "POST", "/v1/completions", '{"prompt":[1],"max_tokens":4,"stream":true}'),
]


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def start(script, mode):
    port = free_port()
    log = open(os.path.join(WORK, f"serve-{port}.log"), "w")
    proc = subprocess.Popen(
        [BIN, "--engine", os.path.join(HERE, "fake_engine.py"), "--pack", os.path.join(WORK, "pack"),
         "--tokenizer", os.path.join(WORK, "tokenizer.json"), "--port", str(port),
         "--engine-env", "FAKE_SCRIPT=" + json.dumps(script), "--engine-env", "FAKE_MODE=" + mode],
        cwd=WORK, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(100):
        try:
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            c.request("GET", "/health")
            if c.getresponse().status == 200:
                return proc, port
        except OSError:
            time.sleep(0.1)
    proc.kill()
    raise SystemExit("FAIL start: baro-serve did not answer /health")


def parse_sse(text):
    out = []
    for ev in text.split("\n\n"):
        for ln in ev.split("\n"):
            if ln.startswith("data:"):
                d = ln[5:].strip()
                out.append("[DONE]" if d == "[DONE]" else json.loads(d))
    return out


rows = []
last = None
proc = None
port = 0
for name, script, mode, method, path, body in G:
    key = (json.dumps(script), mode)
    if key != last:
        if proc:
            proc.kill()  # the Rust front waits 30 s for a stdin EOF it never sends (report-A M1)
            proc.wait(timeout=40)
        proc, port = start(script, mode)
        last = key
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    hdr = {"content-type": "application/json"} if method == "POST" else {}
    c.request(method, path, body=body.encode() if body else None, headers=hdr)
    r = c.getresponse()
    raw = r.read().decode()
    ctype = r.getheader("content-type") or ""
    row = {"name": name, "script": script, "mode": mode, "method": method, "path": path, "request": body,
           "status": r.status, "content_type": ctype}
    if ctype.startswith("text/event-stream"):
        row["kind"], row["sse"] = "sse", parse_sse(raw)
    elif raw:
        row["kind"] = "json"
        try:
            row["body"] = json.loads(raw)
        except ValueError:
            row["kind"], row["text"] = "text", raw
    else:
        row["kind"] = "empty"
    rows.append(row)
if proc:
    proc.kill()
    proc.wait(timeout=40)
with open(OUT, "w") as f:
    for r in rows:
        f.write(json.dumps(r, separators=(",", ":")) + "\n")
print("captured", len(rows), "rows ->", OUT)
