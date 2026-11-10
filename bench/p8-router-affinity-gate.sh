#!/usr/bin/env bash
# CPU-only gate for deterministic cold-prefix placement and serialization.
set -euo pipefail
cd "$(dirname "$0")/.."
out=${1:-.work/p8-router-affinity-gate}
router_bin=${ROUTER_BIN:-serve/target/release/router}
mkdir -p "$out"
[ -x "$router_bin" ] || { echo "FAIL setup: missing executable $router_bin"; exit 1; }
python3 - "$out" "$router_bin" <<'PY'
import http.server
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
import urllib.request

out = pathlib.Path(sys.argv[1])
router_bin = sys.argv[2]
active = 0
maximum = 0
counter_lock = threading.Lock()

class Engine(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def reply(self, value):
        raw = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/v1/state":
            self.reply({"states": []})
        else:
            self.send_error(404)

    def do_POST(self):
        global active, maximum
        size = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(size) or b"{}")
        if self.path == "/tokenize":
            self.reply({"tokens": [11, 22, 33]})
            return
        if self.path != "/v1/completions":
            self.send_error(404)
            return
        with counter_lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.25)
        with counter_lock:
            active -= 1
        self.reply({"choices": [{"text": "ok"}], "prompt": body.get("prompt")})

servers = []
for _ in range(2):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Engine)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    servers.append(server)
endpoints = ",".join(f"{chr(97+i)}=127.0.0.1:{s.server_port}" for i, s in enumerate(servers))
router = subprocess.Popen(
    [router_bin, "--host", "127.0.0.1", "--port", "18408", "--node-id", "p8-gate"],
    stdout=(out / "router.stdout").open("w"),
    stderr=(out / "router.stderr").open("w"),
    env={**os.environ, "BARO_ROUTER_ENGINES": endpoints},
)
try:
    for _ in range(60):
        if "listening on" in (out / "router.stdout").read_text():
            break
        if router.poll() is not None:
            raise RuntimeError(f"router exited {router.returncode}")
        time.sleep(0.1)
    else:
        raise RuntimeError("router did not start")
    time.sleep(11)
    url = "http://127.0.0.1:18408/v1/completions"
    payload = json.dumps({"prompt": "cold prefix gate", "max_tokens": 1}).encode()
    def call():
        request = urllib.request.Request(url, payload, {"content-type": "application/json"})
        urllib.request.urlopen(request, timeout=10).read()
    first = threading.Thread(target=call)
    second = threading.Thread(target=call)
    first.start(); second.start(); first.join(); second.join()
    workloads = json.loads(urllib.request.urlopen("http://127.0.0.1:18408/v1/workloads").read())
    done = [row for row in workloads if row["state"] == "done"]
    assert len(done) == 2, done
    assert all(row["placement"] == "hash" for row in done), done
    assert maximum == 1, maximum
    (out / "SUMMARY.txt").write_text("PASS: hash placement and cold-prefix lock, max upstream concurrency 1\n")
    print("PASS p8-router-affinity-gate: hash placement, cold-prefix serialization")
finally:
    router.terminate()
    try:
        router.wait(timeout=5)
    except subprocess.TimeoutExpired:
        router.kill()
    for server in servers:
        server.shutdown()
PY
