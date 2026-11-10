#!/usr/bin/env bash
# P0a contract gate. The server stays resident for all HTTP checks in one GPU job.
# Usage: bench/p0a-contract-gate.sh OUTDIR [COUNT] [PORT]
# Live usage must be wrapped by gpu-wait, for example:
#   gpu-wait run --vram 24 --timeout 3600 -- bench/p0a-contract-gate.sh OUTDIR
# GATE_DRYRUN is the CPU-only preflight and intentionally stops before launch.
set -euo pipefail
cd "$(dirname "$0")/.."

out=${1:-.work/team-A/codex/p0a}
count=${2:-5}
port=${3:-0}
engine=${BARO_ENGINE:-.work/engine}
pack=${BARO_PACK:-.work/engine-pack-q4}
serve=${BARO_SERVE_BIN:-serve/target/release/baro-serve}
pair_manager=${PAIR_MANAGER_BIN:-$HOME/Projects/imports/Personal-AI-Router/services/build/bin/nvpair-engine-manager}
mkdir -p "$out"
exec > >(tee "$out/contract.log") 2>&1

fail() { echo "FAIL $1: $2"; exit 1; }
[ "$count" -ge 1 ] || fail args "count must be positive"
arm="$out/arm.txt"
cat > "$arm" <<EOF
backend=ollama
count=$count
seed=0
temperature=0
embeddings=live-P0a-e-http
pair_manager=$pair_manager
EOF

# gate-dryrun must stop before a GPU process is launched while proving that
# arm parameters reached the gate.
if [ "${GATE_DRYRUN:-0}" = 1 ]; then
  echo "FAIL GPU: GATE_DRYRUN stops before baro-serve launch"
  exit 97
fi

[ -x "$serve" ] || fail setup "missing $serve"
[ -x "$engine" ] || fail setup "missing $engine"
[ -d "$pack" ] || fail setup "missing pack $pack"
command -v curl >/dev/null || fail setup "curl is required"
command -v python3 >/dev/null || fail setup "python3 is required"

# A gate must never silently exercise an engine older than the Mojo source it
# claims to verify. This catches the common post-merge stale-binary failure
# before reserving the GPU. Record both the source watermark and binary hash
# in the arm so a receipt identifies the exact executable.
newest_mojo=$(find serve -type f -name '*.mojo' -printf '%T@ %p\n' | sort -nr | head -1)
[ -n "$newest_mojo" ] || fail setup "no serve/*.mojo source files found"
newest_mojo_mtime=${newest_mojo%% *}
newest_mojo_path=${newest_mojo#* }
engine_mtime=$(stat -c '%Y' "$engine")
newest_mojo_epoch=${newest_mojo_mtime%%.*}
[ "$engine_mtime" -ge "$newest_mojo_epoch" ] || fail setup "engine older than $newest_mojo_path; rebuild $engine"
engine_sha256=$(sha256sum "$engine" | cut -d' ' -f1)
{
  echo "engine=$engine"
  echo "engine_sha256=$engine_sha256"
  echo "engine_mtime=$(stat -c '%y' "$engine")"
  echo "newest_mojo=$newest_mojo_path"
  echo "newest_mojo_mtime=$(stat -c '%y' "$newest_mojo_path")"
} >> "$arm"
echo "engine read-back: sha256=${engine_sha256:0:12} mtime=$(stat -c '%y' "$engine"); newest Mojo $newest_mojo_path"

"$serve" --engine "$engine" --pack "$pack" --port "$port" \
  > "$out/server.stdout" 2> "$out/server.stderr" &
srv=$!
cleanup() {
  kill -INT "$srv" 2>/dev/null || true
  wait "$srv" 2>/dev/null || true
  pkill -TERM -P "$srv" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 1200); do
  if grep -q '^listening on' "$out/server.stdout"; then break; fi
  kill -0 "$srv" 2>/dev/null || fail start "server exited: $(tail -3 "$out/server.stderr")"
  sleep 0.5
done
url=$(python3 - "$out/server.stdout" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
match = re.search(r"https?://[^\s]+", text)
if not match:
    raise SystemExit("no listening URL")
print(match.group(0))
PY
) || fail start "no listening URL"
actual_port=${url##*:}

curl -fsS "$url/health" > "$out/health.json" || fail health "GET /health failed"
curl -fsS "$url/v1/models" > "$out/models.json" || fail models "GET /v1/models failed"
python3 - "$out/health.json" "$out/models.json" "$arm" "$actual_port" <<'PY'
import json, sys
health = json.load(open(sys.argv[1]))
models = json.load(open(sys.argv[2]))
assert health.get("status") == "ok", health
items = models.get("data", [])
assert len(items) == 1 and items[0].get("id"), models
with open(sys.argv[3], "a") as f:
    f.write(f"port={sys.argv[4]}\n")
    f.write(f"model={items[0]['id']}\n")
print("read-back", health["limits"], items[0]["id"])
PY
model=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["data"][0]["id"])' "$out/models.json")

python3 - "$url" "$model" "$out" <<'PY'
import json, pathlib, sys
import urllib.error
import urllib.request
import math

url, model, out = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
rows = []
prompt = "The capital of France is Paris, a city on the Seine."
for path in ("/api/embeddings", "/v1/embeddings"):
    request = urllib.request.Request(
        url + path,
        json.dumps({"model": model, "input": prompt}).encode(),
        {"content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request) as response:
            status, body = response.status, response.read()
    except urllib.error.HTTPError as error:
        status, body = error.code, error.read()
    assert status == 200, (path, status, body[:500])
    payload = json.loads(body)
    data = payload.get("data")
    assert payload.get("object") == "list" and isinstance(data, list) and len(data) == 1, (path, payload)
    row = data[0]
    vector = row.get("embedding")
    assert row.get("object") == "embedding" and row.get("index") == 0, (path, row)
    assert isinstance(vector, list) and len(vector) == 4096, (path, len(vector) if isinstance(vector, list) else vector)
    assert all(isinstance(x, (int, float)) and math.isfinite(x) for x in vector), (path, "non-finite vector")
    norm = math.sqrt(sum(x * x for x in vector))
    assert abs(norm - 1.0) < 1e-4, (path, norm)
    usage = payload.get("usage", {})
    assert usage.get("prompt_tokens", 0) > 0 and usage.get("total_tokens") == usage.get("prompt_tokens"), (path, usage)
    rows.append({"path": path, "status": status, "dimension": len(vector), "norm": norm,
                 "prompt_tokens": usage["prompt_tokens"]})
batch_request = urllib.request.Request(
    url + "/v1/embeddings",
    json.dumps({"model": model, "input": ["Paris is the capital city of France.", "The 7900 XTX has 24 GB of memory."]}).encode(),
    {"content-type": "application/json"},
)
with urllib.request.urlopen(batch_request) as response:
    batch = json.loads(response.read())
assert response.status == 200 and len(batch.get("data", [])) == 2, batch
assert [row.get("index") for row in batch["data"]] == [0, 1], batch
rows.append({"path": "/v1/embeddings", "batch": 2,
             "dimensions": [len(row["embedding"]) for row in batch["data"]]})
(out / "embeddings-http.json").write_text(json.dumps(rows, indent=2) + "\n")
print("gate 4 HTTP embeddings: 2/2 routes, normalized 4096-d vectors, batch 2/2")
PY

prompts=(
  "Reply with exactly the word amber."
  "What is two plus two? Reply with one digit."
  "Name the first month of the year."
  "Reply with exactly the word router."
  "Give one short word for a cold color."
)
pair_args=(--backend ollama --port "$actual_port" --model "$model" --count "$count" --mode parallel
  --seed 0 --temperature 0 --max-tokens 8 --result-log "$out/pair-results.jsonl")
for ((i=0; i<count; i++)); do
  pair_args+=(--prompt "${prompts[$((i % ${#prompts[@]}))]}")
done
rm -f "$out/pair-results.jsonl"
$HOME/iTools/bin/pair-dispatch "${pair_args[@]}" > "$out/pair-dispatch.log" 2>&1 || fail pair-dispatch "see $out/pair-dispatch.log"
python3 - "$out/pair-results.jsonl" "$count" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
want = int(sys.argv[2])
assert len(rows) >= want, (len(rows), want)
assert all(row.get("ok") is True for row in rows[-want:]), rows[-want:]
print(f"PAIR client requests: {want}/{want} ok")
PY

python3 - "$url" "$model" "$count" "$out" <<'PY'
import json, pathlib, sys, urllib.request
url, model, count, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), pathlib.Path(sys.argv[4])
prompts = [
    "Reply with exactly the word amber.",
    "What is two plus two? Reply with one digit.",
    "Name the first month of the year.",
    "Reply with exactly the word router.",
    "Give one short word for a cold color.",
]
def post(path, body):
    request = urllib.request.Request(url + path, json.dumps(body).encode(),
                                      {"content-type": "application/json"})
    with urllib.request.urlopen(request) as response:
        return json.load(response)
rows = []
for i in range(count):
    prompt = prompts[i % len(prompts)]
    # Gate 1 compares the raw Ollama prompt with OpenAI completions. Without
    # raw=true, /api/generate deliberately applies the chat template and the
    # prompt token counts are not comparable.
    ollama = post("/api/generate", {"model": model, "prompt": prompt, "raw": True, "stream": False,
                                    "options": {"temperature": 0, "seed": 0, "num_predict": 8}})
    openai = post("/v1/completions", {"model": model, "prompt": prompt, "stream": False,
                                      "temperature": 0, "seed": 0, "max_tokens": 8})
    ollama_count = ollama.get("eval_count")
    usage = openai.get("usage", {})
    openai_count = usage.get("completion_tokens",
                              len(openai.get("choices", [{}])[0].get("tokens", [])))
    assert ollama_count is not None and ollama_count == openai_count, (i, ollama_count, openai_count)
    rows.append({"index": i, "ollama_eval_count": ollama_count,
                 "openai_completion_tokens": openai_count})
(out / "gate1-counts.json").write_text(json.dumps(rows, indent=2) + "\n")
print(f"gate 1 token counts: {count}/{count} equal")
PY

venv="$out/ollama-venv"
if [ ! -x "$venv/bin/python" ]; then
  python3 -m venv "$venv"
  "$venv/bin/pip" install --disable-pip-version-check --quiet ollama
fi
OLLAMA_URL="$url" OLLAMA_MODEL="$model" OLLAMA_OUT="$out" \
  "$venv/bin/python" - "$count" <<'PY'
import json, os, pathlib, sys
import ollama
count = int(sys.argv[1])
url = os.environ["OLLAMA_URL"]
model = os.environ["OLLAMA_MODEL"]
out = pathlib.Path(os.environ["OLLAMA_OUT"])
prompts = [
    "Reply with exactly the word amber.", "What is two plus two? Reply with one digit.",
    "Name the first month of the year.", "Reply with exactly the word router.",
    "Give one short word for a cold color.", "Reply with exactly the word cobalt.",
    "What is three plus one? Reply with one digit.", "Name one primary color.",
    "Reply with exactly the word engine.", "Give one short word for a warm color.",
    "Reply with exactly the word state.", "What is five minus two? Reply with one digit.",
    "Name one season.", "Reply with exactly the word client.",
    "Give one short word for a bright color.", "Reply with exactly the word model.",
    "What is six divided by two? Reply with one digit.", "Name one weekday.",
    "Reply with exactly the word latent.", "Give one short word for a dark color.",
]
client = ollama.Client(host=url)
def openai_chat(prompt, stream):
    import urllib.request
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8, "temperature": 0, "seed": 0, "stream": stream}
    request = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"content-type": "application/json"})
    if stream:
        with urllib.request.urlopen(request) as response:
            chunks = []
            for line in response.read().decode().splitlines():
                if line.startswith("data: ") and line[6:] != "[DONE]":
                    chunks.append(json.loads(line[6:])["choices"][0].get("delta", {}).get("content", ""))
            return "".join(chunks)
    with urllib.request.urlopen(request) as response:
        return json.load(response)["choices"][0]["message"]["content"]
rows = []
for i in range(count * 4):
    prompt = prompts[i % len(prompts)]
    chat = client.chat(model=model, messages=[{"role": "user", "content": prompt}],
                       stream=False, options={"temperature": 0, "seed": 0, "num_predict": 8})
    chat_text = chat["message"]["content"] if isinstance(chat, dict) else chat.message.content
    assert chat_text == openai_chat(prompt, False), (i, chat_text)
    chunks = client.chat(model=model, messages=[{"role": "user", "content": prompt}],
                         stream=True, options={"temperature": 0, "seed": 0, "num_predict": 8})
    stream_text = "".join((chunk.get("message", {}).get("content", "") if isinstance(chunk, dict)
                           else chunk.message.content) for chunk in chunks)
    assert stream_text == chat_text == openai_chat(prompt, True), (i, stream_text, chat_text)
    generated = client.generate(model=model, prompt=prompt, stream=False,
                                options={"temperature": 0, "seed": 0, "num_predict": 8})
    generated_text = generated["response"] if isinstance(generated, dict) else generated.response
    assert generated_text == openai_chat(prompt, False), (i, generated_text)
    rows.append({"index": i, "chat": "pass", "chat_stream": "pass", "generate": "pass"})
(out / "gate2-client.json").write_text(json.dumps(rows, indent=2) + "\n")
print(f"gate 2 ollama client: {len(rows)}/{len(rows)} chat/generate stream and non-stream equal")
PY

echo "PASS p0a contract gates 1 and 2"
if [ "$actual_port" != 11434 ]; then
  fail gate3 "PAIR adoption requires baro-serve on port 11434, got $actual_port"
fi
PAIR_MANAGER_BIN="$pair_manager" python3 - "$model" "$out" <<'PY'
import json
import os
import pathlib
import select
import subprocess
import sys
import time

model, out_name = sys.argv[1], sys.argv[2]
out = pathlib.Path(out_name).resolve()
manager = os.environ["PAIR_MANAGER_BIN"]
assert os.access(manager, os.X_OK), manager
stderr_path = out / "pair-manager.stderr"
stdout_path = out / "pair-manager.stdout"
stderr_file = stderr_path.open("w")
stdout_file = stdout_path.open("w")
manager_home = out / "pair-manager-home"
manager_config = out / "pair-manager-config"
manager_home.mkdir(parents=True, exist_ok=True)
manager_config.mkdir(parents=True, exist_ok=True)
proc = subprocess.Popen(
    [manager, "--loaded-poll-interval", "0"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=stderr_file,
    text=True,
    env={**os.environ, "HOME": str(manager_home),
         "XDG_CONFIG_HOME": str(manager_config)},
)

def rpc(request_id, method, params=None):
    request = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        request["params"] = params
    proc.stdin.write(json.dumps(request) + "\n")
    proc.stdin.flush()
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        ready, _, _ = select.select([proc.stdout], [], [], 1)
        if not ready:
            continue
        line = proc.stdout.readline()
        if not line:
            raise AssertionError(f"PAIR manager exited before response {request_id}")
        stdout_file.write(line)
        stdout_file.flush()
        frame = json.loads(line)
        if frame.get("id") == request_id:
            assert "error" not in frame, frame
            return frame.get("result")
    raise AssertionError(f"PAIR manager response timeout id={request_id}")

result = {}
try:
    ollama = rpc(1, "engine:start", {"engine": "ollama"})
    assert ollama.get("running") is True and ollama.get("healthy") is True, ollama
    assert ollama.get("port") == 11434, ollama
    routed = rpc(2, "engine:action", {
        "engine": "ollama", "action": "run_model",
        "params": {"model": model, "prompt": "Reply with exactly the word adopted.", "stream": False},
    })
    assert routed.get("done") is True and isinstance(routed.get("response"), str), routed
    result = {"engine_status": ollama, "routed_response": routed}
finally:
    try:
        if proc.poll() is None:
            proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 3, "method": "shutdown"}) + "\n")
            proc.stdin.flush()
            proc.wait(timeout=10)
    except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5)
    stderr_file.close()
    stdout_file.close()

stderr = stderr_path.read_text()
assert "adopting already-running external engine" in stderr, stderr
result["adoption_log"] = "adopting already-running external engine"
(out / "pair-manager.json").write_text(json.dumps(result, indent=2) + "\n")
print("gate 3 PAIR engine manager: adopted Ollama on 11434 and routed request")
PY
echo "PASS p0a contract gates 1 through 4 (gate 4 HTTP embeddings)"
