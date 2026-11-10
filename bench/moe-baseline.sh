#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

out=${1:?output directory required}
port=${2:-18080}
mkdir -p "$out"
model=$HOME/Models/RegesCore-1.0-35/RegesCore-1.0-35B-UD-Q4_K_S-BARO-e340ee1.gguf
server=$HOME/llama.cpp/build/bin/llama-server

sha256sum "$model" | cut -c1-16 > "$out/model.sha256.prefix"
git -C $HOME/llama.cpp rev-parse HEAD > "$out/llama.commit"
"$server" --version > "$out/llama.version" 2>&1 || true
printf '%s\n' "$server -m $model -c 4096 -ngl 99 -fa on -ctk f16 -ctv f16 -b 2048 -ub 512 -t 8 -np 1 --no-cont-batching --host 127.0.0.1 --port $port" > "$out/server.invocation"

"$server" -m "$model" -c 4096 -ngl 99 -fa on -ctk f16 -ctv f16 -b 2048 -ub 512 -t 8 -np 1 --no-cont-batching --host 127.0.0.1 --port "$port" > "$out/server.log" 2>&1 &
pid=$!
cleanup() { kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; }
trap cleanup EXIT

ready=0
for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:$port/health" > "$out/health.json" 2>/dev/null; then ready=1; break; fi
  kill -0 "$pid" 2>/dev/null || { tail -80 "$out/server.log" > "$out/failure.log"; exit 2; }
  sleep 1
done
[ "$ready" = 1 ] || { tail -80 "$out/server.log" > "$out/failure.log"; exit 3; }
curl -fsS "http://127.0.0.1:$port/props" > "$out/props.json"
cp "$out/server.log" "$out/server-readback.log"

python3 - "$out" "$port" <<'PY'
import json, pathlib, statistics, sys, time, urllib.request

out = pathlib.Path(sys.argv[1])
port = sys.argv[2]
prompts = sorted(pathlib.Path("bench/mtp-prompts").glob("p*.txt"))
if len(prompts) != 20:
    raise SystemExit(f"expected 20 prompts, found {len(prompts)}")

def request(prompt):
    body = json.dumps({"prompt": prompt, "n_predict": 64, "temperature": 0, "top_k": 1, "seed": 1, "ignore_eos": True, "cache_prompt": False, "return_tokens": True}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as response:
        return json.load(response)

rows = []
raw = []
for path in prompts:
    prompt = path.read_text()
    request(prompt)
    measured = []
    for rep in range(1, 6):
        result = request(prompt)
        timings = result.get("timings", {})
        decode = timings.get("predicted_per_second")
        prefill = timings.get("prompt_per_second")
        if not isinstance(decode, (int, float)) or not isinstance(prefill, (int, float)) or decode <= 0 or prefill <= 0 or timings.get("predicted_n") != 64:
            raise SystemExit(f"invalid timings for {path.name} rep {rep}: {timings}")
        measured.append((float(decode), float(prefill)))
        raw.append({"prompt": path.name, "rep": rep, "timings": timings, "n_tokens": result.get("tokens"), "content": result.get("content", "")})
    d = [x[0] for x in measured]
    p = [x[1] for x in measured]
    rows.append({"prompt": path.stem, "prompt_bytes": len(prompt.encode()), "decode_tok_s": statistics.median(d), "decode_reps": d, "prefill_tok_s": statistics.median(p), "prefill_reps": p})

(out / "raw.json").write_text(json.dumps(raw, indent=2) + "\n")
(out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
(out / "finished-at").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n")
PY

echo "completed $out"
