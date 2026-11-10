#!/usr/bin/env bash
# usage: tools/spark-prefill-ref.sh MODEL.gguf OUTDIR [PORT]
# llama.cpp master oracle for bench/spark-prefill-protocol.md. Two separate server passes,
# each wrapped in gpu-wait: pass "fast" (-fa on, f16 KV, -b 2048 -ub 512) -> prompt_ms /
# prompt_per_second per length (n_predict 1); pass "f32" (-fa off, f32 KV) -> 64-token identity
# reference per length. Both passes tokenize via /tokenize and write the id lists; props.json
# per pass is the P1 read-back. Never runs while the engine or another server holds the GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
model=$1; out=$2; port=${3:-8098}
export LENGTHS=${LENGTHS:-64,256,1024,2048}
mkdir -p "$out"
run_pass() {  # $1 tag, $2 n_predict, rest = server flags
  tag=$1; npred=$2; shift 2
  ~/.local/bin/gpu-wait run --priority 90 --vram 8 --timeout 3600 -- \
    ~/llama.cpp-master/build/bin/llama-server -m "$model" -c 4096 -ngl 99 -t 8 "$@" \
    --host 127.0.0.1 --port "$port" > "$out/server-$tag.log" 2>&1 &
  pid=$!
  for i in $(seq 1 240); do curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break; sleep 1; done
  curl -s "http://127.0.0.1:$port/props" > "$out/props-$tag.json"
  python3 - "$out" "$port" "$tag" "$npred" <<'PY'
import json, sys, urllib.request
out, port, tag, npred = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))
import os
for n in [int(x) for x in os.environ.get("LENGTHS", "64,256,1024,2048").split(",")]:
    text = open(f"bench/spark-prefill-prompts/p{n:04d}.txt").read()
    ids = post("/tokenize", {"content": text, "add_special": False})["tokens"]
    open(f"{out}/prompt-tokens-{n}.{tag}.txt", "w").write("\n".join(map(str, ids)) + "\n")
    runs = []
    for r in range(3 if npred == 1 else 1):
        d = post("/completion", {"prompt": ids, "n_predict": npred, "temperature": 0, "top_k": 1, "cache_prompt": False, "return_tokens": True})
        runs.append(d)
    t = [d["timings"] for d in runs]
    json.dump(t, open(f"{out}/timings-{n}.{tag}.json", "w"), indent=1)
    if npred > 1:
        open(f"{out}/ref-tokens-{npred}-{n}.txt", "w").write("\n".join(map(str, runs[0]["tokens"])) + "\n")
    pm = sorted(x["prompt_ms"] for x in t)[len(t) // 2]
    print(f"{tag} n={n} ids={len(ids)} prompt_n={t[0]['prompt_n']} prompt_ms(median of {len(t)})={pm:.1f} prompt_tok_s={t[0]['prompt_n']/pm*1e3:.0f} gen={len(runs[0]['tokens'])}")
PY
  # kill the server itself: $pid is the gpu-wait wrapper; killing only it orphans llama-server
  spid=$(pgrep -f "^$HOME/llama.cpp-master/build/bin/llama-server .*--port $port" | head -1)
  kill "$spid" 2>/dev/null; wait $pid 2>/dev/null || true
  for i in $(seq 1 30); do pgrep -f "^$HOME/llama.cpp-master/build/bin/llama-server .*--port $port" >/dev/null || break; sleep 1; done
  sleep 2
}
passes=${PASSES:-fast,f32}
case ",$passes," in *,fast,*) run_pass fast 1 -fa on -ctk f16 -ctv f16 -b 2048 -ub 512;; esac
case ",$passes," in *,f32,*) run_pass f32 64 -fa off -ctk f32 -ctv f32;; esac
for n in ${LENGTHS//,/ }; do cmp -s "$out/prompt-tokens-$n.fast.txt" "$out/prompt-tokens-$n.f32.txt" && echo "ids $n: fast==f32" || echo "ids $n: DIFFER"; done
