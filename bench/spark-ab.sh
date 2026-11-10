#!/usr/bin/env bash
# usage: bench/spark-ab.sh MODEL.gguf OUTDIR [PORT]
# bench/spark-ab-protocol.md: arm A = .work/spark/spark-engine on bench/mtp-prompts/p*.txt
# (BARO_PROMPT_TEXT), arm B = llama.cpp master fast config on the same text. Run inside
# gpu-wait (priority 90); engine and server never overlap (A finishes before B starts).
# KV=f16 (default, fastest bar per llama-bench 2026-09-08) or q8_0.
set -uo pipefail
cd "$(dirname "$0")/.."
model=$1; out=$2; port=${3:-8098}; eng=.work/spark/spark-engine
mkdir -p "$out"
echo "engine_sha=$(sha256sum $eng | cut -c1-16) model=$model" | tee "$out/arm.txt"
echo "power_cap_uW=$(cat /sys/class/drm/card1/device/hwmon/hwmon*/power1_cap) vddgfx=$(grep -A1 OD_VDDGFX_OFFSET /sys/class/drm/card1/device/pp_od_clk_voltage | tail -1)" | tee -a "$out/arm.txt"
for tf in bench/mtp-prompts/p*.txt; do
  p=$(basename "$tf" .txt)
  env BARO_PROMPT_TEXT="$tf" BARO_GGUF="$model" BARO_PACK=.work/spark/pack-q8 BARO_GEN=64 "$eng" > "$out/$p.A.log" 2>&1
done
~/llama.cpp-master/build/bin/llama-server -m "$model" -c 4096 -ngl 99 -fa on -ctk ${KV:-f16} -ctv ${KV:-f16} -b 2048 -ub 512 -t 8 \
  --host 127.0.0.1 --port "$port" > "$out/server.log" 2>&1 &
pid=$!
for i in $(seq 1 180); do curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break; sleep 1; done
curl -s "http://127.0.0.1:$port/props" > "$out/props.json"
python3 - "$out" "$port" <<'PY'
import json, sys, glob, os, re, statistics as st, urllib.request
out, port = sys.argv[1:]
def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))
rows = []
for tf in sorted(glob.glob("bench/mtp-prompts/p*.txt")):
    p = os.path.basename(tf)[:-4]
    text = open(tf).read()
    ids = post("/tokenize", {"content": text, "add_special": False})["tokens"]
    d = post("/completion", {"prompt": ids, "n_predict": 64, "temperature": 0, "top_k": 1, "cache_prompt": False, "return_tokens": True})
    json.dump({"ids": ids, "tokens": d["tokens"], "timings": d["timings"]}, open(f"{out}/{p}.B.json", "w"))
    la = open(f"{out}/{p}.A.log").read()
    ta = float(re.search(r"tok/s_gen: ([0-9.]+)", la).group(1))
    aid = re.search(r"^prompt ids: (.*)$", la, re.M); aid = [int(x) for x in aid.group(1).split()] if aid else None
    gen = re.search(r"^generated: (.*)$", la, re.M); gen = [int(x) for x in gen.group(1).split()] if gen else None
    tb = d["timings"]["predicted_per_second"]
    rows.append((p, len(ids), ta, tb, "PASS" if aid == ids else ("n/a" if aid is None else "FAIL"), "same" if gen == d["tokens"] else "diff"))
with open(f"{out}/results.txt", "w") as f:
    f.write("prompt n_ids A_tok_s B_tok_s tokenizer_id gen_vs_B\n")
    for r in rows: f.write(f"{r[0]} {r[1]} {r[2]:.2f} {r[3]:.2f} {r[4]} {r[5]}\n")
print(open(f"{out}/results.txt").read())
a = [r[2] for r in rows]; b = [r[3] for r in rows]
sp = lambda x: (max(x) - min(x)) / st.median(x) * 100
print(f"A median {st.median(a):.2f} spread {sp(a):.1f}% | B median {st.median(b):.2f} spread {sp(b):.1f}% | ratio {st.median(a)/st.median(b):.3f} | tokenizer fails {[r[0] for r in rows if r[4]=='FAIL']} | gen same-as-B {sum(r[5]=='same' for r in rows)}/20")
PY
kill $pid; wait $pid 2>/dev/null
