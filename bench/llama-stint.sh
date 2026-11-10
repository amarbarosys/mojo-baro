#!/usr/bin/env bash
# usage: bench/llama-stint.sh ENGINE OUTDIR LLAMA_SERVER [LLAMA_SERVER ...]
#
# Same-stint decode bar against llama.cpp (bench/llama-stint-protocol.md), one gpu-wait job under
# bench/clock-probe.sh: our engine on every bench/mtp-prompts/*.tokens (one process per prompt, BARO_SPEC=0, pack
# .work/engine-pack-q4), then each llama-server on the same ids, then our engine again on the first 5 prompts as
# the drift control. llama.cpp arm: the fast config of bench/spark-ab.sh (-ngl 99 -fa on, KV f16, -b 2048 -ub 512
# -t 8, -np 1), /completion on the id list, n_predict 64 with ignore_eos (our engine always decodes GEN_N 64),
# temperature 0, top_k 1, cache_prompt false, tok/s =
# timings.predicted_per_second, one untimed request first (cold clocks after a load). Arm read-back (P1): power
# cap, vddgfx and load before; engine sha; each server's build, offload, flash_attn and KV lines from its own log.
set -euo pipefail
cd "$(dirname "$0")/.."
eng=$1 out=$2; shift 2
[ $# -ge 1 ] || { echo "usage: bench/llama-stint.sh ENGINE OUTDIR LLAMA_SERVER [...]" >&2; exit 2; }
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
  # gpu-wait carries no PATH: rocm-smi (the clock probe's sampler) lives in /opt/rocm/bin
  exec gpu-wait run --vram 22 --priority 50 --timeout 2400 -- env PATH=/opt/rocm/bin:/usr/bin:/bin LS_N="${LS_N:-}" LS_GGUF="${LS_GGUF:-}" LS_PACK="${LS_PACK:-}" \
    bench/clock-probe.sh "$0" "$eng" "$out" "$@"
fi
model=${LS_GGUF:-$HOME/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_0-pure.gguf}
pack=$(readlink -f "${LS_PACK:-.work/engine-pack-q4}")
mkdir -p "$out"
{
  echo "engine=$eng sha=$(sha256sum "$eng" | cut -c1-16) pack=$pack source=$(grep -o '"source": "[^"]*"' "$pack/identity.json")"
  echo "model=$model"
  echo "power_cap_uW=$(cat /sys/class/drm/card1/device/hwmon/hwmon*/power1_cap) vddgfx=$(grep -A1 OD_VDDGFX_OFFSET /sys/class/drm/card1/device/pp_od_clk_voltage | tail -1) load=$(cut -d' ' -f1-3 /proc/loadavg)"
} | tee "$out/arm.txt"
grep -q "\"source\": \"$model\"" "$pack/identity.json" || { echo "FAIL arm: pack $pack was not built from $model"; exit 1; }

ours() { # label prompt-files...
  local l=$1; shift
  for tf in "$@"; do
    p=$(basename "$tf" .tokens)
    env BARO_PROMPT="$tf" BARO_PACK="$pack" BARO_SPEC=0 BARO_TMAX=16384 "$eng" > "$out/$p.$l.log" 2>&1 \
      || { echo "FAIL $l $p: engine exit $?, $out/$p.$l.log"; exit 1; }
    t=$(grep -oE 'tok/s_gen: [0-9.]+' "$out/$p.$l.log" | cut -d' ' -f2)
    f=$(grep -oE 'mega fail word: [0-9]+' "$out/$p.$l.log" | grep -oE '[0-9]+$' || echo -)
    [ -n "$t" ] || { echo "FAIL $l $p: no tok/s_gen line"; exit 1; }
    grep -q 'TMAX: 16384 ' "$out/$p.$l.log" || { echo "FAIL $l $p: TMAX read-back is not 16384"; exit 1; }
    [ "$f" = 0 ] || [ "$f" = - ] || { echo "FAIL $l $p: mega fail word $f"; exit 1; }
    echo "$p $t $f" >> "$out/$l.txt"
  done
}

llama() { # label server
  local l=$1 srv=$2 port=$((20000 + RANDOM % 20000)) d=$out/$1 pid
  mkdir -p "$d"
  cfg=(-m "$model" -c 16384 -ngl 99 -fa on -ctk f16 -ctv f16 -b 2048 -ub 512 -t 8 -np 1 --no-webui --host 127.0.0.1)
  up() { for _ in $(seq 1 240); do curl -sf "localhost:$1/health" > /dev/null 2>&1 && return 0; sleep 0.5; done; return 1; }
  # read-back load at -lv 4: this llama.cpp keeps the load lines (offload, flash_attn, KV) out of the default
  # verbosity 3, and the timed server keeps the default so debug logging is not in its decode
  "$srv" "${cfg[@]}" --port "$port" -lv 4 > "$d/readback.log" 2>&1 & pid=$!
  up "$port" || { echo "FAIL $l: read-back server not healthy, $d/readback.log"; kill $pid; exit 1; }
  kill $pid; wait $pid 2>/dev/null || true
  {
    echo "server=$srv sha=$(sha256sum "$srv" | cut -c1-16) $(HIP_VISIBLE_DEVICES= "$srv" --version 2>&1 | grep -m1 'version:' || echo 'version: NOT READ')"
    grep -m1 -E 'offloaded [0-9]+/[0-9]+ layers' "$d/readback.log" || echo "offload: NOT READ"
    grep -m1 -iE 'flash_attn *[=:]' "$d/readback.log" || echo "flash_attn: NOT READ"
    grep -m1 -E 'K \((f16|q8_0)\)|type_k' "$d/readback.log" || echo "kv type: NOT READ"
  } | tee "$d/arm.txt"
  "$srv" "${cfg[@]}" --port "$port" > "$d/server.log" 2>&1 & pid=$!
  up "$port" || { echo "FAIL $l: server not healthy, $d/server.log"; kill $pid; exit 1; }
  ./.venv/bin/python3 - "$port" "$d" > "$out/$l.txt" <<'PY' || { echo "FAIL $l: requests, $d"; kill $pid; exit 1; }
import json, os, sys, urllib.request
from pathlib import Path
port, d = sys.argv[1], Path(sys.argv[2])
def post(ids):
    body = {"prompt": ids, "n_predict": 64, "temperature": 0, "top_k": 1, "cache_prompt": False, "return_tokens": True, "ignore_eos": True}
    r = urllib.request.Request(f"http://127.0.0.1:{port}/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=300))
files = sorted(Path("bench/mtp-prompts").glob("p*.tokens"))[:int(os.environ.get("LS_N") or 99)]
post([int(x) for x in files[0].read_text().split()])
for f in files:
    res = post([int(x) for x in f.read_text().split()])
    t = res["timings"]
    if t["predicted_n"] != 64:
        sys.exit(f"{f.stem}: predicted {t['predicted_n']} tokens, not 64")
    (d / f"{f.stem}.json").write_text(json.dumps({"tokens": res["tokens"], "timings": t}))
    print(f"{f.stem} {t['predicted_per_second']:.4f} -")
PY
  kill $pid; wait $pid 2>/dev/null || true
}

prompts=(bench/mtp-prompts/p*.tokens)
[ -z "${LS_N:-}" ] || prompts=("${prompts[@]:0:LS_N}")
ours ours "${prompts[@]}"
n=0
for srv in "$@"; do n=$((n + 1)); llama "llama$n" "$srv"; done
ours ours-tail "${prompts[@]:0:5}"
if grep -h 'NOT READ' "$out"/llama*/arm.txt; then echo "FAIL arm: llama read-back incomplete"; exit 1; fi

./.venv/bin/python3 - "$out" "$n" <<'PY'
import json, statistics as st, sys
from pathlib import Path
out, n = Path(sys.argv[1]), int(sys.argv[2])
def arm(l):
    return {r.split()[0]: float(r.split()[1]) for r in (out / f"{l}.txt").read_text().splitlines()}
def gen(log):
    for line in log.read_text().splitlines():
        if line.startswith("GENERATED"):
            return [int(x) for x in line.split(":", 1)[1].split()]
o = arm("ours")
sp = lambda v: (max(v) - min(v)) / st.median(v) * 100
print(f"ours       median {st.median(o.values()):.2f} tok/s_gen  spread {sp(list(o.values())):.1f}%  n={len(o)}")
tail = arm("ours-tail")
drift = st.median([tail[p] / o[p] for p in tail])
print(f"ours-tail  median ratio to the same prompts in the first block {drift:.4f} (drift control, n={len(tail)})")
for k in range(1, n + 1):
    l = arm(f"llama{k}")
    same = sum(gen(out / f"{p}.ours.log") == json.loads((out / f"llama{k}" / f"{p}.json").read_text())["tokens"] for p in l)
    print(f"llama{k}     median {st.median(l.values()):.2f} tok/s  spread {sp(list(l.values())):.1f}%  n={len(l)}"
          f"  |  ours / llama{k} = {st.median(o.values()) / st.median(l.values()):.3f}"
          f"  |  per-prompt ours faster {sum(o[p] > l[p] for p in l)}/{len(l)}  |  same 64 ids {same}/{len(l)} (not a gate)")
PY
