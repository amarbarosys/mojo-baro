#!/usr/bin/env bash
# T3b step 2c fit (AMDHQ docs/design/latent-os/06-experiments.md `## T3b`, amendment "step 2c"): N
# baro-serve engines (default 4) on one GPU, one weight copy. A exports (BARO_WEIGHTS_EXPORT), the
# others import the same record; all keep token_embd in host memory (BARO_EMBD_HOST=1).
# Checks: A exported, every other engine imported and never read pack.bin; one 64-token greedy
# request per engine alone, ids equal to A's on every engine; then all N requests at once, ids still
# equal; bar: peak device VRAM minus the pre-launch base < QUAD_VRAM_BAR (default 16.0e9); device
# VRAM back within 0.3e9 B of base after all exit; no megakernel fail word. Info: alone and
# concurrent tok/s. QUAD_ENV="K=V ..." adds engine env (the arm, e.g. BARO_MEGA=0).
# Run under gpu-wait as a script with literal paths: the queue passes no env and no stdin.
# usage: bench/t3b-quad.sh ENGINE PACK OUTDIR [PORT]
set -euo pipefail
cd "$(dirname "$0")/.."

eng=$(realpath "${1:?usage: bench/t3b-quad.sh ENGINE PACK OUTDIR [PORT]}")
pack=$(realpath "${2:?usage: bench/t3b-quad.sh ENGINE PACK OUTDIR [PORT]}")
out=$(realpath -m "${3:?usage: bench/t3b-quad.sh ENGINE PACK OUTDIR [PORT]}")
port=${4:-8260}
n=${QUAD_N:-4}
bar=${QUAD_VRAM_BAR:-16.0e9}
read -r -a xenv <<< "${QUAD_ENV:-}"
tmax=${T3B_TMAX:-9216}
shim=$(realpath -m .work/shim-build/libamarbaro_shim.so)
serve=serve/target/release/baro-serve
rec=$out/weights.rec
mkdir -p "$out"
fail() { echo "FAIL $1: $2"; exit 1; }
[ -x "$eng" ] || fail setup "no engine at $eng"
[ -x "$serve" ] || fail setup "no $serve"
[ -f "$shim" ] || fail setup "no shim at $shim"
echo "commit=$(git rev-parse --short HEAD) engine=$eng sha=$(sha256sum "$eng" | cut -c1-16) serve_sha=$(sha256sum "$serve" | cut -c1-16) pack=$pack n=$n env='${QUAD_ENV:-}' bar=$bar BARO_TMAX=$tmax job=${GPU_WAITING_ROOM_JOB:-}" > "$out/arm.txt"
[ "${GATE_DRYRUN:-0}" = 1 ] && { echo "VOID dry run: stop before the engine launch ($eng)"; exit 3; }

vram_used() {
  for d in /sys/class/drm/card[0-9]/device; do
    echo "$(< "$d/mem_info_vram_total") $(< "$d/mem_info_vram_used")"
  done | sort -n | tail -1 | cut -d' ' -f2
}
pids=()
stop_all() {
  for p in "${pids[@]}"; do kill -- -"$p" 2> /dev/null || kill "$p" 2> /dev/null || true; done
  wait 2> /dev/null || true
}
trap stop_all EXIT
base=$(vram_used)
echo "phase=base used=$base" > "$out/vram.txt"
( while :; do echo "t=$(date +%s.%N) used=$(vram_used)" >> "$out/vram.txt"; sleep 0.5; done ) &
pids+=($!)

for i in $(seq 0 $((n - 1))); do
  w=(BARO_WEIGHTS_IMPORT="$rec"); [ "$i" = 0 ] && w=(BARO_WEIGHTS_EXPORT="$rec")
  setsid env "${w[@]}" "${xenv[@]}" BARO_EMBD_HOST=1 BARO_PACK="$pack" BARO_SPEC=0 BARO_TMAX="$tmax" BARO_SHIM_LIB="$shim" \
    "$serve" --engine "$eng" --pack "$pack" --port $((port + i)) > "$out/serve-$i.out" 2> "$out/serve-$i.err" &
  pids+=($!)
  for _ in $(seq 1 180); do curl -sf "http://127.0.0.1:$((port + i))/health" > /dev/null 2>&1 && break; sleep 1; done
  curl -sf "http://127.0.0.1:$((port + i))/health" > /dev/null || fail "health $i" "$out/serve-$i.err"
  echo "engine $i up, used=$(vram_used)" >> "$out/arm.txt"
done
grep -q "weights exported over IPC: [0-9]* bytes" "$out/serve-0.err" || fail readback "engine 0 exported no weights"
for i in $(seq 1 $((n - 1))); do
  grep -q "weights imported over IPC: [0-9]* bytes" "$out/serve-$i.err" || fail readback "engine $i imported no weights"
  grep -q "loading pack:" "$out/serve-$i.err" && fail readback "engine $i read pack.bin"
done
for i in $(seq 0 $((n - 1))); do
  grep -q "seq_cap: 1$" <(grep -m1 -o 'seq_cap: [0-9]*' "$out/serve-$i.err") || fail readback "engine $i seq_cap is not 1"
  grep -q "token_embd in host memory: [0-9]* bytes" "$out/serve-$i.err" || fail readback "engine $i kept token_embd in VRAM"
  grep -h "weights .*ported over IPC\|token_embd in host" "$out/serve-$i.err" | sed "s/^/$i /" >> "$out/arm.txt"
done

rc=0
python3 - "$out" "$port" "$n" "$base" "$bar" <<'PY' || rc=$?
import json, pathlib, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
out, port, n, base, bar = pathlib.Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
prompt = [int(x) for x in open("bench/mtp-prompts/p01-water.tokens").read().split()]
body = json.dumps({"prompt": prompt, "max_tokens": 64, "spec": False, "temperature": 0}).encode()
def call(i):
    req = urllib.request.Request(f"http://127.0.0.1:{port + i}/v1/completions", data=body, headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        toks = json.loads(r.read())["choices"][0]["tokens"]
    return toks, time.perf_counter() - t
log, fails = [], 0
def check(cond, label):
    global fails
    log.append(("PASS " if cond else "FAIL ") + label); fails += 0 if cond else 1
    print(log[-1], flush=True)
alone = [call(i) for i in range(n)]
ref = alone[0][0]
for i, (t, s) in enumerate(alone):
    check(len(t) == 64 and t == ref, f"engine {i} alone: 64 ids == engine 0's ({len(t)} ids, {s:.3f} s)")
t0 = time.perf_counter()
with ThreadPoolExecutor(n) as ex:
    conc = list(ex.map(call, range(n)))
wall = time.perf_counter() - t0
for i, (t, s) in enumerate(conc):
    check(t == ref, f"engine {i} concurrent: ids == engine 0's alone ({s:.3f} s)")
used = [int(l.split("used=")[1]) for l in open(out / "vram.txt") if "used=" in l and l.split("used=")[1].strip().isdigit()]
peak = max(used)
check(peak - base < bar, f"bar: peak {peak / 1e9:.2f} GB - base {base / 1e9:.2f} = {(peak - base) / 1e9:.2f} GB < {bar / 1e9:.2f}")
print(f"INFO alone tok/s per engine {[round(64 / s, 1) for _, s in alone]}")
print(f"INFO concurrent wall {wall:.3f} s, aggregate {n * 64 / wall:.1f} tok/s, per request {[round(64 / s, 1) for _, s in conc]}")
(out / "results.json").write_text(json.dumps({"alone_s": [s for _, s in alone], "conc_s": [s for _, s in conc], "wall": wall, "peak": peak, "base": base, "ref": ref}, indent=1))
(out / "gate.txt").write_text("\n".join(log) + f"\n{'PASS' if fails == 0 else 'FAIL'} t3b-quad: {len(log) - fails}/{len(log)}\n")
print(f"{'PASS' if fails == 0 else 'FAIL'} t3b-quad: {len(log) - fails}/{len(log)}")
sys.exit(0 if fails == 0 else 1)
PY
grep -h "BARO_MEGA:" "$out/serve-0.err" | head -1 >> "$out/arm.txt"
grep -q "NOT-RESIDENT\|mega fail word: [1-9]" "$out"/serve-*.err && { echo "FAIL failword: a grid barrier timed out ($(grep -l 'NOT-RESIDENT\|mega fail word: [1-9]' "$out"/serve-*.err | tr '\n' ' '))"; rc=1; }
stop_all; trap - EXIT
for _ in $(seq 1 20); do now=$(vram_used); [ $((now - base)) -lt 300000000 ] && break; sleep 0.5; done
echo "phase=exit used=$now base=$base" | tee -a "$out/arm.txt"
[ $((now - base)) -lt 300000000 ] || { echo "FAIL exit-vram: device VRAM $now after all engines exited, base $base"; rc=1; }
[ "$rc" = 0 ] || fail gate "$out/gate.txt"
echo "PASS t3b-quad: $out/gate.txt"
