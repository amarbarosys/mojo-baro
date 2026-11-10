#!/usr/bin/env bash
# usage: bench/tier-cpu.sh ENGINE OUT [smoke|force|speed]       (bench/moe-tier-cpu-protocol.md)
# Arms, same binary: A = BARO_TIER=64 pinned, ZC default; B = A + BARO_TIER_CPU=1.
#   smoke  3 prompts x 32 tokens, resident: P1 read-back only (mode line, cpu_misses), no claim.
#   force  teacher-forced agreement of A and B on llama.cpp's 64 greedy ids per prompt; bar frozen
#          in the protocol: B mean >= A mean - 0.5 pp AND B min >= A min - 1.0 pp.
#   speed  ab-resident.sh over the prompts, GEN 64: B/A ratio of median tok/s_gen, this run only.
# Knobs (carried through the queue): QUICK=N first N of the 20 mtp-prompts; EXPLORE=1 skips the
# preflight check and caps the verdict at UNVERIFIED (exit 3). Receipts: OUT/arm.txt, OUT/*.log.
# If speed lands under band: bench/tier-gpu-busy.sh splits GPU busy from wall per token.
set -euo pipefail
cd "$(dirname "$0")/.."
eng=${1:?engine}; out=$(realpath -m "${2:?out dir}"); phase=${3:-smoke}
quick=${QUICK:-0}; explore=${EXPLORE:-0}
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ] && command -v gpu-wait >/dev/null; then
  exec gpu-wait run --vram 20 --timeout 2400 -- env QUICK="$quick" EXPLORE="$explore" "$0" "$eng" "$out" "$phase"
fi
fail() { echo "FAIL $1: $2"; exit 1; }
[ -x "$eng" ] || fail setup "engine $eng not executable"
case $phase in smoke|force|speed) ;; *) fail setup "phase '$phase' is not smoke|force|speed" ;; esac
if [ "$explore" = 1 ]; then echo "EXPLORE=1: preflight check skipped, verdict capped at UNVERIFIED"
else bench/preflight.sh --check || fail preflight "no passing bench/preflight.sh for these contents"; fi
mkdir -p "$out"
pack=${TIER_PACK:-$HOME/.cache/baro/qwen35moe-f2628b18234dbbb4/tier-e5c7fc0d}
gguf=$HOME/Models/RegesCore-1.0-35/RegesCore-1.0-35B-UD-Q4_K_S-BARO-e340ee1.gguf
[ -s "$pack/experts.bin" ] || fail setup "$pack/experts.bin missing or empty (tools/pack-split-experts.py)"
mapfile -t prompts < <(ls bench/mtp-prompts/p*.tokens)
if [ "$quick" -gt 0 ]; then prompts=("${prompts[@]:0:$quick}"); fi
envA="BARO_TIER=64 BARO_TIER_PINNED=1"; envB="$envA BARO_TIER_CPU=1"
{ echo "phase=$phase quick=$quick explore=$explore prompts=${#prompts[@]}"
  echo "engine=$eng sha=$(sha256sum "$eng" | cut -c1-16)"
  echo "pack=$pack index_sha=$(sha256sum "$pack/index.txt" | cut -c1-16) experts_bytes=$(stat -c %s "$pack/experts.bin")"
  echo "armA='$envA' armB='$envB'"
  echo "commit=$(git rev-parse --short HEAD) dirty='$(git status --porcelain | grep -v '^??' | tr '\n' ' ')'"
  echo "power_cap_uW=$(cat /sys/class/drm/card*/device/hwmon/hwmon*/power1_cap 2>/dev/null | head -1)"
  c=/sys/devices/system/cpu/cpu0/cpufreq
  echo "cpu driver=$(cat $c/scaling_driver 2>/dev/null) governor=$(cat $c/scaling_governor 2>/dev/null) epp=$(cat $c/energy_performance_preference 2>/dev/null) min_khz=$(cat $c/scaling_min_freq 2>/dev/null) max_khz=$(cat $c/scaling_max_freq 2>/dev/null)"; } | tee "$out/arm.txt"

readback() {  # $1 log, $2 label, $3 expected mode word: P1 from the engine's own echoes
  grep -q "expert tier: cap 64 .*mode $3" "$1" || fail "readback $2" "no 'expert tier: cap 64 ... mode $3' in $1"
  grep -q 'cpu_misses' "$1" || fail "readback $2" "no tier report line in $1"
  local m; m=$(grep -oE 'cpu_misses [0-9]+' "$1" | awk '{s+=$2} END {print s+0}')
  if [ "$3" = "pinned+cpu" ] && [ "$m" -eq 0 ]; then fail "readback $2" "cpu_misses 0: CPU path never ran"; fi
  if [ "$3" = "pinned+zc" ] && [ "$m" -ne 0 ]; then fail "readback $2" "cpu_misses $m in the ZC arm"; fi
  echo "$2 mode=$3 cpu_misses=$m" | tee -a "$out/arm.txt"
}

if [ "${GATE_DRYRUN:-0}" = 1 ]; then fail dryrun "GATE_DRYRUN stop before the first engine launch"; fi

verdict() {  # $1 the PASS|FAIL line
  if [ "$explore" = 1 ]; then echo "UNVERIFIED (EXPLORE=1): $1"; exit 3; fi
  echo "$1"; case $1 in PASS*) exit 0 ;; *) exit 1 ;; esac
}

if [ "$phase" != force ]; then
  if [ "$phase" = smoke ]; then gen=32; pl=("${prompts[@]:0:3}"); else gen=64; pl=("${prompts[@]}"); fi
  BARO_PACK=$pack GEN=$gen PROMPTS="${pl[*]}" bench/ab-resident.sh "$eng" "$out/$phase" "$envA" "$envB" A B \
    > "$out/$phase.txt" 2>&1 || { tail -5 "$out/$phase.txt"; fail "$phase" "ab-resident.sh failed, see $out/$phase.txt"; }
  readback "$out/$phase/A.log" A pinned+zc
  readback "$out/$phase/B.log" B pinned+cpu
  tail -n +2 "$out/$phase/results.txt"
  line=$(grep -E '^A median' "$out/$phase.txt") || fail "$phase" "no median line in $out/$phase.txt"
  verdict "PASS $phase: $line"
fi

port=18083; refdir=$out/refs; mkdir -p "$refdir"
need=0; for tf in "${prompts[@]}"; do [ -s "$refdir/$(basename "$tf" .tokens).ref.ids" ] || need=1; done
if [ "$need" = 1 ]; then
  echo "refcache miss key=$(sha256sum "$gguf" | cut -c1-16)" | tee -a "$out/arm.txt"
  "$HOME/llama.cpp/build/bin/llama-server" -m "$gguf" -c 4096 -ngl 99 -fa on -ctk f16 -ctv f16 -t 8 -np 1 \
    --no-cont-batching --host 127.0.0.1 --port $port > "$out/llama.log" 2>&1 & lp=$!
  trap 'kill $lp 2>/dev/null || true' EXIT
  for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then break; fi
    kill -0 $lp 2>/dev/null || fail llama "server died, see $out/llama.log"; sleep 1
  done
  for tf in "${prompts[@]}"; do
    p=$(basename "$tf" .tokens); ids=$(tr -s ' \n' ',,' < "$tf" | sed 's/,$//')
    curl -sf "http://127.0.0.1:$port/completion" -H 'Content-Type: application/json' \
      -d "{\"prompt\": [$ids], \"n_predict\": 128, \"temperature\": 0, \"top_k\": 1, \"seed\": 1, \"ignore_eos\": true, \"cache_prompt\": false, \"return_tokens\": true}" \
      | python3 -c 'import json,sys; d=json.load(sys.stdin); n=d["timings"]["prompt_n"]; ids=d["tokens"][n:n+64]; assert len(ids) == 64; print(" ".join(map(str,ids)))' \
      > "$refdir/$p.ref.ids" || fail llama "reference for $p"
  done
  kill $lp; wait $lp 2>/dev/null || true; trap - EXIT
else echo "refcache hit" | tee -a "$out/arm.txt"; fi

echo "prompt arm agree checked" > "$out/force.txt"
for tf in "${prompts[@]}"; do
  p=$(basename "$tf" .tokens)
  for arm in A B; do
    if [ $arm = A ]; then e=$envA; mode=pinned+zc; else e=$envB; mode=pinned+cpu; fi
    log=$out/force-$p-$arm.log
    env BARO_PACK="$pack" BARO_SPEC=0 BARO_MEGA=0 BARO_PROMPT="$tf" BARO_FORCE="$refdir/$p.ref.ids" $e "$eng" > "$log" 2>&1 \
      || fail "force $p $arm" "engine exit, see $log"
    readback "$log" "$p-$arm" $mode > /dev/null
    a=$(sed -n 's/^forced agreement: \([0-9]*\) \/ \([0-9]*\)$/\1 \2/p' "$log")
    [ -n "$a" ] || fail "force $p $arm" "no 'forced agreement' line in $log"
    echo "$p $arm $a" >> "$out/force.txt"
  done
done
column -t "$out/force.txt"
res=$(python3 - "$out/force.txt" <<'PY'
import sys
rows = [l.split() for l in open(sys.argv[1]).read().splitlines()[1:]]
pct = {a: [100 * int(r[2]) / int(r[3]) for r in rows if r[1] == a] for a in "AB"}
mean = {a: sum(v) / len(v) for a, v in pct.items()}; low = {a: min(v) for a, v in pct.items()}
ok = mean["B"] >= mean["A"] - 0.5 and low["B"] >= low["A"] - 1.0
print(("PASS" if ok else "FAIL") + f" force: A mean {mean['A']:.2f}% min {low['A']:.2f}% | B mean {mean['B']:.2f}% min {low['B']:.2f}% | bar B mean >= A-0.5, B min >= A-1.0 | n={len(pct['A'])}")
PY
)
verdict "$res"
