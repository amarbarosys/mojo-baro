#!/usr/bin/env bash
# Short, decisive speed baseline: which arm has the biggest gap, not a publishable ratio (3 prompts per arm, one
# gpu-wait job, ~15 min). A dense Qwythos-9B Q4_0: engine at HEAD on p01-p03 vs llama-bench tg64. B Qwen2.5-7B
# Q4_K_M: spark native K-quant vs q8 (bench/sparkq4k-ab.sh, EXPLORE=1) plus llama-bench on the same GGUF. C q4x
# Strata Q2_0 at the 16K window: bench/q4x-ab.sh on 3 prompts, NG=32, both arms default (B gets an inert variable, q4x-ab refuses equal env strings). ARMS=ABC picks arms.
# Usage: bench/quick-speed.sh OUT
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?usage: bench/quick-speed.sh OUT}; mkdir -p "$OUT"
fail() { echo "FAIL $1: $2"; exit 1; }
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
  exec gpu-wait run --vram 22 --timeout 2400 -- env PATH=/opt/rocm/bin:/usr/bin:/bin:$HOME/iTools/bin ARMS="${ARMS:-ABC}" SKIP_B_OURS="${SKIP_B_OURS:-}" "$0" "$OUT"
fi
LB=${LLAMA_BENCH:-$HOME/llama.cpp-master/build/bin/llama-bench}
QW=$HOME/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_0-pure.gguf
Q7=$HOME/Models/qwen2.5-7b-instruct-import/qwen2.5-7b-instruct-q4_K_M-BARO-3c6c25c.gguf
ENG=${ENGINE:-.work/speed/engine-head}
echo "commit=$(git rev-parse --short HEAD) engine_sha=$(sha256sum "$ENG" | cut -c1-16)" | tee "$OUT/arm.txt"
lbench() { # label gguf
  "$LB" -m "$2" -p 0 -n 64 -fa 1 -ngl 99 -r 3 -o csv > "$OUT/$1.csv" 2> "$OUT/$1.err" || fail "$1" "llama-bench: $(tail -1 "$OUT/$1.err")"
  echo "$1 tok/s $(tail -1 "$OUT/$1.csv" | awk -F, '{gsub(/"/,""); print $(NF-1)}')" | tee -a "$OUT/summary.txt"
}
ARMS=${ARMS:-ABC}
if [[ $ARMS == *A* ]]; then
echo "== A dense 9B Q4_0" | tee -a "$OUT/summary.txt"
for p in p01 p02 p03; do
  tf=$(ls bench/mtp-prompts/$p-*.tokens)
  env BARO_PROMPT="$tf" BARO_PACK=.work/engine-pack-q4 BARO_SPEC=0 "$ENG" > "$OUT/A-$p.log" 2>&1 || fail "A $p" "engine exit, $OUT/A-$p.log"
  echo "A ours $p tok/s $(grep -oE 'tok/s_gen: [0-9.]+' "$OUT/A-$p.log" | cut -d' ' -f2)" | tee -a "$OUT/summary.txt"
done
lbench A-llama "$QW"
fi
if [[ $ARMS == *B* ]]; then
echo "== B 7B Q4_K_M" | tee -a "$OUT/summary.txt"
if [ -z "${SKIP_B_OURS:-}" ]; then
# 3 prompts cannot pass sparkq4k-ab's full-set agreement verdict; a quick run only needs its speed line
MODELS=qwen7b PROMPTS='p0[1-3]-*' EXPLORE=1 bench/sparkq4k-ab.sh "$OUT/B" > "$OUT/B.log" 2>&1 || true
grep -E '^speed' "$OUT/B.log" | tee -a "$OUT/summary.txt" || fail B "no speed line, $OUT/B.log"
fi
lbench B-llama "$Q7"
fi
if [[ $ARMS == *C* ]]; then
echo "== C q4x Strata 16K" | tee -a "$OUT/summary.txt"; mkdir -p "$OUT/C"
PROMPTS="$(ls bench/q4x-prompts/*.prompt | head -3 | tr '\n' ' ')" NG=32 Q4X_AB_INNER=1 ENGINE=${Q4X_ENGINE:-.work/speed/q4x-head} \
  bench/q4x-ab.sh "$OUT/C" "" "QUICK_SPEED_ARM=B" > "$OUT/C.log" 2>&1 || fail C "q4x-ab, $OUT/C.log"
grep -E '^A median' "$OUT/C.log" | tee -a "$OUT/summary.txt"
fi
echo "PASS quick-speed: $OUT/summary.txt"
