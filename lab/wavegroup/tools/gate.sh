#!/usr/bin/env bash
# Lab gate: build decode + tests from this repo, then tokenizer 9/9, planner, rope, weights, ref_story 64/64 and the
# three-arm stint (wavegroup, mojo-baro spark via ALT, llama.cpp). One gpu-wait job for the GPU part.
# usage: tools/gate.sh [OUTDIR]   needs BARO_ENGINE (spark binary) and BARO_PACK (its pack) for the ALT arm
set -euo pipefail
cd "$(dirname "$0")/.."
B=$(cd ../.. && pwd)
M=${GGUF:-$HOME/Models/llama-3.2-1b-instruct-q4_K_M/Llama-3.2-1B-Instruct-Q4_K_M.gguf}
out=${1:-.work/gate}
: "${BARO_ENGINE:?FAIL gate: set BARO_ENGINE}" "${BARO_PACK:?FAIL gate: set BARO_PACK}"
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
  exec gpu-wait run --timeout 2400 -- env BARO_ENGINE="$BARO_ENGINE" BARO_PACK="$BARO_PACK" GGUF="$M" "$0" "$@"
fi
mkdir -p .work "$out"
inc=(-I kernels/gemm -I kernels/gemv -I kernels/llama -I "$B/kernels" -I "$B" -I "$B/serve" -I "$B/tools")
.venv/bin/mojo build "${inc[@]}" -D WMMA_TB=1 -D WMMA_C_F16=0 -o .work/decode kernels/llama/decode.mojo > "$out/build.log" 2>&1 \
  || { echo "FAIL build decode: $(grep -m1 error "$out/build.log")"; exit 1; }
for t in test_tok test_plan test_rope test_weights; do
  .venv/bin/mojo build "${inc[@]}" -o .work/$t kernels/llama/$t.mojo > "$out/$t.build.log" 2>&1 || { echo "FAIL build $t"; exit 1; }
done
tk=$(.work/test_tok "$M" tests/tok_corpus.txt tests/tok_ref.txt | tail -1)
echo "$tk"
echo "$tk" | grep -qE '^tok ([0-9]+) / \1( |$)' || { echo "FAIL tok"; exit 1; }
.work/test_plan | grep -q '^PASS plan' && echo "plan PASS" || { echo "FAIL plan"; exit 1; }
.work/test_rope | grep -q '^PASS rope' && echo "rope PASS" || { echo "FAIL rope"; exit 1; }
.work/test_weights "$M" | grep -q '^PASS weights' && echo "weights PASS" || { echo "FAIL weights"; exit 1; }
got=$(.work/decode "$M" "$(sed -n 1p tests/ref_story.txt)" 64 | sed -n 1p)
[ "$got" = "$(sed -n 2p tests/ref_story.txt)" ] && echo "ref_story 64/64 equal" || { echo "FAIL ref_story"; exit 1; }
ALT=$PWD/tools/baro_arm.sh GGUF="$M" tools/stint.sh .work/decode "$out/stint"
