#!/usr/bin/env bash
# usage: bench/dflash2-d2d.sh OUTDIR [PROMPT_TXT...]   (run inside ONE gpu-wait job; default prompts p01 p02)
# dflash2 D2d: llama.cpp DFlash2 reference dumps for the Mojo draft parity gate. Binary: the dflash2-dump worktree
# (~/Projects/imports/llama.cpp-lanes/dflash2-dump, LLAMA_DFLASH_DUMP patch). Target ngl 52 (boss: 99 and 60 OOM),
# draft on GPU, 7 drafts, greedy, ctx 16384, 64 tokens. One OUTDIR/<prompt>.bin + .log per prompt.
set -euo pipefail
cd "$(dirname "$0")/.."
[ $# -ge 1 ] || { echo "FAIL args: OUTDIR [PROMPT_TXT...]"; exit 1; }
out=$(realpath -m "$1"); shift; mkdir -p "$out"
prompts=("$@"); [ ${#prompts[@]} -gt 0 ] || prompts=(bench/mtp-prompts/p01-water.txt bench/mtp-prompts/p02-python-fib.txt)
B=$HOME/Projects/imports/llama.cpp-lanes/dflash2-dump/build/bin/llama-speculative-simple
M=$HOME/Models/qwen38-27b-nerkyor-coder-q4lynn
[ -x "$B" ] || { echo "FAIL setup: no $B"; exit 1; }
for p in "${prompts[@]}"; do
  n=$(basename "$p" .txt)
  LLAMA_DFLASH_DUMP="$out/$n.bin" "$B" -m "$M/Qwen3.8-27B-Coder390-EfficientThink-Q4LynnStyle-Q8MTP.gguf" -ngl 52 \
    -md "$M/Qwen3.8-27B-nerkyor-DFlash2-Q8_0.gguf" -ngld 99 --spec-type draft-dflash --spec-draft-n-max 7 \
    -c 16384 -n 64 --temp 0 --seed 1 -f "$p" > "$out/$n.log" 2>&1 || { echo "FAIL $n: see $out/$n.log"; exit 1; }
  [ -s "$out/$n.bin" ] || { echo "FAIL $n: empty dump"; exit 1; }
  echo "$n dump $(stat -c %s "$out/$n.bin") B $(grep -E 'accept|n_drafted|decoded' "$out/$n.log" | tr '\n' ' ' | cut -c1-200)"
done
echo "PASS dflash2-d2d"
