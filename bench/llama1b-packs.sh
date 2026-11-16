#!/usr/bin/env bash
# Builds the Llama-3.2-1B packs the spark A/B scripts read (CPU only, about 1 min). Out: .work/llama1b/pack-{kq,q8}.
# kq = engine-pack --dense on the BARO Q4_K_M gguf (K-quant blocks raw). q8 = the control: HEAD's dense packer keeps
# Q4_K/Q6_K raw, so the q8 arm is the stock Q4_K_M gguf requantized to Q8_0 (llama-quantize --allow-requantize, the
# dequantize-then-q8 the old packer did) and packed with the same tool. Replaces the vibethinker/minicpm5/qwen7b packs
# deleted 2026-11-15.
set -euo pipefail
cd "$(dirname "$0")/.."
D=$HOME/Models/llama-3.2-1b-instruct-q4_K_M
G=$D/Llama-3.2-1B-Instruct-Q4_K_M-BARO-38e5113.gguf; S=$D/Llama-3.2-1B-Instruct-Q4_K_M.gguf
Q=${LLAMA_QUANTIZE:-$HOME/llama.cpp/build/bin/llama-quantize}
for f in "$G" "$S" "$Q" .work/engine-pack; do [ -e "$f" ] || { echo "FAIL setup: $f missing"; exit 1; }; done
mkdir -p .work/llama1b
pk() { [ -s "$2/pack.bin" ] && { echo "have $2"; return; }; mkdir -p "$2"
  ./.work/engine-pack "$1" "$2" --dense > "$2.log" 2>&1 || { echo "FAIL pack $2: $(tail -n 1 "$2.log")"; exit 1; }; }
pk "$G" .work/llama1b/pack-kq
[ -s .work/llama1b/q8.gguf ] || "$Q" --allow-requantize "$S" .work/llama1b/q8.gguf Q8_0 > .work/llama1b/q8.log 2>&1 \
  || { echo "FAIL requant: $(tail -n 1 .work/llama1b/q8.log)"; exit 1; }
pk .work/llama1b/q8.gguf .work/llama1b/pack-q8
nk=$(awk '$2 ~ /^q[46]k/' .work/llama1b/pack-kq/index.txt | wc -l); n8=$(awk '$2 == "q8"' .work/llama1b/pack-q8/index.txt | wc -l)
[ "$nk" -gt 0 ] && [ "$n8" -gt 0 ] || { echo "FAIL arms: kq pack has $nk K-quant matrices, q8 pack $n8 q8"; exit 1; }
echo "PASS llama1b-packs: $nk kq / $n8 q8 matrices, kq $(stat -c %s .work/llama1b/pack-kq/pack.bin) q8 $(stat -c %s .work/llama1b/pack-q8/pack.bin) bytes"
