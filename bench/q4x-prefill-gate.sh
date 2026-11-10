#!/usr/bin/env bash
# latent-flash P1 identity gate: final-norm hidden (result_norm, 2560 f32) of every prompt position,
# prefill(ids) vs step() on the same ids, byte-identical, 20 prompts (bench/q4x-prompts/*.prompt =
# llama-server /tokenize of bench/mtp-prompts, frozen 2026-10-09). Prereg exchange/latent-flash-p1-prereg-2026-10-12.md.
# usage: bench/q4x-prefill-gate.sh OUT_DIR [N_PROMPTS]      (runs itself under gpu-wait; Q4X_BIN overrides the binary)
set -euo pipefail
OUT=${1:?out dir}; NP=${2:-20}
REPO=$(cd "$(dirname "$0")/.." && pwd)
GGUF=${Q4X_GGUF:-$HOME/Models/qwen3.8-flash-next-gsq/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf}
BIN=${Q4X_BIN:-.work/p8/q4x}
if [ -z "${Q4X_GATE_INNER:-}" ]; then
  mkdir -p "$OUT"
  exec gpu-wait run --vram 20 --timeout 7200 -- env Q4X_GATE_INNER=1 Q4X_GGUF="$GGUF" Q4X_BIN="$BIN" HOME="$HOME" bash "$REPO/bench/q4x-prefill-gate.sh" "$OUT" "$NP"
fi
cd "$REPO"
export MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=2
{
  echo "date $(date -Is)"; echo "commit $(git rev-parse HEAD)"; echo "binary $BIN sha256 $(sha256sum "$BIN" | cut -c1-16)"
  echo "gguf $GGUF sha256 $(sha256sum "$GGUF" | cut -c1-16)"; echo "env Q4X_TMAX=256 (prefill arm) Q4X_HIDDEN_F32=1 both arms"
  echo "vram_free_gb $(for d in /sys/class/drm/card*/device; do [ -f $d/mem_info_vram_total ] && awk -v t=$(cat $d/mem_info_vram_total) -v u=$(cat $d/mem_info_vram_used) 'BEGIN { if (t > 20e9) printf "%.2f", (t - u) / 2^30 }'; done)"
} > "$OUT/receipt.txt"
PROMPTS=$(ls bench/q4x-prompts/*.prompt | head -n "$NP")
fail=0; names=""; n=0
for p in $PROMPTS; do
  b=$(basename "$p" .prompt); n=$((n + 1))
  ids=$(jq -r 'join(" ")' "$p"); T=$(jq length "$p")
  rm -f "$OUT/$b".{step,pre}.bin "$OUT/$b".{step,pre}.bin.n
  ok=1
  env Q4X_HIDDEN="$OUT/$b.step.bin" Q4X_HIDDEN_F32=1 "$BIN" "$GGUF" 1 $ids > "$OUT/$b.step.log" 2>&1 || ok=0
  env Q4X_HIDDEN="$OUT/$b.pre.bin" Q4X_HIDDEN_F32=1 Q4X_PREFILL=1 Q4X_TMAX=256 "$BIN" "$GGUF" 1 $ids > "$OUT/$b.pre.log" 2>&1 || ok=0
  for arm in step pre; do
    grep -q '^ids ' "$OUT/$b.$arm.log" && grep -q '^prompt ' "$OUT/$b.$arm.log" || ok=0
    [ "$(cat "$OUT/$b.$arm.bin.n" 2>/dev/null)" = "$T" ] || ok=0
  done
  [ "$(stat -c %s "$OUT/$b.step.bin" 2>/dev/null)" = "$((T * 2560 * 4))" ] || ok=0
  cmp -s "$OUT/$b.step.bin" "$OUT/$b.pre.bin" || ok=0
  ms_s=$(grep '^prompt ' "$OUT/$b.step.log" | awk '{for(i=1;i<=NF;i++) if($i=="ms/token") print $(i+1)}')
  ms_p=$(grep '^prompt ' "$OUT/$b.pre.log" | awk '{for(i=1;i<=NF;i++) if($i=="ms/token") print $(i+1)}')
  ids_s=$(grep '^ids' "$OUT/$b.step.log" || true); ids_p=$(grep '^ids' "$OUT/$b.pre.log" || true)
  [ "$ids_s" = "$ids_p" ] && [ -n "$ids_s" ] || ok=0
  if [ $ok = 1 ]; then echo "PASS $b T=$T bytes=$((T * 2560 * 4)) identical; next id equal; ms/token step $ms_s prefill $ms_p" | tee -a "$OUT/receipt.txt"
  else fail=$((fail + 1)); names="$names $b"; echo "FAIL $b T=$T (see $OUT/$b.*.log)" | tee -a "$OUT/receipt.txt"; fi
done
if [ $fail = 0 ]; then echo "PASS $n/$n" | tee -a "$OUT/receipt.txt"; else echo "FAIL $fail/$n$names" | tee -a "$OUT/receipt.txt"; exit 1; fi
