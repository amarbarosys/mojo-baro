#!/usr/bin/env bash
# Where a q4x verify window spends device time: rocprofv3 kernel + memory-copy trace of one prompt, plain decode (A) against
# Q4X_SPEC=K windows (K in KS), one gpu-wait job, one binary. Diagnosis only (no speed claim: the tracer slows launches).
# usage: bench/q4x-rows-trace.sh OUT      env ENGINE=bin KS="2 4" (KS="" traces plain decode only) PROMPT=p03-story NG=64 TMAX=16384; EXTRA="K=V ..." adds env to every arm
set -euo pipefail
OUT=${1:?usage: q4x-rows-trace.sh OUT}; REPO=$(cd "$(dirname "$0")/.." && pwd); . "$REPO/bench/min16k.sh"; . "$REPO/bench/pcie-link.sh"
ENGINE=${ENGINE:?ENGINE}; KS=${KS-2 4}; PROMPT=${PROMPT:-p03-story}; NG=${NG:-64}; TMAX=${TMAX:-16384}; min16k "$TMAX" 16000
GGUF=${Q4X_GGUF:-$HOME/Models/qwen3.8-flash-next-gsq/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf}
MTP=${MTP_PATH:-$HOME/Models/qwen3.8-flash-next/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf}
WARM=${Q4X_WARM:-$REPO/exchange/receipts/q4x-strata/r3a-warm-l01.txt}
if [ -z "${ROWS_TRACE_INNER:-}" ]; then
  mkdir -p "$OUT"
  exec gpu-wait run --vram 20 --timeout 1800 -- env ROWS_TRACE_INNER=1 ENGINE="$ENGINE" EXTRA="${EXTRA:-}" KS="$KS" PROMPT="$PROMPT" NG="$NG" TMAX="$TMAX" Q4X_GGUF="$GGUF" MTP_PATH="$MTP" Q4X_WARM="$WARM" HOME="$HOME" PATH="/opt/rocm/bin:$PATH" bash "$REPO/bench/q4x-rows-trace.sh" "$OUT"
fi
cd "$REPO"; export MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=2
ids=$(jq -r 'join(" ")' bench/q4x-prompts/$PROMPT.prompt)
echo "engine sha256 $(sha256sum "$ENGINE" | cut -c1-16) prompt $PROMPT ng $NG tmax $TMAX $(pcie_link)" | tee "$OUT/receipt.txt"
for a in A $(for k in $KS; do echo K$k; done); do
  e=(Q4X_TMAX="$TMAX" Q4X_WARM="$WARM" $EXTRA); [ "$a" = A ] || e+=(Q4X_MTP="$MTP" Q4X_SPEC="${a#K}" Q4X_SPEC_P=0.5)
  rocprofv3 --kernel-trace --memory-copy-trace -f csv -d "$OUT/$a" -o trace -- env "${e[@]}" "$ENGINE" "$GGUF" "$NG" $ids > "$OUT/$a.log" 2>&1 || { echo "FAIL $a: rocprofv3 or engine exit, $OUT/$a.log"; exit 1; }
  grep -q '^ids ' "$OUT/$a.log" || { echo "FAIL $a: no ids line"; exit 1; }
  echo "$a ids $(grep '^ids ' "$OUT/$a.log" | md5sum | cut -c1-8) $(grep -m1 'spec rounds' "$OUT/$a.log" | cut -c1-150)" | tee -a "$OUT/receipt.txt"
done
echo "PASS rows-trace: $OUT$(pcie_flag)"
