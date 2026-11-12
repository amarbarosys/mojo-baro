#!/usr/bin/env bash
# MTP-draft expert overlap (lane draft-overlap, prereg bench/q4x-draft-overlap-prereg.md): a COUNT, valid on any PCIe link.
# ENGINE_LOG = engine built with -D Q4X_LOG_DRAFT_EXPERTS=1, ENGINE_OFF = same source without it (identity arm, first NI prompts).
# Per run: Q4X_MTP + Q4X_SPEC=K, 20 prompts, K in KS. Read-back (P1) per run: `kv 16384`, expert-cache slots, `spec rounds ... k K pmin P`,
# the `draftov` line (log engine) or its absence (off engine). Identity: log ids == off ids on the first NI prompts at every K.
# usage: bench/q4x-draft-overlap.sh OUT_DIR      env ENGINE_LOG=bin ENGINE_OFF=bin KS="2 4" NP=20 NI=5 NG=64 TMAX=16384 Q4X_SPEC_P=0.5
set -euo pipefail
OUT=${1:?usage: q4x-draft-overlap.sh OUT_DIR}; REPO=$(cd "$(dirname "$0")/.." && pwd); . "$REPO/bench/min16k.sh"
ENGINE_LOG=${ENGINE_LOG:?ENGINE_LOG}; ENGINE_OFF=${ENGINE_OFF:?ENGINE_OFF}; KS=${KS:-2 4}; NP=${NP:-20}; NI=${NI:-5}; NG=${NG:-64}; TMAX=${TMAX:-16384}; PMIN=${Q4X_SPEC_P:-0.5}
GGUF=${Q4X_GGUF:-$HOME/Models/qwen3.8-flash-next-gsq/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf}
MTP=${MTP_PATH:-$HOME/Models/qwen3.8-flash-next/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf}
WARM=${Q4X_WARM:-$REPO/exchange/receipts/q4x-strata/r3a-warm-l01.txt}
min16k "$TMAX" 16000
if [ -z "${OV_INNER:-}" ]; then
  mkdir -p "$OUT"
  exec gpu-wait run --vram 20 --timeout "${GPU_TIMEOUT:-5400}" -- env OV_INNER=1 ENGINE_LOG="$ENGINE_LOG" ENGINE_OFF="$ENGINE_OFF" KS="$KS" NP="$NP" NI="$NI" NG="$NG" TMAX="$TMAX" Q4X_SPEC_P="$PMIN" Q4X_GGUF="$GGUF" MTP_PATH="$MTP" Q4X_WARM="$WARM" HOME="$HOME" PATH="$PATH" bash "$REPO/bench/q4x-draft-overlap.sh" "$OUT"
fi
cd "$REPO"; export MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=2
R=$OUT/receipt.txt
fail() { echo "FAIL $1: $2 (receipt $R)" | tee -a "$R"; exit 1; }
echo "date $(date -Is) commit $(git rev-parse --short HEAD) dirty=$(git status --porcelain -- serve kernels | wc -l) log=$ENGINE_LOG sha256 $(sha256sum "$ENGINE_LOG" | cut -c1-16) off=$ENGINE_OFF sha256 $(sha256sum "$ENGINE_OFF" | cut -c1-16) tmax=$TMAX ng=$NG np=$NP ni=$NI ks='$KS' pmin=$PMIN warm=$(sha256sum "$WARM" | cut -c1-16) mtp=$MTP pcie=$(cat /sys/bus/pci/devices/0000:03:00.0/current_link_speed 2>/dev/null || echo unknown)" | tee "$R"
run() { # engine log prompt-file k
  local ids; ids=$(jq -r 'join(" ")' "$3")
  env Q4X_TMAX="$TMAX" Q4X_WARM="$WARM" Q4X_MTP="$MTP" Q4X_SPEC="$4" Q4X_SPEC_P="$PMIN" "$1" "$GGUF" "$NG" $ids > "$2" 2>&1 || fail run "$2 exited non-zero: $(grep -m1 -iE 'error|exception|assert' "$2" | cut -c1-120)"
  grep -q '^ids ' "$2" || fail run "$2 printed no ids"
  grep -qiE 'not resident|exception|assert' "$2" && fail run "$2: $(grep -m1 -iE 'not resident|exception|assert' "$2")"
  grep -q "kv $TMAX" "$2" || fail readback "$2: no 'kv $TMAX' line"
  grep -q "^spec rounds [0-9]* drafted [0-9]* accepted [0-9]* " "$2" || fail readback "$2: no spec line (MTP path not taken)"
  return 0
}
i=0; bad=0
for pf in $(ls bench/q4x-prompts/*.prompt | head -"$NP"); do
  b=$(basename "$pf" .prompt)
  for k in $KS; do
    l=$OUT/$b.k$k.log; run "$ENGINE_LOG" "$l" "$pf" "$k"
    grep -q '^draftov windows' "$l" || fail readback "$l: log engine printed no draftov line"
    grep -q "spec rounds [0-9]* drafted [0-9]* accepted [0-9]* tokens/round [0-9.]* draft ms/round [0-9.]* verify ms/round [0-9.]* verify misses/round" "$l" || true
    echo "$b k$k $(grep -m1 -o 'expert cache [0-9]* slots' "$l") $(grep '^spec rounds' "$l" | cut -d' ' -f1-8) | $(grep '^draftov windows' "$l")" | tee -a "$R"
    if [ "$i" -lt "$NI" ]; then
      o=$OUT/$b.k$k.off.log; run "$ENGINE_OFF" "$o" "$pf" "$k"
      grep -q '^draftov' "$o" && fail readback "$o: off engine printed a draftov line"
      [ "$(grep '^ids ' "$l")" = "$(grep '^ids ' "$o")" ] && echo "identity $b k$k EQUAL" | tee -a "$R" || { echo "identity $b k$k DIFF" | tee -a "$R"; bad=$((bad + 1)); }
    fi
  done
  i=$((i + 1))
done
for k in $KS; do
  awk -v k="k$k" -v ne=512 -v nl=48 '$2 == k {
      for (f = 1; f <= NF; f++) { if ($f == "windows") w += $(f+1); if ($f == "dsum") ds += $(f+1); if ($f == "dsum_prev") dp += $(f+1); if ($f == "v") v += $(f+1); if ($f == "vhit") vh += $(f+1); if ($f == "vhit_c1") vc += $(f+1); if ($f == "miss") m += $(f+1); if ($f == "misshit") mh += $(f+1); if ($f == "misshit_c1") mc += $(f+1); if ($f == "missB") mb += $(f+1); if ($f == "savedB") sb += $(f+1); if ($f == "savedB_c1") cb += $(f+1) } }
    END { printf "K=%s windows %d mean|D| %.2f | recall %.4f (stale-draft control %.4f, chance %.4f) | precision %.4f | verify misses %d, with e in D %d (%.4f), control %.4f | miss bytes %d, saved bytes %d (%.4f), control %.4f\n", substr(k, 2), w, ds / w, vh / v, vc / v, ds / w / ne, vh / (ds * nl), m, mh, mh / m, mc / m, mb, sb, sb / mb, cb / mb }' "$R" | tee -a "$R"
  cat "$OUT"/p*.k$k.log | awk -v k=$k '$1 == "draftov-layer" { v[$2] += $4; h[$2] += $6 } END { mx = 0; for (l in v) { r = h[l] / v[l]; if (r > mx) { mx = r; ml = l } } printf "K=%s largest per-layer recall %.4f at layer %d\n", k, mx, ml }' | tee -a "$R"
done
[ "$bad" -eq 0 ] || fail identity "$bad log runs differ from the off engine"
echo "IDENTITY PASS: log engine ids equal the off engine on $NI prompts x K '$KS'" | tee -a "$R"; echo "job PASS" | tee -a "$R"
