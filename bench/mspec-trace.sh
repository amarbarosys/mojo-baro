#!/usr/bin/env bash
# usage: bench/mspec-trace.sh ENGINE OUTDIR   (run inside one gpu-wait job)
# MSPEC step 1 (bench/mtp-protocol.md): per prompt, arm A (no spec), B2/B4 bare,
# T2/T4 under rocprofv3 --kernel-trace. Writes P1 receipts + identity to
# OUTDIR/receipts.txt, traces to OUTDIR/<p>.T<k>.csv, then the gap table.
set -euo pipefail
cd "$(dirname "$0")/.."
eng=$1; out=$2
mkdir -p "$out"; : > "$out/prompts.txt"; bad=()
pack=$(readlink -f "${BARO_PACK:-.work/engine-pack-q4}")
{ echo "pack $pack"; sha256sum "$pack/index.txt" "$pack/pack.bin"; echo "engine $(sha256sum "$eng")"; echo "tree $(git rev-parse --short HEAD)"; } > "$out/receipts.txt"
check() {  # log arm spec k
  local l=$1 fail=""
  grep -q "^BARO_SPEC: $2\$" "$l" || fail+=" spec"
  grep -q "^BARO_MEGA: True\$" "$l" || fail+=" mega"
  grep -q "^BARO_MEGA_WIN: False\$" "$l" || fail+=" mega_win"
  grep -q "^pack q4 trunk: True\$" "$l" || fail+=" q4"
  [ "$3" = 0 ] || grep -q "^spec k: $3\$" "$l" || fail+=" k"
  grep -q '^GENERATED' "$l" || fail+=" no-output"
  echo "${fail:-ok}"
}
for tf in bench/mtp-prompts/p*.tokens; do
  p=$(basename "$tf" .tokens); echo "$p" >> "$out/prompts.txt"
  BARO_PROMPT="$tf" "$eng" > "$out/$p.A.log" 2>&1
  ga=$(grep '^GENERATED' "$out/$p.A.log" || true)
  r=$(check "$out/$p.A.log" False 0); echo "$p A receipt=$r" >> "$out/receipts.txt"; [ "$r" = ok ] || bad+=("$p:A:$r")
  for k in 2 4; do
    BARO_PROMPT="$tf" BARO_SPEC=1 BARO_SPEC_K=$k "$eng" > "$out/$p.B$k.log" 2>&1
    rm -rf "$out/tr"; mkdir -p "$out/tr"
    rocprofv3 --kernel-trace --output-format csv -d "$out/tr" -o tr -- env BARO_PROMPT="$tf" BARO_SPEC=1 BARO_SPEC_K=$k "$eng" > "$out/$p.T$k.log" 2>&1
    mv "$(find "$out/tr" -name '*kernel_trace.csv' | head -1)" "$out/$p.T$k.csv"
    for a in B T; do
      l="$out/$p.$a$k.log"
      [ "$(grep '^GENERATED' "$l" || true)" = "$ga" ] && id=PASS || id=FAIL
      r=$(check "$l" True $k); echo "$p $a$k receipt=$r identity=$id $(grep '^mtp:' "$l" || true)" >> "$out/receipts.txt"
      [ "$r" = ok ] || bad+=("$p:$a$k:$r"); [ "$id" = PASS ] || bad+=("$p:$a$k:identity")
    done
  done
done
rm -rf "$out/tr"
./.work/mspec_gaps "$out" | tee "$out/summary.md"
# a failed receipt voids its arm (P1) and a changed output voids the identity: both are failures, never a quiet row
[ ${#bad[@]} -eq 0 ] || { echo "FAIL receipts ${#bad[@]}: ${bad[*]:0:12}, see $out/receipts.txt"; exit 1; }
