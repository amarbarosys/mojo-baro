#!/usr/bin/env bash
# usage: bench/dflash2-d1.sh ENGINE_FW1 ENGINE_FW0 BENCH_ROWS PACK OUTDIR   (run inside ONE gpu-wait job)
# dflash2 D1 falsifier (docs/lanes/2026-11-16-dflash2-report.md): the DFlash2 verify window is 8 rows (anchor + 7
# drafts, llama.cpp common/speculative.cpp:1100-1102), run today as an MTP k=7 window on the target pack.
#   0 kernel   BENCH_ROWS at 27B shapes: amar_fewrow_wmma D=4 m=1..8 vs q4rowb MR 8 (today's m>5 path) vs turbo m<=4
#   1 nospec   ENGINE_FW1 vs ENGINE_FW0, BARO_SPEC=0 BARO_MEGA=0 (no grid barrier on a shared GPU): m=1 reference ids + plain tok/s (FEWROW must not touch m=1)
#   2 k7       ENGINE_FW1 vs ENGINE_FW0, spec k=7: the 8-row window on fewrow vs on q4rowb[SM], ids vs each other
#   3 k2prof   ENGINE_FW1 spec k=2 (MTP baseline on this pack) vs spec k=7 BARO_PROFILE=1 (verify/draft split)
# Then ids of every spec arm against the nospec arm A, per prompt. Exit 1 on any FAIL.
# Env: GEN (256), BARO_TMAX (16384), PROMPTS (all 20 mtp-prompts), LOAD_START_MAX passed to ab-resident.
set -euo pipefail
cd "$(dirname "$0")/.."
[ $# -eq 5 ] || { echo "FAIL args: ENGINE_FW1 ENGINE_FW0 BENCH_ROWS PACK OUTDIR"; exit 1; }
e1=$(realpath "$1"); e0=$(realpath "$2"); br=$(realpath "$3"); pack=$(realpath "$4"); out=$(realpath -m "$5")
mkdir -p "$out"
export BARO_PACK=$pack GEN=${GEN:-256} BARO_TMAX=${BARO_TMAX:-16384} PROMPTS=${PROMPTS:-'bench/mtp-prompts/p[0-9][0-9]-*.tokens'}
[ "$BARO_TMAX" -ge 16384 ] || { echo "FAIL tmax: $BARO_TMAX < 16384"; exit 1; }
echo "fw1 $(sha256sum "$e1" | cut -c1-16) fw0 $(sha256sum "$e0" | cut -c1-16) rows $(sha256sum "$br" | cut -c1-16) pack $pack gen $GEN tmax $BARO_TMAX tree $(git rev-parse --short HEAD)" > "$out/arm.txt"
BARO_PACK=$pack "$br" > "$out/kernel.txt" 2>&1 || { echo "FAIL kernel: see $out/kernel.txt"; exit 1; }
REQ_SPEC=false AB_ENGINE_B=$e0 bench/ab-resident.sh "$e1" "$out/nospec" "BARO_SPEC=0 BARO_MEGA=0" "BARO_SPEC=0 BARO_MEGA=0" fw1 fw0 > "$out/nospec.txt" 2>&1 || { echo "FAIL nospec: see $out/nospec.txt"; exit 1; }
REQ_SPEC=true AB_ENGINE_B=$e0 bench/ab-resident.sh "$e1" "$out/k7" "BARO_SPEC_K=7 BARO_MEGA=0" "BARO_SPEC_K=7 BARO_MEGA=0" fw1 fw0 > "$out/k7.txt" 2>&1 || { echo "FAIL k7: see $out/k7.txt"; exit 1; }
REQ_SPEC=true bench/ab-resident.sh "$e1" "$out/k2prof" "BARO_SPEC_K=2 BARO_MEGA=0" "BARO_SPEC_K=7 BARO_MEGA=0 BARO_PROFILE=1" k2 k7prof > "$out/k2prof.txt" 2>&1 || { echo "FAIL k2prof: see $out/k2prof.txt"; exit 1; }
fails=0
for g in k7/fw1 k7/fw0 k2prof/k2 k2prof/k7prof nospec/fw0; do
  d=$(diff <(cat "$out/nospec/fw1.gen") <(cat "$out/$g.gen") | grep -c '^>' || true)
  echo "ids $g vs nospec/fw1: $d of $(wc -l < "$out/$g.gen") prompts differ" | tee -a "$out/arm.txt"
  [ "$d" = 0 ] || fails=$((fails + 1))
done
[ "$fails" = 0 ] && echo "PASS dflash2-d1 ids" || { echo "FAIL dflash2-d1 ids: $fails arms differ"; exit 1; }
