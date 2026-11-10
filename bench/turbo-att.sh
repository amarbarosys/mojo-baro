#!/usr/bin/env bash
# usage: bench/turbo-att.sh ENGINE PACK OUTDIR MEGA [ITER] [PROMPT] [SPEC]
# wavewait SQTT capture of one WARM decode step of the engine on PACK (MEGA=1: the ITER-th amar_mega_token
# dispatch, default 12, prompt-file run; MEGA=0: every kernel of a 20 ms window ATT_WIN s (default 6) into a resident
# (BARO_SERVE=1) run that generates NGEN tokens (default 400) with spec per request = SPEC, so the window lands in decode;
# ATT_KERN=regex instead traces the ITER-th dispatch of every kernel whose mangled name matches, warm, in that same run).
# Wraps ~/Projects/mojo/wavewait/tools/att-capture.sh (CU 3, SE mask 0x1, 256 MB buffer, its own gpu-wait job).
# Read-backs: BARO_MEGA / TMAX / pack kind lines in OUTDIR/engine.log, trace size vs buffer from att-capture.
set -euo pipefail
cd "$(dirname "$0")/.."
eng=$1 pack=$2 out=$3 mg=$4 it=${5:-12} p=${6:-p01-water} spec=${7:-0}
want=False; [ "$mg" = 1 ] && want=True
kq=False; grep -q ' q4_k ' "$pack/index.txt" && kq=True
mkdir -p "$out"; [ -z "$(ls -A "$out/run" 2>/dev/null)" ] || { echo "FAIL att: $out/run not empty"; exit 1; }
extra="--kernel-iteration-range [$it-$it]"; kern='amar_mega_token'
if [ "$mg" = 0 ]; then kern='.*'; extra="-P ${ATT_WIN:-6}:0.02:1"; fi
if [ -n "${ATT_KERN:-}" ]; then kern=$ATT_KERN; extra="--kernel-iteration-range [$it-$it]"; fi
ngen=${NGEN:-400}; rs=false; [ "$spec" = 1 ] && rs=true
printf '{"id":1,"prompt":[%s],"n":%d,"spec":%s}\n' "$(tr -s ' \n' ',,' < "bench/mtp-prompts/$p.tokens" | sed 's/,$//')" "$ngen" "$rs" > "$out/req.jsonl"
echo "eng=$(sha256sum "$eng" | cut -c1-16) pack=$pack kq=$kq mega=$mg spec=$spec iter=$it prompt=$p kernel=$kern extra='$extra' commit=$(git rev-parse --short HEAD)" | tee "$out/arm.txt"
if [ "$mg" = 1 ]; then
  cmd=(env BARO_MEGA=1 BARO_PACK="$pack" BARO_SPEC="$spec" BARO_TMAX=16384 BARO_PROFILE=5 BARO_PROMPT="bench/mtp-prompts/$p.tokens" "$PWD/$eng")
else
  cmd=(bash -c "env BARO_MEGA=0 BARO_PACK='$pack' BARO_SPEC=$spec BARO_TMAX=16384 BARO_SERVE=1 '$PWD/$eng' < '$PWD/$out/req.jsonl'")
fi
ATT_SLIM=1 ATT_KERNEL="$kern" ATT_EXTRA="$extra" ATT_TIMEOUT=600 \
  "$HOME/Projects/mojo/wavewait/tools/att-capture.sh" "$PWD/$out/run" -- "${cmd[@]}" \
  | tee "$out/att.out" || { echo "FAIL att-capture: see $out/att.out"; exit 1; }
log=$(ls -t "$out/run.log" 2>/dev/null | head -1)
grep -q "BARO_MEGA: $want" "$log" || { echo "FAIL readback: BARO_MEGA want $want, log $log"; exit 1; }
grep -q 'TMAX: 16384 ' "$log" || { echo "FAIL readback: TMAX, log $log"; exit 1; }
grep -q "pack kq trunk: $kq" "$log" || { echo "FAIL readback: pack kq trunk != $kq"; exit 1; }
want_s=False; [ "$spec" = 1 ] && want_s=True
grep -q "BARO_SPEC: $want_s" "$log" || { echo "FAIL readback: BARO_SPEC want $want_s"; exit 1; }
grep -E 'tok/s_gen|tok_s|mega profile|mega fail word|^GENERATED|"done"' "$log" | cut -c1-200 | tee -a "$out/arm.txt"
echo "PASS turbo-att $out"
