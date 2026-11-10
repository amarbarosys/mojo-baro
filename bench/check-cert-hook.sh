#!/usr/bin/env bash
# bench/cert-hook-protocol.md gates: the per-token certificate hook in serve/window.mojo (BARO_CERT=SEED, launch path).
#   identity: BARO_CERT on vs off give the same GENERATED ids on 5 prompts, and every cert line says pass 7/7.
#   mutant:   an engine built from a kernel copy that adds 0.05 to row 5 of every q4 GEMV output must print cert lines
#             that FAIL (pass < 7/7) on every token.
set -euo pipefail
cd "$(dirname "$0")/.."
mode=${1:?usage: check-cert-hook.sh identity|mutant}
OUT=.work/cert-hook; mkdir -p "$OUT"
MOJO=${MOJO:-./.venv/bin/mojo}
export MODULAR_CRASH_REPORTING_ENABLED=false
fail() { echo "FAIL $1: $2"; exit 1; }
PACK=$(readlink -f "${PACK:-.work/engine-pack-q4}")
[ -f "$PACK/pack.bin" ] || fail pack "$PACK/pack.bin missing"
BUILD="serve/engine.mojo -I . --target-accelerator ${BARO_TARGET_ACCELERATOR:-gfx1100}"
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
  timeout 1500 "$MOJO" build $BUILD -I kernels -o "$OUT/engine" >"$OUT/build.log" 2>&1 || fail build "$OUT/build.log"
  if [ "$mode" = mutant ]; then
    rm -rf "$OUT/mut"; mkdir -p "$OUT/mut"; cp -r kernels "$OUT/mut/kernels"
    sed '/def amar_matmul_skinny_q4rowb/,/def amar_matmul_skinny_q8dot/ s/rebind\[Cp.ElementType\](t\[r\])/rebind[Cp.ElementType](t[r] + (Float32(0.05) if row == 5 else Float32(0)))/' kernels/matmul_skinny.mojo >"$OUT/mut/kernels/matmul_skinny.mojo"
    cmp -s kernels/matmul_skinny.mojo "$OUT/mut/kernels/matmul_skinny.mojo" && fail mutant "sed changed nothing"
    timeout 1500 "$MOJO" build $BUILD -I "$OUT/mut/kernels" -o "$OUT/engine-mut" >"$OUT/build-mut.log" 2>&1 || fail build "$OUT/build-mut.log"
  fi
  exec gpu-wait run --vram 22 --timeout 900 -- bash "$0" "$mode"
fi
export MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=2
run() { # ENGINE CERT PROMPT LOG
  env BARO_CERT="$2" BARO_PACK="$PACK" BARO_MEGA=0 BARO_SPEC=0 BARO_TMAX=16384 BARO_PROMPT="$3" "$1" >"$4" 2>&1 </dev/null || fail run "$4"
}
rcpt="$OUT/gate-$mode.txt"
{ echo "checkout $(git rev-parse --short HEAD) pack $PACK"; echo "link $(cat /sys/bus/pci/devices/0000:00:01.1/current_link_speed)"; } >"$rcpt"
if [ "$mode" = identity ]; then
  echo "engine $OUT/engine sha256 $(sha256sum "$OUT/engine" | cut -c1-16) env BARO_MEGA=0 BARO_SPEC=0 BARO_TMAX=16384" >>"$rcpt"
  n=0
  for p in bench/mtp-prompts/p0[1-5]-*.tokens; do
    b=$(basename "$p" .tokens)
    run "$OUT/engine" "" "$PWD/$p" "$OUT/id-off-$b.log"
    run "$OUT/engine" 1 "$PWD/$p" "$OUT/id-on-$b.log"
    grep -q '^cert on seed 1 ' "$OUT/id-on-$b.log" || fail readback "$b: no 'cert on seed 1' line, the flag did not take"
    [ "$(grep -c '^cert tok' "$OUT/id-off-$b.log")" -eq 0 ] || fail readback "$b: cert lines with the flag off"
    g_off=$(grep -m1 '^GENERATED' "$OUT/id-off-$b.log" || true); g_on=$(grep -m1 '^GENERATED' "$OUT/id-on-$b.log" || true)
    [ -n "$g_off" ] || fail run "$b: no GENERATED line"
    [ "$g_off" = "$g_on" ] || fail identity "$b: GENERATED differs, $OUT/id-*-$b.log"
    all=$(grep -c '^cert tok' "$OUT/id-on-$b.log"); ok=$(grep -c '^cert tok .* pass 7/7 ' "$OUT/id-on-$b.log")
    [ "$all" -gt 0 ] && [ "$all" -eq "$ok" ] || fail cert "$b: $ok of $all cert lines pass 7/7"
    mr=$(grep '^cert tok' "$OUT/id-on-$b.log" | awk '{if ($NF > m) m = $NF} END{print m}')
    echo "$b: $(wc -w <<<"${g_off#GENERATED:}") ids identical, $ok/$all cert lines pass 7/7, max rho $mr" | tee -a "$rcpt"
    n=$((n + 1))
  done
  echo "PASS cert-hook identity: $n prompts, token ids identical with the certificate on and off, every cert line pass 7/7" | tee -a "$rcpt"
else
  echo "engine $OUT/engine-mut sha256 $(sha256sum "$OUT/engine-mut" | cut -c1-16) mutant +0.05 row 5 of every q4 GEMV output" >>"$rcpt"
  p=bench/mtp-prompts/p01-water.tokens
  run "$OUT/engine-mut" 1 "$PWD/$p" "$OUT/mut.log"
  all=$(grep -c '^cert tok' "$OUT/mut.log"); bad=$(grep -c '^cert tok .* pass 0/7 ' "$OUT/mut.log")
  [ "$all" -gt 0 ] || fail cert "no cert lines from the mutant engine"
  [ "$bad" -eq "$all" ] || fail mutant "only $bad of $all cert lines failed 7 of 7 matrices"
  mr=$(grep '^cert tok' "$OUT/mut.log" | awk '{if (m == "" || $NF < m) m = $NF} END{print m}')
  echo "PASS cert-hook mutant: $bad of $all tokens print pass 0/7 (min maxrho $mr, kappa 0.125)" | tee -a "$rcpt"
fi
