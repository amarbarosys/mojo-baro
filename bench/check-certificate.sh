#!/usr/bin/env bash
# bench/cpu-parity-protocol.md item 1: float Freivalds certificate on the 7 GEMVs of layer 3. SEED 2 honest run must
# pass at the frozen KAPPA (bench/cert-kappa.txt); kernel mutants (one GEMV output +DELTA at row 5) must be caught.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=.work/cert; mkdir -p "$OUT"
MOJO=${MOJO:-./.venv/bin/mojo}
PACK=${PACK:-$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.work/engine-pack-q4}
PACKABS=$(readlink -f "$PACK")
export MODULAR_CRASH_REPORTING_ENABLED=false
fail() { echo "FAIL $1: $2"; exit 1; }
[ -f "$PACKABS/pack.bin" ] || fail pack "$PACKABS/pack.bin missing"
KAPPA=$(cat bench/cert-kappa.txt)
B="-D BARO_SHARED_EXP=1 -I cpu -I serve"
timeout 900 "$MOJO" build $B -I kernels kernels/test_certificate.mojo -o "$OUT/cert" >"$OUT/build.log" 2>&1 || fail build "$OUT/build.log"
for d in 0.05 0.005; do
  rm -rf "$OUT/mut$d"; mkdir -p "$OUT/mut$d"; cp -r kernels "$OUT/mut$d/kernels"
  sed "/def amar_matmul_skinny_q4rowb/,/def amar_matmul_skinny_q8dot/ s/rebind\[Cp.ElementType\](t\[r\])/rebind[Cp.ElementType](t[r] + (Float32($d) if row == 5 else Float32(0)))/" kernels/matmul_skinny.mojo >"$OUT/mut$d/kernels/matmul_skinny.mojo"
  cmp -s kernels/matmul_skinny.mojo "$OUT/mut$d/kernels/matmul_skinny.mojo" && fail mutant "sed changed nothing"
  timeout 900 "$MOJO" build $B -I "$OUT/mut$d/kernels" "$OUT/mut$d/kernels/test_certificate.mojo" -o "$OUT/cert-mut$d" >"$OUT/build-mut$d.log" 2>&1 || fail build "$OUT/build-mut$d.log"
done
gpu-wait run --timeout 900 -- sh -c "$OUT/cert $PACKABS 48 $KAPPA 2 >$OUT/honest.txt 2>&1 || true; for d in 0.05 0.005; do $OUT/cert-mut\$d $PACKABS 8 $KAPPA 2 >$OUT/mut\$d.txt 2>&1 || true; done" || true
for f in honest mut0.05 mut0.005; do
  grep -q '^arm cert' "$OUT/$f.txt" || fail arm "$f: no arm line (job died before the GPU?)"
  grep -q '^max-ratio down' "$OUT/$f.txt" || fail incomplete "$f: run did not finish"
done
grep -q '^CERT PASS 336 of 336' "$OUT/honest.txt" || fail honest "$(grep -c '^cert' "$OUT/honest.txt") checks, $(grep Unhandled "$OUT/honest.txt")"
caught() { awk -v k="$KAPPA" '/^cert /{n[$2]++; if ($NF > k) c[$2]++} END{t=0; for (m in n) if (c[m] == n[m]) t++; print t}' "$1"; }
c1=$(caught "$OUT/mut0.05.txt"); c2=$(caught "$OUT/mut0.005.txt")
[ "$c1" -eq 7 ] || fail mutant "delta 0.05 caught on $c1 of 7 matrices"
echo "PASS certificate: honest 336/336 <= kappa $KAPPA, mutant +0.05 caught on $c1/7 matrices at every step, +0.005 on $c2/7"
