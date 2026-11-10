#!/usr/bin/env bash
# bench/cpu-parity-protocol.md step 1: one attention layer (14 stages x 8 steps), cpu/ops.mojo vs the launch-path GPU
# kernels under -D BARO_SHARED_EXP=1, bitwise on the GPU's own input per stage. A planted mutant (rope sincos
# argument off by 1e-7 relative) must fail rope and nothing else. The default-flag GPU output (digest) must equal
# main's attn.mojo and elementwise.mojo: the shared helpers are `comptime if SHARED_EXP` only.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=.work/cpu-layer; mkdir -p "$OUT/mut" "$OUT/main"
MOJO=${MOJO:-./.venv/bin/mojo}
PACK=${PACK:-$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.work/engine-pack-q4}
PACKABS=$(readlink -f "$PACK")
export MODULAR_CRASH_REPORTING_ENABLED=false
fail() { echo "FAIL $1: $2"; exit 1; }
[ -f "$PACKABS/pack.bin" ] || fail pack "$PACKABS/pack.bin missing"
sed 's/var sc = fsincos32(th)/var sc = fsincos32(th * 1.0000001)/' cpu/ops.mojo >"$OUT/mut/ops.mojo"
cmp -s cpu/ops.mojo "$OUT/mut/ops.mojo" && fail mutant "sed changed nothing"
rm -rf "$OUT/main"; mkdir -p "$OUT/main"; cp -r kernels "$OUT/main/kernels"; for f in attn elementwise; do git show main:kernels/$f.mojo >"$OUT/main/kernels/$f.mojo"; done
B="-I kernels -I cpu -I serve"
timeout 900 "$MOJO" build -D BARO_SHARED_EXP=1 $B kernels/test_cpu_layer.mojo -o "$OUT/layer" >"$OUT/build.log" 2>&1 || fail build "$OUT/build.log"
timeout 900 "$MOJO" build -D BARO_SHARED_EXP=1 -I kernels -I "$OUT/mut" -I serve kernels/test_cpu_layer.mojo -o "$OUT/layer-mut" >"$OUT/build-mut.log" 2>&1 || fail build "$OUT/build-mut.log"
timeout 900 "$MOJO" build $B kernels/test_cpu_layer.mojo -o "$OUT/layer-def" >"$OUT/build-def.log" 2>&1 || fail build "$OUT/build-def.log"
timeout 900 "$MOJO" build -I "$OUT/main/kernels" -I cpu -I serve "$OUT/main/kernels/test_cpu_layer.mojo" -o "$OUT/layer-main" >"$OUT/build-main.log" 2>&1 || fail build "$OUT/build-main.log"
gpu-wait run --timeout 900 -- sh -c "for v in layer layer-mut layer-def layer-main; do $OUT/\$v $PACKABS >$OUT/\$v.txt 2>&1 || true; done" || true
for v in layer layer-mut layer-def layer-main; do
  grep -q '^arm gpu=launch-path kernels' "$OUT/$v.txt" || fail arm "$v: no arm line (job died before the GPU?)"
  [ "$(grep -c '^stage ' "$OUT/$v.txt")" -eq 14 ] || fail incomplete "$v: expected 14 stage lines"
done
grep -q '^PASS test_cpu_layer' "$OUT/layer.txt" || fail bitwise "$(grep '^stage .* mismatches [1-9]' "$OUT/layer.txt" | tr '\n' ';')"
grep -q '^PASS' "$OUT/layer-mut.txt" && fail mutant "mutant passed"
other=$(grep '^stage .* mismatches [1-9]' "$OUT/layer-mut.txt" | grep -v -c '^stage rope_' || true)
[ "$other" -eq 0 ] || fail mutant "mutant failed stages other than rope"
grep -q '^stage rope_q mismatches [1-9]' "$OUT/layer-mut.txt" || fail mutant "mutant did not fail rope_q"
d1=$(grep '^gpu-digest' "$OUT/layer-def.txt" || true); d2=$(grep '^gpu-digest' "$OUT/layer-main.txt" || true)
[ -n "$d1" ] && [ "$d1" = "$d2" ] || fail default "default-flag GPU digest differs from main: '$d1' vs '$d2'"
echo "PASS cpu-layer: 14 stages x 8 steps cpu == gpu bitwise (BARO_SHARED_EXP=1), mutant caught on rope only, default-flag GPU digest == main ($d1)"
