#!/usr/bin/env bash
# bench/cpu-parity-protocol.md steps 1 and 1b: one attention layer, one SSM layer, the greedy pick and 5 pack tensors
# (27 stages), cpu/ops.mojo + cpu/pack.mojo vs the launch-path GPU kernels and loader under -D BARO_SHARED_EXP=1,
# bitwise on the GPU's own input per stage. A planted mutant (rope sincos argument and delta d, each off by 1e-7
# relative) must fail rope and delta and nothing else. The default-flag GPU output (digest) must equal main's attn.mojo,
# elementwise.mojo and ssm.mojo: the shared helpers are `comptime if SHARED_EXP` only.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=.work/cpu-layer; mkdir -p "$OUT/mut" "$OUT/main"
MOJO=${MOJO:-./.venv/bin/mojo}
PACK=${PACK:-$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.work/engine-pack-q4}
PACKABS=$(readlink -f "$PACK")
export MODULAR_CRASH_REPORTING_ENABLED=false
fail() { echo "FAIL $1: $2"; exit 1; }
[ -f "$PACKABS/pack.bin" ] || fail pack "$PACKABS/pack.bin missing"
sed 's/var sc = fsincos32(th)/var sc = fsincos32(th * 1.0000001)/; s/\* gb\[unsafe_offset=nv + h\]$/* gb[unsafe_offset=nv + h] * 1.0000001/' cpu/ops.mojo >"$OUT/mut/ops.mojo"
[ "$(diff cpu/ops.mojo "$OUT/mut/ops.mojo" | grep -c '^>')" -eq 2 ] || fail mutant "sed did not change exactly 2 lines"
cmp -s cpu/ops.mojo "$OUT/mut/ops.mojo" && fail mutant "sed changed nothing"
rm -rf "$OUT/main"; mkdir -p "$OUT/main"; cp -r kernels "$OUT/main/kernels"; for f in attn elementwise ssm; do git show main:kernels/$f.mojo >"$OUT/main/kernels/$f.mojo"; done
B="-I . -I kernels -I cpu -I serve"
CAP="systemd-run --user --scope -q -p CPUQuota=150% -p MemoryMax=10G -p MemorySwapMax=0"
cool() { while [ "$(sensors | awk '/Tctl/{gsub(/[^0-9.]/,"",$2); print int($2)}')" -ge 84 ]; do echo "Tctl >= 84 C, waiting"; sleep 30; done; }
cool; $CAP timeout 900 "$MOJO" build -D BARO_SHARED_EXP=1 $B kernels/test_cpu_layer.mojo -o "$OUT/layer" >"$OUT/build.log" 2>&1 || fail build "$OUT/build.log"
cool; $CAP timeout 900 "$MOJO" build -D BARO_SHARED_EXP=1 -I . -I kernels -I "$OUT/mut" -I cpu -I serve kernels/test_cpu_layer.mojo -o "$OUT/layer-mut" >"$OUT/build-mut.log" 2>&1 || fail build "$OUT/build-mut.log"
cool; $CAP timeout 900 "$MOJO" build $B kernels/test_cpu_layer.mojo -o "$OUT/layer-def" >"$OUT/build-def.log" 2>&1 || fail build "$OUT/build-def.log"
cool; $CAP timeout 900 "$MOJO" build -I . -I "$OUT/main/kernels" -I cpu -I serve "$OUT/main/kernels/test_cpu_layer.mojo" -o "$OUT/layer-main" >"$OUT/build-main.log" 2>&1 || fail build "$OUT/build-main.log"
gpu-wait run --timeout 900 --shared --vram 9 -- sh -c "for v in layer layer-mut layer-def layer-main; do $OUT/\$v $PACKABS >$OUT/\$v.txt 2>&1 || true; done" || true
for v in layer layer-mut layer-def layer-main; do
  grep -q '^arm gpu=launch-path kernels' "$OUT/$v.txt" || fail arm "$v: no arm line (job died before the GPU?)"
  [ "$(grep -c '^stage ' "$OUT/$v.txt")" -eq 28 ] || fail incomplete "$v: expected 28 stage lines"
done
grep -q '^PASS test_cpu_layer' "$OUT/layer.txt" || fail bitwise "$(grep '^stage .* mismatches [1-9]' "$OUT/layer.txt" | tr '\n' ';')"
grep -q '^PASS' "$OUT/layer-mut.txt" && fail mutant "mutant passed"
other=$(grep '^stage .* mismatches [1-9]' "$OUT/layer-mut.txt" | grep -E -v -c '^stage (rope_|delta_)' || true)
[ "$other" -eq 0 ] || fail mutant "mutant failed stages other than rope and delta"
for st in rope_q delta_o; do grep -q "^stage $st mismatches [1-9]" "$OUT/layer-mut.txt" || fail mutant "mutant did not fail $st"; done
d1=$(grep '^gpu-digest' "$OUT/layer-def.txt" || true); d2=$(grep '^gpu-digest' "$OUT/layer-main.txt" || true)
[ -n "$d1" ] && [ "$d1" = "$d2" ] || fail default "default-flag GPU digest differs from main: '$d1' vs '$d2'"
echo "PASS cpu-layer: 28 stages (attention + SSM layer x 8 steps, f32 rmsnorm, argmax, pack) cpu == gpu bitwise (BARO_SHARED_EXP=1), mutant caught on rope and delta only, default-flag GPU digest == main ($d1)"
