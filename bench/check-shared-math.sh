#!/usr/bin/env bash
# kernels/bexp.mojo fexp32 and flog32: CPU and GPU give the same bits on 40k Float32 inputs, within 1 ulp of a
# correctly rounded mpmath oracle per range; a planted mutant (flog32 without the k ln2_lo term) must fail.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=.work/f32math; mkdir -p "$OUT/mut"
MOJO=${MOJO:-./.venv/bin/mojo}
export MODULAR_CRASH_REPORTING_ENABLED=false
fail() { echo "FAIL $1: $2"; exit 1; }
sed 's/dk \* F(9.0580006145e-06)/F(0)/' kernels/bexp.mojo >"$OUT/mut/bexp.mojo"
cmp -s kernels/bexp.mojo "$OUT/mut/bexp.mojo" && fail mutant "sed changed nothing"
timeout 200 "$MOJO" build -I kernels tools/f32math/cli.mojo -o "$OUT/cli" >"$OUT/build.log" 2>&1 || fail build "$OUT/build.log"
timeout 200 "$MOJO" build -I "$OUT/mut" tools/f32math/cli.mojo -o "$OUT/cli-mut" >"$OUT/build-mut.log" 2>&1 || fail build "$OUT/build-mut.log"
python3 -I tools/f32math/oracle.py --gen 40000 7 >"$OUT/xs.txt"
python3 -I tools/f32math/oracle.py <"$OUT/xs.txt" >"$OUT/oracle.txt" || fail oracle tools/f32math/oracle.py
"$OUT/cli" <"$OUT/xs.txt" >"$OUT/cpu.txt" || fail cpu cli
"$OUT/cli-mut" <"$OUT/xs.txt" >"$OUT/mut.txt" || fail mutant cli-mut
gpu-wait run --timeout 120 -- sh -c "$OUT/cli gpu <$OUT/xs.txt >$OUT/gpu.txt" || fail gpu "cli gpu"
d=$(paste -d' ' "$OUT/cpu.txt" "$OUT/gpu.txt" | awk '$1!=$3||$2!=$4' | wc -l)
[ "$d" -eq 0 ] || fail bitwise "cpu and gpu differ on $d of $(wc -l <"$OUT/xs.txt") inputs"
for m in cpu gpu mut; do python3 -I tools/f32math/split.py "$OUT/xs.txt" "$OUT/$m.txt" "$OUT/oracle.txt" >"$OUT/$m.split"; done
tee <"$OUT/cpu.split" "$OUT/receipt.txt"
worst() { awk -v p="$2" '$0 ~ p {print $NF}' "$1" | sort -g | tail -1; }
for p in "exp full" "log"; do
  for m in cpu gpu; do
    w=$(worst "$OUT/$m.split" "$p"); [ "${w%.*}" -le 1 ] || fail ulp "$m $p max $w ulp > 1"
  done
  w=$(worst "$OUT/mut.split" "$p")
  [ "$p" = exp\ full ] || [ "${w%.*}" -gt 1 ] || fail mutant "mutant passed ($p max $w ulp)"
done
echo "PASS shared-math: cpu == gpu bitwise on all inputs, fexp32/flog32 within 1 ulp, mutant caught"
