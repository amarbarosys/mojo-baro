#!/usr/bin/env bash
# train-gate: M0+M1 gate for miniagi/train/. Builds, regenerates oracle cases, runs the op/model/AdamW
# parity runner, trains 200 steps twice in Mojo and once in torch, then checks
#   - every parity case PASS
#   - Mojo rerun bit-identical (loss bits per step)
#   - steps 0..EARLY-1: loss within TOL_EARLY relative of torch (a wrong grad/optimizer shows here)
#   - fast arm (fp16 WMMA forward GEMMs): early within TOL_EARLY_FAST, whole run within the same floor
#   - whole run: worst relative gap <= FLOOR_X x torch's own gap after a 1e-7 init perturbation
#     (fp32 training is chaotic: torch-vs-perturbed-torch reaches 3.6% by step 192, 2026-10-09,
#     so a flat per-step tolerance over 200 steps cannot hold even for torch against itself)
# Prints PASS n/n or FAIL k/n <names>, exits non-zero on any miss.
# Oracle venv: see tools/train_oracle.py docstring.
set -euo pipefail
cd "$(dirname "$0")/.."
W=.work/train
TOL_EARLY=${TOL_EARLY:-1e-5}
TOL_EARLY_FAST=${TOL_EARLY_FAST:-1e-3}
EARLY=${EARLY:-20}
FLOOR_X=${FLOOR_X:-2}
STEPS=${STEPS:-200}
PY=.work/oracle-venv/bin/python
mkdir -p "$W"
fails=(); n=0
check() { n=$((n + 1)); if "${@:2}"; then echo "PASS $1"; else echo "FAIL $1"; fails+=("$1"); fi; }

[ -x "$PY" ] || { echo "FAIL oracle-venv: $PY missing (tools/train_oracle.py docstring)"; exit 1; }
./.venv/bin/mojo build miniagi/train/test_parity.mojo -I miniagi/train -I kernels -o .work/train_parity > "$W/build.log" 2>&1 \
  || { echo "FAIL build parity: see $W/build.log"; exit 1; }
./.venv/bin/mojo build miniagi/train/main.mojo -I miniagi/train -I kernels -o .work/train_main >> "$W/build.log" 2>&1 \
  || { echo "FAIL build main: see $W/build.log"; exit 1; }
gpu-wait run --timeout 300 -- "$PY" -P tools/train_oracle.py cases "$W/cases" > "$W/oracle.log" 2>&1 \
  || { echo "FAIL oracle cases: see $W/oracle.log"; exit 1; }
[ -s "$W/corpus/arith.u8" ] || { echo "FAIL corpus: $W/corpus/arith.u8 missing (mini-AGI corpora.arithmetic, bytes < 256)"; exit 1; }

check parity bash -c "gpu-wait run --vram 2 --timeout 900 -- ./.work/train_parity > $W/parity.log 2>&1 && ! grep -q '^FAIL' $W/parity.log"
I=$W/cases/model/init.f32; D=$W/corpus/arith.u8
for r in a:0 b:0 fast:1; do
  gpu-wait run --vram 4 --timeout 1800 -- ./.work/train_main "$I" "$D" "$STEPS" 1e-3 "$W/loss-mojo-${r%:*}.txt" 512 "${r#*:}" > "$W/train-${r%:*}.log" 2>&1 \
    || { echo "FAIL train ${r%:*}: see $W/train-${r%:*}.log"; exit 1; }
done
for mode in train noise; do
  gpu-wait run --timeout 3600 -- "$PY" -P tools/train_oracle.py "$mode" "$I" "$D" "$STEPS" 1e-3 "$W/loss-torch-$mode.txt" > "$W/torch-$mode.log" 2>&1 \
    || { echo "FAIL torch $mode: see $W/torch-$mode.log"; exit 1; }
done

check rerun-bit-identical cmp -s "$W/loss-mojo-a.txt" "$W/loss-mojo-b.txt"
worst() { paste "$1" "$2" | awk -v from="${3:-0}" -v to="${4:-999999}" '$1 >= from && $1 < to { r = ($4 - $2) / $2; if (r < 0) r = -r; if (r > m) m = r } END { printf "%.6g", m }'; }
early=$(worst "$W/loss-torch-train.txt" "$W/loss-mojo-a.txt" 0 "$EARLY")
run=$(worst "$W/loss-torch-train.txt" "$W/loss-mojo-a.txt")
floor=$(worst "$W/loss-torch-train.txt" "$W/loss-torch-noise.txt")
echo "  early worst $early (tol $TOL_EARLY) · run worst $run · torch noise floor $floor (x$FLOOR_X)"
check loss-early awk -v a="$early" -v t="$TOL_EARLY" 'BEGIN { exit !(a <= t) }'
check loss-vs-floor awk -v a="$run" -v f="$floor" -v x="$FLOOR_X" 'BEGIN { exit !(f > 0 && a <= x * f) }'
fearly=$(worst "$W/loss-torch-train.txt" "$W/loss-mojo-fast.txt" 0 "$EARLY")
frun=$(worst "$W/loss-torch-train.txt" "$W/loss-mojo-fast.txt")
echo "  fast (fp16 WMMA forward): early worst $fearly (tol $TOL_EARLY_FAST) · run worst $frun"
check fast-loss-early awk -v a="$fearly" -v t="$TOL_EARLY_FAST" 'BEGIN { exit !(a <= t) }'
check fast-loss-vs-floor awk -v a="$frun" -v f="$floor" -v x="$FLOOR_X" 'BEGIN { exit !(f > 0 && a <= x * f) }'
grep -h '^fast' "$W"/train-a.log "$W"/train-fast.log | sed 's/^/  /'

if [ ${#fails[@]} -gt 0 ]; then echo "FAIL ${#fails[@]}/$n ${fails[*]}"; exit 1; fi
echo "PASS $n/$n"
