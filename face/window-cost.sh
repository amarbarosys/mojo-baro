#!/usr/bin/env bash
# usage: face/window-cost.sh OUTDIR FPS [PACK] [ENGINE]     (docs/lanes/2026-11-12-face-plan.md F0 step 2)
# Window cost: one RESIDENT 9B engine, the 20 mtp-prompts (64 tokens, f32 KV, TMAX 16384), arm A = no window, arm B =
# the face in an offscreen SDL3 window (no desktop window, SDL_VIDEODRIVER=offscreen) rendering FPS frames a second on
# the same card, via bench/ab-resident.sh SEQ=1 (blocks of 5, order alternating, seq-decide bootstraps the paired
# median ratio). Read back from the running system: the face's own measured fps (OUT/face-fps.log; an arm whose window
# ran below 50% of FPS is refused: below that it was not running; a rate between is a finding, printed), vram_max / gtt_max per arm (OUT/arm.txt), engine sha256, BARO_* lines.
# One gpu-wait job, 14 GB. Defaults: PACK .work/engine-pack-q4 (Qwythos 9B Q4_0), ENGINE .work/engine-9b.
set -euo pipefail
cd "$(dirname "$0")/.."
out=$(realpath -m "${1:?usage: face/window-cost.sh OUTDIR FPS [PACK] [ENGINE]}"); fps=${2:?FPS}
pack=$(realpath "${3:-.work/engine-pack-q4}"); eng=$(realpath "${4:-.work/engine-9b}")
[ -x "$eng" ] || { echo "FAIL setup: engine $eng not executable"; exit 1; }
[ -x .work/face ] || { echo "FAIL setup: .work/face not built (face/gate.sh builds it)"; exit 1; }
mkdir -p "$out"; : > "$out/face-fps.log"
here=$PWD
gpu-wait run --timeout 1500 --vram 14 -- env LOAD_START_MAX="${LOAD_START_MAX:-4}" BARO_PACK="$pack" BARO_TMAX=16384 SEQ=1 BLOCK=5 \
  AB_ENGINE_B="$here/face/engine-with-window.sh" FACE_ENGINE="$eng" FACE_FPS="$fps" FACE_OUT="$out" FACE_BIN="$here/.work/face" \
  bash -c "cd $here && bench/ab-resident.sh $eng $out 'FACE=none' 'FACE=win' nowin win" > "$out/ab.log" 2>&1 || { tail -5 "$out/ab.log"; echo "FAIL ab-resident: see $out/ab.log"; exit 1; }
tail -4 "$out/ab.log" | head -1
# the window ran at the asked rate in every block (the last fps line before each stop)
awk -v want="$fps" '/^window start/ { if (last != "") print last; last = "" } /^fps/ { last = $2 } END { if (last != "") print last }' "$out/face-fps.log" > "$out/face-fps.final"
awk -v want="$fps" '{ if ($1 < 0.5 * want) { bad = 1; print "FAIL window: a block ran at " $1 " fps, below 50% of " want } } END { exit bad }' "$out/face-fps.final" \
  || exit 1
echo "window fps per block: $(tr '\n' ' ' < "$out/face-fps.final")(asked $fps)"
# paired summary from the per-prompt tok/s files (seq-decide is not used: its binary does not load under gpu-wait's env)
paste "$out/nowin.toks" "$out/win.toks" | awk '
  function med(a, n,   i, j, t) { for (i = 1; i <= n; i++) for (j = i + 1; j <= n; j++) if (a[j] < a[i]) { t = a[i]; a[i] = a[j]; a[j] = t }
    return n % 2 ? a[(n + 1) / 2] : (a[n / 2] + a[n / 2 + 1]) / 2 }
  { A[NR] = $1; B[NR] = $2; R[NR] = $2 / $1; if (R[NR] < 0.95) slow++ }
  END { ma = med(A, NR); mb = med(B, NR); mr = med(R, NR)
    printf "n %d  nowin median %.2f  win median %.2f  median of paired ratios %.3f  prompts below 0.95: %d\n", NR, ma, mb, mr, slow }' | tee "$out/summary.txt"
grep -E "vram_max" "$out/arm.txt" | sort | uniq -c | sed 's/^ *//' | tee -a "$out/summary.txt"
