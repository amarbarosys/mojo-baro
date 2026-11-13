#!/usr/bin/env bash
# usage: face/dash_gate.sh [OUTDIR]     (default .work/dash-gate)
# F2 gate (docs/lanes/2026-11-12-face-plan.md): a real engine behind a real baro-serve front; the face renders 3 headless
# frames (no window, no display) after sending 6 short completions itself (FACE_DASH_PROBE); tesseract (img-read) reads
# the panel back from EVERY frame. Required, each within 5%:
#   OCR "tok/s X"            vs  tok_s_gen of the last done line (the front's reply, printed by the face)
#   OCR "U / T GiB"          vs  rocm-smi VRAM used / total of the biggest card, read right after the frames
# plus the state tag, the model card, the path word and "fail 0". DASH_GATE_SKEW=1.2 scales the expected values: the
# gate must then exit non-zero (the falsifier). Needs the engine at ENGINE (default .work/engine-9b) and the pack at PACK.
# Everything GPU runs in ONE gpu-wait job (engine hold). Exit non-zero on any miss: FAIL <step>: <reason>.
set -euo pipefail
cd "$(dirname "$0")/.."
out=${1:-.work/dash-gate}; mkdir -p "$out" .work
fail() { echo "FAIL $1: $2"; exit 1; }
mojo=./.venv/bin/mojo
engine=${ENGINE:-$PWD/.work/engine-9b}
pack=${PACK:-$PWD/.work/engine-pack-q4}
serve=${BARO_SERVE:-$HOME/Projects/mojo/mojo-baro/serve/target/release/baro-serve}
port=${DASH_PORT:-18433}
skew=${DASH_GATE_SKEW:-1}
cap() { systemd-run --user --scope -q -p MemoryMax=8G "$@"; }
[ -x "$engine" ] || fail setup "no engine at $engine"; [ -e "$pack" ] || fail setup "no pack at $pack"; [ -x "$serve" ] || fail setup "no baro-serve at $serve"
cap $mojo build -I . -I face face/dash_test.mojo -o .work/dash_test > "$out/dash_test.build.log" 2>&1 || fail build "dash_test, see $out/dash_test.build.log"
timeout 30 .work/dash_test > "$out/dash_test.log" 2>&1 || fail dash_test "see $out/dash_test.log"
cap $mojo build --target-accelerator amdgpu:gfx1100 -I . -I face face/app.mojo -o .work/face > "$out/face.build.log" 2>&1 || fail build "face, see $out/face.build.log"
rm -f "$out/compare.txt" "$out"/face-*.ppm "$out"/face-*.png "$out"/*.ocr.txt "$out/run.log" "$out/server.log" "$out/smi.txt"
# the queued job: front + engine up, 3 frames with 6 probes, rocm-smi, front down
cat > "$out/job.sh" <<EOF
set -euo pipefail
cd "$PWD"
BARO_TMAX=16384 $serve --engine $engine --pack $pack --port $port > $out/server.log 2>&1 &
sp=\$!
trap 'kill \$sp 2>/dev/null; wait \$sp 2>/dev/null || true' EXIT
ok=0
for i in \$(seq 180); do [ "\$(/usr/bin/curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$port/health || true)" = 200 ] && ok=1 && break; kill -0 \$sp 2>/dev/null || break; sleep 1; done
[ \$ok = 1 ] || { echo "FAIL front: no 200 on /health, see $out/server.log"; exit 1; }
FACE_DASH_PROMPT='[$(tr ' ' ',' < $pack/prompt-tokens.txt | sed 's/,*$//')]' FACE_DASH_PROBE=6 FACE_DASH_TOKENS=64 SDL_VIDEODRIVER=offscreen .work/face --shots $out --front 127.0.0.1:$port > $out/run.log 2>&1
/opt/rocm/bin/rocm-smi --showmeminfo vram > $out/smi.txt 2>&1
/usr/bin/curl -s http://127.0.0.1:$port/health > $out/health.json
EOF
cap gpu-wait run --vram "${DASH_VRAM:-16}" --timeout 900 -- bash "$out/job.sh" > "$out/gpu-wait.log" 2>&1 || fail run "see $out/gpu-wait.log, $out/server.log, $out/run.log"
done_line=$(grep '^dash done' "$out/run.log" | tail -1) || fail run "the face saw no done line, see $out/run.log"
exp_toks=$(sed -E 's/.* raw=([0-9.]+) .*/\1/' <<<"$done_line")
echo "$done_line" | grep -q 'fail=0' || fail done "done line has a fail word: $done_line"
# rocm-smi: the card with the most VRAM
read -r smi_used smi_total < <(awk '/Total Memory \(B\)/ {t[$1]=$NF} /Total Used Memory \(B\)/ {u[$1]=$NF} END {b=""; for (g in t) if (b=="" || t[g]>t[b]) b=g; print u[b], t[b]}' "$out/smi.txt")
[ -n "$smi_used" ] && [ "$smi_total" -gt 0 ] || fail rocm-smi "unreadable, see $out/smi.txt"
exp_used=$(awk -v b="$smi_used" -v k="$skew" 'BEGIN {printf "%.4f", b/1073741824*k}')
exp_total=$(awk -v b="$smi_total" 'BEGIN {printf "%.4f", b/1073741824}')
exp_toks=$(awk -v t="$exp_toks" -v k="$skew" 'BEGIN {printf "%.4f", t*k}')
within() { awk -v a="$1" -v b="$2" 'BEGIN {d=(a-b)/b; if (d<0) d=-d; exit !(d<=0.05)}'; }
need=("idle" "Model card" "path " "fail 0" "tok/s")
for f in 1 2 3; do
  [ -s "$out/face-$f.ppm" ] || fail frame "face-$f.ppm missing"
  magick "$out/face-$f.ppm" "$out/face-$f.png" || fail convert "frame $f"
  img-read --min-chars 5 "$out/face-$f.png" > "$out/face-$f.ocr.txt" 2>&1 || fail ocr "img-read frame $f"
  for s in "${need[@]}"; do grep -qiF "$s" "$out/face-$f.ocr.txt" || fail ocr "frame $f: '$s' not read back, see $out/face-$f.ocr.txt"; done
  got_toks=$(grep -oE 'tok/s [0-9]+\.[0-9]' "$out/face-$f.ocr.txt" | head -1 | awk '{print $2}')
  [ -n "$got_toks" ] || fail ocr "frame $f: no 'tok/s N.N' value, see $out/face-$f.ocr.txt"
  within "$got_toks" "$exp_toks" || fail tok/s "frame $f: panel $got_toks vs done line $exp_toks (more than 5% apart)"
  read -r got_used got_total < <(grep -oE '[0-9]+\.[0-9] ?/ ?[0-9]+\.[0-9] ?GiB' "$out/face-$f.ocr.txt" | head -1 | sed -E 's#([0-9.]+) ?/ ?([0-9.]+).*#\1 \2#')
  [ -n "${got_used:-}" ] || fail ocr "frame $f: no 'U / T GiB' value, see $out/face-$f.ocr.txt"
  within "$got_used" "$exp_used" || fail vram "frame $f: panel used $got_used GiB vs rocm-smi $exp_used GiB (more than 5% apart)"
  within "$got_total" "$exp_total" || fail vram "frame $f: panel total $got_total GiB vs rocm-smi $exp_total GiB (more than 5% apart)"
  echo "frame $f: tok/s panel $got_toks vs done line $exp_toks; VRAM panel $got_used / $got_total vs rocm-smi $exp_used / $exp_total GiB" | tee -a "$out/compare.txt"
done
echo "PASS dash gate: dash_test + 3 headless frames on a live front and engine, tok/s and VRAM within 5% of the done line and rocm-smi ($out)"
