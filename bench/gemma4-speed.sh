#!/usr/bin/env bash
# M4 decode speed of lane-gemma4-port: ours (serve/gemma4.mojo one-shot, BARO_GEN tokens after a 14-id prompt)
# against llama.cpp llama-bench tg on the same F16 GGUF, full offload, interleaved A B A B A B in one gpu-wait job.
# Both decode at near-empty context. Numbers are tok/s; the verdict line prints the median ratio.
# Usage: bench/gemma4-speed.sh GGUF PACK OUT [N=256]   (queues itself under gpu-wait)
set -euo pipefail
cd "$(dirname "$0")/.."
GGUF=$1; PACK=$2; OUT=$3; N=${4:-256}
if [ -z "${G4_INJOB:-}" ]; then
  [ -x .work/gemma4/engine ] || { echo "FAIL setup: no engine"; exit 1; }
  mkdir -p "$OUT"
  .work/gemma4/llama-tok encode bench/mtp-prompts/p01-water.txt "$GGUF" > "$OUT/p01.ids"
  exec gpu-wait run --timeout 900 --vram 14 -- env G4_INJOB=1 "$PWD/$0" "$GGUF" "$PACK" "$OUT" "$N"
fi
{ uptime; sha256sum .work/gemma4/engine; } > "$OUT/env.txt"
for r in 1 2 3; do
  env MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=60 BARO_PACK="$PACK" BARO_PROMPT="$OUT/p01.ids" BARO_GEN="$N" \
    .work/gemma4/engine > "$OUT/ours-$r.log" 2>&1 || { echo "FAIL ours run $r: $(tail -2 "$OUT/ours-$r.log")"; exit 1; }
  grep -oP 'tok/s_gen: \K[0-9.]+' "$OUT/ours-$r.log" >> "$OUT/ours.txt"
  ~/llama.cpp/build/bin/llama-bench -m "$GGUF" -ngl 99 -p 0 -n "$N" -r 1 -o csv > "$OUT/llama-$r.csv" 2> "$OUT/llama-$r.log" \
    || { echo "FAIL llama run $r: $(tail -2 "$OUT/llama-$r.log")"; exit 1; }
  python3 -I -c 'import csv,sys; r=list(csv.DictReader(open(sys.argv[1]))); print(r[-1]["avg_ts"])' "$OUT/llama-$r.csv" >> "$OUT/llama.txt"
done
python3 -I - "$OUT" "$N" <<'PY'
import sys, statistics as st
o, n = sys.argv[1], sys.argv[2]
a = [float(x) for x in open(f"{o}/ours.txt")]; b = [float(x) for x in open(f"{o}/llama.txt")]
print(f"ours tok/s {' '.join(f'{x:.1f}' for x in a)} median {st.median(a):.1f}")
print(f"llama.cpp tg{n} tok/s {' '.join(f'{x:.1f}' for x in b)} median {st.median(b):.1f}")
print(f"RESULT gemma4-speed: ours / llama.cpp = {st.median(a) / st.median(b):.3f} (decode {n} tokens, F16, full offload)")
PY
