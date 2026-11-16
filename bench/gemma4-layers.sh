#!/usr/bin/env bash
# M2 gate of lane-gemma4-port: our per-layer hidden (l_out-<il>) vs llama.cpp's on one prompt, cosine per position.
# Bar (frozen before the first run): min cosine over the compared positions >= 0.9999 at layers 0, 1, 4 (full),
# 15 (first shared-KV layer), 34 (last). llama.cpp prunes rows before the first shared-KV layer, so layers >= 15 hold
# only the last position there; earlier layers compare every position. Reference KV is f32 (llama.cpp f16 KV alone moves
# layer 11 at pos 567 to cos 0.9998 against its own f32-KV run, 2026-11-16).
# Usage: bench/gemma4-layers.sh GGUF PACK PROMPT.txt OUT        (OUT must not exist; paths relative to the repo)
#   CPU: tokenize + llama.cpp dump (patched llama-debug, LLAMA_DUMP_DIR); GPU (queued here): our dump; compare.
set -euo pipefail
cd "$(dirname "$0")/.."
GGUF=$1; PACK=$2; PROMPT=$3; OUT=$4
LB=${LLAMA_DUMP_BIN:-.work/gemma4/llama-dump/build/bin}
ENGINE=${ENGINE:-.work/gemma4/engine}
[ -e "$OUT" ] && { echo "FAIL setup: $OUT exists (dumps append)"; exit 1; }
[ -x "$ENGINE" ] || { echo "FAIL setup: no engine $ENGINE"; exit 1; }
mkdir -p "$OUT/llama" "$OUT/ours"
"$LB/llama-tokenize" -m "$GGUF" -f "$PROMPT" --ids --log-disable 2>/dev/null | tr -d '[] ' | tr ',' '\n' > "$OUT/ids.txt"
echo "prompt ids: $(wc -l < "$OUT/ids.txt")"
LLAMA_DUMP_DIR="$OUT/llama" "$LB/llama-debug" -m "$GGUF" -f "$PROMPT" -c 4096 -b 4096 -ub 4096 -t 16 -ngl 0 -ctk f32 -ctv f32 -fa off \
  --tensor-filter '^l_out-[0-9]+$' > "$OUT/llama.log" 2>&1 || { echo "FAIL llama: $(tail -3 "$OUT/llama.log")"; exit 1; }
[ -s "$OUT/llama/index.txt" ] || { echo "FAIL llama: empty dump"; exit 1; }
gpu-wait run --timeout 600 --vram 12 -- env MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=60 BARO_PACK="$PWD/$PACK" \
  BARO_PROMPT="$PWD/$OUT/ids.txt" BARO_GEN=1 BARO_DUMP_DIR="$PWD/$OUT/ours" "$PWD/$ENGINE" > "$OUT/ours.log" 2>&1 \
  || { echo "FAIL ours: $(tail -3 "$OUT/ours.log")"; exit 1; }
[ -s "$OUT/ours/index.txt" ] || { echo "FAIL ours: empty dump, $(tail -3 "$OUT/ours.log")"; exit 1; }
python3 -I - "$OUT" <<'PY'
import sys, numpy as np
o = sys.argv[1]
def rd(d):
    data = np.memmap(f"{d}/data.bin", dtype=np.float32, mode="r"); t = {}
    for l in open(f"{d}/index.txt"):
        name, _, off, n = l.split(); t[name] = data[int(off) // 4: int(off) // 4 + int(n)]
    return t
L, O = rd(f"{o}/llama"), rd(f"{o}/ours")
H = 1536; worst = {}
for il in range(35):
    a = O[f"l_out-{il}"].reshape(-1, H).astype(np.float64); b = L[f"l_out-{il}"].reshape(-1, H).astype(np.float64)
    skip = len(a) - len(b); a = a[skip:]  # llama keeps only the output rows on the shared-KV layers
    cos = (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
    worst[il] = cos.min()
    print(f"l_out-{il:2d} rows {len(b):4d} min cos {cos.min():.7f} at pos {int(cos.argmin()) + skip} mean {cos.mean():.7f}")
bad = [il for il in (0, 1, 4, 15, 34) if not worst[il] >= 0.9999]
print(("FAIL" if bad else "PASS") + " gemma4-layers: gate layers 0 1 4 15 34 min cos " + " ".join(f"{worst[i]:.6f}" for i in (0, 1, 4, 15, 34)) + f", bar 0.9999, below: {bad}")
sys.exit(1 if bad else 0)
PY
