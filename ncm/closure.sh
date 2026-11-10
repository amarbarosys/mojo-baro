#!/usr/bin/env bash
# NCM closure gate: rebuild ncm.mojo FROM the sources a BARO gguf carries (baro.kernel.src.*, nothing from the repo),
# run the file's own self-check (ncm.run.puzzles vs ncm.run.ref.tokens, the PyTorch oracle's tokens), exit non-zero
# unless all 64 boards match.   usage: ncm/closure.sh MODEL-BARO.gguf
set -euo pipefail
G=$(readlink -f "${1:?usage: ncm/closure.sh MODEL-BARO.gguf}")
REPO=$(cd "$(dirname "$0")/.." && pwd)
D=$(mktemp -d "$REPO/.work/ncm-closure.XXXX")
python3 -I - "$G" "$D" <<'P' || { echo "FAIL extract"; exit 1; }
import os, struct, sys
f = open(sys.argv[1], "rb"); u = lambda fmt: struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
assert f.read(4) == b"GGUF" and u("<I") == 3
u("<Q"); nkv = u("<Q"); size = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
s = lambda: f.read(u("<Q")).decode()
out = 0
for _ in range(nkv):
    k, t = s(), u("<I")
    if t == 8:
        v = s()
        if k.startswith("baro.kernel.src."):
            p = os.path.join(sys.argv[2], k[len("baro.kernel.src."):]); os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "w").write(v); out += 1
    elif t == 9:
        et, n = u("<I"), u("<Q")
        for _ in range(n): s() if et == 8 else f.read(size[et])
    else:
        f.read(size[t])
print("extracted", out, "sources")
sys.exit(0 if out else 1)
P
find "$D" -name '*.mojo' | sed "s#$D/#  #"
"$REPO/.venv/bin/mojo" build -I "$D/tools" --target-accelerator gfx1100 "$D/ncm/ncm.mojo" -o "$D/ncm-bin" || { echo "FAIL build from the gguf's sources"; exit 1; }
gpu-wait run --timeout 300 --vram 4 --label ncm-closure --cwd "$D" -- "$D/ncm-bin" "$G" | tail -1
echo "PASS ncm closure: $(basename "$G") rebuilds its engine and reproduces the oracle tokens"
