#!/usr/bin/env bash
# NCM sudoku parity gate: Mojo codes (all 12 levels) and tokens == the PyTorch oracle dump on 64 boards,
# decoder logits within 1e-3; FULL=1 also runs all 2000 boards and requires the oracle's 1988 solved.
# Oracle dump: ~/Projects/research/ncm/tools/ncm_dump.py W/oracle 64 (ROCm torch venv there).
set -euo pipefail
cd "$(dirname "$0")/.."
W=${NCM_W:-$HOME/Models/ncm-checkpoints/sudoku/easy}
mkdir -p .work/ncm
./.venv/bin/mojo build -I tools --target-accelerator gfx1100 ncm/ncm.mojo -o .work/ncm/ncm || { echo "FAIL build"; exit 1; }
gpu-wait run --timeout 300 --vram 4 --label ncm-check --cwd "$PWD" --env NCM_DUMP=1 -- ./.work/ncm/ncm "$W/mojo/ncm-sudoku-easy.gguf" 64 64 "$W/mojo" | tail -1
python3 -I - "$W" <<'P' || exit 1
import sys, numpy as np
W, N, lv = sys.argv[1], 64, [1, 2, 4, 6, 9, 13, 18, 26, 37, 50, 67, 89]
o = np.fromfile(f"{W}/oracle/codes.i32", np.int32).reshape(N, 322); m = np.fromfile(f"{W}/mojo/codes.i32", np.int32)[:N * 322]
off, bad = 0, 0
for k, l in enumerate(lv):  # mojo stores level-major inside the batch
    d = int((o[:, off:off + l] != m[N * off:N * off + N * l].reshape(N, l)).sum()); bad += d
    if d: print(f"FAIL level {k}: {d} codes differ")
    off += l
t = int((np.fromfile(f"{W}/oracle/tokens.i32", np.int32).reshape(N, 89) != np.fromfile(f"{W}/mojo/tokens.i32", np.int32)[:N * 89].reshape(N, 89)).sum())
e = float(np.abs(np.fromfile(f"{W}/oracle/declogits.f32", np.float32) - np.fromfile(f"{W}/mojo/declogits.f32", np.float32)).max())
print(f"codes differing {bad}/{N * 322}, tokens differing {t}, decoder logits max abs diff {e:.2e}")
sys.exit(0 if bad == 0 and t == 0 and e < 1e-3 else 1)
P
if [ "${FULL:-0}" = 1 ]; then
  s=$(gpu-wait run --timeout 600 --vram 6 --label ncm-check-full --cwd "$PWD" -- ./.work/ncm/ncm "$W/mojo/ncm-sudoku-easy.gguf" 2000 256 "$W/mojo" | tail -1); echo "$s"
  grep -q "solved 1988 / 2000" <<<"$s" || { echo "FAIL full: oracle solves 1988 / 2000"; exit 1; }
fi
echo "ok ncm parity"
