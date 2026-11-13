# Brief: WMMA failure diagnostics in test_wmma_rdna34 (codex, effort medium, S)

Repo: the worktree you start in (mojo-baro). `cd` there first. Owner asleep: no windows, no images opened.

## Goal
When `kernels/test_wmma_rdna34.mojo` fails on a stranger's card (RDNA4 never ran here), the output must say WHERE it is
wrong, so a pasted report can be diagnosed without the card.

## Do
1. On mismatch, print up to 8 wrong cells: `bad r=R c=C got=G want=W`, then for each wrong cell the (row, col) of the
   reference matrix whose value equals `got` within the same 2^-19 sum|ab| bar, if any (`matches want at r2=.. c2=..`).
   That turns a scrambled-row bug into a readable permutation.
2. Print the lane and register that wrote each wrong cell under the current layout: lane `= c + 16 * half`, register
   index from the row formula in `store_acc` (gfx11 row `2*i + half`, gfx12 row `i + 8*half`). Read `store_acc` for the
   formula; do not change it.
3. Keep the PASS line and exit codes unchanged (exit non-zero on FAIL).
4. Falsifier: build a mutant copy under `.work/` with the gfx12 row formula forced on gfx11; it must FAIL and print a
   consistent permutation (every wrong cell `matches want` somewhere).

## Check
- Build for gfx1100 and gfx1201 (`./.venv/bin/mojo build kernels/test_wmma_rdna34.mojo --target-accelerator <t>`).
- GPU: only `gpu-wait run --shared --vram 1 --timeout 120 -- <binary>` (needs the herd/gpu-wait outside the sandbox: if
  you cannot reach it, say so; the boss runs it).
- `tools/ci-checks.sh` passes.

## Rules
Test files may have docstrings. No commits if your sandbox cannot write `.git`: leave the diff in the worktree; the boss
commits. Report to `docs/lanes/2026-11-12-wmma-diag-report.md` in the worktree: what changed, build results, mutant
output (first 10 lines), what is UNVERIFIED. Blocked means two approaches tried with evidence.
