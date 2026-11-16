# Brief: make the window megakernel fast at 27B (amar_mega_window), then extend it to 8 rows

Worktree `~/Projects/mojo/mojo-baro-lanes/verify8` (branch lane-verify8). Report: `docs/lanes/2026-11-16-megawin27-report.md`.
When done or blocked: `herd tell a lane pane "DONE megawin27 -> docs/lanes/2026-11-16-megawin27-report.md"`.
Tools: codebase-memory MCP first for navigation; every GPU run `gpu-wait run --timeout S --priority 90 -- ...`;
wavewait via `bench/ww.sh` (kernel changes need a capture); smallest tests. No merge, no push, no edits on main.

## Why
Plain decode on the launch path leaves the GPU idle ~17% (1,100 kernels/token, 3.4 us gaps; `.work/v8/kt`).
The persistent megakernel recovers it at m=1: 39.0 -> 42.1 tok/s (`.work/v8/gap-ab2`, `bench/gap-ab.sh`).
Spec paths need the WINDOW megakernel (`kernels/mega.mojo::amar_mega_window`, `MEGA_MR = 3` in serve/registry.mojo),
but at 27B it LOSES: MTP k=2 launch 73.3 tok/s vs mega_win 32.4 (same acceptance; ids differ on 3/3 prompts).

## Evidence (`.work/v8/ww-megawin`, `tools/wavewait app ... --dispatch D --top 0`)
- waitcnt vmcnt 63.2% (cause mem 63%), barrier 16.3%, valu 12.8%, spill 0.5% with scratch_load/store in hot code.
- Hot PC ranges 0x22000-0x29000 (FFN/GEMV phases; strides 0x11000 = 17408*4). Weight-row loops issue
  `global_load` then `s_waitcnt vmcnt(0)` at once: no loads in flight across iterations.
- The m=1 launch-path GEMV `kernels/turbo_rows.mojo::amar_turbo_rows` reaches ~800 GB/s on the same weights.

## Steps (test after each)
1. Find which window-megakernel phases own the cycles (build with line info if possible, or map PC ranges).
2. Fix the GEMV phases first: several loads in flight (unroll/prefetch like turbo_rows), remove spills
   (VGPR budget, launch bounds). Gate: ids vs launch-path MTP k=2 (teacher-forced agreement if not bit-exact,
   same-arm control, bar frozen before the run), fail word 0 on every run, then `bench/gap-ab.sh` mtp1 > mtp0 (73.3).
3. Only after step 2 wins: MEGA_MR 8 (DFlash2 window) + the 5 tap copies (serve/window.mojo `tap_d`, `df_tap_slot`)
   inside the kernel; DFlash2 currently forces mega_win off (`serve/engine.mojo` ~1873).
Freeze predictions in `.work/verify8-astra/prereg.md` before each timed run.
