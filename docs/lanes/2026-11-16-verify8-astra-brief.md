# Brief: why is the 8-row q4 GEMM half the bandwidth of the 1-row GEMV? (investigate + recommend)

Worktree: `~/Projects/mojo/mojo-baro-lanes/verify8` (branch lane-verify8). Card: RX 7900 XTX, gfx1100, wave32.
Deliverable: `docs/lanes/2026-11-16-verify8-astra-report.md` with root cause (with evidence), one recommended fix
(code sketch or patch on a branch, not merged), and the predicted GB/s. When done: `herd tell a lane pane "DONE verify8-astra -> docs/lanes/2026-11-16-verify8-astra-report.md"`.

## Tools (mandatory)
- Code navigation: codebase-memory MCP first (`search_graph`, `trace_path`, `get_code_snippet`, `query_graph`;
  project `home-user-Projects-mojo-mojo-baro`, same code as this lane). grep only for literals.
- GPU: every run via `gpu-wait run --timeout S --priority 90 -- <cmd>`. Short runs only. Wavewait: `bench/ww.sh`.

## Problem
DFlash2 speculative decode verifies 8 tokens per cycle. Verify window ~46 ms vs 25.7 ms for one plain token.
Profile: GEMMs are ~75% of the window. The 8-row q4 GEMM `amar_fewrow_wmma` (`kernels/fewrow.mojo`, routed from
`serve/window.mojo::gemm_w` when m > 4) reaches ~410 GB/s; the 1-row GEMV `amar_turbo_rows` (`kernels/turbo_rows.mojo`)
reaches ~800 GB/s on the same weights. Goal: 8-row GEMM near the 1-row bandwidth (weights read once for all 8 rows).

## Measured (bench `bench/bench_turbo_rows.mojo`, real 27B weights, shapes 17408x5120 gate, 5120x17408 down)
| arm | GB/s |
|---|---|
| turbo m=1 | ~800 |
| fewrow WMMA D=4, m=1..8 | 400-425 (flat in m) |
| fewrow + R row tiles per block (A fragment and ones-WMMA shared, R=2/4) | 270-430: falsified |
| fewrow + half-waves load different blocks then swap (permlanex16) to drop duplicate B loads | ~390: falsified |

Wavewait of fewrow in the live engine (`.work/v8/ww-base/{classes,top}.txt`): waitcnt vmcnt 36.5%, barrier 21.4%,
valu 21.7%, vmem load 15.7%; top stall `s_waitcnt vmcnt(7)`, then the final `s_barrier`. Per wave only NB/8/D
loop trips (5 for K=5120, 17 for K=17408) before the cross-wave LDS reduce.

Build: `./.venv/bin/mojo build --target-accelerator gfx1100 bench/bench_turbo_rows.mojo -I . -I kernels -I serve -D BARO_H=5120 -D BARO_FFN=17408 -o .work/v8/bench-x`
Run: `gpu-wait run --timeout 300 --priority 90 -- env BARO_PACK=$(realpath ../nerkyor-vs-forge/.work/nerkyor/pack-q4) .work/v8/bench-x`

## Constraints
Numerics may change (verify gate is teacher-forced agreement, tolerance 0.005), but say if bit-exactness vs the current
fewrow is lost. Smallest tests only. No edits on main; no merge; no push.
