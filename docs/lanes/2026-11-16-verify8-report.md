# verify8: why the DFlash2 verify window is slow (2026-11-16)

| step | result | receipt |
|---|---|---|
| live profile of the 8-row window | GEMMs ~75% of the window; SSM recurrence only 21% of the SSM block | `~/Projects/mojo/mojo-baro-lanes/dflash2/.work/dflash2/prof2` |
| 8-row q4 GEMM, bf16 WMMA | 424 / 399 GB/s (gate / down) vs 805 / 778 for the 1-row turbo GEMV; VALU-bound (64.5%) | `docs/lanes/2026-11-16-verify8-astra-report.md` |
| R row tiles, half-wave split loads, split-K, centering, bit-built bf16, refill pipeline | all falsified | same |
| transposed Q/S | +4.5 / +5.9%, bit-exact, needs a pack layout | same |
| int8 WMMA (`amar_fewrow_iu8` + `amar_q8_cols`) | 538 / 447 GB/s incl. quantize; max abs err 0.37% of max abs ref; not bit-exact | 91f72c9, `.work/v8/bench-iu8.log` |
| self-describing bake + closure of nerkyor 27B | PASS 64/64 after the bake carried `baro.run.build.flags` (tools/bake.sh never wrote it) | `.work/v8/bake2`, `~/Models/qwen38-27b-nerkyor-coder-q4lynn/nerkyor-27B-BARO-91f72c9.gguf` |
| CPU or GPU limit, plain decode launch path | host 99.2% in the HSA wait spin (corewait --pid); GPU 82.6% busy, ~1,100 kernels/token, 3.4 us median gap = ~4.5 ms/token idle | `.work/v8/cwd-plain`, `.work/v8/kt` |

Open: the int8 GEMM is not wired into the engine (no engine gate, no wavewait capture of it in the engine yet). Next lever: the
inter-kernel gaps, which every spec path pays because spec turns the megakernel off.

## Merge gate (2026-11-16)
- preflight: PASS f489438aa1b9, `.work/merge/preflight.log`
- run-tests.sh at ef8bb09: rc=0, PASS all tests, `.work/merge/run-tests.log`
