# Report: lane rdna4-epi (gfx12 accumulator rows, emulated on gfx1100)

Branch `lane-rdna4-epi`. Brief `docs/lanes/2026-11-12-rdna4-epilogue-brief.md`, plan `docs/lanes/2026-11-12-rdna4-epi-plan.md`.
Skills loaded: rdna4-port, rdna4-isa, gpu-kernels (rules #7 to #9), mojo-baro, gate-authoring.
Receipts (local, gitignored): `.work/rdna4-epi/` in the worktree; copies in `exchange/receipts/rdna4-epi/`.

## Commits
- 997a57d acc_row helper, gfx12 row emulation on gfx1100, epilogue sites
- ac47447 remaining-steps plan, two rules
- c545a27 attn prefill row sum in gfx11 order under gfx12 rows (gate-5 fix)
- 8bf852f rule: reductions over WMMA acc rows go through gfx11_rows
- 5197d88 settings: list BARO_EMU_GFX12_ROWS

## Gates
| gate | bar | result | receipt |
|---|---|---|---|
| 1a gfx11 ISA identity | 8 targets byte-equal vs 91c933f | PASS 8/8 (engine 288 objects) | `rebuild.log` |
| 1b parity default | fails=0 skips=0 | PASS 7/7 | `run-def-*.log` |
| 2 parity emulation | fails=0, digits equal to default | PASS 7/7, test_prefill digits identical | `run-emu-*.log` |
| 3 mutants | 13 caught | PASS 13/13 | `mutants.txt` |
| 4a gfx1201 build | 8 builds | PASS | `rebuild.log` |
| 4b probe table | unchanged | PROBE-SAME | `probe.log` |
| 5 engine agreement | ctl 100%, emu 100% on 20/20, mutant < 100% each, TMAX 16384 | PASS (first run FAIL 17/20, fixed by c545a27) | `gpu2.log`, `engine/` |

Gate 1a identical ISA means no perf A/B and no wavewait capture are needed (rdna4-port s10).
Gate 5 uses 64 forced ids per prompt: the fixed-length comparison that `bench/min16k.sh` exempts from the 16000 max_tokens floor.

## Gate-5 failure and fix
First run: emu 63/64 on p01, p04, p09. Not decode: all 90 code objects that differ between def and emu contain v_wmma; every
decode object is byte-equal, and decode has no float atomics. Cause: `amar_attn_prefill_wmma` summed `pv.reduce_add()` over the
lane's rows; gfx12 lanes hold rows i+8*half, so the f32 sum grouped differently (1 ulp in test_prefill: 0.0017692209 vs
0.001769221). Fix: `gfx11_rows(pv).reduce_add()`; gfx11 bytes unchanged. After: 64/64 on 20/20.

## Mutants (gate said)
- rdna34_store: test_wmma_rdna34: Unhandled exception caught during execution: FAIL wmma-gfx: 253/256 mismatch
- prefill_q4: test_prefill: Unhandled exception caught during execution: parity failure: q4 wmma 128x128 M=1024 vs
- prefill_q8: test_prefill: Unhandled exception caught during execution: parity failure: q8 wmma 128x128 M=1024 vs
- prefill_lds: test_prefill: Unhandled exception caught during execution: bit-exact failure: q4 lds 128x128 M=1024 
- prefill_kq: test_kq_prefill_gemm: FAIL kq-prefill-ref: 16755393 elements outside |err| <= 2^-8 |ref| + 2^-19 sum
- mmq_rows: test_mmq: Unhandled exception caught during execution: parity failure: mmq 128x128 (WM4 WN2 TM2 TN4)
- mmq_scale: test_mmq: Unhandled exception caught during execution: parity failure: mmq 128x128 (WM4 WN2 TM2 TN4)
- moe_wmma: test_moe_wmma: Unhandled exception caught during execution: FAIL test_moe_wmma: 2 checks
- wmma_lds: bench_fp16: {"variant": "wmma_lds_fp16", "m": 512, "n": 512, "k": 512, "ms": 0.071645195, "gflops": 
- wmma_pipe: bench_fp16_pipe: {"variant": "wmma_pipe_fp16", "m": 512, "n": 512, "k": 512, "ms": 0.019508841754385
- attn_mask: test_prefill: Unhandled exception caught during execution: parity failure: attn prefill wmma
- attn_repack: test_prefill: Unhandled exception caught during execution: parity failure: attn prefill wmma
- attn_out: test_prefill: Unhandled exception caught during execution: parity failure: attn prefill wmma

## Sites
| site | change | status |
|---|---|---|
| `matmul_prefill.mojo:93` (q4), `:169` (q8) | `acc_row(row0 + tm * 16, i, half)` | VERIFIED-EMU (test_prefill, mutants prefill_q4, prefill_q8) |
| `matmul_prefill_lds.mojo:158` | same | VERIFIED-EMU (test_prefill 1b, mutant prefill_lds) |
| `matmul_prefill_kq.mojo:202` | same | VERIFIED-EMU (test_kq_prefill_gemm, mutant prefill_kq) |
| `matmul_mmq.mojo:277` rows | same | VERIFIED-EMU (test_mmq, mutant mmq_rows) |
| `matmul_mmq.mojo:157` scale staging | LDS slot tid from global slot `mmq_scale_slot(tid % BM)` under GFX12_ROWS | VERIFIED-EMU (test_mmq, mutant mmq_scale) |
| `matmul_mmq.mojo:24` `mmq_scale_slot`, `:199` `half * 8` | unchanged on purpose (buffer format, see handoff) | VERIFIED-EMU through :157 |
| `moe_wmma.mojo:156` | `acc_row(wm * TM * 16 + tm * 16, i, half)` | VERIFIED-EMU (test_moe_wmma, mutant moe_wmma) |
| `matmul_wmma_lds.mojo:161` | `acc_row(block_row + ..., i, half)` | VERIFIED-EMU (bench_fp16 correct flag, mutant wmma_lds); no parity test in the suite |
| `matmul_wmma_pipe.mojo:879` | same | VERIFIED-EMU (bench_fp16_pipe correct flag, mutant wmma_pipe); no parity test in the suite |
| `attn.mojo:747,754` mask | `acc_row(t0, i, half) <= lim` | VERIFIED-EMU (test_prefill 2b, mutant attn_mask) |
| `attn.mojo:764,765` P-to-B repack | `bf[acc_row(0, i, 0)]`, `bf[acc_row(0, i, 1)]` | VERIFIED-EMU for the layout; stdlib half-slice not modelled (handoff, attention verdict) |
| `attn.mojo:777` output row | `acc_row(dt * 16, i, half)` | VERIFIED-EMU (mutant attn_out) |
| `kernels/test_freivalds.mojo:106`, `miniagi/latent/lops.mojo:98` | not touched, outside the file list | UNPORTED: wrong rows on gfx12 |

| `attn.mojo:756` row sum | `gfx11_rows(pv).reduce_add()` | VERIFIED-EMU (gate 5, test_prefill digits) |

## Design calls that differ from the brief

1. `acc_row(base, i, half)`, not `acc_row(i, half)`. The two-argument form made the site `(row0 + tm*16) + (2*i + half)`; LLVM then built
   `half | 2i` first and the gfx11 epilogue bytes moved (13 of 32 code objects in `test_prefill` differed, same VGPR counts). With the base inside
   the helper the gfx11 expression is `base + 2*i + half` as before and every code object is byte-equal. The ISA gate caught this; it can fail.
2. `mma` now binds `d` and `c` to 8 elements (`def mma[T: DType, //](mut d: SIMD[T, 8], a: SIMD, b: SIMD, c: SIMD[T, 8])`). An unbound `SIMD`
   cannot be passed to a helper that needs the width; `SIMD[T, N]` with `N: Int` is rejected too (the length parameter is not an Int).
3. `matmul_mmq`: `mmq_scale_slot` and the `half * 8` scale row are unchanged. The slot order is the Ad/An buffer format and `test_mmq.mojo`
   checks it with its own copy (`slot_of`, line 79): changing the slot under gfx12 rows would fail that test, and the test may not be edited.
   Instead, under `GFX12_ROWS` the LDS staging loads global slot `block_row + mmq_scale_slot(tid % BM)` into LDS slot `tid`, so the LDS copy is in
   row order and the `half * 8` read (8 contiguous slots per half) is right for both layouts. Mutant (remap removed) is killed by `test_mmq`.

## Out of scope
- `kernels/test_freivalds.mojo:106`, `miniagi/latent/lops.mojo:98` still decode `2*i + half` (outside the file list).
- Baseline defect: `test_text` SIGILL (exit 132) on the unchanged tree 91c933f.
- gfx12 numerics stay UNVERIFIED: no gfx12 card; only the emulation and the gfx1201 build are checked.

## run-tests.sh
`./run-tests.sh` under gpu-wait (STATE_ENGINE = def engine): `PASS run-tests: all tests, pack .work/engine-pack-q4`, `suite exit 0`,
259 PASS lines, `FAIL: 0 (none)`. Receipt `.work/rdna4-epi-gate.txt`. Before (91c933f): 15 PASS, then test_text SIGILL, exit 132.
The worktree needs `.work/front-text` linked from main, or text_oracle stops the suite at a missing pack tokenizer.
Preflight: `PASS preflight 2382b469c3dd` (`.work/rdna4-epi/preflight-post.log`).
