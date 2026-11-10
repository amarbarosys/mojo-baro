# Decode attention, generalised (frozen 2026-09-11, before any kernel change)

Binds: `bench/PROTOCOL-RULES.md` P1-P6, `bench/coldcache-protocol.md`, repo `CLAUDE.md`.

## Why this round

The llama.cpp harvest (`results/ggml-harness-headroom-2026-09-11.md`, note
`2026-09-11-llama-cpp-kernel-harvest`) puts decode attention furthest from the 7900 XTX
roofline of any kernel our models run: `flash_attn_ext_vec` at 13-31% of 960 GB/s,
cache-cold. It is also the one kernel a granite engine is missing: `kernels/attn.mojo`
hard-codes Qwythos (`HD=256`, `NQH=16`, `NKVH=4`, `KVT=f32`).

## Scope

A decode attention kernel generic over `HD`, `NQH`, `NKVH` and `KVT` (f32, f16, bf16) at
comptime, runtime `scale`, one query token, reading the paged KV layout `kv_off` defines
(128-token pages, layer-in-page, LatentOS page format). Split across blocks for long
caches the way `attn_phases` does in `kernels/mega.mojo` (partials + combine), exact
single-span path for short ones.

Out of scope: prompt attention, masks, sliding window, gating, the megakernel phase
itself (a later round wires the winner in).

Base: `main` (contains `1e7ab91`, the page-arithmetic fix). **Nothing from `lane-attn`**:
its Round A V addressing is the bug `1e7ab91` deleted. Its hoisted-load idea is allowed
only on per-element `kv_off`.

## Arms

| arm | what | instrument |
|---|---|---|
| **R** | llama.cpp `flash_attn_ext_vec` via `bench/ggml-harness/op_bench flash_attn`, re-run in the SAME stint as O | rocprofv3 device time per iteration |
| **O** | ours, standalone bench with the same KV rotation (>= 4 x 96 MB) and the same shapes | rocprofv3 device time per iteration |

KV dtype f16 on both arms. Shapes (NH NKVH HD KV), one query token, KV = 4096:

| shape | model | R receipt (us, 2026-09-11) |
|---|---|---|
| S1 16 4 256 4096 | qwen35 (Qwythos, Ornith, 27B) | 55.99 |
| S2 40 8 64 4096 | granite-4.2-3b | 34.78 |
| S3 28 4 128 4096 | qwen2.5-7b | 67.03 |

Plus a scaling receipt at KV = 512, 1024, 16384 on S1 (split vs exact path crossover).

## Gates (all must pass before any timed run counts)

1. **P1 read-back:** O's bench echoes HD, NQH, NKVH, KVT, KV length, grid and block dims,
   split count, bytes rotated; R's harness line (`arm ... | rotated ... exceeds`) recorded.
2. **Numerics:** O vs an fp64 numpy reference of softmax(QK^T * scale) V over the same f16
   KV bytes: max abs error <= 2e-3 relative to max|out|, on all three shapes and on
   KV = 1, 127, 128, 129, 4096 (page edges).
3. **No regression on the shipped path:** instantiated at HD=256 / NQH=16 / NKVH=4 /
   KVT=f32, O is bit-identical to today's `amar_attn_decode` on `kernels/test_attn_block.mojo`,
   and the engine gate stays green (`./run-tests.sh`, `tools/model-ref.py` 64/64,
   mega == launch).

## Frozen predictions (device us per iteration, cache-cold)

Ideal time = KV bytes / 960 GB/s: S1 17.5, S2 8.7, S3 8.7 us.

| shape | R | O predicted | ratio R/O |
|---|---|---|---|
| S1 | 55.99 | 24-30 | 1.9-2.3x |
| S2 | 34.78 | 13-18 | 1.9-2.7x |
| S3 | 67.03 | 13-18 | 3.7-5.2x |

**Land:** O beats R by >= 1.5x on at least two shapes and is not slower on any, all gates
pass. **Close negative:** O < 1.1x on two or more shapes. Between: report, no land, one
more round only with a new frozen prediction.

## Stop rules

- Gate 3 mismatch: the generic kernel changed the shipped arithmetic. Find the difference;
  do not re-freeze the reference.
- R's rotation reads "BELOW": the R row is void, re-run, never compared.
- Any run whose fail word or parameter echo is missing is void (repo `CLAUDE.md`).

## Result (2026-09-11, lane-dattn)

**Verdict: BETWEEN, no land.** Ours beats R by 1.43x (S1), 1.70x (S2), 1.48x (S3). The land
rule needs >= 1.5x on two shapes: only S2 clears it (S3 misses by 0.02, S1 by 0.07). Close
negative needs < 1.1x on two shapes: none. Per this protocol: report, no land, one more round
only with a new frozen prediction.

Run: `bench/dattn-confirm.sh` confirmation c4, commit `7721fe0`, llama.cpp `1744c6bde`, one
gpu-wait job: gates first (fail-closed), then 10 repeats alternating arm order. Arm O configs
frozen by commit `ffc7684` from the stint-3 sweep. Device us per iteration: each arm's own
kernels from the 21st main-kernel dispatch on, divided by the main-kernel count, from the trace.

| shape | R us (spread) | O us (spread) | R/O | ideal us | O % of roof | R % of roof |
|---|---|---|---|---|---|---|
| S1 16/4/256 KV4096 | 37.29 (0.9 %) | 26.16 (1.0 %) | **1.43** | 17.5 | 67 | 47 |
| S2 40/8/64 KV4096 | 28.24 (0.3 %) | 16.63 (1.5 %) | **1.70** | 8.7 | 52 | 31 |
| S3 28/4/128 KV4096 | 41.32 (0.3 %) | 27.85 (0.6 %) | **1.48** | 8.7 | 31 | 21 |

S1 scaling receipt (split vs exact):

| KV | R us (spread) | O split us (spread) | R/O | O exact us (spread) |
|---|---|---|---|---|
| 512 | 36.91 (3.4 %) | 10.57 (1.9 %) | 3.49 | 28.63 (0.1 %) |
| 1024 | VOID (37.6 %) | 11.45 (1.9 %) | - | 44.87 (0.1 %) |
| 4096 | 37.29 (0.9 %) | 26.16 (1.0 %) | 1.43 | VOID (21.6 %) |
| 16384 | 104.74 (0.2 %) | 84.75 (0.3 %) | 1.24 | not run |

The split path beats the exact path at every length measured; there is no crossover in 512-16384.

Frozen predictions against the result:

| prediction | frozen | observed | |
|---|---|---|---|
| O S1 | 24-30 | 26.16 | HELD |
| O S2 | 13-18 | 16.63 | HELD |
| O S3 | 13-18 | 27.85 | MISSED |
| R/O S1 | 1.9-2.3 | 1.43 | MISSED |
| R/O S2 | 1.9-2.7 | 1.70 | MISSED |
| R/O S3 | 3.7-5.2 | 1.48 | MISSED |

Every ratio miss traces to R, not O: R's frozen receipts (55.99 / 34.78 / 67.03) were timed at
200 iterations, targets shorter than the clock ramp; at 5000 iterations R reads 37.29 / 28.24 /
41.32, 1.2-1.6x faster. The same defect voided confirmation run c3 (spread 15-37 %). The round's
premise ("13-31 % of roofline") was measured the same way and overstated R's gap.

Receipts (P3):

| checked | read from | observed |
|---|---|---|
| O arm parameters | bench_dattn arm line per target | HD, NQH, NKVH, KVT=f16, KV, path, nsplit, nld, rot, span, grid, block, combine dims as frozen |
| iterations | trace, main-kernel dispatch count | 5020 per target, both arms (5000 timed + 20 warmup) |
| rotation | arm line, both arms | 402.7 MB every target, "exceeds" 4 x 96 MB |
| O resources | trace | split S1 176 VGPR / 0 B scratch, S2 192 / 0, S3 192 / 316 B; LDS 8704 / 3072 / 7680 |
| R resources | trace | flash_attn_ext_vec 216 / 96 / 136 VGPR, 0 scratch |
| gate 2 | tools/dattn-ref.py | 760/760, worst 1.43e-5 (bound 2e-3) |
| gate 3 | test_attn_block, rebuilt in the job | 0 of 4096 words differ |
| engine gate | tools/mega-gate.sh in the job | ALL PASS (identity q8 / q8d / q4, spec 0/1, perf ratio 1.201) |
| clocks | rocm-smi sampler, 0.5 s | busy sclk median 3296 MHz (662-3325), busy power median 157 W, cap 290 W |

Report: `exchange/lane-dattn-report.md`.
