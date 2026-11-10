# Baseline — what is verified working

Current truth for `mojo-baro`. Every number here was measured on this machine and
is reproducible with the command given. Before any kernel
task, start here: **do not re-derive it, and do not trust a
number that is not in this file or produced by `bench/run.py`.**

Last verified: 2026-11-02 for the 27B Turbo decode and MoE prefill q6_k rows (lanes turbokernel,
rowbw, moeprefill), 2026-10-30 for the K-quant megakernel on Mojo 1.2 (the toolchain moved to 1.2.0 on
2026-10-29; every row dated before that is a Mojo 1.1 or earlier number), 2026-10-26 for the
K-quant megakernel and prefill rows on Mojo 1.1, 2026-10-19 for the spark K-quant / f32 rows (`bench/sparkq4k-protocol.md`), 2026-09-15 for the qwen35moe decode row (`bench/moe-persist-protocol.md`,
main-tree re-gate), 2026-09-12 for the long-context prefill rows
(`docs/prefill-long-ctx-2026-09-11.md`), and 2026-09-02 for the fp16 WMMA rows
(`bench/wmma-fp16-protocol.md` Round 3). Rows not named here were not re-measured
on those dates: check the protocol each one cites before trusting it.

## Hardware and toolchain

| | |
|---|---|
| GPU | AMD RX 7900 XTX, **gfx1100 (RDNA3)** — 24 GB, ~22.3 GB free to MAX |
| Warp size | **32** (RDNA3, *not* 64 like CDNA) |
| LDS | 64 KB/block |
| ROCm | 7.2, hipBLASLt present at `/opt/rocm/lib/libhipblaslt.so` |
| Mojo | **1.2.0**, repo-local venv `./.venv/bin/mojo` (`~/.local/bin/mojo` links to it) |
| MAX | 26.7.0 |
| Rust | 1.97.1, service layer SHIPPED: `serve/` crate `baro-serve` (axum + tokio), the OpenAI-compatible HTTP front for `serve/engine.mojo`. One engine per card: `BARO_POOL` > 1 does not fit on a single 24 GB GPU, because one engine's MAX runtime reserves about 23.5 to 24.8 GB |

Nothing is installed machine-wide. `uv sync` creates `.venv` from the tracked
`pyproject.toml`, which pins `max[all]==26.7.0` (Mojo 1.2.0) from PyPI. **Not the
nightly index**: stable releases are not carried there, so a nightly pin does not
resolve. Move the toolchain with `mojo-upgrade apply mojo-baro VERSION`: it builds
every main on the old and new toolchain (flags from `.mojo-upgrade`) and swaps only
when no file goes from PASS to FAIL.

## Layers

```
Python/app  →  Mojo (GPU kernels)  →  C++ shim (vendor SDK)  →  hipBLASLt
                      ↑
                    Rust (network/API shell): serve/ crate baro-serve, shipped
```

Rust talks only to Mojo's C-ABI surface. It must **not** bind the C++ shim
directly: two independent owners of one hipBLASLt context is a lifetime bug.

## What runs

| Component | Path | Verify with |
|---|---|---|
| C++ hipBLASLt shim | `shim/` | `./run-tests.sh` |
| Mojo FFI binding | `kernels/amarbaro.mojo` | `./run-tests.sh` |
| GEMM kernels | `kernels/matmul.mojo` | `./bench/run.py` |
| Bench + correctness engine | `bench/run.py` | `./bench/run.py` |
| Parameter sweep | `bench/sweep.py` | `./bench/sweep.py` |

ROCm ecosystem verdicts worth knowing (2026-08-27 survey):
**ATOM has no RDNA3 build path** (gfx942/gfx950 only); **aiter does build and run on
gfx1100** (JIT, since 2026-08-28) but its prebuilt asm kernels are CDNA-only and CK flash
attention needs the `--targets` fix (aiter PR #5786) to dispatch on gfx11; **hip-ep's autotune
LUT is gfx1151-only**, and MIGraphX EP is already installed system-wide.

`./run-tests.sh` builds the shim and checks an fp16 GEMM through the C ABI
against a host reference. `./bench/run.py` gates on correctness *before*
reporting throughput and exits non-zero if any variant is wrong.

## Engine (2026-09-04)

The default pack is **q4** (`BARO_PACK`, default `.work/engine-pack-q4`); the
q8 numbers in this section are the q8 arm, kept because they are what the
kernel work below was measured on. The q8 pack is built by
`tools/engine-pack.py MODEL.gguf OUTDIR --q8` and `tools/q8-check.py` proves it
bit-equal to `llama-quantize Q8_0`. On that arm, all 2D
weight GEMMs are `amar_matmul_skinny_q8row`: one wave per weight row over the
weight-native [out, in] int8 layout plus fp16 block-32 scales, 855 GB/s on
the ffn shape (62.6 us per 100 MB-equivalent stream). Decode: **68.8 tok/s_gen**,
64/64 greedy identity with llama.cpp Q8_0 and the bf16 reference; llama.cpp
Q8_0 no-spec bar 74.1. The bf16 pack path (41.7 tok/s) is gone from the
engine; its numbers stay in `bench/q8-protocol.md`. **Re-measured 2026-10-26, same stint (`bench/llama-stint-protocol.md` run 3): 80.38 tok/s_gen against llama.cpp Q8_0 75.46, 1.065x, 20-prompt medians, 16384 window both sides.** The 68.8 / 74.1 pair above is the 2026-09-04 state.

**MTP speculative decode (`BARO_SPEC=1`; default k=2 via `BARO_SPEC_K` or
`spec-k.txt`).** Draft = `blk.32` head; rows verified in one m=k+1 trunk
window, SSM/conv state in a (k+1)-slot ring so rollback is free.

**Speculation composes with sampling since A1 (2026-09-15, `b3c0d90`,
`bench/spec-sample-protocol.md`).** The window runs the real rule (accept with
min(1, p/q) on the truncated distributions, residual draw on the first
rejection, bonus token from p) instead of refusing to speculate at
temperature > 0. 20-prompt medians, one stint, dense q4, k=2:

| arm | median tok/s_gen | acceptance |
|---|---|---|
| T=0.7 top_p 0.9, spec on | **147.15** | 0.691 |
| T=0.7 top_p 0.9, spec off | 109.19 | |
| T=0, spec on | 150.24 | 0.660 |
| T=0, spec off | 134.97 | |

Two things to read off that table rather than the headline. **Sampling itself
costs 19% of decode at this vocab** (109.19 against 134.97 with no speculation
on either side): the device sampler scans 248,320 logits per draw where the
greedy path does one reduction. And **sampled decoding with speculation is
faster than greedy decoding without it** (147.15 against 134.97). Falsifier:
`bench/spec-sample-ab.sh <engine> OUT 0.7 0.9` on this card; a T=0.7 spec
median below the T=0.7 no-spec median refutes the row.

Headline is the **20-prompt median**, per `bench/PROTOCOL-RULES.md` P4 —
a single-prompt speculative number is an instrument receipt, never a verdict:

| set | ours | llama.cpp Q8_0 | ratio |
|---|---|---|---|
| **20 real prompts (median, k=2)** | **100.7** | **123.5** | **0.78x** |
| 5-token race prompt (k=4) | 145.6 | 109.8 | 1.33x |

So: ahead on the preregistered race prompt, **behind on real text**. Both
engines' speculative output matches their own greedy output on the race
prompt; on the 20-prompt set ours matches on 20/20, llama.cpp's on 16/20.

The earlier headline of **127.96 tok/s_gen (1.89x over 67.77, acceptance
50/53)** was measured on the 5-token race prompt alone and is superseded —
that prompt's repetitive tail inflates acceptance to ~94%. Kept here only so
the number is recognisable when it turns up in older notes.
Protocol, k sweep and bug log: `bench/mtp-protocol.md` Result 2.

Weight-native wave-per-row is the fastest measured stream on this card
(qingming-gfx1100-gemv 917 GB/s fp32; ours 856 bf16 / 855 q8). The earlier
"wt-layout is coalescing-bound" verdict was about a thread-per-column kernel.

## Measured kernel performance

512×512×512, fp32, 200 iterations, 10 warmup. Run-to-run spread <0.5%.

| variant | GFLOP/s (512³) | notes |
|---|---|---|
| `hipblaslt` | ~5200 | **vendor baseline**, tuned (algo × splitK × wgm) |
| `amar_matmul_naive` | ~1250 | one thread per output element |
| `amar_matmul_tiled` | ~2270 | 16×16 shared-memory tiles |
| `amar_matmul_regtile` | ~5130 | BM32 BN32 BK8 TM2 TN2, swept |

**regtile and tuned hipBLASLt are a tie**, trading places by size:

| size | hipBLASLt | regtile | winner |
|---|---|---|---|
| 512³ | 5370 | 5103 | vendor +5.2% |
| 1024³ | 6905 | 6790 | vendor +1.7% |
| 2048³ | 7089 | 7349 | ours +3.7% |
| 4096³ | 6758 | 7236 | ours +7.1% |
| 8192³ | 6263 | 6290 | tie +0.4% |

Both peak around 2048–4096³ and decline after; neither scales past 4096³, which
points at L2/HBM traffic rather than compute. `naive` degrades monotonically
(1129 → 658) as the working set outgrows cache.

Both sustain ~7.1 TFLOPS at large sizes, roughly **23% of the ~30.7 TFLOPS
plain-FMA peak** (~61 TFLOPS is the dual-issue figure).

**Do not repeat the retracted claim.** An earlier version of this file recorded
regtile as ~2× faster than hipBLASLt. That was measuring an *untuned vendor
call*, not a fast kernel. Removing three defects in our own shim — per-call
workspace allocation, trusting the heuristic's ordering, and never setting
splitK/wgm — took hipBLASLt from 2497 to 5201 GFLOP/s at 512³ and erased the
lead entirely. Matching a tuned hipBLASLt is still a good result; it is a much
smaller claim than the one first recorded.

A vendor baseline that looks easy to beat is a bug in your harness until
proven otherwise.

## Hard-won facts — do not relearn these

**Mojo 1.0 API.** `fn`, `alias`, `let`, `inout`, `owned` are all gone; use `def`,
`comptime`, `var`, `mut`. Imports take a `std.` prefix.

**Mojo 1.2.0 rejects three spellings 1.1 accepted.** Each fix builds on both:

| 1.2.0 error | Write |
|---|---|
| recursive reference to `CStringSpan.__init__` | `String(unsafe_from_utf8=X)`, not `String(StringSlice(unsafe_from_utf8=X))` |
| `existing function with conflicting signature` on `read`/`write` | `external_call["__read"/"__write", ...]` |
| `package 'io' does not contain 'FileDescriptor'` | `from std.sys import stderr`; `print(..., file=stderr)` |

**`DLHandle` no longer exists.** FFI goes through `external_call`, and the binary
must be linked against the shared library at build time:
`-Xlinker -L<dir> -Xlinker -lamarbaro_shim -Xlinker -rpath -Xlinker <dir>`.
`mojo run`'s JIT will **not** resolve external symbols and `LD_PRELOAD` does not
reach it — build AOT.

**`Pointer` is non-null by design** and cannot hold a C handle that may come back
null. Carry opaque handles as `Int`, or use `Optional[Pointer[...]]`.

**Kernel scalar arguments must be fixed-width.** `Int`/`UInt` are not
`DevicePassable` — use `Int32` and convert inside the kernel.

**Accumulators must be SIMD values with comptime-unrolled indices.** Holding them
in `stack_allocation` gives you *scratch memory*, which on AMD lives in device
memory. The first register-tiled kernel did this and ran at 570 GFLOP/s —
**slower than naive**. Same algorithm, 6.4× apart.

**Benchmark at ≥200 iterations.** At 20, launch overhead dominated: both variants
reported roughly half their true throughput with ~10% run-to-run spread.

**CMake: link `hip::host`, not `hip::device`,** for a shim with no device code.
`hip::device` injects `--offload-arch` flags a non-clang host compiler rejects.

**t-strings reject format specifiers** — build JSON with String concatenation.

**hipBLASLt needs three things done right or it benchmarks as garbage.**
(1) Cache the workspace and selected algorithm on the context — a per-call
heuristic plus `hipMalloc`/`hipFree` costs more than the GEMM and read 1373
GFLOP/s, *below our naive kernel*. (2) The heuristic returns candidates in
predicted, not measured, order — time all of them. (3) **`splitK` and `wgm` are
only reachable through `hipblaslt_ext`**, not the C API, and are worth more than
algorithm choice: they took 3097 → 5201 GFLOP/s. Winning `splitK` decays with
size (4 at 512³, 1 at 2048³), consistent with small problems failing to fill the
GPU with workgroups.

**Never let the GPU idle inside a benchmark.** The card sits at ~28 MHz / 31 W
and ramps to ~3000 MHz in **~0.4 s of back-to-back work**, then holds. It drops
again the moment anything interrupts. Consequences, all measured:

- A fixed 10-iteration warmup (~10 ms at 512³) never leaves idle clocks.
- The bias tracks *measurement order*: whichever variant runs first is penalised
  most. Fixing it moved naive +33%, regtile +9%, hipblaslt +4%.
- **Each variant re-warms immediately before it is timed.** One warmup at the
  start is not enough — the host-side correctness check between variants is long
  enough to lose the clocks.
- Do not call `rocm-smi` inside a timing loop. Doing so idled the GPU between
  samples and produced a fake 11k↔27k GFLOP/s oscillation at half true speed;
  power read 31–54 W against a 290 W cap, which is the tell. Sample clocks from
  a separate process.

**MAX claims ~90% of VRAM on `DeviceContext()` creation.** It is a pool, not a
leak: it appears instantly, sits at ~22.7 GB whether the problem is 4096³
(192 MB of matrices) or 8192³ (768 MB), and stops at whatever is free. Cap it
with `MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT` — the bench harness
defaults it to 10.

| cap | VRAM held | regtile GFLOP/s (1024³) |
|---|---|---|
| 100% | 23.06 GB | **segfault in `hipblasLtCreate`** |
| 10% | 2.72 GB | 7703 |
| 2% | 0.88 GB | 7591 |

At 100% there is nothing left for hipBLASLt to allocate its handle and the
process dies. Capping costs no throughput — it measured slightly *faster*.
This is the constraint on co-tenancy: uncapped, one MAX process owns the card.

**fp32 WMMA does not exist on gfx1100 — it is the ISA, not Mojo.** Verified
three ways: `llvm-mc -mcpu=gfx1100` accepts `v_wmma_f32_16x16x16_f16` and
rejects `..._f32`; `BuiltinsAMDGPU.def` has zero f32-input WMMA entries; and
Mojo's own constraint reads *"RDNA WMMA does not support FP32 inputs (only
FP16/BF16 -> FP32)"*. **Any fp16 TFLOPS figure (aiter's ~82–89) is therefore not
comparable to this fp32 benchmark** — they use different hardware inside the
same chip.
(The aiter fp16 figure is aiter's Triton `gemm_a16w16` with a gfx1100 tuned table: 2026-10-09
on aiter main 2b812e5 it measured 79.9-85.3 TFLOPS at 4096^3 with a tuned gfx1100
`DEFAULT.json`, 20.7 with the copied gfx942 table, and fails outright with stock main,
which ships no gfx1100 GEMM table.)

**fp16 WMMA kernel — measured 2026-09-02 (`bench/bench_fp16_pipe.mojo`, `kernels/matmul_wmma_pipe.mojo`).**
Pipelined rewrite: 4x2 warps, 2x4 wave tiles (128x128 block, 256 threads), K32,
two LDS buffers, one barrier per K-step, **two-deep global prefetch** (tile k+2
in registers while k+1 waits in the second set), XOR-swizzled A and transposed B,
no pads, default VGPR cap (188 VGPR, 0 spills, 32 KB LDS = 2 blocks per WGP).
Interleaved race medians of 5 (`bench/race-fp16.sh`), vendor = hipBLASLt fp16 via shim:

| size | ours (fp16 C) | hipBLASLt fp16 | old `wmma_lds` | verdict |
|---|---|---|---|---|
| 512³ | 21619 (128x128) / **31841** (32x64 tile, 1-deep) | 27203 | 10948 | needs size dispatch |
| 2048³ | **93841** | 81698 | 56593 | +14.9%, ranges disjoint |
| 4096³ | **97957** | 89716 | 66197 | +9.2%, ranges disjoint |

Size dispatch landed 2026-09-02 (`6a4e38e`): the kernel takes warps/wtile as
parameters; the bench picks the tile from the number of 128x128 blocks the
grid would launch: >= 96 -> 128x128 8-wave, >= 64 -> 64x128 8-wave (2x4/2x2),
else 64x64 4-wave (2x2/2x2). `bench/fp16-templates.sh` (10 s warm-up, one build
per size, receipts in JSON), ours vs hipBLASLt:

| size | 256³ | 512³ | 768³ | 1024³ | 1536³ | 2048³ | 2560³ | 3072³ | 3584³ | 4096³ |
|---|---|---|---|---|---|---|---|---|---|---|
| ours | 6372 | 30642 | 64204 | 74824 | 93632 | 91300 | 97974 | 99307 | 105786 | 90705 |
| hipBLASLt | 6288 | 26324 | 54924 | 63623 | 69332 | 80203 | 87147 | 97224 | 85671 | 82437 |
| ratio | 1.01 | 1.16 | 1.17 | 1.18 | 1.35 | 1.14 | 1.12 | 1.02 | 1.24 | 1.10 |

Warm-up matters: at the old 1 s warm-up the same binary read 66k at 4096³
(clocks not settled); benches now warm 10 s and log `warmup_s`.

Clock under WMMA load (`bench/clock-probe.sh`, 2026-09-02, 4096³ 128x128 8-wave,
two runs): sclk holds 2559–2709 MHz, median ~2.6 GHz, at 278–320 W package
power and 72–77 °C junction. Lighter kernels hold ~3000 MHz at the same power
(the 44k kernel of 2026-09-01 read 3005–3008 MHz at 291–307 W), so the matrix
path is power-bound and the peak to measure against is 96 CU × 512 FLOP/clk ×
2.6 GHz ≈ 128 TFLOP/s (512 FLOP/clk/CU = AMD's 122.8 TFLOPS spec ÷ 2.5 GHz
÷ 96 CU), not 123 at spec boost or 147 at 3 GHz. The shipped kernel runs at
70–76% of that at 4096³ (89.5k/91.8k in the probes, 97957 race median) and 82%
at 3584³. That 15–20% is the whole remaining kernel-side headroom and none of
the textbook items claims it: LDS staging, XOR swizzle and edge-branch removal
are already in; fp16 fragments are already register-packed (the 4x4 config at
252 VGPR with 0 spills cannot exist unpacked: 128 acc + 128 bfr + 64 staging
before addressing); a vectorized epilogue is ≤1% at 4096³ (64 scalar stores
per wave against ~8000 main-loop instructions), and the D-fragment layout (a
lane owns rows 2i+half of one column) cannot widen a store without a permute
or an LDS-staged transpose.

What moved it, in order (each measured alone at 4096³): pipelining +5%; a 128x128
tile only once LDS fits two blocks per 64 KB WGP (pads out, swizzles in) +11%;
conflict-free transposed B +3%; dropping edge-bounds branches +6%; two-deep
prefetch +11% (the 1-deep register prefetch of 2026-09-01 was flat because it
had one LDS buffer and two barriers; depth pays only on top of double-buffered
LDS). Transposed B with *padding* was -16% (16-way store conflicts), which is
why the 2026-09-01 "NT loses" verdict was wrong about the cause. Vendor kernel
identity (`TENSILE_DB=0x8000`): `MT96x96x32 WG 4 waves MIWT3_3 PGR2 PLR1 TLDS1`.

**Two compiler facts that gate every kernel here.** (1) LDS is 64 KB per WGP
(`sharedMemPerMultiprocessor`), so a 36 KB block runs alone; 32 KB runs two.
(2) Mojo defaults `max_flat_workgroup_size` to 1024, which caps VGPRs at 192;
`@__llvm_metadata(`rocdl.flat_work_group_size`=StaticTuple[Int32, 1](NTHREADS))`
raises it to 256 (receipt in the code-object notes, `tools/isa-receipt.py`).
Per-config, not free: it lets 4x4 wave tiles compile spill-free (91k) but makes
the 8-wave 2x4 kernel 6% slower, and the 98k champion runs with it OFF (`LB=0`).

The 2026-09-01 numbers (66197 at 4096³, "occupancy-bound, five variants lose")
stand as history in the protocol file; the kernel `matmul_wmma_lds.mojo` is kept.
Step-by-step receipts: `bench/wmma-fp16-protocol.md`.

**fp16 WMMA in Mojo works, and the fragment shape is the trap.** RDNA3 wave32
wants **a/b = 16 wide, c/d = 8 wide** (matching
`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` typed `V8fV16hV16hV8f`). The
8-wide fragments `TensorCore.load_a` builds are the CDNA/NVIDIA shape and fail
with *"no valid implementation of mma"*. Measured lane mapping:

```
A: lane L, elem i(0..15) -> A[L % 16][i]
B: lane L, elem i(0..15) -> B[i][L % 16]
D: lane L, elem i(0..7)  -> D[2*i + L // 16][L % 16]
```

D is not the obvious layout — consecutive `i` steps two rows and the half-waves
interleave. Verified 16×16×16 tile: 0 mismatches, max_err 5.96e-07. Working
probes in `.work/wmma/`. Also: `get_mma_shape` has **no working RDNA entry** for
any dtype/shape_id — pass `Index(16,16,16)` to `TensorCore` explicitly. And the
whole `TensorCore` surface takes `LayoutTensor` (built via `Layout.row_major`),
which does not unify with the `TileTensor` our kernels use; calling `mma`
directly with hand-built fragments avoids both problems.

**gfx1100 is thinly tuned in hipBLASLt.** 8 fp32 (`SS_SS`) Tensile libraries vs
134 for gfx942; 95 total gfx1100 files vs 1111. The fp32 heuristic offers only
**4** candidate algorithms at any size. This is the contribution opportunity.

**MAX `DeviceBuffer.unsafe_ptr()`** yields a raw device address that the shim's
hipBLASLt calls accept directly — no copy needed to compare against vendor.

## Tuning findings

Small tiles win on RDNA3, contradicting NVIDIA-derived tiling guidance. Every
top-5 swept configuration uses BM=32 or 64; hand-picked BM128/BN128/TM8/TN8
**regressed to 1651 GFLOP/s** (0.45×), almost certainly register pressure
collapsing occupancy. Occupancy is buying more than data reuse here.

Do not trust large-tile intuition on this card. Measure it.

## Rules for kernel work

1. **Correctness first.** A fast wrong kernel is a failure. `bench/run.py`
   checks before it times.
2. **One number, one commit.** Every result is tagged with the commit that
   produced it. `bench/sweep.py` keeps its own `sweep.jsonl` because a 198-point
   parameter search is search output, not an experiment record.
3. **Numeric parameter search belongs in `bench/sweep.py`,** not done by hand. It is exact, free, and already beat a hand-tuned config by 1.38×.
4. **New strategies go in their own file** (`kernels/matmul_<strategy>.mojo`) and
   register a variant in `bench/bench.mojo`. Never edit another strategy's kernel.

### fp16 roofline on this card (2026-09-04, protocol Round 6-7)

WMMA-only microbenchmark under the 290 W cap: **R = 125.4 TFLOP/s**, 436
FLOP/clk/CU at 3.0 GHz (`bench/bench_wmma_peak.mojo`). The pipe kernel at
4096^3 is 89-91k GFLOP/s = 0.71 R at 2.6 GHz (0.82 R per clock);
hipBLASLt is 0.65 R at 2.9 GHz. The gap is 13% clock (energy per FLOP)
and 19% issue. Five levers were raced in Round 7 and none kept; details
and receipts in `bench/wmma-fp16-protocol.md`.

## Qwen3.8-27B Turbo decode (2026-11-02, lane turbokernel, `docs/lanes/turbokernel-report.md`)

Qwen3.8-27B-TurboFCFusion Q4_K_M (qwen35, H 5120, 64 layers, NQH 24, NH_V 48, FFN 17408, MTP head), engine built with the bake's shape
flags (`-D BARO_H=5120 -D BARO_N_LAYERS=64 -D BARO_NQH=24 -D BARO_FFN=17408 -D BARO_SSM_NH_V=48 -D BARO_YARN=0`), 20 mtp-prompts, 64 tokens,
f32 KV, TMAX 16384, resident arms (`bench/turbo-base.sh`, `bench/ab-resident.sh`):

| path | q4 pack (`--q4`, 17.2 GB) | kq trunk (`--kq`, 18.2 GB) |
|---|---|---|
| launch m = 1 | 38.15 | 34.42 |
| megakernel m = 1 | 44.40 (1.164x launch) | 34.47 (1.002x) |
| spec k = 2, launch window | 62.38 / mega 62.88 | VOID: the MTP draft reads q4/q8 only |
| **spec k = 2 with `-D BARO_TURBO=1`** | **68.72** vs 61.12 same job (1.124x, ids 19/20) | - |
| **+ R0/R2 small kernels (rowbw)** | same job 67.76 vs 65.73 (1.031x, host load 6-8, ids 19/20, the miss is a same-arm flip) | clean-box number not measured |

Byte floor 17.2 GB per token on the kq trunk (55.8 tok/s at 960 GB/s), 14.7 GB on the q4 pack after the host-side embedding. The m = 3
window is row-scaling-bound: `amar_matmul_skinny_q4rowb[2,3]` costs 1.41x / 1.29x its m = 1 self on the FFN shapes (P5 receipt
`bench/bench_turbo_rows.mojo`); `kernels/turbo_rows.mojo` (`amar_turbo_rows`, the A rows staged once per block in LDS, same fma chain,
every word equal, `kernels/test_turbo_rows.mojo`) takes gate 90.6 -> 80.8 us and down 100.7 -> 82.7 us at m = 3. Opt-in by build flag,
the bake carries it in `baro.run.build.flags`. Known on this shape: p03/p08-class prompts flip between two runs of ONE binary (same-arm
repeat `.work/turbokernel/det-q4-mega/`), so identity is read on the stable prompts and the kernel-level byte test is the strict check;
the megakernel's barrier share is 11.2% (5120 / 8 = 640 row groups over 96 blocks, the delta scan on 48 of 96 blocks), the next m = 1 lever.

## Megakernel decode path (default since 2026-09-06, `BARO_MEGA=1`)

Decode at m=1 without spec runs as ONE persistent launch per token
(`kernels/mega.mojo::amar_mega_token`, G=96 x 512 threads, bounded grid
barriers): all 32 layers + final norm + head GEMM + argmax. Prefill (m>1) and
the MTP window keep the launch path; `BARO_MEGA=0` restores it everywhere
(parity reference). Gate: `tools/mega-gate.sh` (build, kernel parity test,
run-tests, identity on every runnable pack with spec on/off, ref tokens,
3-run perf). Receipt 2026-09-06: launch 67.13 -> mega 81.98 tok/s_gen
(+22%), spread < 0.5%, all identities equal; `BARO_PROFILE=5` prints the
per-sub-block device profile, `BARO_DUMP=path` dumps X per layer for both
arms. Round receipts: `bench/megakernel-protocol.md`.

**Sampled decode on the megakernel since 2026-09-16 (lane-A6, `54fb4bb`,
`bench/chat-protocol.md` A6): no-spec T=0.7/top_p 0.9 132.30 tok/s_gen
against no-spec T=0 134.65 in the same stint (98.3%), was 108.32.** The 19%
"sampler cost" found by A1 was the megakernel bypass at `temperature > 0`;
the sampler kernel itself costs nothing measurable against the launch path's
argmax. A sampling request now runs `mega_token_*` with `fold_head = 0` and
the launch-path head plus `amar_sample_row` after it; T<=0 is byte-identical
(force-ab 20/20). Penalties, top_logprobs and grammar requests take the same
route since `39f6a4c` (A6.3): penalized T=0 109.85 -> 133.38, top_logprobs
107.45 -> 129.80, identity 20/20 on every arm (`bench/pen-ab.sh`); grammar cost
1.253x -> 1.010x (`bench/grammar-cost.py`), 66/66 valid. Only a plain greedy
request folds the head. The speculative window stays on the launch path:
the q4 window megakernel (`BARO_MEGA_WIN=1`, now correctly instantiated for
the q4 pack) measured 0.761x of it on k=2 spec (115.39 vs 151.68, A6.4), worse
than the q8 record (W2/W3). Report `exchange/2026-09-16-A6-report.md`.

**KV cache paged through a block table since 2026-09-17 (lane-A2 step 1, `09a8cd9`,
`bench/a2-protocol.md`).** 128-token pages; every KV kernel reads the physical page from a device
table (`kvtab_d`, identity for a single request, `BARO_KVTAB=reverse` is the gate arm that proves
the indirection is live). Forced agreement 60/60 at 8k/16k/32k under both tables
(`bench/a2-gate.sh`), decode after 32k 100.7 vs 101.6 unpaged, short context 0.994x, state files
saved in logical page order. `serve/kvpage.mojo` holds the allocator A3 will use.

**int8 KV since 2026-09-17 (lane-a2s2, `80d4b2c`, opt-in `-D BARO_KVQ=int8`):** per-row int8 with
page-tail scales, 16640 bytes per token against 65536 f32. Bar (P14) from llama.cpp q8_0 KV (min
93.8%) and our bf16 (min 96.9%): `MIN_PCT` 93.75. int8 forced agreement min 96.9% at 8k/16k/32k under
identity AND reverse tables, decode after 32k **117.69 vs 100.54 f32 (1.171x)**, short context
1.011x, RULER niah_single 100.0 at 64k and 128k, `BARO_STATE_SAVE` refuses int8 (step 3). The default
f32 build is ISA-identical to the pre-merge engine. Report `exchange/lane-A2S2-report.md`.

**ComfyUI node since 2026-09-17 (`exchange/lane-COMFY-report.md`):** `~/Projects/imports/ComfyUI/custom_nodes/comfyui-baro/`
(own git) adds BaroChat and BaroJSON over `baro-serve`, time-sliced: the MAX runtime holds about 22 GB
per engine process whatever the pack (5.3 GB pack, 21.75 GB held), so the engine runs between
ComfyUI's model loads (`/free`, engine under gpu-wait per request, PR_SET_PDEATHSIG child). Gates:
identity 64/64, an image whose PNG carries the expanded prompt, BaroJSON 41/41 valid, kill -9 orphan test.

**Same-stint bar against llama.cpp (2026-10-12, `bench/llama-stint-protocol.md`): ours 148.89 vs llama.cpp master
`08ffba21e` 111.34 tok/s, 1.337x, ours faster 20/20**, llama.cpp `1744c6bde` 112.04 in the same job, drift control
0.9977; Q4_0-pure both sides, no spec, llama.cpp fast config (fa on, f16 KV). Receipts `exchange/receipts/llama-stint/`.

**q4 m=1 champion since 2026-10-12 (`c5034a4`): 148.80 tok/s_gen no-spec, 20-prompt median**, identity 20/20,
every prompt 1.017-1.029x, against 145.37 for the pre-change build in the same job (ratio 1.024; load 5.4-6.8
recorded; receipt `exchange/receipts/loop-008/champ-main.log`). The change removes 88 grid barriers per token from
the megakernel (FFN gate+up+swiglu in one phase, attention output gate folded into the head write, SSM conv /
scalars / l2norm folded into the delta phase), same per-element arithmetic order. It is the first change written
by the self-optimising loop's proposer (opus, iteration 008 positive control, `bench/loop-protocol.md`). Previous:
144.50 (`76df085`, 2026-10-03, q4_dot_lds UNROLL 2 -> 1, `exchange/megabytes-prereg-2026-10-03.md`), and below.

**q4 m=1 champion since 2026-09-11 (`3824e20`, merge of lane-dattn): 136.37
tok/s_gen no-spec, 20-prompt median** (was 133.9). Post-merge A/B against
pre-merge main (`.work/engine-premerge`, same pack and env): 127.98 -> 136.37
(1.066x), identity 20/20; receipt `.work/postmerge-dattn/ab20.log`, merged
spread 4.4 % from one prompt (p15 130.8), every other prompt 135.9-136.7. Not
an attention gain (the new decode attention runs only above T = 1088): the
megakernel recompile re-rolled the q4 dot loop into a faster schedule,
`isa-loops` dual 114/78/78/53 -> 124/79/79/59, 0 scratch. Any later
`kernels/mega.mojo` edit can re-roll it back; read the fingerprint first.
Long context from the same merge: decode after p8192 112.70 -> 124.48, after
p32768 85.22 -> 101.36 (`bench/dattn-wire-protocol.md`).

**q4 m=1 champion since 2026-09-08 (`9e6feaa`): 133.9 tok/s_gen no-spec,
20-prompt median** (was 130.6). The per-phase rmsnorm+quantise and its grid
barrier are folded into the consuming GEMV's LDS prologue (`stage_rms`, 65
barriers per token gone) and the m=1 q4 kernel runs the chunked delta
(`RELOAD=True`, 239 VGPRs, 0 scratch). Bit-identical: gate 13/13, identity
20/20, q4 vs model-ref 64/64. Per token (`BARO_PROFILE=5`): 7498 -> 7358 us;
delta phase 464 -> 269. Known pool left by this round: the shared q4 dot loop
lost dual-issue pairs (102 -> 74) and gained `s_delay_alu` (184 -> 454), ffn
gate/up +180 us; read `isa-loops` (delay/dual columns) on every megakernel
build -- a loop that changes there with unchanged source is the register
allocator, and only the real-pack A/B settles it. `rocdl.waves_per_eu`
cannot be set through `@__llvm_metadata` (six spellings rejected).


## qwen35moe decode

**Expert tier pinned by default since 2026-09-17** (`BARO_TIER_PINNED=1`; `=0` for the page cache): experts in host RAM decode at 67.12 tok/s against 48.57 page-cached, identity 20/20, `exchange/lane-MOE3-report.md` item 0; the cost was the blocking pread, not the bus.

**Zero-copy misses, opt-in since 2026-09-17** (`BARO_TIER_ZC=1`, `404ac04`, `bench/moe-tier-protocol.md`): a missed q4_k expert is read by the gate/up and down kernels straight from the pinned store while each lane stores what it loaded into the cache slot, so the per-piece DMA before the launch goes away; 67.71 -> 72.02 tok/s_gen (1.064), identity 20/20, `kernels/test_moe_block.mojo` zc arm bit-exact. The three q6_k down layers keep the DMA. In-kernel PCIe reads need 8 or more pieces in flight (probe `bench/tier_zerocopy_probe.mojo`: 24.6 to 27.2 GB/s at 8 to 24 pieces, 14 to 15 at 2).

**Doorbell decode + byte-sized slots, default under ZC since 2026-10-09 (`7a0477a`, Strata port P2 + P3, `bench/moe-tier-cpu-protocol.md` round 4).** No per-layer `synchronize`: the GPU hands the router's ids/weights/x to the host through `hipHostMalloc(Coherent)` memory and a one-wave kernel spins until the host's LRU plan lands (P0 probe `kernels/test_doorbell.mojo`: 1.95 us round trip vs 23.0 us for kernel + copy + sync; MAX host buffers are non-coherent, 3.9 ms). Slots sized per layer in bytes: q4_k layers hold 73, q6_k 64, same VRAM. 20-prompt resident median **67.34 vs 63.76 tok/s_gen (1.056x)**, tokens identical 20/20, hit rate 0.765 -> 0.779. Host CPU contention lands directly on decode (a load peak of ~3.8 cost p04-p07 10-15% in one run). `BARO_TIER_BELL=0` / `BARO_TIER_BYTES=0` restore the old path.

**Expert store is one RAM copy, mmap + hipHostRegister, default under BARO_TIER_PINNED (lane P4, `exchange/lane-P4-MMAP-report.md`).** `experts.bin` is mapped read-only (MAP_SHARED) and registered ReadOnly|Mapped, so the page-cache pages are the pinned store instead of a second 18.3 GB pinned buffer filled by pread. Probe `kernels/test_hostregister.mojo` at the full 18.3 GB file: GPU read 28.82 vs 28.83 GB/s pinned, sum equal to the CPU sum. Peak AnonPages 23.2 -> 6.0 GB; token smoke ids md5 `92381a907c63` on both arms, 512-token prefill prompt equal across arms. H2D copies out of the registered range hung the doorbell stream's synchronize in-engine (not in the probe), so the q6_k down pieces go through a CPU memcpy into the staging ring. `BARO_TIER_CPU=1` keeps the pinned buffer; `BARO_TIER_MMAP=0` restores pread.

**CPU miss path `BARO_TIER_CPU`, wired, opt-in, slower than ZC (2026-10-09, `bench/moe-tier-cpu-protocol.md`).** `serve/expert_cpu.mojo` computes a q4_k tier miss's gate/up/down on the CPU from the pinned store while the GPU runs the hits (Strata's split): int8 per-32 activations through AVX-512 VNNI `vpdpbusd`, 8 workers (16 workers cost 183 us per `parallelize` call on the 7800X3D, 8 cost 9 us). Standalone 42-47 GB/s (`serve/test_expert_cpu.mojo`); in-engine 30.9 (round 1) and 34.3 GB/s (round 2). End to end against the ZC arm, 20-prompt resident median, identity (teacher-forced agreement) PASS every round: round 1 0.997, round 2 **0.806** (`49424e5`, deferred refill, q6 layers on ZC). rocprofv3 per token: B 9.81 ms of kernels vs A 15.29, wall 20.12 vs 19.17 ms, so the GPU idles on the host thread; that, not the CPU kernel or PCIe, is the limiter. Round 3 `BARO_TIER_CPU_MAX=K` (`5fbc636`) caps CPU misses per layer, the rest stay on ZC; its first sweep is void (arm A moved 49 to 66 tok/s between runs). Memory: the CPU path adds 58 MB RSS and 40 MB VRAM; RSS in pinned tier mode is the 18.3 GB store (19.43 GB). The store loads with 8 parallel preads since `153c9a8` (read 1.05 s warm, startup about 4.9 s from 5.4-7.5); the file's page cache still holds a second 18 GB copy (dropping it with fadvise makes the next start read NVMe, 13.4 s). **Round 5 (2026-10-09, `serve/cpu_worker.mojo`, `bench/moe-tier-cpu-protocol.md`):** the CPU misses run on a pthread (spin, `usleep(100)` after 2 ms idle) so the host only plans: host plan time per layer 105.8 -> 18.5 us, tokens bit-identical to Round 4 (A1 and B1, 20/20), agreement PASS (83.05 / 82.89 mean), but B1/A1 **0.842** (load 6 during B; smoke at load 3.4 shows no change from Round 4): the host was never the limiter. The CPU job (172 us in-engine, 60 standalone) is on the critical path because the GPU hit kernels after the plan are shorter than it.

**Champion (2026-09-15, R6.0b `38ee0b7`, `bench/moe-persist-protocol.md`):
111.89 tok/s_gen, 20-prompt median** (spread 1.2%), launch path
(`BARO_MEGA=0`), no spec, 727 launches per token. Up from 107.28 (R6.0
`1e270b5`, 857 launches) and 94.79 (R4 `7145b71`, 1117 launches), all
bit-identical to the previous engine on 20/20 prompts, fail word 0. R6.0 and
R6.0b fold per-layer elementwise chains into fewer launches; R6.0b landed at
1.043x, below the preregistered +5% kill line, on the maintainer's recorded override.

**Persistent MoE token kernel (2026-09-16, R6, `kernels/mega_moe.mojo`,
opt-in):** `BARO_MEGA=1` on the MoE profile runs the 40 layers in one launch
(8 launches per token with the head), bit-exact with the launch path (dump
compare 64 tokens x 80 slots identical, teacher-forced 64/64 on 20/20 vs
`38ee0b7`). 20-prompt A/B 110.90 -> 113.94 tok/s_gen, ratio 1.027. It stayed off only
because of a +5% minimum-gain line that is void since 2026-10-09 (CLAUDE.md: any measured,
identity-holding gain lands); default `BARO_MEGA=0` until the 2026-10-09 rewrite is measured.
Refuses `BARO_TIER` and `BARO_EXPERTS`. Receipt that it ran: the engine's
`mega barrier gen` line, 470 per token. Details and the remaining pools in
`bench/moe-persist-protocol.md` R6.2 and `exchange/lane-R6-report.md`.
llama.cpp on the same GGUF: **109.92** (20-prompt median, 109.49 to 110.01).
The same-stint ratio is only for R6.0 (106.95, 0.973x); 111.89 was measured in
a different stint, so the 1.018x it implies is not a same-stint receipt.
Receipts: `.work/moe-perf/lc-r60b-main.log`, `ab-r60b-main.log`.

R6.1 (`f900bbb`, 2026-09-16): a one-launch-per-token MoE kernel
(`kernels/mega_moe.mojo`, `BARO_MEGA=1`, default off) is at parity (per-layer
dump identical, forced identity 20/20 at 64/64 against `38ee0b7`). Its speed
is not measured yet (R6.2, `exchange/lane-R6-report.md`).

**History, round 1 (2026-09-15, `bench/moe-perf-protocol.md`).** RegesCore-35B, pack `.work/moe-w1/pack` (Q4_K experts, Q8_0 projections, Q6_K
head), launch path (`BARO_MEGA=0`), no spec. **93.46 tok/s_gen, 20-prompt
median** (was 42.88), three landed arms in one session: vectorized Q4_K
expert dot (`8130f65`), vectorized Q8_0 row dot (`d49bfc3`), one-wave router
top-8 (R3). llama.cpp 109.4 on the same GGUF. Teacher-forced agreement
53.15/64 mean (band 52.70..53.70 around the W3 bar). Every MoE weight path
used to dequantize one element per lane per load; the bytes-per-token floor
is about 3.2 ms (2.7 GB), so the remaining gap is launch-bound small kernels
in the SSM sub-block and expert time, not bytes.

**Served path (2026-09-15, `bench/moe-served-protocol.md`).** The same binary
behind `baro-serve`, one `POST /v1/completions` per prompt, prompts as token
ids, 64 tokens: **92.89 tok/s_gen 20-prompt median (range 90.56 to 93.07),
0.993x the one-shot 93.52 (range 86.97 to 94.11) measured in the same stint,
and 74.74 tok/s over the whole HTTP round trip** (range 49.81 to 83.92).
Served tokens are identical to the one-shot tokens on 20/20 prompts. Per
request the medians are prefill 159.3 ms, decode 0.678 s, wall 0.843 s, so
serving costs 5.5 ms (HTTP, JSON, the line protocol, queueing) and the rest
of the gap to the decode-only number is prefill. Quote 92.89 against another
engine's decode rate and 74.74 against another server's end-to-end latency;
they are not the same measurement. Falsifier, runnable on any RDNA3 card with
this pack: `bench/served-prompts.sh .work/moe-served/engine .work/moe-w1/pack
OUT "BARO_SPEC=0 BARO_MEGA=0"`; a served/one-shot ratio outside 0.98 to 1.02,
or any identity failure, refutes this row.

### Where a qwen35moe decode token goes (2026-10-09, `747e9ef`)

Read these three before choosing any MoE decode kernel to work on:

- `tools/bytes-per-token.mojo PACK_DIR`: bytes read per token by tensor class. q8d pack:
  dense q8 2.14 GB (79%), active experts 0.57 GB (21%), 2.71 GB/token, roof 295 tok/s at 800 GB/s.
- `bench/moe-token-profile.sh ENGINE PACK OUT [ENV...]` (under gpu-wait): per-kernel us/token,
  launches/token and wall-minus-kernel gap, by the two-length difference. At `747e9ef` resident:
  654 launches, kernel 7041 us, gap 2018 us, wall 9059 us/token. An isolated kernel test does not
  predict this (q8d batching: 1.35-1.68x isolated, 0.92x engine, reverted `428cb17`).
- `isa-hot` (iTools, gpu-telemetry) on `tools/isa-receipt.py ENGINE --dump DIR`: kernels ranked by
  v_cndmask select trees, spills and flat loads. Open: `amar_ssm_delta_step` spills 1082-1241
  scratch instructions per instantiation and runs only 32 workgroups.

Decode at `747e9ef`: resident (q8d pack, TMAX 1024) ~124 tok/s; tier A1 config 74.68 (20 prompts).

## Self-describing bakes 2026-09-15 (`8184f7d`, closure PASS from the file, nothing external)

- `Qwythos-9B-Claude-Mythos-5-1M-MTP-BF16-BARO-8184f7d.gguf` sha256
  `8a0d20fba26b34b0a...` (`.work/bake/shas.txt`): `tools/gguf-verify.sh` PASS
  64/64, rebuilt engine 118.9 tok/s_gen on p09 (spec on, one prompt).
- `RegesCore-1.0-35B-UD-Q4_K_S-BARO-8184f7d.gguf` sha256 `a3d719282d1c5aa89...`:
  PASS 64/64, rebuilt engine 93.7 tok/s_gen on p09.
- Both carry `baro.hw.*` (card, driver, ROCm, power cap, 20-prompt median,
  config, protocol), `baro.kernel.model`, and the vendored `latentos/`
  package in `baro.kernel.src.latentos/*`, so the closure rebuilds the
  harness with no path outside the file (`8184f7d` vendored LatentOS; the
  `aa3f147` bakes needed `~/AMDHQ/src` and are superseded).
  `tools/gguf-verify.sh MODEL.gguf` is the contributor's one-command check
  (`docs/amd-family.md`).

## Quality vs llama.cpp (2026-09-16, `bench/quality-protocol.md`)

Same GGUF both arms. Perplexity: WikiText-2 test, 8 x 512-token chunks, llama-perplexity's second-half
scoring window (2,040 tokens), one token stream from llama.cpp's tokenizer. Task: `bench/data/e8_tasks.json`
(100 GSM8K math + 20 JSON), thinking off, T=0, identical token ids to both arms, exact match. Ours = the
engine and pack `tools/baro serve` caches.

| model | engine | PPL ours / llama.cpp (ratio) | task ours / llama.cpp | delta | verdict |
|---|---|---|---|---|---|
| Llama-3.2-1B-Instruct-Q4_K_M | spark | pending (spark logprobs landed after this sweep) | 22/120 / 24/120 | -1.7 pp | task PASS |
| lily-cybersecurity-7b-v0.2-Q6_K | spark | pending (spark logprobs landed after this sweep) | 13/120 / 15/120 | -1.7 pp | task PASS |
| Qwen2.5-7B-Instruct-Q4_K_M | spark | pending (spark logprobs landed after this sweep) | 46/120 / 46/120 | +0.0 pp | task PASS |
| Qwen2.5-Coder-7B-Instruct-Q4_K_M | spark | pending (spark logprobs landed after this sweep) | 21/120 / 20/120 | +0.8 pp | task PASS |
| Qwythos-9B-Claude-Mythos-5-1M-MTP-BF16 | dense | 9.301 / 8.410 (1.106) | 70/120 / 80/120 | -8.3 pp | PASS |
| RegesCore-1.0-35B-UD-Q4_K_S | moe | 6.233 / 6.242 (0.999) | 11/120 / 12/120 | -0.8 pp | PASS |

- The Qwythos row compares our q4 pack (`baro.run.pack.flags = --q4`, what the 136 tok/s champion runs)
  against llama.cpp on the BF16 file: the 1.106 PPL ratio and -8.3 pp are mostly 4-bit quantization, not
  engine parity (the q4-both-sides rows land at 0.999 and within 2 pp).
- Not run this round: Ornith-1.5-9B, Qwythos-9B-v2, Spark-X2.5-4B (cut for time), Granite-4.2-3B (harness
  script edited mid-run, row void). Spark-family perplexity waits on a rerun now that `serve/spark.mojo`
  has logprobs.

## Dense imports through `model-import` (2026-09-17, `bench/dense-run.sh`, forced agreement vs llama.cpp)

`bench/dense-run.sh` judges `MIN_PCT` (default 90) since `b4271cd`; before that PASS meant "ran" and it
printed PASS on 0/64. Teacher-forced agreement, 20 prompts, min / mean.

| model | ours q4 vs llama q4 | notes |
|---|---|---|
| Qwen2.5-7B-Instruct | 96.9 / 99.4 PASS | bake `qwen2.5-7b-instruct-q4_K_M-BARO-b4271cd.gguf` verifies 64/64 from the file |
| Qwen2.5-0.5B-Instruct | 68.8 / 85.0 UNVERIFIED | was 0 before the q8 GEMV tail fix `2d1ebbe` (H=896 is not a multiple of the 512-lane tail step) |

The 0.5B gap is not the quant: llama.cpp's own q4_K_M against its bf16 reads 82.8 / 91.4 on the same
prompts (the class bar), and our pack built from the bf16 GGUF against llama.cpp bf16 reads 75.0 / 86.4.
The remaining 5 to 6 points are our numerics (bf16 activations; the 1B floor was 95.3, the 7B 96.9) or a
shape-specific defect on NQH 14 / NKVH 2 / H 896. Next step if wanted: the per-layer oracle comparison
(`tools/llama-oracle.py`) on one prompt. No 0.5B bake until then. Receipts: `~/Models/qwen2.5-0.5b-instruct-import{,2,3}/gate/`.

## Spark engine: native K-quant weights, f32 activations (2026-10-19, lanes sparkq4k + sparkf32, `bench/sparkq4k-protocol.md`)

`tools/engine-pack.py --dense` writes Q4_K / Q6_K matrices as their gguf blocks (no q8 requant) and `serve/spark.mojo`
runs them through `amar_gemv_kq` (`kernels/gemv_kq.mojo`, byte test `kernels/test_gemv_kq.mojo`); every GEMV input
(rmsnorm, attention and gate*up outputs) is f32 (`-D BARO_ACT_BF16=1` rebuilds the bf16 numerics). 20 prompts, no-spec
`tok/s_gen` medians, 64 generated tokens, f32 KV, TMAX 16384. Receipts `exchange/receipts/sparkq4k/`, `sparkf32/`.

| model (Q4_K_M) | pack bytes q8 -> K-quant | tok/s q8 -> K-quant bf16 -> K-quant f32 | vs llama.cpp CPU (min / mean) | vs exact model (f64 forward, `bench/sparkq4k-truth.sh`) |
|---|---|---|---|---|
| vibethinker-1.5b | 2455 -> 1825 MB | 315.7 -> 362.6 -> 385.8 (1.149x, then 1.050x, same-job pairs) | 81.2 / 96.8, p04 52/64 (the f64 model reads 52/64 there) | **100.0 / 100.0, 64/64 on all 20** (bf16 inputs 87.5 / 99.2) |
| MiniCPM5-2B | 3300 -> 2361 MB | 213.5 -> 216.3 -> 233.3 (f32 row from another job) | 92.2 / 98.0 PASS | not run |
| Qwen2.5-7B-Instruct | 9244 -> 6247 MB | 96.1 -> 130.0 (1.353x; f32 row not run) | 95.3 / 99.4 PASS (bf16 inputs) | not run |

**2026-10-26, Llama-3.2-1B-Instruct-Q4_K_M (wavegroup levers ported):** fused gate/up, split-K QKV, batched prefill,
rope_freqs, and split decode attention from 128 tokens (`BARO_ATT_SPLIT_T`). Same stint, 21 prompts x 64 tokens: spark
563.3 / wavegroup 571.1 / llama.cpp 416.1 tok/s before the attention split; split vs one-block 21 prompts g256 1.029x,
g64 1.005x, ids 21/21; 4K context 158.3 -> 422.0 tok/s, ids equal. Fused decode rope 1.028x (byte-equal). Stint after both:
spark 572.1 / wavegroup 572.0 / llama.cpp 414.3 tok/s (spark 1.380x llama.cpp), spark ids = wavegroup 21/21. Receipts `lab/wavegroup/docs/receipts/2026-10-26-*`.

The vibethinker gate is the exact-model arm (protocol amendment A1): llama.cpp CPU disagrees with an f64 forward of the
same gguf on 12 of 64 p04 tokens, so a bar against it was unreachable; it stays as a reported control row. Commands:
`bench/sparkq4k-ab.sh OUT` (q8 vs K-quant, three models, one job), `bench/sparkf32-ab.sh OUT` (bf16 vs f32 activations),
`bench/sparkq4k-truth.sh TRUTHDIR OUT` (exact-model arm; tokens from `exchange/receipts/sparkq4k/truth-gen.py`).

## Megakernel on K-quant packs (2026-10-26, lane mega-kq, `bench/mega-kq-check.sh`)

`amar_mega_token[KQ=True]` reads Q4_K / Q6_K rows of an `engine-pack --kq` trunk with `gemv_kq.kq_lane_part`'s
per-element order, so `BARO_MEGA=1` generates the launch path's ids. Ornith-1.5-9B Q4_K_M, 20 mtp-prompts, 64
tokens, f32 KV, TMAX 16384, launch and megakernel interleaved in one gpu-wait job (`.work/kq/mega11-check.out`):

| arm | tok/s_gen median (min / max) | ids vs launch |
|---|---|---|
| launch path (`BARO_MEGA=0`) | 96.87 (93.34 / 98.11) | reference |
| megakernel (`BARO_MEGA=1`) | 99.55 (98.99 / 100.02) | 20/20 equal |

How it got there (commit 4661cb3; the first wiring 257a33e read 56.65, 0.586x): the loads of two superblocks
of two rows are issued before any decode (`kq_lane_part_ld2`, depth `-D BARO_KQ_D`, default 2, the register
optimum, depth 3 spills more and loses), every single-weight kq phase pairs rows per wave, and the rms-fed phases
read a per-block bf16 copy of the normalized row from LDS instead of a grid-barriered `CurB`. Probe one prompt
with `bench/mega-kq-probe.sh ENGINE PACK LOG` (prints the per-phase stamps and the ids md5). Open: the ssm
delta scan re-rolls between builds (468 to 808 us, about 3% of a token); the chunked form (`RELOAD=True`)
reads 261 us but re-rolls gate_up, so the kq instance stays on `RELOAD=False`.

**On main (Mojo 1.2) the kq megakernel is slower than the launch path: 87.90 vs 98.29 tok/s** (`mega-kq-check`,
2026-10-30). The same source built with Mojo 1.1 reads 99.4 on p01, with 1.2 85.9; every kq GEMV phase is slower.
Cause open (`exchange/lane-MEGA-BARRIER-report.md`). Since lane mega-barrier (500a9fc) the kq instance computes z
during the delta scan, on the 64 of 96 blocks the scan leaves idle: same-job 20-prompt medians, ids 20/20
(`bench/mega-kq-ab.sh`), mega11 98.57 -> 100.44 on Mojo 1.1, 85.90 -> 88.13 on Mojo 1.2.

## Batched prefill on K-quant packs (2026-10-26, lane kq-prefill, `bench/llama-stint-protocol.md` run 9)

A kq trunk (`engine-pack --kq`) replayed the prompt one row at a time (7.6 s per 1024 tokens, run 8). `prefill_forward`
now routes each GEMM on the entry's kind; Q4_K / Q6_K matrices are dequantised into an f16 scratch with the GEMV's own
unpack (`prefill_rows.kq_dequant`) and multiplied by the f16 rows with spark's LDS-pipelined WMMA GEMM
(`kq_gemm`, `amar_matmul_wmma_pipe[TB_=1, CF16=0]`); the NH_V = 32 projections take a one-warp-wide tile. Scratch
+142.6 MB on kq packs only. Ornith-1.5-9B Q4_K_M, `bench/kq-prefill-stint.sh`, llama-bench pp on the same gguf,
receipts `exchange/receipts/kq-prefill/stint2/`:

| prompt tokens | ours (mean of 2) | llama-bench | ours / llama time | run 8 (replay) |
|---|---|---|---|---|
| 1024 | 0.288 s | 0.298 s | 0.97x | 7.625 s |
| 16384 | 5.036 s | 5.689 s | 0.89x | 124.2 s |

Not bit-exact with the replay (WMMA summation order), so the gate is agreement with a same-arm control:
`PREFIX=bench/prefill-prompts/p1024.tokens CTL_ENV=BARO_PREFILL=0 bench/ornith-agree.sh ENGINE kq kq OUT` reads
candidate median 93.75 / min 78.12 against control 93.75 / 78.12 (the min bar of 85 fails on the control itself with the
prefix). Decode untouched: `bench/kq-decode-check.sh` ids 20/20 equal main in both the replay and the prefill arm,
medians 99.21 / 99.41 vs main 99.36. Unit test `kernels/test_kq_prefill_gemm.mojo` + `tools/kq-prefill-ref.py` (fp64,
real pack rows, m = 1, 17, 128, 1024). Open: dequant is 12% of the chunk at 1K (`BARO_PROFILE=1`); the follow-up is a
K-quant decode inside the LDS GEMM (no scratch round trip).

## K-quant decode inside the prefill GEMM (2026-10-26, lane kq-mmq, `bench/llama-stint-protocol.md` run 10)

Run 9's f16 scratch round trip is gone: `amar_matmul_prefill_kq` (`kernels/matmul_prefill_kq.mojo`) decodes Q4_K / Q6_K
sub-blocks to f16 inside the B staging of the LDS-pipelined WMMA GEMM (the `matmul_prefill_lds` schedule), casts the
bf16 rows to f16 in the A staging and accumulates into the residual directly. Each 32-wide K step takes the two
16-runs that share one 16-byte nibble load, so every weight byte is loaded once; the header / scales load once per
superblock and the step loop unrolls per kind (`KQ_U_Q4K = 8`, `KQ_U_Q6K = 4`, swept). Kernel receipts
(`bench/bench_kq_prefill.mojo`, cold weights, m = 1024, vs run 9's dequant + pipe): ffn_gate Q4_K 1.43x, attn_gate
Q4_K 1.31x, attn_qkv Q6_K 1.15x, ffn_down Q6_K 1.04x. Scratch -142.6 MB on kq packs. Ornith-1.5-9B Q4_K_M,
`bench/kq-prefill-stint.sh`, receipts `exchange/receipts/kq-mmq/stint/`:

| prompt tokens | ours (mean of 2) | llama-bench | ours / llama time | run 9 |
|---|---|---|---|---|
| 1024 | 0.259 s | 0.298 s | 0.87x | 0.288 s (0.97x) |
| 16384 | 4.539 s | 5.662 s | 0.80x | 5.036 s (0.89x) |

Gates as run 9: agreement with the prefixed control 93.75 / 78.12 = control (PASS on the control-relative bar), decode
ids 20/20 equal main in both arms (medians 99.83 / 100.01 vs main 99.80), oracle PASS with run 9's numbers (the decoded f16 values are the same). Open: the Q6_K decode is
about 17% heavier per FLOP than Q4_K's and the K = 12288 down shape adds about 16% more; neither is alignment (forcing
aligned loads changed nothing) nor a workgroup tail (the 256-WG Q4_K shape costs the same per FLOP as the 768-WG one).

## Prefill tile tune, stage 0 (2026-10-09, lane `tile-tune`, `bench/tile-tune-protocol.md`)

Open. The prefill GEMM dispatch rule (`gemm_prefill_q4`) was tuned on the ffn shape
only; `bench/bench_tile_sweep.mojo` sweeps 6 bit-exact tile variants on all 5 dense
weight shapes. Predictions P1-P4 frozen in `472f1a4`. Stage 0 ran 2026-10-09 (all variants bit-exact
vs the rule; receipts `exchange/receipts/tile-tune/ts-*.out`; box shared with other
lanes, ffn spread up to 1.6x at m <= 128). Rule time / best time:

| m | q 8192x4096 | o 4096x4096 | kv 1024x4096 | ffn 12288x4096 | down 4096x12288 |
|---|---|---|---|---|---|
| 64 | 1.11 | 1.00 | 1.28 | 1.34 | 1.07 |
| 128 | 1.00 | 1.00 | **2.04** | 1.03 | 1.00 |
| 256 | 1.01 | 1.00 | **1.68** | 1.09 | 1.01 |
| 512 | 1.02 | 0.99 | 1.18 | 1.10 | 0.99 |
| 1024 | 0.96 | 1.00 | 1.15 | 1.03 | 0.98 |

P1 HIT (kv 1.68x at m=256, 1.18x at 512; winners wmma16x256 / wmma32x256 / lds64x128).
P2 MISS (o and down at parity m=128..256). P3 MISS (ffn: wmma16x256 1.50x at m=32,
inside a noisy band; stage 1 A/B decides). P4 HOLDS except kv (lds64x128 1.15x).
Stage 1 built (`d351fe2`): `baro.tune.pf` KV -> pack `tune.txt` -> `serve/tune.mojo` at
boot, opt-in per pack, malformed rows raise (`serve/test_tune.mojo` in ci-checks).
`bench/tune-ab.sh`, identity on every run of every table:
- table v1 (`tune-v1.txt`, receipts `ab1`): 1.004 / 1.009 / 1.013 at 128 / 256 / 512, **0.927 at
  1024, 0.945 at 8192**: the ffn_down m>512 pick (lds64x128) beat the rule only by noise in the
  isolated sweep and lost 5-7% in the engine.
- table v2 (`tune.txt`, keeps the rule unless a variant beat it; `ab2`): 0.998 / 1.001 / 1.000 at
  512 / 1024 / 8192.
- table v2, 10 paired runs (`ab3`, bar frozen first): 128 1.004 with B faster 9/10 = WIN; 256
  1.006, 7/10 = NULL.
Net: +0.4% prefill at 128 tokens, nothing measurable elsewhere.
Caveat (pane a lane pane, `exchange/gridfill-prereg-2026-10-09.md`, `0f05062` on main): the
sweep drifts ~30% within one job (down m=256 rule 720 us first, 938 us last in a
reverse-order rerun). Only the kv 1024x4096 rows (2.04x / 1.68x) clear that drift; the
q/o/ffn/down picks (< 1.3x) are drift-level choices. The paired A/B cancels drift, so the
+0.4% stands. GEMM is 86-91% of dense prefill: the lever is the GEMM kernels themselves
(15-38 TFLOP/s of ~120), not tile choice. The bake path (`BARO_TUNE_TABLE`
in tools/bake.sh, tools/baro pack build) is UNVERIFIED: no bake has been run through it.

## OrcaSAQ-27B bake expansion and CP512 result (2026-10-17)

`tools/baro` selected `engine-pack.mojo --q4` from the baked GGUF metadata. The Mojo port
already exists and this bake used it. Its Q4 path keeps `token_embd.weight` unquantized:
the input Q6_K tensor is 1,042,944,000 bytes, while the BF16 packed tensor is
2,542,796,800 bytes, an increase of 1,499,852,800 bytes (1.50 GB). The Python oracle
has the same behavior. A smaller embedding needs a new packed representation and a
matching embedding lookup path; see `exchange/lane-ORCA27B-VRAM2-BUFS-report.md`.

The separate `SEQ_CAP=4` allocation finding remains: `sstate_d` reserves 5,435,817,984
bytes. The CP512 prefill-cap candidate was rejected after balanced resident A/B job
`mvcr9rbez68e` stopped at 10 pairs with a 99% speed-ratio CI `[0.996679, 0.999936]`,
below parity. Identity differed at p03, which was not covered by same-binary controls.
The candidate's total-VRAM peak difference was provisional because arm-start values
varied. Measurements are in `exchange/receipts/orca27b-vram2-bufs/summary.txt`,
committed as `0926e74`.

## q4x (Qwen3.8-Flash-Next) HTTP, 16K, long-context decode (2026-10-18, lane q4x-serve16k)

Numbers behind `docs/CAPABILITIES.md` (q4x section), receipts under `exchange/receipts/q4x-serve16k/`, binary `.work/q4x-serve16k/q4x-as` sha256 `c19d34cf74ad9975`.

| what | number | gate |
|---|---|---|
| HTTP | p01 511 and p02 16000 ids equal the `Q4X_STDIN` ids, text equals detokenized ids, cancel at 148 ids, tmax 16384 | `bench/q4x-serve-check.sh MODE=http`, job PASS |
| 16K agreement (2 prompts x 32 ids, `Q4X_CHUNK=8192`, `Q4X_TMAX=16640`) | OURS 96.88%, CTRL 96.88%, self-control 32/32 | `bench/q4x-gate.sh`, bar ours >= ctrl - 2 pp, PASS |
| decode at 16384-token context | 77.90 -> 27.42 ms/token (B/A 0.352), ids equal | `bench/q4x-ab.sh` A B B A, 2 prompts, spread 0.6 to 1.2% |
| 16000-token HTTP generation, mean | 48.1 -> 20.2 ms/token | `bench/q4x-serve-check.sh MODE=http` (p02) |
| decode at short context, 20 prompts x 64 ids | 20.006 vs 20.041 ms/token (1.0017), ids equal 20/20 | `bench/q4x-ab.sh`, `Q4X_ATTSPLIT=0` vs 512 |
| split attention kernel at T=16384 | 4762 -> 514 us/call, 0 of 6144 bf16 outputs differ at 9 lengths | `kernels/test_q4x_attn_split.mojo` |

Open: one-forward prefill at 16640 spills to GTT (12 to 16 GB) and the third run faulted, so use `Q4X_CHUNK=8192`; transient GTT peaks of 15.2 and 4.2 GB in 2 of 6
chunked runs: cause found in lane open3 (2026-10-18), the card is full (23.4 of 24 GiB, margin 512 MiB) and the kernel evicts the engine's buffers to GTT; they were not harmless (prefill up to 582 s, one spike decode at 285 ms/token); `Q4X_ECACHE_MARGIN_MB` now defaults to 2048, 0 of 6 spikes against 6 of 6 at the 2K tier, +7% decode, the 16K tier was not re-run; the 16K A/B uses 2 prompts, not 20 (2 min of prefill each); the 20-prompt medians are at short context.
Next pool at short context (trace): launch-floor gaps 4.2 ms/token over 1244 dispatches, host waits after the router 2.4 ms/token, memcopies 3.0 ms/token.


## MoE prefill q6_k down kernel and the two prefill traces (2026-11-02, lane moeprefill, `bench/moeprefill-protocol.md`)

Traces (`bench/prefill-trace.sh`, p1024 minus p0128, one 1023-row chunk, receipts `.work/moeprefill/tr-*` on the lane):
- **MoE tier prefill is staging-copy-bound, not kernel-bound.** qwen35moe, tier-split pack, `BARO_TIER=64`, TMAX 16384:
  prefill_s 2.306 s for 1023 rows and 1.636 s for 127 rows, so about 1.5 s per chunk is the 18.3 GB expert staging copy
  (`expert_tier.stage_layer`, about 12 GB/s). The chunk's prefill kernels sum to about 0.31 s (14% of the wall). Top kernels per
  896 rows: `amar_ssm_delta_rows` 65.0 ms (30 launches), `moe_down_q6k_grouped` 38.5 ms (3 launches), trunk `matmul_prefill_lds`
  52 ms, `moe_wmma_q4k_grouped` 38.8 ms (gate/up + down), shared expert rows 42.6 ms, `moe_pairs_group` 23.0 ms, router 11.2 ms.
  `BARO_PF_GEMM=exact` costs 1328 ms per 896 rows (gate/up grouped 540, q8d rows 403, down grouped 197).
- **kq dense prefill** (Ornith Q4_K_M, 1023 rows): prefill_s 0.252 s, `amar_matmul_prefill_kq` 139 of 165 ms of prefill kernel
  time (84%), 87 ms of the wall outside kernels.
- The whiteboard's "MoE G2 resident 2860 tok/s at 1k" (2026-09-27) ran `.work/engine-pack-q4` (6.6 GB, a dense pack) per its
  own arm file; no resident MoE prefill speed has been measured. Resident does not fit next to the desktop (21 GB pack, 24 GB card).

Lever landed (`652d4c7`): `moe_down_q6k_grouped` (the three q6_k down layers of every prefill chunk, default path) decoded one
element per `q6k_value` call; it now loads 4 ql, 2 qh, the 16 scales and d once per 256-block per wave32 lane and keeps the m=1
`q6k_row_dot` element order and value expression. `bench/bench_moe_q6k_rows.mojo` at the chunk shape (1023 tokens, 1279 groups,
random Q6_K bytes): 0 differing bytes against `amar_moe_down_q6k`, **20856 -> 4674 us per launch (0.22x)**; `kernels/test_moe_rows.mojo`
18 outputs bit-exact; ISA 163 VGPRs, 0 spills, workgroup size annotated. Engine A/B `bench/moe-prefill-ab.sh` (tier mode, 20
prompts x 768 tokens, ABBA, same job): prefill_s median **0.9500 -> 0.9246 s, B/A 0.9732, paired median 0.9715, B faster on 20/20
prompts, 16 ids equal on all four runs of all 20 prompts**; in-engine rocprofv3 receipt for the three launches 45571 -> 12878 us (0.28x),
p1024 GENERATED equal across arms. Frozen predictions: P2 absolute band hit, ratio band 0.3 to 0.6 missed low
(0.22); P4 band 0.98 to 1.00 missed low (0.973, the copy overlaps less than assumed); P1 missed high (20.9 ms standalone vs 12.8 ms
in-engine per layer). Decode m=1 kernels untouched. Open: the exact delta scan (24% of MoE prefill kernel time, latency-bound serial
chain, bit-exact by order), the staging copy rate, and `moe_pairs_group` (one block per layer, 0.6 ms).

## MoE expert cache capacity diagnostic (2026-10-21)

A direct tier-only probe measured 22.35% expert misses and 0.1196 GiB fetched per decode step at ~5.5k context with cap64. Cap128 cut the long-request miss rate to 5.34% warm and used 10,443,816,960 B of cache. A/B token IDs matched, but the observed B/A 0.3753 speed ratio is void: this boot's PCIe root link was at 2.5 GT/s. `./run-tests.sh` stopped at a profile/pack mismatch in two attempts. No performance claim is supported; see [`exchange/lane-moe-miss-report.md`](../exchange/lane-moe-miss-report.md).

## MTP draft expert prefetch

The draft router predicts 7.36% of verify expert routes and would save 7.33%
of verify miss bytes across 40 completed prompt/K arms. Both fail the frozen
prefetch kill line (50% overlap, 10% saved bytes): STOP, no prefetch kernel.
Evidence and limits: `exchange/lane-mtp-prefetch-falsifier-report.md` on
`lane-moe-miss`; source receipt belongs to sibling `lane-draft-overlap`.

### MoE expert cache capacity, 16 GT/s link (2026-10-21, after the reboot)
Direct tier mode (`BARO_TIER`, pinned, ZC, `BARO_MEGA=0`), same job, 4 requests per arm, token streams identical
(938 ids, `cmp`). Two runs: cap64 68.9 / 68.1 tok/s, cap128 91.4 / 91.5 tok/s, time B/A 0.75.
VRAM cost: the cache takes 5.22 GB at cap64 and 10.44 GB at cap128; with cap128 about 6.9 GiB stays free.
Use `BARO_TIER=128` when nothing else needs more than about 6 GB of VRAM; the default stays unchanged.
The 2.5 GT/s numbers in exchange/lane-moe-miss-report.md (B/A 0.3753) are VOID.

### q4x-host levers and warm dispatch fusion, quiet CPU, 16 GT/s (2026-10-21)
20 prompts, NG 64 unless noted, identity 20/20 in every run, ABBA; load average under 4 at start.
- `Q4X_DEVARG=1` (device argmax): B/A 0.959, prereg kill > 0.95: KILLED as a default (about 4%, stays opt-in).
- `Q4X_MISSPAR=3` (miss copies on 3 streams): B/A 0.993, prereg kill > 0.99: KILLED. Miss-copy gaps (2.35 ms/token in the
  16 GT/s idle table) do not shrink with more streams.
- Dispatch fusion warm, NG 256 (main vs pre-fusion build): B/A 0.889. Earlier B/A 0.907 (DEVARG) and 0.865 (MISSPAR) were
  measured while three CPU lanes held all 16 threads and are void.
