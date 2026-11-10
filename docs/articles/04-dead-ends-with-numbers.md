# Five things that did not work on RDNA3, with the receipt for each

The spec sheet says int8 tensor-core instructions on this card run at
2x the rate of bf16. I measured `v_wmma_i32_16x16x16_iu8` against
`v_wmma_f32_16x16x16_bf16`, register-only loop, interleaved, seven
repeats: **1.007x. Not 2x. Effectively identical.** That one number closed
an entire line of work before it started, and it's the cleanest example
of what this article is: five ideas that had a real, preregistered
prediction, a real measurement, and a real "no" — with the receipt kept
instead of quietly dropped. A perf repo that only shows what worked is
showing you survivorship, not results.

Same box as the rest of this series: AMD RX 7900 XTX, gfx1100 (RDNA3),
ROCm 7.2, Mojo 1.0.0 / MAX 26.5.0.

## 1. int8 WMMA does not get the spec-sheet 2x on gfx1100

`bench_wmma_peak_i8.mojo` isolates just the tensor-core instruction, no
memory traffic, no scheduling around it: both int8 and bf16 land at 133
T-ops/s, 512 ops/clk/CU. The "2x int8" figure that shows up in AMD's
marketing material is real for some instruction shape on some part of
this GPU family — it is not real for this instruction, on this card. The
practical consequence: an int8 x int8 GEMM cannot beat the same tile
schedule run in bf16 here, because the tensor-core throughput that would
have to carry the win isn't there. Halving the fragment size just adds a
per-block dequantization scale to the epilogue for nothing.

## 2. int8 MMQ prefill: confirmed, in a real kernel, not just a microbenchmark

I built it anyway, because a microbenchmark isn't the same claim as an
LDS-pipelined, quantized GEMM in the actual prefill path, and llama.cpp's
own MMQ kernel does beat a naive bf16 GEMM on other hardware — so the
question was whether the *schedule*, not the dtype, was where their edge
came from. Same schedule (8 waves, 2x4 wave tile, one LDS-staged K-block
per step, two buffers, XOR swizzle), instantiated once for int8 and once
for bf16, bit-exact against the existing prefill kernel on the bf16 arm.
Result: **the int8 form runs at 0.91x of the bf16 form on the identical
schedule** — slower, not faster. The B-operand load (the weight side) is
only half of the inner loop's traffic; the other half is the A-operand
(activations) and the dequant/scale epilogue, neither of which shrinks
when you quantize the weight. llama.cpp's real advantage on this card was
always the pipelined LDS schedule itself, which this kernel now also has
in bf16 — the dtype was never the lever.

## 3. Split-K decode: bit-identical, and 9% slower

Splitting each GEMM row's K-dimension in half — one wave computes the
first half, another the second, a lock-free handshake combines them —
looked like a way to add more parallel work per phase in the megakernel's
fixed grid. It's bit-identical by construction (both halves sum to the
same value in the same order as the unsplit form, 20/20 against the
launch path, 64/64 against the reference), and it shipped as a working,
correct kernel. It measured **119.0 → 108.1 tok/s_gen, -9%**. Every split
phase got slower — the SSM output projection +29%, the attention output
projection +45%, the FFN down-projection +18% — because halving the
K-dimension halves how much memory-latency-hiding work each wave has
in flight before it has to wait on the handshake; the per-phase fixed
cost that finer-grained work has to pay didn't shrink to match. The
first version of the handshake was worse still — an acquire-ordered load
that invalidated the whole compute unit's cache on every row, forcing
thousands of redundant re-fetches from L2 per phase — and even after
fixing that bug to a relaxed poll plus an atomic partial read, the
approach itself was still a net loss. Reverted; the shared inner-loop
body it introduced was kept because it was useful independent of the
split.

## 4. Kernel micro-fusion: three rounds, one marginal win, two reverted

Before the megakernel existed, the plan for cutting per-token launch
count was smaller: fuse pairs of adjacent kernels that write related
outputs. Three preregistered attempts, each with its own frozen
land rule:

| fusion | idea | measured | land rule | verdict |
|---|---|---|---|---|
| F1 | gate+up FFN projections in one kernel | +1.0%, spread 0.9% | ≥ +1%, spread < 5% | landed, at the floor |
| F2 | wider thread blocks for narrow-N SSM/attention GEMMs | 0.998x (flat) | ≥ +5% | **not landed** |
| F3 | fuse two SSM projections sharing an input row | +0.8% | ≥ +2% | **not landed** |

F1 is the only one that cleared its own bar, and it cleared it exactly at
the floor of its predicted range. F2's prediction was the most
aggressive of the three (+13-19%, reasoned from a bytes-per-region
model that said the SSM and attention GEMMs were only reaching 50% of
HBM) and it measured as pure noise — the model was wrong about where
the missing bandwidth was; the real gap turned out to be in non-GEMM
kernels inside those sub-blocks, not the GEMM launch itself. Three
rounds of a technique that only ever returns ~1% per pair, with real
failure modes, is what closed this approach entirely — the eventual win
on launch count came from a structurally different idea (one persistent
kernel replacing the whole sequence, not fusing pairs within it), covered
in the second article in this series.

## 5. Multi-row batching: wins in isolation, loses at the config that ships

An int8-dot FFN kernel that batches multiple decode rows through one GEMM
call is a real, measured win — **+7.3% on a 4-token speculative race
window, and 21% faster in prefill**. Wired into the actual speculative
decode path it ships in, at the k=2 draft width this engine actually
runs, it's a **loss**: the 20-prompt median went from 103.68 to 102.93
tok/s_gen, and token identity against the reference dropped from 20/20 to
16/20. `BARO_DOT` stays off by default.

The reason is acceptance, not arithmetic. A k=2 speculative window
verifies 3 rows at once (draft + 2 proposed continuations), which is
exactly the shape where this kernel's GEMM-side win is smallest and two
extra activation-quantization launches per layer cost more than the win
recovers. The kernel isn't wrong — it wins at m≥5 windows and in prefill,
where the batched-row count is large enough to amortize its fixed
overhead. It's wrong *for the window size this engine's own acceptance
rate produces*. Batching more rows per call only pays if something
upstream is actually going to accept that many rows, and nothing about
raising the row count changes what the model itself accepts.

## The pattern across all five

Every one of these had a number attached to it before it ran, and every
one of them is closed by a number, not a hunch — three by an explicit
land rule stated in advance, two by a preregistered A/B on the real
weight pack. None of the five is a "maybe later": int8 tensor-core work
is closed by hardware fact, not implementation quality; split-K and
micro-fusion are closed by their own measured cost; multi-row batching is
closed for the specific config that ships today and stays open for any
future config with a wider accepted window. If any of these five gets
re-proposed, the receipt above is the reason to ask what changed since —
not to take the second pitch on faith.
