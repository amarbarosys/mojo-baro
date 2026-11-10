# One persistent kernel for a whole decode step on RDNA3: +22%, and a wrong barrier size took the desktop down first

A software grid barrier hung the GPU hard enough that every GL client on
the desktop died — eight terminal windows, the compositor, the panel — and
`sudo reboot` itself hung on the way out. That happened *inside* the round
that shipped a persistent, per-token megakernel for this engine's decode
path at **+22%** (67.13 → 81.98 tok/s_gen). Both things are true about the
same week of work, and the second doesn't happen without understanding the
first.

Same box throughout: AMD RX 7900 XTX, gfx1100 (RDNA3), ROCm 7.2, Mojo
1.0.0 / MAX 26.5.0, 290 W power cap, -100 mV undervolt, sclk held ~3.0 GHz.

## The verdict I had already closed

Two days earlier I'd measured the floor for a single kernel launch on this
card — 2.57 us, back to back, clocks held — multiplied it by the ~646
launches a decode token issues, and got 6.9% of a 41.7 tok/s bf16 token.
Below the round's own 8% stop rule. I closed it: *do not propose launch
fusion or a megakernel for decode tok/s*, and wrote that down as a standing
rule so I wouldn't re-litigate it.

Two things about that verdict changed under it without the verdict itself
being wrong at the time. First, quantization: the q8 pack moved decode from
41.7 to 68.8 tok/s, which means the token got shorter — 14.5 ms instead of
24 — so the same 646-launch floor is now **10.8%** of it, not 6.9%. A fixed
cost against a shrinking denominator grows as a fraction even though
nothing about the cost itself moved. Second, and this is the part the
original round hadn't measured: what a persistent kernel actually costs
to synchronize between phases, as opposed to launching a new one.

## The number that reopened it

A launch isn't free, but neither is a barrier. I measured a hand-rolled,
sense-reversing grid-wide barrier (an atomic counter plus a generation
word, `std.atomic.Atomic[u32, scope="agent"]`) against the same card's
2.43-2.57 us launch floor:

| arm | cost |
|---|---|
| launch floor, G=96 | 2.43 us |
| barrier @ 96 blocks | **0.60 us** |
| barrier @ 192 blocks | 0.75 us |
| barrier @ 384 blocks | 1.10 us |

A barrier is 0.25-0.45x the launch it would replace, at every grid size I
could resident-fit. That's the number that reopens a verdict I'd already
filed as closed: replacing a launch boundary with a barrier isn't
launch-for-launch parity, it's a discount, and the discount times 646
opportunities per token is what raises the ceiling to **7.7%** — projected
68.8 → ~74 tok/s from launch cost alone, before touching anything else. I
checked `hipLaunchCooperativeKernel`, the vendor-native cooperative-launch
path; it's supported on gfx1100 but buys about 0.1 us over the hand-rolled
version, not worth the API surface.

## What "resident" means, and the crash

A grid barrier only works if every block it's waiting on is actually
running at once — if the GPU can't fit the whole grid resident
simultaneously, a block that hasn't been scheduled yet will never reach the
barrier, and every block that's spinning on it waits forever. So the first
real question isn't "does the barrier work," it's "how many blocks can this
kernel's register footprint actually fit resident on this card." For this
kernel's VGPR count, the answer is 192 blocks. I know that number exactly
because I found it the wrong way first.

The protocol I'd written for this measurement said: probe at the predicted
ceiling, then probe higher, and the higher one should deadlock — that's
supposed to be the proof the ceiling is real. I launched a probe at 240
blocks, 48 over the ceiling, expecting a timeout. A deadlocked grid barrier
on a single-GPU desktop is not a timeout. The command processor never
drains the graphics ring, the ring times out, the driver issues a MODE1
reset, VRAM contents are gone, and every client holding a GL context —
eight terminal windows, the compositor, the panel — dies at once.
`sudo reboot` hung on compositor teardown; the machine needed a power
cycle to come back.

The fix, in order: compute the ceiling from the kernel's actual VGPR count
before ever launching a barrier (`floor(768 / vgpr_granule) waves per SIMD
× 4 SIMDs / 8 waves per block × 96 CUs` — for this kernel, 192); give every
spin-wait barrier a bounded spin count and a fail word, so overshooting
prints `NOT-RESIDENT` and returns cleanly instead of spinning forever; and
probe *upward* from the computed ceiling minus a safety step, never
downward from a guess. All three are now load-bearing rules in this repo,
not just this round's fix — a kernel that can, by construction, run forever
has to be able to give up.

## Building it in stages, each one a real measurement

I didn't write one 32-layer megakernel and hope. Each stage was gated
separately, on the real weight pack, before the next was attempted.

**Stage 0** asked a question that turned out to have a free answer: does a
fixed, occupancy-sized grid (block-strided over the work, G=96) cost
anything relative to the native `ceildiv(N, 8)` launch grid, before any
barrier is added? It doesn't — it's **5-16% faster**, on every GEMM shape
tested, because one block per CU with no tail wave beats the native grid's
uneven remainder. That's a gain the frozen prediction had assumed to be
zero.

**Stage 1** made one SSM layer — seven phases, six barriers — into a
single persistent kernel and timed it against the 14 separate launches it
replaces: 120.8 us → 87.1 us, **0.72x**, with the residual and conv-window
and SSM-state buffers checked bit-identical against the launch sequence on
random inputs.

**Stage 2** folded the whole token — all 32 layers — into one launch, head
and embedding still separate: **67.08 → 79.45 tok/s_gen, +18.4%**, 20/20
identity against the launch path. The first attempt at this stage actually
failed identity on 1 of 20 prompts — one token, deep into a 64-token
generation, off by 1 ulp — because an unrolled rmsnorm sum-of-squares let
the compiler contract the FMA differently than the loop form it was
replacing. Same math, different instruction selection, different rounding.
Reverting to the loop form fixed it and it's been bit-identical since; the
lesson (same order of operations is not the same result until you've
checked the compiled form) is one I'd already learned once on the q4
kernel and had to relearn here.

**Stage 3** folded the LM head GEMM and argmax into the same kernel — one
launch per decode token, period: **67.62 → 81.88 tok/s_gen, +21.1%**. The
headline number this shipped with, re-verified on a later stint: **67.13 →
81.98, +22%**.

One thing I tried and reverted: chunking the SSM delta-scan's register
usage to fit the tighter 192-block ceiling helped in isolation (a
synthetic 4-layer test said 17 → 10 us, equal totals) and *hurt* on the
real pack — interleaved A/B over a full stint measured 80.94 → 79.71
tok/s_gen, -1.5%, because the whole kernel's register allocation shifted
in response and the FFN GEMM loops (which never touch the delta scan)
picked up 180 us of worse scheduling elsewhere. A synthetic microbenchmark
and a real 32-layer kernel do not share a register allocator's decisions;
only the real-pack A/B is a verdict.

## What's left on the table

A later round folded the per-phase RMSNorm and its barrier directly into
the GEMM's LDS staging step — one fewer phase, one fewer barrier, 65 of
them gone per token — for another +2.6% on top of the q4 path (this is
part of the 133.9 tok/s_gen number in the previous article). The pool
that's still open: the shared int4 dot-loop's instruction schedule lost
dual-issue pairs across these rounds (102 → 74 VOPD pairs) and picked up
more `s_delay_alu` stalls, which is worth chasing but nothing in Mojo today
lets me set `waves_per_eu` to force the scheduler's hand — I tried six
different metadata spellings and the compiler rejected all six.

## The falsifier

Every stage above has its own gate and its own commit; `bench/bench_gridbar.mojo`
reproduces the barrier-vs-launch table, `tools/mega-gate.sh` reproduces the
identity checks end to end, and `bench/megakernel-protocol.md` has every
stage's frozen prediction next to its result, including the one that
missed. If your card's VGPR count differs from this one's, your residency
ceiling will differ too — compute it before you probe it.
