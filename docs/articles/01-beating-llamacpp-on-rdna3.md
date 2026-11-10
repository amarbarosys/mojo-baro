# A from-scratch Mojo engine decodes a 9B model at 134 tok/s on a 7900 XTX — 1.2x llama.cpp, bit-identical

I built a decode engine for a 9B hybrid SSM+attention model (Qwythos-9B,
Qwen3.5-class, MTP head) in Mojo, targeting one GPU: the RX 7900 XTX,
gfx1100, RDNA3. No CUDA, no CDNA, no vendor library doing the heavy lifting.
On greedy decode at Q4_0, it runs at **133.9 tok/s_gen** — a 20-prompt
median, not a cherry-picked run — against llama.cpp's own Q4_0-pure build
on the same box, same prompts, at **110.0**. That's **1.2x**. Every token
along the way matches a from-scratch numpy reference, greedy, no drift.

This is not a toy kernel benchmark. It's a whole decode loop: tokenizer,
KV cache, RMSNorm, an SSM delta scan, grouped-query attention, SwiGLU FFN,
32 layers, running on hardware AMD's own library stack barely tunes for.
Here's what's actually fast, what I had to give up to get there, and what's
still behind llama.cpp — stated as plainly as the win.

**Hardware and software, so the numbers mean something off this one box:**
AMD RX 7900 XTX, gfx1100 (RDNA3), 24 GB GDDR6, wave32; ROCm 7.2; Mojo 1.0.0
/ MAX 26.5.0; 7800X3D host. llama.cpp is built and run on the same machine,
same GPU, all layers offloaded (`-ngl 99`), flash attention on, q8_0 KV
cache, 8 threads, 8192-token context — the same config for every comparison
in this article, not a default invocation.

## Why decode on this model is a memory problem

Qwythos-9B is a hybrid: 24 SSM layers and 8 attention layers out of 32,
plus an MTP draft head, 248320-token vocab. At batch size 1 every one of
those layers reads its weights from GDDR6 exactly once per generated
token and does a comparatively tiny amount of arithmetic on them — there
is no batching to amortize the read. That's true whether the layer is an
SSM delta-scan, a GQA attention block, or a SwiGLU FFN; the mix changes
which kernel runs, not the fact that the kernel is bandwidth-bound. Which
is why the two levers that actually moved the number are "read fewer
bytes" (quantization) and "waste less time between reads" (fewer kernel
launches) rather than anything about arithmetic throughput.

## The identity check, first

Before any speed number: does it produce the right tokens? Two separate
checks, and they're not the same claim. The engine has two execution paths
for a decode step — a sequence of per-kernel launches (the reference path)
and a single persistent megakernel (the fast path, more on this below).
Run both on 20 held-out prompts and they must generate the identical token
stream: **20/20**. Separately, the launch path is checked against a
from-scratch numpy fp32 forward pass over the dequantized Q4_0 weights,
greedy, one prompt: **64/64** tokens match. Neither check is the other —
one proves the fast path didn't diverge from the slow path, the other
proves the slow path is actually right. Both pass. That's the floor
everything else stands on.

## Wave-per-row: one thread-wave, one weight row

The GEMM is the whole game in decode. At batch size 1, every layer's
weight matrix gets read from GDDR6 exactly once and multiplied by one
activation vector — this is a memory-bandwidth problem wearing a
matrix-multiply costume, and the kernel that wins is the one that streams
weight bytes with the least waste, not the one with the cleverest tiling.

The kernel is `amar_matmul_skinny_q*row`: one GPU wave per output row,
walking the weight matrix in its native `[out, in]` layout so every lane
reads consecutive bytes — no transpose, no re-layout pass, no padding.
Weights are quantized (int8 block-32 scales for q8, packed nibbles for
q4); activations stay bf16. Measured standalone on the FFN shape, cold
cache, rotating buffers so nothing hides in Infinity Cache: **855 GB/s**
for the q8 form. That's not a synthetic peak — it's the actual kernel,
100.7 MB moved in 118 us.

Whole-engine bandwidth is a fuzzier number than that one kernel figure,
because "how many bytes does a token really touch" depends on which
tensors you count. An earlier pass through this repo's own numbers quoted
q8 decode at 74% of an "HBM roof" — that figure turned out to double-count
bytes the engine doesn't read every token (a 2 GB embedding table, an
unused draft layer) and mislabeled GDDR6 as HBM. Recomputed with the
correct byte count against the card's nominal 960 GB/s: **60.4%** for this
engine's q8 path, **65.2%** for llama.cpp's Q8_0 on the same prompts. Both
numbers are below what the earlier framing claimed, and neither is a
measured sustained-bandwidth ceiling — it's a spec-sheet denominator. I'm
stating the corrected numbers here because the wrong ones already made it
into an earlier internal doc, and I'd rather retract in public than let a
flattering-but-wrong percentage stand.

The q4 form of the same kernel is the one behind the headline number. It
packs weights as Q4_0 nibbles — two 4-bit values per byte, one fp16 scale
per 32-value block, bit-for-bit the same layout `llama-quantize --pure
Q4_0` produces, checked byte-equal before anything else was measured.
Getting the inner loop to match its own q8 sibling's instruction mix (an
`fma`-contracted multiply-accumulate form, not a separate multiply then
add) took a specific round of ISA reading, because two loops that look
identical in Mojo source can compile to different instruction forms and
silently diverge a few ulps deep — a bit-identity check catches it,
"looks the same in the diff" does not.

On q8 specifically, decode runs 68.8 tok/s_gen against llama.cpp Q8_0's
74.1 — **0.93x, behind**. The win in the headline comes from Q4_0: halving
the bytes read per token roughly proportionally raises the ceiling, and
the q4 kernel holds its efficiency well enough to clear llama.cpp's own
Q4_0 build. Two different quantizations, two different verdicts on the
same hardware — I'm showing both rather than only the one that flatters.

## One persistent kernel per token

The other lever is launch count. A naive decode step for a 32-layer model
issues on the order of 600 separate kernel launches — one per GEMM, per
norm, per elementwise op. Each launch has a real, measured floor on this
card (2.57 us, back-to-back, at held clocks), and 600+ of them add up.

The fix: fold the entire decode step — all 32 layers, final norm, head
GEMM, argmax — into one persistent kernel that block-strides across a
fixed, occupancy-sized grid, synchronizing between phases with a bounded
grid-wide barrier instead of a launch boundary. Every phase body is
bit-for-bit the same code as the standalone kernel it replaces, so token
identity holds by construction, and it does: 20/20 against the launch
path, every time it's been gated. Net effect on the q8 pack: **67.13 ->
81.98 tok/s_gen, +22%**. I go into how that was built — and the point
where a wrong barrier size hung the GPU hard enough to kill the desktop —
in the next article; here it's one lever among several, not the whole
story.

## What's still behind, stated plainly

Two things this engine does not win, and I'd rather put them in this
article than let the headline number stand alone.

**Speculative decode is behind on real text.** With an MTP (multi-token
prediction) draft head verifying two tokens per step, the 20-prompt
median is 100.7 tok/s_gen on the q8 track against llama.cpp's own MTP at
123.5 — **0.78x**. On q4 it's 129.25 against 169.5 — **0.76x**. Both
numbers are worse than my no-spec baseline's ratio against llama.cpp,
which tells you speculative decode is where llama.cpp's implementation
is currently just better, not where the underlying kernel work
transfers. It isn't a clean loss, though: on that same q4 prompt set
llama.cpp's own speculative output matches its own greedy output on only
7/20 prompts (median acceptance 75.5%, spread 39% run to run); mine
matches 20/20. It's slower and more consistent — both true, and I'm not
picking one to report. (An earlier internal number — 145.6 tok/s_gen,
1.33x ahead — came from a single 5-token prompt whose repetitive tail
inflates acceptance to ~94%. It's a real number, but it describes that
prompt, not the engine; the 20-prompt median is the one that counts.)

**Prefill is behind by a lot.** Filling a 32k-token context takes this
engine roughly 50 seconds against llama.cpp's 12.8 — three to five times
slower, depending on context length. Decode is where the kernel work in
this article lives; prefill is a different problem (batched GEMM at
large m, not memory-bound single-row streaming) and hasn't had the same
attention yet.

## The falsifier

Every number above traces to a commit in this repo and a script you can
re-run: `bench/mtp-prompts.sh` for the 20-prompt medians, `tools/mega-gate.sh`
for both identity checks, `tools/llama-mtp-prompts.sh` for the llama.cpp
side on the same prompt set. If you have a 7900 XTX (or any gfx1100 part)
and get a different ratio, I want to know — the config for both engines
(quant, thread count, KV type, draft width) is in `docs/BASELINE.md`, and
if it's stale where you're reading it, that's a bug in the doc, not in the
number.
