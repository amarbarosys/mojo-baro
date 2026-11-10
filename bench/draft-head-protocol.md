# A4 draft-head training protocol, frozen before first run

> **Binding: [`bench/PROTOCOL-RULES.md`](PROTOCOL-RULES.md).** P1: every arm
> parameter is read back from the running system and recorded before the
> timed run. No receipt, no arm.

Plan: `docs/NEXT-PLAN.md` A4. Brief: `briefs/2026-09-17-A4-draft-head.md`.

## Why the E13 null does not predict an A4 null

E13-mini (`~/AMDHQ` `030bd64`, NO SIGNAL: `d = +2.0` pp, held-out loss -2.5%
in 150 steps) trained a projector to solve a task the trunk cannot do at all
without help: read 8 of its own raw final-layer vectors, fed back as
substitute *input embeddings* for 8 autoregressive steps (out of the
token-embedding manifold, LatentMAS-style), then answer a multi-step math
problem tens of tokens later from that alone. The supervision is one
end-of-sequence correctness bit per item, the input distribution is
self-referential and never seen in pretraining, and the projector's target
(making an out-of-manifold vector readable as an embedding) is a genuinely
open research question: both cited precedents (2510.15244, 2609.00474)
needed real training to clear it on other model pairs, and this repo's own
E8 round 5 showed the untrained vectors carry nothing at all (10/100, chance).

A4 is the opposite shape on every axis that made E13 hard. The input is
`h_t`, the trunk's own final-norm hidden state at a real text position: the
exact vector the trunk's own `lm_head` already reads at every position during
normal decoding, not a foreign self-referential vector. The target is the
literal next token in real text, dense supervision at every position of every
document, not one bit per item after tens of intervening tokens. And the
architecture already has a real, pretrained answer to this exact question:
`blk.32` (NextN) ships in this checkpoint and already gets 42% second-token
acceptance out of the box (`bench/mtp-protocol.md`), so the untrained
starting point here is a strong prior on a linearly-decodable signal, not the
10/100-chance floor E13 started from. DeepSeek-V3's own published number for
this exact architecture family (85 to 90%, 2412.19437) says the ceiling is
far above 42%, on real precedent, not a hopeful analogy. None of E13's three
sources of difficulty (out-of-manifold input, sparse distant supervision,
unproven target readability) are present here, so E13's null carries no
weight against A4's hypothesis. This is stated, not assumed: it is a
prediction the smoke and full run either confirm or refute.

## Order step 1: shape and architecture verdict: FITS, no kernel change

`tools/engine-pack.py`'s `mtp_names()` packs 15 tensors for `blk.32` in a
fixed order (`e+0` .. `e+14`): `attn_norm, attn_q, attn_k, attn_v,
attn_q_norm, attn_k_norm, attn_output, post_attention_norm, ffn_gate,
ffn_up, ffn_down, nextn.eh_proj, nextn.enorm, nextn.hnorm,
nextn.shared_head_norm`, read verbatim by `serve/window.mojo`'s
`blk32_forward` at those same indices. Training this lane's head means
re-optimizing these 15 tensors' values; their shapes, their slot in the pack,
and every kernel that consumes them (`kernels/*.mojo`, untouched by this
lane's own rule) are unchanged by construction. There is no shape question to
resolve: the open question was only whether the training side could
reproduce blk.32's exact forward math faithfully enough to train it.

It can, cheaply, because `docs/mtp-notes.md` SS1-2 established that blk.32 is
"structurally identical to one full-attention trunk decoder block" (joint
QG projection with a sigmoid gate, per-head RMSNorm on Q/K, GQA, mrope RoPE,
dense SwiGLU FFN) wrapped in one new input stage (`eh_proj` over
`concat(enorm(embed(tok)), hnorm(h))`) and one new output stage
(`shared_head_norm` before the shared `lm_head`). HF `transformers` 5.16's
`transformers/models/qwen3_5/modeling_qwen3_5.py` already implements exactly
the full-attention variant of this block as `Qwen3_5DecoderLayer` /
`Qwen3_5Attention` / `Qwen3_5MLP`, the SAME classes the frozen trunk already
uses for its own full-attention layers (every 4th, `is_full = (bid+1) %
fa_interval == 0`), so blk.32's attention plus FFN needs no hand-written
torch reimplementation: instantiate one more `Qwen3_5DecoderLayer` of the
trunk's own `full_attention` config, wrap it with `eh_proj` / `enorm` /
`hnorm` / `shared_head_norm`, and load blk.32's real GGUF weights into it.

The blocker E13 did NOT have: `~/AMDHQ/tools/latent-os/gguf_to_hf_qwen35.py`
already contains this exact tensor-name-and-permutation mapping for every
full-attention layer (`build_state_dict`'s `is_full` branch, lines 162 to
168); it is simply never invoked for `bid=32` (`build_state_dict` explicitly
marks every `blk.32.*` tensor `dropped`, lines 206 to 208, and the HF
checkpoint confirms it: `model.safetensors.index.json` has zero `mtp.*`
keys, checked this session). Reusing that same mapping for `bid=32` is a
small, mechanical extension (same code path, one more block index), not new
design. Read-back receipt for this claim: `PYTHONPATH=~/llama.cpp/gguf-py`,
`GGUFReader` on `Qwythos-9B-Claude-Mythos-5-1M-MTP-BF16-BARO-04867e2.gguf`,
15/15 `blk.32.*` tensor names present with the shapes in `docs/mtp-notes.md`
SS1 (verify before training, not asserted here).

**Conclusion: no shape or kernel obstacle. The head trains as itself (its own
15 tensors, its own architecture), not as a foreign projector standing in for
it.** `bench/e13_projector.mojo`'s Linear-GELU-Linear shape is NOT reused
directly: it does not match blk.32's slot (single 8192->4096 `eh_proj` plus
a full attention plus FFN block, not a 4096->4096 MLP). What is reused is
E13's pattern: dump real activations from the q4 engine (`761472a`'s
mechanism, `final_norm_hidden` specifically, see below), train a frozen-trunk
torch side, torch-vs-Mojo parity check before trusting any gradient,
preemptible GPU slices (`627e8da`/`ca10db1`'s shape of
build-then-verify-then-train).

## Data: dense (h_t, token_{t+1}) pairs from real held-out text

`bench/e13_engine_dump.mojo`'s `collect_latent_raw` does NOT apply here: it
runs a k-step self-feeding latent rollout (E13's own out-of-manifold input),
not a read of the trunk's real per-position hidden state. The right primitive
already exists and is simpler: `serve/realign.mojo::final_norm_hidden`
re-derives `model.norm(residual)`, from `b.x_d` after any `step_window` call.
New tool `bench/draft_dump.mojo`: for each held-out document, `reset_and_load`
plus a per-token `step_window` loop (the same loop `run_to_prompt_end`
already runs), calling `final_norm_hidden` after processing token `i` to get
row `h[i]`. No new kernel: `final_norm_hidden` and `step_window` are
unchanged, existing device functions.

**Exact alignment (verified against both real call sites in
`serve/window.mojo`, not assumed).** Every real `blk32_forward` call passes
`pos == tok_pos == P` (both call sites, prefill and the self-chained draft
loop), with `hsrc = h_{P-1}` (the trunk's hidden state from BEFORE token
`P`), and its output is written to `Toks[P+1]` (confirmed from the draft
loop's own `tokcp_k(..., st.pos+j+1, ...)` write-back immediately after a
`blk32_forward(..., st.pos+j, st.pos+j, ...)` call). So blk.32 computes an
alternate, deeper path to the SAME `h_P` the trunk's own attention would
produce from `(Toks[P], h_{P-1})`, and predicts `Toks[P+1]` through the
shared `lm_head`, exactly mirroring what the trunk's own `h_P` already
predicts. For training: dump row `h[i]` (= `h` after processing `tokens[i]`)
pairs with INPUT token `tokens[i+1]` (`P = i+1`) and LABEL `tokens[i+2]`
(`P+1`), valid for `i` in `0..n-3`. An earlier draft of this protocol had
`tok_pos = pos+1`; that version's own GPU check (below) is what caught it.

Held-out text: the same real-text source already vendored for quality
sweeps and tokenizer work, not GSM8K, not the 20-prompt gate set (the gate
set stays a gate). 2000 (h, token) pairs for the smoke, drawn from held-out
documents distinct from anything scored later; the full run draws far more
(Order step 4, out of scope until the smoke passes).

## Training: frozen trunk, one `Qwen3_5DecoderLayer` plus the four wrapper
## tensors, dense next-token cross-entropy

`tools/mtp_head.py` (this lane, mojo-baro): `MTPHead` module is `eh_proj`
(`nn.Linear(2*H, H, bias=False)`), `enorm` / `hnorm` / `shared_head_norm`
(`Qwen3_5RMSNorm(H)`, same "+1" convention as the trunk's own norms), and one
`Qwen3_5DecoderLayer(cfg, layer_idx=32)` built with `block_type =
"full_attention"` forced (blk.32 is always full attention regardless of the
interleave pattern, per `docs/mtp-notes.md` SS1). Forward, over a whole
document's positions at once (ordinary causal batch, not incremental decode:
blk.32's own attention needs the causal history of its OWN eh_proj output
across the sequence, which a batched forward gives for free): `x =
eh_proj(concat(enorm(embed(tok_t)), hnorm(h_{t-1})))` at every position, one
`Qwen3_5DecoderLayer` pass with a standard causal mask, `shared_head_norm`,
then the FROZEN trunk `lm_head` for logits, cross-entropy against
`token_{t}` (dense, every position, not one label at the end). `embed` and
`lm_head` are the trunk's own frozen `token_embd.weight` / `output.weight`,
loaded read-only from the HF checkpoint (already verified vocab-identical to
the GGUF, per the E13-convert receipt). Weights for `eh_proj` / norms /
decoder layer initialize from blk.32's real GGUF values (extracted by
`tools/mtp_head.py`'s `extract_blk32`, the `gguf_to_hf_qwen35.py` pattern
mapping named above): fine-tuning the shipped head, not training from
scratch.

**Parity gate (must pass before any training step touches real gradients).**
`tools/mtp_head.py --mode parity`: run `MTPHead` (untrained, straight off the
GGUF) on a handful of real held-out positions and compare its logits'
argmax against the real q4 engine's own `blk32_forward` output on the SAME
positions (`bench/draft_dump.mojo --mode parity`, a small new dump reusing
`blk32_forward` exactly as `step_window` already calls it, capturing
`dtok_d` for known positions). A FAIL here voids the whole lane before any
GPU training minute is spent: an untrained torch replica must reproduce
what the untrained real head does, or nothing trained on the replica means
anything for the real engine.

**Result so far (this session, 6 documents from `bench/data/e8_tasks.json`,
69 positions, CPU torch replica vs the real q4 engine, `.work/a4/parity-dump.bin`
+ `.work/a4/parity-report.json`): torch argmax matches the engine's own
argmax 43/69 (62%); torch matches the true next token 7/69 (10%); the real
engine matches the true next token 9/69 (13%).** The 13% is well below
`bench/mtp-protocol.md`'s documented 42% baseline, and the void check this
protocol's smoke section requires (baseline within 5 pp of 42%) would FAIL
as measured here. Two explanations are open, not yet distinguished: (1) the
20-prompt set the 42% figure was measured on is natural-language chat text,
while these 6 documents are `e8_tasks.json`'s math/JSON items, a real domain
shift that could genuinely depress acceptance; or (2) a remaining wiring
defect in this session's `bench/draft_dump.mojo`/`tools/mtp_head.py`, on top
of the `tok_pos = pos+1` bug this same session already found and fixed by
running it (an earlier version's 45/68 torch-vs-engine and 2/68
engine-vs-true-next numbers, both now superseded). The 62% torch-vs-engine
agreement is not by itself strong evidence either way: consecutive real
hidden states are correlated enough that even the wrong alignment produced
a similar 66% figure. **Not resolved this session: rerun this same parity
check against the actual 20-prompt gate set through the real spec-decode
harness (`bench/mtp-prompts.sh`-class, `cfg.spec=True`, not a hand-rolled
`blk32_forward` call) to get an apples-to-apples reproduction of 42% before
trusting either number.** Confound named in advance, not hidden: the torch
replica runs float32 against the real path's q4 GEMMs, so a bounded,
non-zero argmax gap is expected regardless; how much of the residual gap is
that versus domain shift versus a remaining bug is exactly what the
gate-set rerun would settle.

## Smoke (frozen, Order step 3): kill line RESTATED against the reproduced
## statistic (2026-09-17, before any training minute)

The first version of this section framed the bar as "42% -> 50%", a
per-position figure this session's own real-engine rerun (`exchange/lane-A4-report.md`)
showed is not what `bench/mtp-prompts.sh` measures (that harness reports
`accepted`/`drafted` over full spec-decode windows, aggregate 65.97% on the
real 20 prompts, not a 42% per-token rate measured against literal prompt
text). The kill line is restated against the statistic that is now actually
reproduced, not the one that was never directly measured:

**Kill line.** Same harness, `bench/mtp-prompts.sh`, unmodified. Quick
5-prompt subset (`p01` through `p05`, `k=2`), untrained head and trained
head measured in the SAME STINT (one script invocation per arm, same
engine binary, same session, so neither arm's number is a stale receipt
from a different build): **the trained head's aggregate `accepted/drafted`
must be at least 4 percentage points above the untrained head's own
aggregate on the same 5 prompts, measured in that stint.** SIGNAL is that
lift; anything less is NO SIGNAL, not a kill of the hypothesis (same logic
as E13-mini's own verdict: 10 minutes of training cannot converge, so a
miss here says the probe did not show the effect, not that the effect is
absent). **Additionally, on the FULL 20-prompt set, greedy identity
(arm B's `GENERATED` == arm A's) must stay 20/20 PASS**: the accept rule
guarantees T=0 output is unchanged by construction (a rejected draft always
falls back to the target's own token), so a FAIL here means the write-back
or repack broke something structural, not an acceptance-rate question, and
voids the run regardless of the lift number.

**Void (checked first).** The untrained-head arm of this same stint's
5-prompt run is more than 5 pp off this session's own reproduced baseline
per-prompt numbers (`exchange/lane-A4-report.md`: p01 34/59, p02 39/50,
p03 32/62, aggregate 61.4% on p01-p03; p04/p05 not yet run at 5-prompt
granularity, recorded when the stint runs): the harness or build drifted,
rerun before scoring, never compare against a different day's numbers.

**Sequoia frozen prediction (Sequoia 2402.12374), recorded before any
training step:** expected tokens per verify pass `(1 - a^(K+1)) / (1 - a)`
over window cost `t(K)`, `a = 0.8`, `K = 2`: `2.44 / 1.45`. This is the
economic bar the FULL run (Order step 4) is judged against, not the smoke;
the smoke's own bar is the 4 pp lift above, stated separately because
150-step-class training (E13-mini's own scale) cannot reach convergence.

**Result: SIGNAL (>= 4 pp lift, identity 20/20 intact) leads to Order step 4
(preemptible full training, `gpu-wait run --preemptible --priority 10`,
3 GPU-hour budget, card shared with lane a3's gates which outrank training).
NO SIGNAL is reported as such, same as E13-mini: this probe cannot kill the
hypothesis, only say the minutes-scale check did not show the effect; a
longer run is the maintainer's call. A full-set identity FAIL kills the run
regardless of the lift number and is diagnosed before anything else, per
CLAUDE.md's own rule that a gate the candidate can write is not a gate: the
write-back path is new code this session, unverified until this run
verifies it.**

**Write-back receipt requirement (frozen here, checked at run time):**
`tools/mtp_head.py --mode writeback`'s patched GGUF must differ from the
source GGUF ONLY in the byte ranges of `blk.32`'s 15 tensors; every other
byte identical. Checked by a byte-level diff of the two files restricted to
outside those 15 offset ranges (zero differing bytes required there), not
by trusting the writeback code's own offset arithmetic.

## Recipe correction (frozen 2026-09-17, before this session's second
## training minute)

The first smoke's recipe (lr 2e-4, one optimizer step per document, no
accumulation, no clipping, no warmup) damaged the head: 5-prompt
`accepted/drafted` went 67.78% -> 3.06% (`exchange/lane-A4-report.md`), and
the degradation showed up identically in an exact-precision torch replica,
not only the quantized engine, so it reads as genuinely bad weights, not a
write-back defect. the maintainer authorized ONE corrected-recipe smoke, same
frozen kill line (5-prompt `accepted/drafted` lift >= 4 pp, full-20-prompt
identity 20/20 intact), with this recipe, frozen before any training step:

- **`lr = 2e-5`** (10x lower than the first attempt).
- **16-document gradient accumulation per optimizer step**
  (`~/AMDHQ/tools/latent-os/e13_train.py`'s own `--accum 16` convention):
  each document's loss divided by 16 before `.backward()`, one
  `optimizer.step()` per 16 documents, not per document. The first
  attempt's per-document stepping was a mismatch with the `lr` it borrowed
  from E13 (E13's `lr=2e-4` was always paired with 16-item accumulation;
  reusing that `lr` without the accumulation applied a step sized for an
  averaged, low-variance gradient to a single-document, high-variance one).
- **Gradient-norm clipping at 1.0** (`torch.nn.utils.clip_grad_norm_`,
  applied to the summed 16-document gradient, before `optimizer.step()`).
- **10% linear warmup**: the first `ceil(0.1 * n_steps)` optimizer steps
  scale `lr` linearly up to the target value, full `lr` from the step after
  that.
- **8 optimizer steps** (128 documents total, `16 * 8`), not the first
  smoke's single-pass-to-2000-pairs design: a loss TREND needs more than
  one or two points, and 16-document accumulation makes the earlier
  "~2000 pairs" budget too small to produce several real steps. Total
  supervised positions this time are therefore larger than 2000 (128
  documents at ~198 pairs/document average, per this session's own GSM8K
  dump receipt, is roughly 25000), a deliberate, stated departure from the
  first smoke's data-size framing, in service of the same kill line.

**Void, checked BEFORE the real-engine gate runs at all (frozen, not
negotiable after the fact): the mean per-document loss at the LAST
optimizer step must be lower than at the FIRST, or the run is VOID.**
A VOID run is reported as VOID; the write-back/repack/gate sequence does
not run against a checkpoint that failed this check, exactly as a bad
harness build is never scored against the kill line.

## P1 read-back before any timed run

Engine pack sha256 (patched-GGUF pack, once past the smoke); `BARO_SPEC`,
`BARO_DRAFT_Q4`; the draft head's own weight checksum as printed at load
(the same receipt `bench/mtp-protocol.md` already requires); GPU minutes
used per gate, reported in the lane report per `docs/NEXT-PLAN.md`'s GPU
rule.
