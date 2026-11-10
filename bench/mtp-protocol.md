# MTP draft-head protocol — frozen before first run (M5 item 5)

> **Binding: [`bench/PROTOCOL-RULES.md`](PROTOCOL-RULES.md).** P1 in particular:
> every parameter defining an arm is read back from the running system and
> recorded BEFORE the timed run. No receipt, no arm.


Question: does the model's own blk.32 draft head (NextN/MTP, `qwen35`
architecture, `nextn_predict_layers=1`) earn its launch cost in our
engine — i.e. does spending extra GPU launches on a draft-then-verify
loop beat plain greedy decode at our engine's own actual shape?

## Correctness precondition (gates ALL speed numbers)

No speedup number may be recorded until the draft head passes a
numerical parity gate against llama.cpp's draft logits (`ctx_dft`
`res->t_logits`, `common/speculative.cpp`'s `draft()` path, per
`docs/mtp-notes.md` §2-§3):

1. Top-1 argmax match on every position sampled during a fixed
   16-token draft trace.
2. Cosine similarity of the full logit vector >= threshold TBD-at-gate-
   time, held to the same bar `q8b` parity used (9.2e-4-class), recorded
   in the Result section — not loosened after the fact.
3. Max relative error bound recorded alongside cosine — cosine-alone is
   the known silent-corruption signature (high cosine, differing
   argmax = wrong token picked with a deceptively "close" logit vector).

Any one of the three failing voids the run: no tok/s, no acceptance
rate, no multiplier gets recorded that day. Fix the kernel, re-freeze
if the fix changes the instrument, re-run.

## Metric caveat (binds prediction 2 below)

llama.cpp's `tok/s` (`tools/server/server-common.h:404-427`,
`timings.predicted_per_second`) is generation-time-INCLUDING nothing
but decode: `(n_decoded - 1) / t_decode_s`. Our engine's own
`tok/s_gen` metric (see `bench/decode-race-protocol.md`) is computed
the same way, generation-time-EXCLUDING prefill — but the two are NOT
directly comparable across engines beyond what `decode-race-protocol.md`
already disclosed and froze. This protocol does NOT re-measure or
re-litigate llama.cpp's numbers. All predictions below are **self-
relative**: our-engine-with-MTP `tok/s_gen` vs our-engine-without-MTP
`tok/s_gen`, both measured by our own metric, same instrument, same run
session. llama.cpp's 44.1 / 109.8 tok/s figures (below) are cited only
as prior art informing the prediction ranges, never as the denominator
of any claim made here.

## Facts on the record (measured previously, not re-derived here)

- llama.cpp bar, frozen `adbb48c`, verdict `6b99693`
  (`bench/decode-race-protocol.md`): no-spec 44.1 tok/s (median of 4,
  spread 0.2%); with its MTP head 109.8 tok/s (median of 4, spread
  0.7%), draft acceptance 52/62 = 84% every run, multiplier 2.49x.
- HBM roofline ~960 GB/s / 17.9 GB-per-token weight traffic =
  53.6 tok/s non-speculative ceiling.
- Our engine, no MTP: 38.97 tok/s by our own metric, 64-token greedy
  output bit-identical to llama.cpp.
- GPU: llama-server holds ~22 GB VRAM at idle — MUST be down for this
  run, GPU exclusive (same discipline as `coldcache-protocol.md` v3/v4).

## Instrument

64-token gate run (`./.work/engine`, prompt "The capital France is",
greedy, `n_predict=64`), llama-server DOWN, GPU exclusive, existing
regression-gate prompt/tokens (`.work/engine-pack/ref-tokens.txt`).
Two arms, same process/binary build, same prompt:

- **Arm A (no MTP)**: current engine path, plain greedy decode.
- **Arm B (with MTP)**: blk.32 draft head wired per `docs/mtp-notes.md`
  §2-§4 (`process()`/`draft()`/`accept()` loop mirrored from
  `common/speculative.cpp`), same greedy sampling, same prompt.

Report per arm: `tok/s_gen` ((n_decoded-1)/decode_s, our own metric,
generation-time only), and for arm B additionally: draft acceptance
count/rate, drafted-vs-accepted token counts per verification window.
5 repeats per arm, discard first (cache/clock warm), report
median of remaining 4.

## Frozen predictions

1. **Stability**: per-arm spread on the 4 kept repeats < 5% of median.
   If spread exceeds 5% on either arm, no speedup claim may be made
   that day — re-run under the stability gate before recording
   anything (copy discipline: `coldcache-protocol.md` v2/v3).
2. **Acceptance rate**: 60-85%. Reasoning: llama.cpp gets 84% on this
   exact model/head; ours is a fresh implementation of the same weights
   through different kernels, so parity is expected but not assumed —
   range widened below llama.cpp's number to allow for real
   implementation slop while still requiring the head to be doing
   useful work.
3. **Speedup**: arm B `tok/s_gen` / arm A `tok/s_gen` in 1.6-2.4x.
   Reasoning: llama.cpp's own multiplier is 2.49x; our engine starts
   from a lower no-spec baseline (38.97 vs 44.1) and untuned draft-path
   kernels, so the range is shifted down and narrowed versus
   llama.cpp's observed ceiling, not copied from it.

## Falsifier

Arm B / Arm A speedup < 1.3x — the draft head is not earning its
extra launches in our engine; MTP work stops until the underlying
inefficiency (kernel launch overhead, draft-path fusion, acceptance
rate) is diagnosed and re-preregistered.

## Claim rule

A speedup number may be published ONLY if:

- the correctness precondition (top-1 AND cosine AND max-relative-error
  gate) passed and is recorded in the Result section, AND
- both arms' greedy 64-token output stayed bit-identical to the
  existing `ref-tokens.txt` gate — MTP is a launch-scheduling
  optimization, not a model-behavior change; if the output moves at
  all, the run is void regardless of how fast it was, AND
- per-arm spread stayed under the prediction-1 threshold.

Any deviation from this protocol (different prompt, different
`n_predict`, different sampling, GPU not exclusive) requires a new
preregistration before any number from that run may be cited as "the"
MTP result.

## Result

**Not yet run. This section is a placeholder — filled in only after
the correctness precondition passes and the instrument above executes
exactly as specified. No prose, no number, no verdict belongs here
until then.**

## Amendment 2026-09-04 (frozen before the draft loop is written, tree 87e0f81)

Base has moved: the engine is q8-only at **68.77 tok/s_gen** (`q8-protocol.md`),
64/64 identical to llama.cpp Q8_0 and bf16. Arm A = that binary/pack. Arm B
= same binary with `BARO_SPEC=1`, draft window k from `spec-k.txt` (4).
The correctness precondition of this file was met on 2026-09-01 (draft head
argmax and logits validated vs llama.cpp, `tools/draft-ref.py`); the DRAFT
line still prints `from_token 369 draft_argmax 369` on the q8 engine.

Design (mirrors `common/speculative.cpp` process/draft/accept, greedy):
- State per iteration: `pos` = last known token; trunk ring/KV valid through
  `pos-1`; `Hnm` rows hold the trunk's post-output-norm hidden for the last
  window; draft KV (`kc32/vc32`) valid through the previous iteration.
- **process**: one blk.32 pass, M = tokens not yet in the draft KV with
  truth h (token p pairs with h(p-1)): after prefill the P-1 prompt tokens,
  later the n_acc+1 accepted+bonus tokens. Its last row IS draft step 0:
  argmax -> d1, its own post-shared-head-norm hidden -> h for step 1.
- **draft**: k-1 further M=1 blk.32 passes at pos+j with (d_j, draft h_{j-1}).
- **verify**: trunk window M = min(k+1, remaining) rows at pos..pos+k over
  [Toks[pos], d1..dk]; argmax into `dtok_d`, host compares, n_acc = leading
  matches, bonus = argmax(row n_acc) written to Toks[pos+n_acc+1]; ring and
  pos advance by n_acc+1. Rejected rows' trunk KV / ring slots / draft KV are
  overwritten by the next window, never read.
- Greedy verification accepts iff equal, so the 64 tokens are identical to
  arm A by construction; any drift voids the run (claim rule unchanged).

Byte model per iteration at k=4, 84% acceptance (llama.cpp's rate):
verify 10.7 GB + process ~0.2 GB + LM head x4 (1.06 GB each) + 3 draft
bodies x 0.2 GB ~= 15.7 GB for ~4.4 tokens -> 3.6 GB/token vs 10.7 today.

| prediction | value | rule |
|---|---|---|
| acceptance (drafted->accepted) | 60-85% (unchanged) | recorded |
| arm B / arm A | **1.6x-2.4x** (110-165 tok/s_gen) | land >= 1.5x AND 64/64 AND spread < 5% |
| falsifier | < 1.3x: stop, diagnose, re-preregister | |

Both arms 5 repeats, discard first, median of 4; server down; `BARO_SPEC`
and `spec k` printed by the run are the P1 receipts, plus per-run
`drafted/accepted` counts.

## Result (2026-09-04, amendment arms; run on the tree that landed as the next two commits)

Correctness precondition: 64/64 on every run of both arms (10/10), plus
`BARO_SPEC_K=1` 64/64. Receipts per run: `spec k: 4`, `BARO_SPEC: True/False`,
`mtp: drafted 53 accepted 50 k 4` (arm B). bge-m3 llama-server stopped.
Logs `.work/mtp-gate/{A,B}{1..5}.log`, `summary.txt`.

| arm | kept 4 (tok/s_gen) | median | spread |
|---|---|---|---|
| A (`BARO_SPEC` unset) | 67.79 67.71 67.75 68.50 | **67.77** | 1.2% |
| B (`BARO_SPEC=1`, k=4) | 127.99 127.61 127.92 128.02 | **127.96** | 0.3% |

Speedup **1.888x** (predicted 1.6-2.4x): LANDS. Acceptance 50/53 = 94%,
above the 60-85% band (5-token prompt, repetitive 64-token greedy tail;
recorded, not claimed as general). llama.cpp MTP bar 109.8 -> ours 1.17x.

Two bugs found on the way, both would have voided the run:
1. First speculative window after prefill skipped the accept check
   (`prefill_done` gate), writing all drafted-row argmaxes unverified;
   token 5 wrong. Fixed with a per-window `win_spec` flag.
2. `amar_matmul_skinny_q8row[MR>1]` looped rows with a runtime `M`,
   dynamic-indexing the accumulator array into scratch: a 5-row window
   cost ~8x a 1-row pass (first arm B: 31.9 tok/s, falsifier fired).
   Compile-time `MR` loop with `r < M` guard; prefill 0.116 -> 0.046 s.

## Result 2 (2026-09-04, real prompts, k sweep, llama.cpp head-to-head)

Not preregistered; recorded because the 5-token number above does not
generalise. Set: `bench/mtp-prompts/` (20 prompts, 7-59 tokens, tokenized
by the model's own tokenizer), 64 greedy tokens each, one run per arm
(single runs: this is a ranking, not a claim). Identity gate = arm B
`GENERATED` equal to arm A's; passed 100/100 runs. Logs `.work/mtp-final/`,
llama.cpp `.work/llama-mtp-prompts/`.

Two engine fixes landed between the first real-prompt run and this table
(commit after this section): the batched q8row GEMM is now instantiated
at the window's actual row count (2, 3, 5) instead of the 8-row template,
and keeps 16-wide loads for those sizes. First run, k=4: median 82.2
(1.19x), two prompts below 1.0x. After: table below.

| k | median tok/s_gen | median B/A | min-max | median acceptance |
|---|---|---|---|---|
| A (no spec) | 68.3 | | | |
| 1 | 98.2 | 1.43x | 1.22-1.53 | 85% |
| **2** | **100.7** | **1.47x** | 1.20-1.80 | 69% |
| 3 | 94.7 | 1.39x | 1.03-1.84 | 58% |
| 4 | 92.4 | 1.35x | 0.90-1.85 | 51% |

Default k moves from 4 to 2 (`spec-k.txt`, `BARO_SPEC_K`). k=4 stays
better only on code/list/number prompts (p02, p04, p19, p20). The
5-token race prompt on this tree: k=4 145.6, k=2 125.8.

llama.cpp Q8_0 on the same prompts (`tools/llama-mtp-prompts.sh`, draft
n-max 4): no-spec 74.1, MTP median 123.5, ratio 1.66x, acceptance 58%,
and its speculative output differs from its own greedy output on 4/20
prompts. Ours at k=2: 100.7, ahead on 2/20 prompts, median 0.78x of
llama.cpp MTP. Verdict for the README: ahead on the preregistered
5-token race (145.6 vs 109.8), behind on real text (100.7 vs 123.5).

Where the remaining gap is (profile on p09, BARO_PROFILE=1): draft path
~2 ms/window; a 2-row trunk window still costs ~1.28x a 1-row pass
(bandwidth model says ~1.05x). SSM sub-block is per-row serial (5
launches x 24 layers per extra row); multi-row GEMM at QV=16 not yet
at the m=1 stream rate. Both are the next MTP levers, before q4.

## Amendment MSPEC step 1 (frozen 2026-09-11 before any trace run, tree `dc9e06b`)

Plan: `~/Brain/mojo/mojo-baro/briefs/2026-09-11-mega-spec-window.md`. Question:
is the launch-gap share of the speculative verify window large enough that a
multi-row megakernel for m = k+1 could pay? This amendment measures; it builds
nothing.

**Prior on the record.** `megakernel-mrow-protocol.md` (2026-09-06, q8 pack,
`BARO_PROFILE=3`, one prompt): 646 launches x 2.43 us = 1.57 ms = 6.0% of a
26 ms k=2 window, estimated from the launch floor, not measured on a timeline.
Its W3 round built `amar_mega_window[MR=3]` and closed it: the megakernel's m=3
GEMM phases scale 1.6x from m=1 vs 1.5x native, so it lost to the launch path.
Zero gaps is therefore an upper bound on what a window megakernel can buy, not
an expected gain. The default pack is now q4 (`.work/engine-pack-q4`), whose
trunk window is shorter than q8's, so the same launch count is a larger share.

**Instrument.** `.work/engine` built from this tree. For each of the 20
`bench/mtp-prompts/` prompts, one process per arm, GPU through `gpu-wait`:
- `A`: no spec, bare (identity reference, no-spec tok/s context).
- `B2`, `B4`: `BARO_SPEC=1 BARO_SPEC_K=2|4`, bare (the tok/s the bound scales).
- `T2`, `T4`: the same two arms under `rocprofv3 --kernel-trace --output-format csv`.

P1 receipts read from each run's own log, per arm: `BARO_SPEC:`, `spec k:`,
`BARO_MEGA:`, `BARO_MEGA_WIN:` (must be False: the window runs the launch
path), `pack q4 trunk:` (True), `mtp: drafted/accepted`, `tok/s_gen`; pack
identity = resolved `BARO_PACK` path plus sha256 of its manifest files, recorded
once per session. Identity gate: `GENERATED` of every B and T run equals A's.
A run failing its receipt or identity is void.

**Definitions (from the kernel trace, per spec iteration).**
- *Verify window* = the trunk pass of `step_window` under `win_spec`: from the
  start of its `embed_k` dispatch to the end of its `argmax_d` dispatch.
  Kernel count n, kernel sum S (sum of dispatch durations), wall W (span), gap
  share g = 1 - S/W.
- *Iteration* = from the start of the draft process pass (`blk32_forward`) to
  the start of the next iteration; adds the draft path and the host accept
  sync. Its idle share is reported beside g, not used for the decision.
- Per prompt, g is aggregated as sum(W - S) / sum(W) over its spec windows.

**Decision metric and kill line.** Median over the 20 prompts of the per-prompt
g at k=2 (the shipping default). **g < 10%: stop at step 1**, no build.
g >= 10%: report and wait for the maintainer's go; step 2 is not started by this lane
either way.

**Upper bound on the gain.** Per prompt, tok/s_ub = B tok/s_gen /
(1 - sum(W - S) / decode_s of the traced run). Reported as the median over 20
prompts for k=2 and k=4, beside the B median.

**Trace overhead check.** If the median T/B tok/s ratio is below 0.95, the
tracer is inflating gaps: the measured g is then an upper bound and says so
in the Result; the decision still uses it (an inflated g below 10% kills
harder).

| prediction (frozen) | k=2 | k=4 |
|---|---|---|
| dispatches per verify window | 600-700 | 600-700 (same code path, m=5) |
| verify window wall W, ms | 10-15 | 12-19 |
| gap share g, median | **7-17%, point 11%** | **5-13%, point 9%** |
| tok/s_ub / B | 1.05-1.15x | 1.04-1.12x |
| T/B tok/s ratio (tracer cost) | 0.93-1.00 | 0.93-1.00 |

Reasoning: ~650 dispatches at 1.5-3 us of dispatch gap each is 1.0-1.9 ms;
q4 is ~57% of q8's bytes, so the m=3 launch-path window is ~11-14 ms against
q8's 21.3 ms. Falsifier for the reasoning (not the decision): g outside
4-20% at k=2 means the per-dispatch gap model is wrong and the Result says
what the trace shows instead.

### Result MSPEC step 1 (2026-09-11, tree `7df2c61`, engine sha256 `ddbb7fcc`)

Receipts (`.work/mspec/run1/receipts.txt`): 100/100 speculative runs read back
`BARO_SPEC: True`, `spec k: 2|4`, `BARO_MEGA: True`, `BARO_MEGA_WIN: False`,
`pack q4 trunk: True`; identity PASS 100/100 against arm A's `GENERATED`; pack
`.work/engine-pack-q4`, sha256 index `d2c11dff`, pack.bin `7e44a51f`. Traced
and bare runs of the same arm drafted and accepted the same counts. Per-prompt
table `.work/mspec/run1/summary.md` (`bench/mspec_gaps.mojo`); a Python csv
oracle with the same definitions (`oracle.txt`) gives the same g on all 40
traces, 0 malformed windows. `rocprof-kernels` was not the instrument: it
aggregates per kernel and has no timeline segmentation, which the per-window
gap share needs.

| median over 20 prompts | k=2 | k=4 |
|---|---|---|
| dispatches per verify window | 678 | 678 |
| verify window W, ms | 14.27 | 19.31 |
| **gap share g** (min-max) | **15.65%** (15.49-16.23) | **14.95%** (14.69-20.51) |
| g_lo, tracer cost charged to gaps (not preregistered) | 8.35% | 9.21% |
| host accept sync per window, us | 58 | 60 |
| B tok/s_gen | 150.96 | 127.01 |
| tok/s_ub (ub/B) | 174.71 (1.16x) | 144.68 (1.14x) |
| ub_lo/B | 1.08x | 1.08x |
| T/B (tracer cost) | 0.93 | 0.95 |

Arm A (no spec) median 137.02 tok/s_gen: on q4, k=2 speculation is 1.10x and
k=4 is 0.93x of no-spec. No llama.cpp arm ran this session, so no ratio against
it is claimed (P4).

Scoring: dispatches HIT/HIT; W HIT / MISS (19.31 vs 12-19); g HIT (point 11
missed by +4.7 pp) / MISS (14.95 vs 5-13); ub/B MISS/MISS (1.16, 1.14 vs
1.05-1.15, 1.04-1.12); T/B HIT/HIT. Reasoning falsifier (g outside 4-20% at k=2)
did not fire.

**Verdict by the frozen rule: g = 15.65% >= 10%, step 1 does not kill; step 2
waits for the maintainer.** The trace-overhead check fired (T/B 0.93 < 0.95), so g is an
upper bound. This amendment said what an inflated g below 10% means and nothing
about one above it. Charging the tracer's whole extra decode time to window
gaps gives 8.35%; charging only the windows' share of dispatches (678 of ~753 per
k=2 iteration) gives ~9.2% on p01. The untraced share lies in [8.35, 15.65]% and
the best estimates sit below the kill line, so the rule passes on the upper
bound only.

Findings:
- Gap per dispatch: 3.3 us traced at k=2, ~1.8 us at g_lo. The m=1 launch floor
  (2.57 us back-to-back, `launch-fusion-protocol.md`) sits between.
- The gap grows with rows at a fixed dispatch count: 2.23 ms at k=2, 2.89 ms at
  k=4. Part of what the trace calls a gap scales with m (drain/ramp at dependent
  boundaries), which a megakernel's grid barriers also pay.
- W3 (`megakernel-mrow-protocol.md`) measured the m=3 megakernel's GEMM phases at
  1.6x of m=1 against 1.5x native, and its head +300 us. A loss of that size
  against a realistic +8% ceiling leaves no margin.
- p20 at k=4, traced run: one 120.6 ms stall between two dispatches (rmsnorm_cast
  -> matmul_skinny, row 2822 of 17081), decode_s 1.59 vs 0.56 bare. Host-side,
  cause not diagnosed. Dropping it moves the k=4 median g by 0.04 pp.

## Amendment FR-Spec (frozen 2026-09-11 before any build or run, tree `3c1f877`)

Question: does restricting the MTP draft head to a frequency-ranked vocabulary
subset (FR-Spec, MiniCPM4 paper arXiv 2506.07900 section 4.1.1) raise
speculative tok/s_gen on the q4 pack without changing the output?

*Path note (2026-09-11, before any run): the two tools named below were ported from the planned Python to Mojo (CLAUDE.md s9 default); names corrected, nothing else changed.*

**Change.** The draft head (`blk32_forward`, `do_head`) computes logits over
`FR_K = 62080` rows (25% of 248320) of `output.weight`, taken from the q4
pack's own rows in frequency order, then maps the argmax back to the full id
with one int gather. The verify step still uses the full vocabulary, so a draft
outside the subset is simply rejected; output must not change. Switch:
`BARO_FR=1` on a pack built by `tools/fr-draft.mojo`. Frequency table:
`tools/fr-vocab.mojo` over `bench/ruler/data/essays.txt` plus the repo's
tracked `serve/`, `kernels/`, `tools/*.py`, `docs/*.md`, `bench/*.md`
text (about 2.3 MB). The 20 `bench/mtp-prompts/` prompts are not in it.

**Estimate behind the predictions.** The q4 head is about 572 MB (248320 x 4096
nibbles plus fp16 scales), about 0.7 ms per draft-head call at ~800 GB/s. A k=2
window makes 2 draft-head calls in 14.27 ms (MSPEC step 1 median), about 10%;
k=4 makes 4 in 19.31 ms, about 15%. FR_K = 25% removes 75% of that.

**Instrument.** `bench/mtp-prompts.sh` on the 20 prompts, same engine binary,
same session, GPU through `gpu-wait`: arm A (no spec), B2/B4 (`BARO_SPEC=1`,
k=2/4, full head), F2/F4 (`BARO_SPEC=1 BARO_FR=1`, k=2/4). P1 receipts from each
log: `BARO_SPEC:`, `spec k:`, `BARO_FR:` (True only on F arms), `pack q4
trunk: True`, `mtp: drafted/accepted`, `tok/s_gen`.

**Predictions (20-prompt medians, F vs B at the same k).**
- P-FR1 identity: GENERATED of every F run equals arm A. Hard gate; any
  mismatch voids the round.
- P-FR2 k=2: tok/s_gen +3% to +9%.
- P-FR3 k=4: tok/s_gen +5% to +13%.
- P-FR4 acceptance (accepted/drafted) falls by at most 3 points at each k.
- P-FR5 corpus coverage: the top 62080 ids cover >= 95% of corpus tokens.

**Falsifier.** k=2 gain under +1%, or acceptance down more than 5 points: the
subset from this corpus does not pay on this model, and the change stays off by
default. Claim rule: default-on only if P-FR1 holds and both P-FR2 and P-FR3 hold.

### Result FR-Spec (2026-09-11, engine `.work/engine-fr` from this commit's tree)

Pack `.work/engine-pack-q4-fr` (`tools/fr-draft.mojo`, checked by
`tools/test_fr_draft.py`: trunk identical, 62080 rows byte-exact). Ids
`.work/fr-ids.txt` sha256 `c99eb697bfa63080`..., corpus 720,004 tokens with only
20,808 distinct ids, so 41,272 of the 62,080 rows were lowest-id filler.
Arms in one gpu-wait job: arm A plus B2/B4 (`BARO_FR` unset) then arm A plus
F2/F4 (`BARO_FR=1`); every B log reads `BARO_FR: False`, every F log
`BARO_FR: True k 62080`, `pack q4 trunk: True`. Arm A median 137.24 / 137.12.

| | full head | FR head | change | prediction | verdict |
|---|---|---|---|---|---|
| identity k=2 / k=4 | 20/20 / 20/20 | 20/20 / 20/20 | | 20/20 (P-FR1) | held |
| tok/s_gen k=2 (median) | 150.80 | 161.78 | +7.29% (per prompt 1.014 to 1.125) | +3 to +9% (P-FR2) | held |
| tok/s_gen k=4 (median) | 129.37 | 140.43 | +8.55% (per prompt 0.945 to 1.175) | +5 to +13% (P-FR3) | held |
| acceptance k=2 | 65.9% | 65.0% | -0.9 pt | >= -3 pt (P-FR4) | held |
| acceptance k=4 | 47.9% | 46.8% | -1.1 pt | >= -3 pt (P-FR4) | held |
| corpus coverage by top 62080 | | 100% | | >= 95% (P-FR5) | held, but vacuous: every distinct corpus id fits |

Claim rule met (P-FR1, P-FR2, P-FR3). Not yet default-on: the frequency table
comes from a small corpus with lowest-id filler; the next round ranks the
unseen ids by BPE merge order and widens the corpus, then re-runs this A/B.

## Amendment FR-Spec round 2 (frozen 2026-09-11 before its run)

Round 1's table came from a 720k-token corpus with 20,808 distinct ids; the other
41,272 rows were lowest-id filler. Round 2 changes only the id table:
`tools/fr-vocab.mojo` now ranks ids the corpus never produced by the BPE merge
that builds them (measured nearly a no-op on this tokenizer: 61,613 of 62,080
rows unchanged, because its ids already follow merge order), and the corpus
grows to 48.2M tokens (essays, repo text, the maintainer's markdown notes, the
llama.cpp sources; 77,807 distinct ids; the top 62,080 cover 99.95%). Ids
`.work/fr-ids-ac.txt` sha256 `e56773aeedbf29a0`...; 45,562 rows shared with round 1.
Pack `.work/engine-pack-q4-fr-ac`, same builder and checker. Instrument, arms and
receipts as round 1; B arms re-run in the same job.

Predictions (F vs B at the same k, 20-prompt medians):
- P-FR6 identity 20/20 at k=2 and k=4 (hard gate).
- P-FR7 acceptance change vs full head no worse than round 1 (-0.9 pt at k=2,
  -1.1 pt at k=4), and within +-1 pt of it.
- P-FR8 tok/s_gen gain within 2 points of round 1 (+7.29% k=2, +8.55% k=4): the
  head size is unchanged, so only acceptance can move the result.
Falsifier: acceptance worse than round 1 at both k, which would mean a bigger
general corpus ranks the model's actual tokens worse than the small one.

### Result FR-Spec round 2 (2026-09-11, same engine, pack `.work/engine-pack-q4-fr-ac`)

Receipts as round 1 (all 80 speculative logs carry the right `BARO_FR`); arm A
median 136.92 / 136.99.

| | full head | FR head | change | prediction | verdict |
|---|---|---|---|---|---|
| identity k=2 / k=4 | 20/20 / 20/20 | 20/20 / 20/20 | | 20/20 (P-FR6) | held |
| acceptance k=2 | 65.9% | 64.6% | -1.3 pt (round 1: -0.9) | no worse than round 1 (P-FR7) | **failed** |
| acceptance k=4 | 47.9% | 46.2% | -1.7 pt (round 1: -1.1) | no worse than round 1 (P-FR7) | **failed** |
| tok/s_gen k=2 | 149.09 | 164.49 | +10.33% | within 2 pt of +7.29% (P-FR8) | missed high |
| tok/s_gen k=4 | 127.37 | 141.53 | +11.12% | within 2 pt of +8.55% (P-FR8) | missed high |

Falsifier triggered: acceptance is worse than round 1 at both k, so a larger
general corpus ranks this model's own tokens worse than the small one did. The
larger tok/s gain is partly run-to-run noise (round 1's full-head arms read 150.80
and 129.37). Next: rank by the model's own output (a generated corpus,
option B) rather than by general text. FR-Spec stays off by default.

## A1 speculative-sampling gates at real vocab (preregistered 2026-09-15, before the first run)

`kernels/test_sample_device.mojo` gains two gates beside its sampler pair.
Full round: `bench/spec-sample-protocol.md`. Thresholds frozen here before
the gates ran, per P2:

- **gate3, per draw, exact.** For each of the three real logits rows and each
  of two shapes (`T1_k0_p1`, `T0.7_k20_p0.8`), 64 draws: the token and the
  accept flag from `amar_spec_accept` equal `spec_accept_ref` on the same
  `(seed, counter)`. **Threshold: 0 mismatches of 64.** Not a rate, not a
  distribution: one disagreeing draw fails the gate. The draft distribution is
  the NEXT row's logits (wrapping), so `q` is far from `p` and most draws take
  the residual branch rather than the accept branch. The device and host
  probability rows are compared too (`max |device - host|` is printed as a
  receipt, not a threshold, since both gates run on the device rows).
- **gate4, distribution, 20,000 draws.** Draw `x ~ q` on device, run the
  accept rule on device, bin the EMITTED token against the independent numpy
  oracle for `p` (`tools/sample-nucleus-oracle.py`, the same oracle file gate2
  uses), chi-square with expected counts under 5 pooled, **critical value at
  p = 0.001**. This is the theorem's own claim: tokens emitted by accept plus
  residual are distributed as `p`, so the same oracle that gates direct
  sampling gates this.

Why these two and not one: aggregate agreement with a per-draw disagreement is
exactly the sampler defect this repo shipped once already
(`exchange/2026-09-15-m5-sampler-diagnosis.md`). gate4 passing while gate3
fails is a failure.

### Result (2026-09-15): both gates pass at real vocab, on both draft arms

`kernels/test_sample_device.mojo` at `.work/b1/tsd3.log`, one `gpu-wait` job,
about 30 GPU seconds for the whole file (5 sampler shapes plus 2 spec shapes,
3 rows, 20,000 draws each).

- **gate3: 0 mismatches of 64 on every row, shape and arm** (12 runs). The
  device probability rows match the host reference to 5.96e-08 at
  T0.7_k20_p0.8 and 4.17e-07 at T1_k0_p1, which is the f32 rounding floor,
  reported as a receipt and not a threshold.
- **gate4: every chi-square well under its p=0.001 critical value** (12 runs),
  for example p01-water near T1_k0_p1 chi2 55.96 against crit 99.68 on df 60,
  and p02-python-fib near T0.7_k20_p0.8 chi2 1.96 against 14.13 on df 2.

The two draft arms matter and the first run proved it: with the draft taken
from another prompt's row ("far"), acceptance is **0 of 64** and 155 of 20,000,
so that arm tests the residual branch and nothing else. The "near" arm (same
row at 1.3x temperature) accepts 33 to 58 of 64 and 12,061 to 18,292 of 20,000.
Running only the far arm would have left the accept branch uncovered while
every gate printed PASS.
