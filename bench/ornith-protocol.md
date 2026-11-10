# Ornith-1.5-9B (K-quant source) in the engine

Preregistered 2026-09-11, before any Ornith timed run, tree at the lane-ORNITH commit that
lands `tools/engine-pack.py`'s K-quant path. Binds to `bench/PROTOCOL-RULES.md` P1-P6.
Question: does a K-quant-sourced (Q4_K_M) engine pack decode Ornith-1.5-9B correctly, and at
what speed against llama.cpp running the same GGUF natively?

## Why this model needs its own protocol

`~/Models/ornith-1.5-9b-q4_K_M/Ornith-1.5-9B-Q4_K_M.gguf` is `qwen35`, the same shape as
Qwythos-9B (H=4096, FFN=12288, NQH=16, NKVH=4, HD=256, SSM inner 4096, state 128, 16 groups,
conv 4, full attention every 4th layer, VOCAB 248320, tokenizer sha-identical to Qwythos), but
it exists ONLY as K-quants (223 Q4_K, 35 Q6_K, 184 F32 — no bf16/f32 source). `tools/engine-pack.py`
dequantises each K-quant tensor to f32 with gguf-py (`gguf.quants.dequantize`), rounds to bf16,
then quantises through the existing `--q8` path unchanged. `tools/model-ref.py`'s numpy stack is
shape-driven, not source-format-driven, so it needs no change to run over the Ornith pack — its
`q8` tensor path (cached dequantise, seconds/token) is what makes a 20-prompt sweep practical; the
plain-bf16 path re-dequantises every weight on every call with no cache and was ruled out on time.

**Amendment (mid-lane, coordinator-approved 2026-09-11):** step 3a's teacher-forced agreement
needs `BARO_FORCE` in `serve/engine.mojo`, which did not exist (only `serve/spark.mojo` had it).
`serve/engine.mojo`'s decode path selects every token on-device inside `mega_token_k` /
`mega_win_k` / `argmax_k` — there is no per-step host round trip to intercept like spark's
per-layer host loop. Added at the host level instead, with no kernel change: the outer decode
loop in `serve/engine.mojo` (`while wst.pos < n_total - 1: step_window(...)`) already runs one
`step_window` call per generated token when `BARO_SPEC=0`, so after each call — only when
`BARO_FORCE` is set and the position is past the prompt — the token `step_window` just wrote
into `toks_d` is copied back to host (recorded as "predicted"), then a host->device copy of the
reference id overwrites it before the next step reads it as context. `BARO_FORCE` + `BARO_SPEC=1`
raises (teacher forcing is defined for the no-spec path only, matching `spark.mojo`). With
`BARO_FORCE` unset this adds zero device work: the whole block is behind `if len(force) > 0`.
Own commit, named in the report.

## Arms

- **ours**: `.work/engine` built from `serve/engine.mojo` (this lane's `BARO_FORCE` addition),
  `BARO_PACK=.work/engine-pack-ornith-q8` (`tools/engine-pack.py MODEL.gguf OUTDIR --q8`, K-quant
  path), greedy (`BARO_SPEC=0`) unless noted.
- **ours/numpy**: `tools/model-ref.py decode 64`, `BARO_PACK` pointed at the SAME q8 pack as
  `ours` — its `q8` tensor path is cached and dequantises once per tensor (`_Q8_CACHE`), so this
  runs in seconds/token, unlike a fresh bf16 numpy pass over ~9B params per token (measured
  impractical: no caching on the plain-bf16 `T()` path, ruled out for a 20-prompt x 64-token
  sweep). Not a timed arm; its greedy stream over the identical dequantised weights `ours` uses is
  the "our own math" reference (same role as G1 in `bench/q8-protocol.md`), and is INDEPENDENT of
  whether the K-quant->q8 chain lost anything relative to llama.cpp's own quantisation — it only
  tests whether the kernels compute what the pack's numbers say they should.
- **llama.cpp**: `~/llama.cpp/build/bin/llama-server -m Ornith-1.5-9B-Q4_K_M.gguf -c 8192 -ngl 99
  -fa on -b 2048 -ub 512 -t 8 -ctk q8_0 -ctv q8_0`, greedy (`temperature 0, top_k 1`), token-id
  prompt (`tools/llama-ref-run.sh` pattern), its own native K-quant kernels — not requantised
  through our pack.
- Prompt set: `bench/mtp-prompts/*.tokens` (20 prompts) — reusable as-is, tokenizer sha-identical
  to Qwythos per `exchange/lane-moe-report.md`. `GEN_N = 64` (`serve/registry.mojo`).
- Read-back before any number is read (P1): engine sha, `BARO_PACK` path and pack index tensor
  count (442) printed by the engine at load, `BARO_FORCE` id count printed when set, llama.cpp
  `/props` (`ctk`/`ctv`, `n_ctx`), `prompt_n == N` every run. llama-server down whenever `ours`
  holds the GPU and vice versa (P1, `bench/mtp-protocol.md`).

## Gates and frozen predictions

**G1 (step 2, already run and PASSED):** `tools/test_engine_pack_kquant.py` — packed-q8
dequantised vs gguf-py's dequantise of the source K-quant bytes, max error within q8 rounding
(d/2 plus one bf16-rounding ulp) on 4 sample tensors (a Q4_K and a Q6_K trunk tensor, `output.weight`,
and a `blk.32` NextN tensor). Result: PASS, 0 elements over tolerance, observed error 27-34% of
the tolerance band on all 4.

**G2 (`BARO_FORCE` no-op when unset):** 20-prompt A/B, this lane's engine vs `main`'s engine, same
Qwythos q4 pack, greedy, temperature 0. Land rule: 20/20 identical `GENERATED` lines — a single
new `if len(force) > 0:` branch around unchanged code should not be able to move output, but the
branch itself must be proven never taken, not assumed. Falsifier: any of the 20 differs.

**G3 (kernel self-consistency, cheap, run before 3a):** `ours` greedy (no `BARO_FORCE`) vs
`ours/numpy` greedy, one representative prompt, 64 positions. This is the same identity class
`bench/q8-protocol.md` G1 already meets on Qwythos (engine == numpy over the same dequantised
pack), so greedy equality is the right check here (CLAUDE.md's "never greedy past ~256 ids" is
about betting a *cross-implementation* identity claim on a long coin-flip tail, not about a
same-numbers kernel-vs-reference check at 64 ids). **Prediction: 64/64 or a single-digit number of
late near-tie divergences, not a kernel bug.** Falsifier: divergence before position ~32, or on
more than one prompt if a second is checked — that would point at the K-quant path, not float
ordering, and 3a's numbers would need a kernel bisect first.

**Step 3a — teacher-forced agreement vs llama.cpp.** `llama.cpp` generates its own 64-token greedy
stream per prompt (temperature 0); that stream is fed to `ours` as `BARO_FORCE`. Agreement =
fraction of the 64 positions where `ours`' own argmax (recorded as "predicted", before the forced
id overwrites the context) equals llama.cpp's token at that position. This is a description of how
often the two quantisation paths would have made the same choice, not an equality gate (per
CLAUDE.md, no greedy-identity pass/fail is frozen past 64 ids either way) — **no absolute threshold
is frozen**, because there is no independent ground truth here to call one arm "right": G3 already
shows `ours`' kernels match `ours`' own numbers, so a low agreement number means quantisation-path
divergence between K-quant->q8 and llama.cpp's native mixed-K-quant, not a bug. Recorded alongside
G3's near-100% self-consistency so a reader can tell the two apart. **Prediction: 60-90% median
agreement** (two independent ~4-8 bit quantisations of the same weights, on a model neither path
has been tuned against) — reported, not gated. Falsifier: <30% median, which would say one of the
two paths is doing something qualitatively wrong, not just accumulating more rounding error.

**Step 3b — decode speed, 20-prompt median (P4).** Derivation: the Ornith q8 pack measured **9.99
GiB** (`.work/engine-pack-ornith-q8/`), matching the Qwythos-shape q8 prediction (10.35 GB,
`bench/q8-protocol.md`) almost exactly — same architecture, same byte budget. The q8 decode path's
last recorded number on this shape was 68.8 tok/s_gen (2026-09-04, `docs/BASELINE.md`), before the
shared-infrastructure wins landed since (megakernel launch fusion +22%, rmsnorm fold): those apply
to the q8 dot loop too, but the q4-specific chunked-delta schedule that took q4 to 133.9/136.37 does
not. **Prediction: ours 70-95 tok/s_gen.** Falsifier: outside 55-120 (either a regression the shared
wins should have prevented, or a suspiciously large unexplained gain).

llama.cpp Q4_K_M averages ~4.5 bits/weight against Q8_0's 8 bits/weight (~1.7x fewer stream bytes),
scaling the existing Q8_0 no-spec bar (74.1 tok/s_gen, `docs/BASELINE.md`). **Prediction: llama.cpp
110-150 tok/s_gen.** Falsifier: outside 80-200.

No ratio claim is frozen between the two arms — the point of this round is the numbers, not a
predetermined verdict.

**Step 3c — MTP on identical to no-spec (P4).** `bench/mtp-prompts.sh .work/engine .work/ornith-mtp`
at `k=2` (repo default, `BARO_SPEC_K`): arm B (`BARO_SPEC=1`) `GENERATED` must equal arm A
(`BARO_SPEC=0`) `GENERATED`, every one of the 20 prompts — the existing gate, unmodified by this
lane, already track-recorded at 20/20 on Qwythos. **Prediction: 20/20.** Falsifier: any prompt's
arm B differs from arm A — MTP correctness is a kernel property (draft head + acceptance), not a
weight-source property, so a new model diverging here would be a real finding, not noise.

**Step 3d — chat smoke.** One `/v1/chat/completions` request through the Rust front
(`serve/src/main.rs`, `serve/PROTOCOL.md`) against the Ornith pack. No quantitative prediction:
pass/fail is a well-formed response whose `content` a human reads as a coherent answer to the
prompt asked.

## Not in this round

Requantising Ornith through anything other than `--q8`; `--q4-draft` (MTP draft-head Q4, untouched
by the K-quant path — would `KeyError` if invoked on a K-quant `output.weight`, since it still reads
`ge.GGML_BYTES` directly; out of scope, not fixed); serving concurrent requests; any kernel change
(the item's Rules forbid it outright; `BARO_FORCE` needed none, per the amendment above).

## Result (2026-09-11, engine sha `e469d1c`, `bench/ornith-run.sh` — `.work/ornith-run/`)

Read-back (P1): engine build 0 errors, `BARO_PACK` pack 442 tensors (printed at load); llama.cpp
`/props` `n_ctx: 8192`; llama-server ran 4 parallel slots (round-robin slot ids 0-3 in
`llama-server.log`, single request at a time from this script so that does not affect per-request
timing) — not read back before this run, noted here as a receipt gap for the next one; llama-server
down while `ours` held the GPU and vice versa throughout.

- **G3 (kernel self-consistency): 64/64, first divergence at position 64 — i.e. none.** Matches the
  prediction exactly. `ours`' kernels compute exactly what the q8 pack's numbers say to, on Ornith's
  actual tensor shapes and values, not just on Qwythos's.
- **Step 3a (teacher-forced agreement vs llama.cpp): median 63/64 (98.4%), range 57-64/64.** Well
  above the 60-90% predicted band and nowhere near the 30% falsifier — the two independent
  quantisation paths (K-quant->q8 vs llama.cpp's native mixed-K-quant) agree far more often than
  the conservative prediction assumed. One outlier, `p16-chat` at 57/64 (89%), still comfortably
  above the falsifier; not investigated further (3a is reported, not gated, per the protocol above).
- **Step 3b (tok/s_gen, 20-prompt median):** `ours` **80.78** (range 76.8-81.0) — inside the
  predicted 70-95 band. `llama.cpp` **88.8** (range 83.1-90.6, from `llama-server.log`'s own
  `eval time ... tokens per second`, not the `/completion` response's `timings` block, which this
  script did not capture — a gap for next time) — **below the predicted 110-150 band**, though still
  inside the 80-200 falsifier. The Q4_K_M-vs-Q8_0 byte-scaling argument overestimated llama.cpp's
  edge; `ours`/`llama.cpp` = **0.91x**, close rather than the implied ~0.6x. No kernel or config
  change follows from this alone (P6: the prediction's own scaling assumption is the more likely
  miss, not either engine); flagged for the next round that touches decode speed on this shape.
- **Step 3c (MTP identical to no-spec): 20/20.** Matches the prediction exactly —
  `.work/ornith-run/mtp/results.txt`.
- **Step 3d (chat smoke): PASS**, response read: *"The user is asking a simple factual question: the
  capital of France. I should answer in one sentence as requested. \</think\> The capital of France
  is Paris."* — coherent, correctly answers the question, reasoning-then-answer shape expected of
  the Ornith family (per the `ornith` skill).

**Verdict: Ornith-1.5-9B runs correctly and at a usable speed through the K-quant engine pack.**
No kernel or engine bug found; the one genuine miss was this protocol's own tok/s prediction band
for llama.cpp, not the systems under test.

## Rebase onto main (2026-09-11, coordinator-requested pre-merge)

Main had moved to `fc28bf1` (lane-CHAT merged: C1 control block/cancel/stop, M1b checkpoints,
C3 host sampler, C4 engine pool). `git rebase main` conflicted in `serve/engine.mojo`'s decode
loop: CHAT added `cancelled`/`stopped`, hint-aware prefill chunking, and extended
`Chain.save`'s signature with two required `Bool`s (`pinned`, `boundary`); this lane's
`BARO_FORCE` block sits in the same loop body. Resolved keeping both: `pos_before` is captured
before whichever `step_window` variant runs (plain or hint-chunked), the forced-token
read-back/overwrite runs immediately after using the resulting `wst.pos`, and the checkpoint save
calls keep CHAT's new signature verbatim (this lane's old 5-arg call would not have compiled
against it). CHAT's post-loop `cancel_pending`/`stop_seqs` handling is untouched and sits after
the merged block, as before. Compiles clean (0 errors).

Also fixed, coordinator-flagged: `--q4-draft` KeyError'd on a K-quant `output.weight` (read raw
bytes through `ge.GGML_BYTES` directly, bypassing the K-quant path) — same `dequantize_kquant`
fallback as the main pack loop, spot-checked against the source Q6_K bytes (S-sized, not gated,
MTP draft-head speed stays out of this round).

Re-verified against current main, both required by the coordinator before merge:

| gate | result |
|---|---|
| G2 (`BARO_FORCE` no-op), vs a fresh build of `fc28bf1`, Qwythos q4 pack, greedy | **PASS, 20/20** (`.work/g2-ab-rebase/results.txt`) |
| `run-tests.sh` | **exit 0**, 85 kernels, 38 in registry, 0 orphans (unchanged) |

Step 3's numbers above (G3, 3a-3d) were not rerun — nothing in the rebase touches the K-quant
pack path, the forcing semantics, or any kernel; the merge conflict was ordering/wiring around
CHAT's unrelated additions (cancel/stop/checkpoint-signature), not a change to what this lane's
code computes. `bench/ornith-run.sh` remains valid to rerun if that assumption is ever doubted.
