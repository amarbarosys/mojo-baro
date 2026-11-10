# Dense families protocol

Preregistration for the dense-families plan (`~/Brain/mojo/mojo-baro/briefs/2026-09-11-dense-families.md`).
Each lane owns one section; predictions are frozen by the commit that adds them, before any build or run.

## PROFILE: comptime profile module from GGUF metadata (steps 1-2)

**Process note: this section is written after the step-1 build and gate ran, not before** --
the item started mid-conversation from the coordinator with KATT still in flight, and the
build/gate work (`tools/gen-profile.mojo`, the spark2_5 profile wiring) was already done and
verified before this preregistration was written up. Flagged here rather than silently
back-dated; no other lane's rule violation is implied.

Scope: `tools/gen-profile.mojo` (new), `serve/spark.mojo` (import constants from a generated
profile instead of a hardcoded block; recipe flags behind comptime ifs), `kernels/spark_kernels.mojo`
(ROPE_NEOX dispatch in `amar_rope_plain`/`amar_rope_kv_append`, new `amar_bias_add`).

Predictions (step 1, spark2_5 default profile):
1. `tools/gen-profile.mojo` run against all 5 targets' real GGUFs reproduces the plan's table
   (L, H, FFN, heads q/kv, HD) exactly, VOCAB read off `token_embd.weight`'s own tensor shape.
2. Spark rebuilt with `-I` pointed at the generated spark2_5 profile is bit-identical to the
   pre-change build on: the two frozen fixture gates (`.work/spark/ref` text 64/64,
   `.work/spark/chat-ref` chat 43/43) and a fresh 20-prompt greedy A/B against a baseline build
   of the same pre-change source (`bench/mtp-prompts/p*.txt`, `BARO_GEN=64`), 20/20.
3. `./run-tests.sh` exits 0 with the same PASS count as before the change (KATT's floor: 102).

Predictions (step 2, recipe-flag scaffolding only -- not exercised end to end):
4. QKV_BIAS's new `amar_bias_add` path is numerically inert (bit-identical output) when wired
   in with an all-zero synthetic bias on the spark2_5 shapes, proving the new offset-stride and
   kernel-launch code executes correctly without changing decode when the bias is zero.
5. ROPE_NEOX=False (the "norm"/interleaved rope path, needed for llama/lily) is not exercisable
   against a real model this round -- Spark's own weights are neox-native -- so it is verified
   only against a numpy rope reference on synthetic data, not an end-to-end model.
6. GRANITE_MULT's embedding/residual/logit scale multipliers are NOT wired into `spark.mojo`
   this round (design deviation, see report): they fold cleanly into pack-time weight scaling
   instead of runtime kernel branches, and that packer does not exist yet. Only `ATTN_SCALE` is
   wired (already generic; no boolean branch needed since it's a parameter substitution).

Kill: any mismatch in 1-3 stops the lane after three repair attempts and is reported, not
silently downgraded to "close enough".

## PROFILE step 3: per-target build + verify (frozen before the Llama-3.2-1B run)

Both blockers landed on main before this section was written (KATT `088c940` head-dim
parameterization, TOK `3c1f877` SPM + llama-bpe/qwen2/granite-docling), so step 3 is in scope
now. New in this step, found while building the packer (not in the plan's original flag list):
**HAS_GATE** (spark2_5's per-head attention output gate has no analogue in llama/qwen2/granite;
without a way to disable it every non-spark target's attention output would be silently
multiplied by sigmoid(0)=0.5) and **activation** (spark2_5 uses GELU, llama/qwen2/granite use
SiLU; `amar_gemv_q8` gets a new EPI=3 branch). Both are commits before this run, not part of it.

New tool: `tools/engine-pack.py --dense MODEL.gguf OUTDIR` -- Q/K/V weight fusion (three
separate GGUF tensors concatenated into one `[QKV, H]` matrix, matching how the fused-QKV
kernel splits its output back into Q/K/V sub-ranges) plus Ornith's existing K-quant dequant,
for the plain-transformer archs KATT and this profile system now support. Verified by: the
packed model runs, produces syntactically well-formed generation, and decodes (via
`tools/baro-tokenize.mojo`) to fluent on-topic text for a prompt about water molecule
structure -- necessary but not sufficient; the BARO_FORCE run below is the real gate.

Method: 20 prompts (`bench/mtp-prompts/p*.txt`) tokenized once by our own tokenizer (already
gated against `llama-tokenize` by TOK), the same ids sent to both llama.cpp
(`/completion`, `n_predict 64`, `temperature 0`, `top_k 1`) and our engine
(`BARO_PROMPT=<ids>` no-spec for tok/s_gen, then `BARO_FORCE=<llama's greedy ids>` for
agreement). Server: `llama-server -ngl 99 -fa on -ctk f16 -ctv f16` (fastest config per
`spark-ab-protocol.md`), native K-quant GGUF (no requant needed, unlike spark2_5's Q8_0 arm).

Frozen predictions, Llama-3.2-1B-Instruct-Q4_K_M:
1. Forced agreement >= 90% of generated positions on at least 18/20 prompts (a Q4_K_M
   requant candidate is not expected to hit spark2_5's 98-100% bar; llama.cpp's own f16-KV
   config disagrees with its f32 reference on some fraction of the same 20-prompt set per
   this repo's CLAUDE.md, so <100% agreement is not by itself a failure).
2. tok/s_gen positive and finite on all 20 prompts (no crash, no NaN-shaped hang); no
   specific ratio vs llama.cpp claimed -- this lane is a correctness port, not a perf round.
3. Chat completions smoke: NOT applicable. `serve/src/engine.rs` drives a long-running
   request/response protocol that only `serve/engine.mojo` implements; `serve/spark.mojo` is
   a one-shot CLI with no such protocol, and wiring it up is out of this lane's file list.
   Reported, not attempted.

Kill: prediction 1 failing (under 18/20 prompts clearing 90%) after three repair attempts is
reported as a real disagreement, not re-thresholded down to make it pass.

### Result: Llama-3.2-1B-Instruct-Q4_K_M (2026-09-11, `bench/dense-run.sh`, `.work/dense/llama32-1b-run2/`)

| # | prediction | result |
|---|---|---|
| 1 | >=90% agreement, >=18/20 prompts | **PASS, 20/20**: range 61-64/64 (95.3-100%), one prompt 25/25 (llama's own greedy completion ended at 25 tokens) |
| 2 | tok/s_gen positive/finite, no crash | **PASS**: 451-458 tok/s_gen across all 20 prompts, no crash |
| 3 | chat smoke | **N/A as predicted**, not attempted |

Verdict: llama-arch (interleaved rope, tied embeddings, SiLU, no bias, no gate) is correct
end to end through the new `--dense` packer and the profile-driven engine. Server:
`llama-server -ngl 99 -fa on -ctk f16 -ctv f16`, native Q4_K_M (no requant).

### Result: Qwen2.5-7B-Instruct-Q4_K_M (2026-09-11, `.work/dense/qwen25-7b-run/`)

Same method and thresholds as Llama-3.2-1B above. This target exercises QKV_BIAS for the
first time (the fused-QKV bias-add kernel and the offset-stride generalization) and
ROPE_NEOX=True with a non-tied output head, neither previously run end to end.

| # | prediction | result |
|---|---|---|
| 1 | >=90% agreement, >=18/20 prompts | **PASS, 20/20**: range 62-64/64 (96.9-100%), three prompts ended their llama.cpp completion early (14/14, 23/23, 33/33) and matched fully |
| 2 | tok/s_gen positive/finite, no crash | **PASS**: 96.8-97.8 tok/s_gen across all 20 prompts (7B vs 1B model, expected slower) |
| 3 | chat smoke | **N/A as predicted**, not attempted |

Verdict: QKV_BIAS is correct (the bias-add kernel and the generalized per-layer offset
stride both work as designed), and qwen2-arch (neox rope, biased QKV, non-tied output) is
correct end to end.

### Result: granite-4.2-3b-BF16 (2026-09-11, `.work/dense/granite42-3b-run2/`) -- one repair round

**First attempt FAILED**: 40/64 forced agreement average (range 26-51/64, all well under the
90% floor). Root cause, found by reading `llama_model_rope_type` in llama.cpp's
`llama-model.cpp`: `LLM_ARCH_GRANITE` is grouped with `LLM_ARCH_LLAMA` under "normal RoPE,
pairs of consecutive head values" (`LLAMA_ROPE_TYPE_NORM`), not with `LLM_ARCH_QWEN2`'s
half-offset NeoX group -- the opposite of what `tools/gen-profile.mojo` assumed. Granite's
GGUF conversion permutes Q/K the same way Llama's does. This was a plan-derived guess never
checked against the actual source before this run, exactly the failure mode
`mojo-nightly-lane-builder`/this plan's own methodology warns about ("recipe differences read
from llama.cpp source as SPEC, code written fresh") -- the table was assembled from general
recollection instead of the source, for this one arch.

Fix: `is_rope_norm_arch` in `tools/gen-profile.mojo` now includes `"granite"`. Re-ran clean.

| # | prediction | result |
|---|---|---|
| 1 | >=90% agreement, >=18/20 prompts | **PASS, 20/20** (after the fix): range 63-64/64 (98.4-100%), one prompt 26/26 |
| 2 | tok/s_gen positive/finite, no crash | **PASS**: 164.5-169.0 tok/s_gen (BF16 source, no quantization) |
| 3 | chat smoke | **N/A as predicted**, not attempted |

Verdict: granite-arch (NORM rope like llama, non-tied output, the distinct attention-scale
multiplier from GRANITE_MULT) is correct end to end once the rope table is fixed.
embedding_scale and residual_scale are both 1.0 in this checkpoint, so this run does not
exercise the pack-time weight-folding gap noted in step 2's predictions above -- that remains
unverified for a hypothetical Granite checkpoint with non-unity multipliers.

### Result: lily-cybersecurity-7b-v0.2-Q6_K (2026-09-11, `.work/dense/lily7b-run/`)

Mistral-arch under `general.architecture = llama` (TOK's SPM tokenizer, `tokenizer.ggml.model
= llama`, `add_bos true`), last of the four targets. Uses the now-confirmed llama/granite
NORM rope, same as Llama-3.2-1B.

| # | prediction | result |
|---|---|---|
| 1 | >=90% agreement, >=18/20 prompts | **PASS, 20/20**: range 62-64/64 (96.9-100%), two prompts ended early (5/5, 27/27) matching fully |
| 2 | tok/s_gen positive/finite, no crash | **PASS**: 94.1-95.0 tok/s_gen across all 20 prompts |
| 3 | chat smoke | **N/A as predicted**, not attempted |

Verdict: all four dense-family targets now verified end to end. SPM tokenization (TOK) and
NORM rope (this lane, fixed on Granite) both confirmed correct for the Mistral-family GGUF.

## PROFILE step 3 summary: 4/4 targets PASS (Llama-3.2-1B, Qwen2.5-7B, Granite, Lily)

All at >=95% forced agreement except Granite's first (rope-type bug, fixed) attempt. One
repair round total across the whole step. Not exercised by any of the four real checkpoints:
GRANITE_MULT's embedding/residual-scale pack-time folding (both multipliers are 1.0 in the
one Granite checkpoint available), and the chat-completions smoke (spark.mojo has no Rust
front protocol implementation, out of this lane's files).

## KATT: head dimension as a comptime parameter

Scope: the attention kernels the Spark path calls, `amar_attn_decode_swa_gated` and
`amar_rope_kv_append` in `kernels/spark_kernels.mojo`, and the helpers they share in
`kernels/attn.mojo` (`attn_head_span`, `kv_off`). Spark has no separate prefill attention
kernel: its multi-row path is the same decode kernel over grid y, so nothing else is in scope.
The head counts ride along as parameters because `kv_off`'s page stride and the GQA group
(`NQH // NKVH`) are part of the same indexing; every target has different head counts.
Defaults equal today's constants (HD 256, NQH 16, NKVH 4), so no call site changes.

Base: `0447adb`. Test: `kernels/test_spark_attn.mojo` runs rope+append then gated decode
attention at (HD, NQH, NKVH) = (64, 32, 8), (128, 28, 4), (256, 16, 4), two query rows,
NAT 2 with att_i 1, T spanning three KV pages, full window and a 100-token window.
Oracle: `tools/spark-attn-ref.py` (numpy, float64) on the dumped inputs.

Frozen predictions:

1. **Parity.** For every HD and both window modes, every output element satisfies
   `|got - ref| <= 2^-8 * |ref| + 1e-5` (half a bf16 ulp plus float32 slack). Zero violations.
2. **HD 256 bit-identity.** (a) The HD 256 test output from the new source is byte-identical
   (`cmp`) to the same test built against the base kernels. (b) The disassembly of every
   AMDGPU code object in the Spark binary (`serve/spark.mojo`) and in the qwen35 engine
   binary (`serve/engine.mojo`, which shares `attn_head_span`) is identical before and after.
   Control: two base builds of the Spark binary produce identical disassembly.
3. **ISA receipt.** At each HD, both kernels show 0 VGPR spills, 0 SGPR spills and 0 private
   segment bytes. VGPR count at HD 64 and 128 is at most the HD 256 count.
4. **Census.** `kernel-census --check` exits 0.
5. **Suite.** `./run-tests.sh` exits 0 before and after; the after run has exactly one more
   test binary (the new test) and its numpy oracle step.

Kill: any violation in 1 or 2 stops the lane after three repair attempts; 3 failing at HD 64
or 128 is reported, not repaired by changing the kernel family.

### KATT results (2026-09-11, commits `26fffe3`, `356d22f`; evidence `.work/KATT-gate.txt`)

| # | prediction | result |
|---|---|---|
| 1 | parity, 0 violations | PASS: 0/4096, 0/7168, 0/8192 per mode at HD 64/128/256; max abs err 1.5e-3 (HD 64 swa); 99.88-100% of elements equal the bf16 rounding of the float64 reference |
| 2a | HD 256 test output byte-identical to base | PASS: `o_full`, `o_swa`, appended row identical (`cmp`) |
| 2b | engine disassembly unchanged | PASS: 16/16 Spark and 186/186 qwen35 engine code objects identical after normalisation; base-vs-base control identical |
| 3 | 0 spills, VGPR(64,128) <= VGPR(256) | PASS: attention VGPR 34/36/40, rope 15/15/15, 0 spills, 0 scratch |
| 4 | census rc 0 | PASS |
| 5 | suite green, +1 test binary | PASS: exit 0 before and after; 101 -> 102 PASS lines |
