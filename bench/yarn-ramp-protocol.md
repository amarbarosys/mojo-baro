# YaRN ramp direction: preregistration

Frozen BEFORE any arm is built or run. the maintainer: "do YaRN ramp inversion".

## Claim under test

`kernels/attn.mojo` `amar_rope_yarn` mixes interpolated and extrapolated
angles in the opposite direction to llama.cpp and HF.

Ours (`kernels/attn.mojo:266-268`):

    ramp  = clamp((j - YARN_LOW) / (YARN_HIGH - YARN_LOW))
    theta = theta_in * (1 - ramp) + theta_ex * ramp

so `ramp` is the weight on EXTRAPOLATION and it is 0 for low `j`. Low `j` is
the HIGH-frequency end, so we interpolate the fast-rotating pairs and
extrapolate the slow ones.

llama.cpp (`ggml/src/ggml-cpu/ops.cpp`, `rope_yarn_ramp` + `rope_yarn`):

    ramp_mix = (1 - clamp((i0/2 - low) / (high - low))) * ext_factor
    theta    = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix

so their weight on EXTRAPOLATION is 1 for low `j`. HF
`_compute_yarn_parameters` matches llama. Ours is inverted.

## Not in doubt, checked before freezing

The Qwythos GGUF really does ask for YaRN, so this code path is live:
`qwen35.rope.scaling.type = "yarn"`, `scaling.factor = 4.0`,
`rope.scaling.original_context_length = 262144`, `rope.freq_base = 1e7`,
`rope.dimension_count = 64`.

The constants are right, only the direction is suspect. Recomputing
llama's correction dims from those metadata values with the usual
beta_fast=32 / beta_slow=1:

    corr_dim(b) = n_dims * ln(n_ctx_orig / (b * 2pi)) / (2 * ln(base))
    corr_dim(32) = 64 * ln(262144/201.06) / (2*ln(1e7)) = 14.24 -> low  = floor = 14
    corr_dim(1)  = 64 * ln(262144/6.283)  / (2*ln(1e7)) = 21.12 -> high = ceil  = 22

which is exactly `YARN_LOW = 14.0` and `YARN_HIGH = 22.0`. FREQ_SCALE 0.25
= 1/4.0 and MSCALE 1.1386294361119891 = 1 + 0.1*ln(4) also match llama's
`mscale *= 1 + 0.1*log(1/freq_scale)`.

## Arms

- A: current `main`, unchanged kernel.
- B: A with one line changed, `ramp = 1 - clamp(...)`.

Same pack (`.work/engine-pack-q4`), same prompts, same llama.cpp reference
ids, `BARO_SPEC=0` on every run (teacher forcing is a no-spec gate and the
engine raises otherwise). Both engine sha256s recorded in the arm file.

Reference: llama.cpp on the Qwythos BF16 GGUF, greedy, 64 tokens,
`cache_prompt: false`. Our pack is Q4_0 against their BF16, so neither arm
will reach 64/64; that quantisation gap is SHARED by both arms and cancels
in the comparison. The comparison is A vs B, not either against 64.

## Prediction, frozen

If the ramp is inverted, B scores strictly higher teacher-forced agreement
than A across the 20-prompt set, and the gap is large rather than marginal:
the two mixes differ by 0.75 * theta_ex on every pair with j < 14, which is
most of the rotation at any position.

PASS: B's mean agreement over 20 prompts exceeds A's by more than 2 tokens
per prompt, and B is not worse than A on more than 2 of the 20 prompts.

FALSIFIER: B equal to or worse than A. Then the ramp direction is not the
defect on this model and the change is reverted, whatever the source
comparison says. If that happens, do NOT keep iterating on rope: the
next suspect is that ext_factor is effectively 0 in llama for this model,
which would make both mixes moot and point the disagreement elsewhere.

## What is deliberately NOT changed in this round

`tools/attn-ref.py`, `tools/draft-ref.py`, `tools/model-ref.py` and
`kernels/mega.mojo` carry the same ramp. They are only updated once the
measurement decides, so the numpy oracles keep describing arm A until then
and cannot silently drag the comparison with them.

## Result: FALSIFIER FIRED, ramp NOT changed

Valid arm, third attempt (the first two were void, below):

    armA=.work/engine-yarnA(7753b733fcabf5f8)   A ran 7753b733fcabf5f8
    armB=.work/engine-yarnB2(dbb3a84ce6a3c2c5)  B ran dbb3a84ce6a3c2c5
    A: 47 57 49 51 55 53 58 58 38 59 41 58 47 49 51 51 56 47 57 56   mean 51.90
    B: 47 56 50 51 57 53 58 56 38 61 40 58 48 49 50 50 56 45 60 56   mean 51.95
    delta +0.05; B better 5/20, worse 6/20, same 9/20

The frozen bar was "+2 tokens per prompt and worse on at most 2 of 20".
Observed is noise around zero. Per the preregistration the change is
REVERTED and rope is not iterated on. Both kernels are back to `main`'s
ramp; nothing landed.

WHAT THIS DOES AND DOES NOT SETTLE. It settles that the ramp direction has
no measurable effect on 64-token decode at positions under ~130 on this
model. It does NOT clear the ramp at long context, which is where the
defect was originally measured (rotated K dims off by 0.75*theta_ex(j) for
pairs 0-13, error growing with position, seen in the LatentOS use-1 state
diff at 32k). This gate is insensitive to that regime by construction: the
whole 20-prompt set fits in ~130 positions. A long-context arm is the only
thing that would decide it, and the preregistration did not include one.

So the honest status is: inverted against llama.cpp and HF on inspection,
no measurable consequence at short context, unmeasured at long context.
Not a bug worth landing blind, and not yet cleared either.

## Two void runs, recorded because they both looked like clean results

VOID 1. Arm B patched only `kernels/attn.mojo`. Dense decode runs the
megakernel's own copy of the same eight lines at `kernels/mega.mojo:747`,
so the changed code never executed during decode. Result read
delta -0.05 with 15 of 20 prompts byte-identical, which is impossible if 14
of 32 rotary pairs had changed; the data shape is what exposed it, not the
binary hash, which differed as expected.

VOID 2. Arm B2 patched both kernels and built a genuinely new binary
(`dbb3a84c`), but the harness loop ran `.work/engine-yarn$arm` while a `sed`
had rewritten only the arm-header line. The receipt named `dbb3a84c`; the
binary that ran was still `96752fb4`. Identical output to VOID 1 on a
supposedly different binary is what exposed it.

Both are now rules: `bench/PROTOCOL-RULES.md` P7 (the arm file is written by
the run, and equal hashes are refused) and P8 (a null result is not a
finding until the changed code is proven reached).
