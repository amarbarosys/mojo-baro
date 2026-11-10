# MoE lane: fix plan to close W3 gate 2

Written 2026-09-12 after bug #3 was localised to layer 0's SSM output.
the maintainer adopted it ("is worth, plan for fixes"). Correctness is the only prize
left in this lane: W2 measured q4k experts at 213.43 us/token/layer against a
registered 55-100, decode is 42.97 tok/s against llama.cpp's 109.445, and
qwen35moe prefill is deliberately slow by design (`d79d78a`). Nothing here
targets a number.

Every step names the check that decides it. No step is "done" on a build, a
green compile or a plausible diff.

## Phase 0 -- four XS items, independent of the blocker

Each is its own commit. Batched first because they are cheap, they are
independent of each other, and two of them make the gate itself trustworthy.

0.1 `bench/moe-gate2-force.sh` hardcodes `BARO_PREFILL=0` (since `9e5cc60`,
    two commits before the m=1 preregistration). That env forces m=1 replay
    by itself, so the script cannot tell whether the engine's own gating
    works: the lane's `d79d78a` receipt would have looked identical without
    the commit it was meant to validate. Drop the env.
    CHECK: the script reproduces the full 20-prompt table with replay forced
    by the CODE path only (`prefill rows: 0` in all 20 headers).

0.2 `serve/engine.mojo:482` gates prefill on `MEGA_ALLOWED`, which means
    "qwen35moe" only because that is the one profile with it `False`
    (`model_qwen35moe.mojo:18` False, `model_qwen35.mojo:18` True). The
    first future dense profile that turns the mega kernel off silently
    loses batched prefill. Scope it on MoE-ness instead (`IS_MOE`).
    CHECK: dense `bench/force-ab.sh` 20/20 at 64/64 against a main-built
    engine, AND all 20 qwen35moe headers still report `prefill rows: 0`.

0.3 Spec decode default flip (1.124x, identity 20/20 already measured).
    `bench/dense-run.sh` and `bench/ornith-run.sh` inherit the default and
    use `BARO_FORCE`; the engine raises at `engine.mojo:667`. Pin
    `BARO_SPEC=0` in BOTH scripts FIRST, then flip the default.
    CHECK: both scripts run green after the pin, then force-ab 20/20 at
    64/64, then the 20-prompt decode median moves by the predicted 1.124x.

0.4 `bench/force-ab-serve.sh` was committed at `ba9b832` and has never been
    run once. Either it works or it is deleted.
    CHECK: it produces a populated results table against a live server, or
    it is fixed until it does.

## Phase 1 -- the blocker: bug #4, layer 0 SSM

Evidence that put it here, single token, zero states, both sides:
attn_residual sum ours -0.202546 vs llama -0.204999 (1.2%, agrees only by
cancellation) while elements 0,1,2 are 0.0184/-0.0137/-0.0041 against
0.0173/-0.0156/-0.0023, i.e. 6%, 12% and 78% out. Implied RMS matches to 1%
(0.02202 vs 0.02226), so the direction is wrong, not the scale. This is
upstream of the MoE block, and 30 of the 40 layers run it.

Why it was never suspected: the dense profile is exact at 64/64 on both the
mega and non-mega paths, so the SSM code looked proven. It is not proven at
THIS profile's geometry (inner width 4096 against external H 2048,
`attn_gate.weight` 2048x4096, `ssm_out.weight` 4096x2048).

1.1 Add four capture points to `kernels/ssm.mojo` behind their own env flag,
    mirroring llama's tensor names so the comparison is name-for-name:
    `conv_output_silu`, `beta_sigmoid` / `a_softplus`, `attn_gated`,
    `linear_attn_out`.
    CHECK: builds clean on BOTH profiles, and dense force-ab stays 20/20 at
    64/64 with the flag off (the capture must not perturb the dense path).

1.2 Single-token layer-0 comparison against the oracle already on disk
    (`tools/llama-oracle.py`, `.work/moe-w3/llama-1tok.json`, 1398 tensors
    from `evalcb1.log`).
    CHECK: the first sub-step whose ELEMENTS (not sum) diverge beyond the
    quantisation floor is named in the report. Compare printed elements:
    layer 0 proved a sum can agree to 1.2% while components are 78% out.

1.3 Fix whatever 1.2 names.
    CHECK, in this order: layer 0 residual elements within tolerance of
    llama; then the first four layers; then the full 20-prompt table.

FALSIFIER for the phase: if layer 0's SSM elements come back correct and the
divergence is somewhere else entirely, stop and re-localise rather than
iterating inside `ssm.mojo`. Four bugs in this lane have each hidden the
next, so a fifth is likely and cheap to mis-attribute.

## Phase 2 -- close gate 2, frozen conditions, unchanged

Not reordered and not softened. All three, or the gate is not closed.

2.1 20 prompts at 64/64 teacher-forced against llama.cpp.
2.2 `./run-tests.sh` green on a CLEAN fixture. Remove `.work/draft-logits.bin`
    and `.work/draft-hn.bin` first: a moe-written pair makes `test_sample_ref`
    fail spuriously (H 2048 against main's 4096), which already cost this
    lane a false failure.
2.3 One non-empty Rust-front `/v1/chat/completions` answer with `BARO_PACK`
    set to the MoE pack.

## Phase 3 -- after the gate, not before

3.1 YaRN ramp inversion. Now a measurement rather than an argument: the
    oracle prints `Qcur`/`Kcur` before and after rope, so ours can be
    compared directly instead of reasoning from llama.cpp source. Fix is 5
    lines (`kernels/attn.mojo:266`, `kernels/mega.mojo:747`,
    `tools/attn-ref.py:53`, `tools/draft-ref.py:105`, `tools/model-ref.py:87`)
    plus a re-baseline of ~30 protocol files.
    CHECK: teacher-forced agreement on 20 prompts before and after, plus one
    long prompt.
    Deliberately after gate 2: landing it now makes any MoE failure
    unattributable.
3.2 The 5 preserved WIP branches (`lane-PROFILE` 460db75, `lane-chat`
    1535784, `lane-prefill-bif` 0196a81, `lane-tern-gemv` 3fe426a,
    `worktree-agent-a7d4d7d6` fda6529): merge, rebase or kill each on its
    own merits. Ungated and unreviewed; WIP does not mean good.
3.3 W4 our-side arm, only once gate 2 is closed. Its prefill number measures
    a knowingly slow correctness path and must be labelled as such, never
    compared against llama.cpp's batched prefill.

## Standing rules for this plan

- Verify every claim by rebuilding from the COMMITTED tree. This lane's
  receipt predated its commit five times.
- Verify a null result by binary hash. A FREQ_SCALE experiment once produced
  a byte-identical binary and would have read as a clean exoneration.
- A sum is not a comparison. Compare elements.
- A gate the candidate can write is not a gate (`CLAUDE.md`).
