# Review of kernels/gemma4_attn.mojo (main session, 2026-11-16, read-only audit of the uncommitted draft)

## Looks right (verify with your test, do not take on trust)
- Online softmax per wave and across waves (wm/wl in LDS, rescale by exp(m_wave - m_block)); split partials (m, l, o)
  + log-sum-exp merge.
- NEG = -3.4e38 is finite (dattn.mojo:19): exp(NEG - NEG) = 1, empty waves/splits give 0, not NaN. Keep NEG finite.
- Window: t_lo = max(T - WINDOW, 0) = last 512 positions incl. current, matches llama.cpp (pos_q - pos_k < n_swa) IF
  t_len counts the new token. State that contract in the report.
- Butterfly reduce: readlane(p, i*DUP) matches t_me = t0 + lane // DUP; l sums only lane % DUP == 0.
- Scale 1.0, int8 refused at comptime, slot via att_i, Q/O [head, HD].

## MUST: KV pool contract (memory corruption if ignored)
dkv_tab_off[HD_, 1, NAT] (dattn.mojo:37) pages with stride KVPAGE * HD using the kernel's own HD_. Gemma4 mixes HD 256
(S) and HD 512 (F): one shared pool / att_i space would overlap. Required: two pools, S (HD 256) and F (HD 512), each with
its own NAT and att_i counted within its kind; shared layers pass the source's index in its pool (layer 13 -> S pool,
14 -> F pool). Write this into the kernel's header comment and the report; add a test arm with both pools live.

## Performance risks (check before any speed claim)
- VGPR at HD 512: kr and vr are NLD x W floats per lane, W = 16 -> NLD 8 = 256 VGPR for K/V alone, plus q, o, part
  (lines 72-78). Read `.vgpr_count` / `scratch_` from the ISA receipt (rdna3-isa checklist 1 and 10). Likely fix: NLD 4
  at HD 512, or do not hold V until the scores are done.
- Wave combine: 8 serial passes with a barrier each (lines 131-138). Fine at decode, costs latency at small T; a single
  pass after one barrier is cheaper. Minor.

## Small
- amar_attn_merge_gemma4: no bounds checks, assumes block_dim = HD_ / 8 (32 or 64): comptime assert or document.
- bexp exp (shared fast path): include large-magnitude scores in the test so a fast-exp error shows.
- Test must cover: T in {1, 2, 511, 512, 513, 4096, 16384} x WINDOW {512, 0} x HD {256, 512}; nsplit > number of
  chunks (empty splits); the off-by-one window mutant; both pools live at once.
