# P5a draft-head self-distillation smoke protocol

This is frozen before any P5a dump, training, or acceptance run. It follows
`bench/PROTOCOL-RULES.md`, `docs/PLATFORM-PLAN.md` P5a, and the A4 receipt shape in
`exchange/lane-A4-report.md`. Every GPU command uses `gpu-wait`; no receipt means no claim.

## Scope and ownership

- Sonnet owns `bench/draft_dump.mojo`, extending dump mode only. Kernels remain untouched.
- Codex owns `tools/mtp_head.py`, `tools/mtp_train.py`, this protocol, and the receipt/report.
- Codex adds `tools/p5a_dump_roundtrip.py` as the CPU-only binary-format falsifier.
- Receipts live under `.work/team-B/<agent>/p5a/`; the report is
  `exchange/lane-P5A-report.md`.

## Frozen v2 dump format

All integers are little-endian `u32`, all probabilities and hidden values are little-endian
`f32`. The file begins with `u32 version = 2`. Each document then contains `u32 n_tok`,
`n_tok` token IDs, and `u32 n_records`. Each record contains, in order:

    u32 position
    u32 input_token
    u32 target_argmax
    u32 top8_count                 # exactly 8
    u32 top8_ids[8]                # descending target logit, token ID tie-break
    f32 top8_probs[8]              # normalized over these eight entries
    f32 hidden[H]                  # trunk final-norm h at this position

For record position `P`, `hidden` is `h_(P-1)`, `input_token` is `tokens[P]`, and the target
distribution is the trunk's output after processing `tokens[P]`, predicting `tokens[P+1]`.
Records use `1 <= P <= n_tok-2`; `target_argmax` is `top8_ids[0]`. The CPU reader must reject
wrong version, count, truncation, non-finite values, duplicate top-8 IDs, or non-normalized
probabilities. The round-trip falsifier writes a tiny synthetic v2 file, reads it back, and
checks all fields including position and hidden-row alignment before any GPU dump.

## Loss and data

The trunk is frozen. The cross-entropy target is the recorded trunk greedy target
`target_argmax`, and the added self-distillation term uses normalized-top8 KL with equal
weight:

    loss = cross_entropy(student_logits, target_argmax) + KL(q8 || p8)

`q8` is the recorded normalized `top8_probs`. `p8` is the student's softmax restricted to
the same eight token IDs, renormalized over those IDs. No tail mass is inferred or silently
discarded outside this explicitly normalized-top8 definition. Training uses `h_(P-1)` and
`tokens[P]` from each v2 record, so the target and student predict the same next position.

The smoke uses the same held-out source as A4 and does not train on the acceptance gate. The
acceptance subset is exactly `bench/mtp-prompts/p01.tokens` through `p05.tokens`; the full
identity set is all 20 files consumed by `bench/mtp-prompts.sh`. The trained and untrained
arms use the same engine binary, pack, session, and harness invocation. The receipt records
paths, pack checksum, draft-head checksum, model settings, prompt list, and the denominator.

### Dated ruling

2026-09-17, before any P5a training run: the maintainer ruled that the corrected plan document governs
this protocol. CE must target the trunk's own greedy pick, `target_argmax`, rather than the
human next token `tokens[P+1]`, because the acceptance gate measures agreement with that same
greedy pick. This amendment resolves the earlier protocol wording and is binding for the
trainer and its receipts.

## Preflight and gates

1. Run `python tools/p5a_dump_roundtrip.py` on the synthetic fixture. Exit 0 is required.
2. Run the dump reader on a real v2 sample and report record count, position range, finite
   values, top-8 normalization, and target argmax consistency. This is CPU-only.
3. Freeze the untrained five-prompt baseline and full 20-prompt greedy identity in one
   receipt before training. The A4 comparison uses aggregate `accepted/drafted`.
4. Run the smoke with `gpu-wait run --preemptible --priority 10`, the plan's 20-minute
   smoke budget, and record the admitted job and GPU minutes. Train and evaluate untrained
   and trained arms in the same stint.
5. SIGNAL requires trained aggregate `accepted/drafted` at least 4 percentage points above
   the untrained aggregate on p01 through p05, plus full-set greedy identity 20/20. A full
   identity failure kills the run regardless of lift. A miss of the lift is recorded as
   `NO SIGNAL` and parks P5a under the plan's kill line.
6. Only SIGNAL permits the optional full run: `gpu-wait run --preemptible --priority 10`,
   capped at the plan's 3 GPU-hours, with the 20-prompt acceptance and k=2 throughput claim.

Any mismatch in format, target alignment, baseline, identity, or acceptance denominator is a
kill line before GPU training. The report must include every command, exit code, receipt path,
and any unverified scope.
