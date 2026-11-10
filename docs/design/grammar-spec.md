# Grammar-constrained speculative decoding (JSON lane item 3)

Design only, no code changes. Answers item 3 of
`~/Brain/mojo/mojo-baro/briefs/2026-09-16-json-enforcement-lane.md`: per-row
window masks and matcher rollback, so a `response_format` request can keep
speculation instead of running the plain launch path at 1.253x the plain
arm's per-token cost (`bench/grammar-protocol.md` Gate 5, frozen).

## 1. Greedy (T=0) spec path

Drafts come from the MTP head, `blk32_forward` (`serve/window.mojo:166-317`),
called once for the processed rows (`serve/window.mojo:894-897`) and once per
draft step `j` in `1..m-1` (`serve/window.mojo:917-926`). Each call's
`do_head` block computes the row's argmax with `argmax_d`
(`serve/window.mojo:312`), which is `amar_argmax_pos`
(`kernels/elementwise.mojo:297-340`): a plain per-row max over `N` columns,
**no mask parameter at all**. The chosen id lands in `Dtok` and is copied into
`toks_d` by `tokcp_k` (`serve/window.mojo:925`).

Verify argmax happens in `step_window`'s `elif win_spec:` branch
(`serve/window.mojo:1512-1535`): the trunk's `m`-row logits go through
`argmax_d` again (`serve/window.mojo:1514`, same unmasked `amar_argmax_pos`)
into `Dtok`, copied to host (`b.dtok_h`) alongside the drafted ids already on
host (`b.win_h`), and the accepted count is a byte-for-byte prefix match:
`while n_acc < m - 1 and b.dtok_h[n_acc] == b.win_h[n_acc]`
(`serve/window.mojo:1522-1523`). Acceptance is pure token-id equality between
the draft and the target's own greedy argmax; there is no probability
comparison on this path.

**What changes.** Both argmaxes need to become masked, and masked with two
*different* masks:

- The **verify-side** argmax (line 1514) must use row `i`'s mask, built from
  the matcher's state assuming the first `i` draft tokens were accepted. This
  is the mask that actually enforces the grammar: since a masked argmax can
  never select a forbidden id, `win_h[i]` (the target's opinion) can never be
  a forbidden token, so a forbidden draft can never equal it and is rejected
  by the existing prefix-match, unchanged.
- The **draft-side** argmax (line 312, called from the loop at 894-926) does
  not strictly need masking for correctness (an unmasked forbidden draft is
  already rejected at verify by the point above), but leaving it unmasked
  wastes a whole draft-and-verify round trip on a token that was never going
  to be accepted. `kernels/sample.mojo`'s masked machinery already exists for
  this: `greedy_tok[MASK: Bool]` (`kernels/sample.mojo:239-274`) does exactly
  a masked block-argmax, with `NO_IDX` for an all-zero mask row, and is
  already wired into `sample_row_body`'s `temperature <= 0` branch
  (`kernels/sample.mojo:794-803`, the kernel behind `amar_sample_row_masked`,
  `kernels/sample.mojo:877-893`). No new algorithm is needed on the device
  side, only a call site: use `amar_sample_row_masked` (or a thin wrapper
  around `greedy_tok[MASK]`, since `blk32_forward`'s head is a fused
  GEMM+argmax, not a standalone sampler call) in place of `argmax_d` for the
  draft head when a grammar is attached.

**Should drafts be masked?** Yes, and it is closer to free than it looks. The
per-row mask has to be built regardless, to mask the *verify* side (point
above). Reusing that same mask to constrain the draft costs one more masked
argmax call per draft step (device-side, same order of cost as the argmax it
replaces) and can only raise acceptance, never lower it: a masked draft is by
construction a legal continuation, so it can only fail to match the target's
own (also masked) argmax for the same "model disagreement" reasons an
unconstrained greedy draft can already fail to match, never for a
grammar reason. An unmasked draft adds a whole extra rejection class (legal
target token, illegal draft token) with zero chance of ever being accepted.
Masking the draft is a pure win once the mask exists for verify anyway.

Masking the draft also simplifies the host-side matcher walk (see item 3):
because a masked draft is always grammar-legal, walking the matcher forward
with `accept()` on the *actual* drafted id at each step is a real advance,
not a hypothetical one that has to be undone. Rollback is then needed only at
the true rejection point found by verify (target disagreed with an otherwise
legal draft), not speculatively for the whole window on every call.

## 2. Sampled (T>0) spec path (A1)

`elif win_spec and cfg.sample.temperature > 0:` (`serve/window.mojo:1454-1511`).
`sample_probs_k` (`amar_sample_probs`, `kernels/sample.mojo:896-957`) fills
`Pt`, the target's truncated probability row, per verify row. `Pd` (the
draft's truncated row) was filled earlier at draft time by
`draft_draw` -> `sample_probs_1` (`serve/window.mojo:824-838`, same
`amar_sample_probs` kernel, called per draft step `j`,
`serve/window.mojo:907` and `924`). Accept/residual/bonus is
`amar_spec_accept` (`kernels/sample.mojo:960-1030`), called at
`serve/window.mojo:1489-1491` over `m - 1` rows.

`amar_spec_accept`'s three branches, all keyed only on `Pt`/`Pd`, never on the
grammar directly:

- **Accept**: `unif(...) * pd < pt` (`kernels/sample.mojo:986`), i.e.
  `min(1, pt/pd)` as a Bernoulli test on `pt(x)`, `pd(x)` at the drafted id `x`.
- **Residual**: Gumbel-max over `log(pt[e] - pd[e])` restricted to
  `r.gt(0)` (`kernels/sample.mojo:999-1009`), i.e. samples proportional to
  `max(0, pt - pd)`.
- **Bonus fallback** (residual finds nothing, `tok == NO_IDX`): Gumbel-max
  over `log(pt[e])` restricted to `pt[e] > 0` (`kernels/sample.mojo:1011-1027`).

**Does masking only the target `p` keep the rule exact? Yes**, and neither
`amar_spec_accept` nor `amar_sample_probs` needs to change (the round is
"calls those kernels, does not edit them" per
`bench/spec-sample-protocol.md:23-25`, and this holds here too):

- Mask `Pt` by zeroing forbidden ids before/while `amar_sample_probs` builds
  it for the target row (the same masked-normalization `amar_sample_row_masked`
  already does for the plain grammar path, `kernels/sample.mojo:877-893`,
  just applied to the probability-row kernel instead of the draw kernel).
  Leave `Pd` (the draft's row) unmasked.
- **Accept**: at a forbidden id, `pt(x) = 0`, so `unif(...) * pd < 0` is false
  for any `pd >= 0`. Reject, unconditionally. Exact.
- **Residual**: at a forbidden id, `r = pt - pd = 0 - pd(x) <= 0`, so
  `r.gt(0)` is false and that id never enters the residual candidate set
  (`kernels/sample.mojo:1000`). The residual distribution over the *legal*
  ids is `max(0, pt - pd)` restricted to where `pt > 0`, which is exactly
  `norm(max(0, pt_masked - pd))` renormalized over the legal set: masking
  only `pt` before the kernel runs reproduces this with no kernel change,
  because every forbidden id's contribution is already forced to `<= 0` and
  dropped by the existing `r.gt(0)` gate.
- **Bonus fallback**: draws only where `pt[e] > 0` (`kernels/sample.mojo:1017`),
  i.e. from masked `pt` directly. Exact by construction.

So the design is: build one masked-probability row (mask `Pt`, leave `Pd`
alone), and the existing `amar_spec_accept` already implements "accept only
legal drafts, residual/bonus only over legal ids" with zero kernel edits.
Whether to *also* mask `Pd`'s row at draft time (so the draft is drawn from a
grammar-legal distribution, `amar_sample_probs` on a masked row instead of
`draft_draw`'s current unmasked call, `serve/window.mojo:824-838`) is the same
tradeoff as the greedy case: not required for correctness (an illegal draft
id gets `pt(x)=0` and is rejected with probability 1 by the accept rule
above), but raises acceptance by never spending a draft on an id that cannot
be accepted, at the cost of one more masked-normalization pass per draft
step. Recommended, same as greedy.

## 3. Host side

Per generated token today (non-spec grammar path, `m == 1` only): one
`fill_mask` call (`grammar/matcher.mojo:66-69`, DFS over the token trie,
measured ~28-30us steady state, `docs/grammar.md:112-129`) plus one
`enqueue_copy` of a single `VOCAB`-bit row
(`serve/window.mojo:1628-1631`, `gmask_h`/`gmask_d`,
`serve/window.mojo:624-625`: "`m == 1` only this round, so one row is enough
while grammar requests run with spec off"). One host->device round trip per
emitted token.

**Item 3 build**, using the masked-draft design from questions 1-2 (draft
step always legal, so the matcher walk during drafting is a real advance,
not a hypothetical one):

1. Before drafting starts: `snap = matcher.snapshot()`
   (`grammar/matcher.mojo:33-35`, O(depth): copies `cur_rule`, `cur_state`,
   `call_stack`, `terminated`, `docs/grammar.md:66-72`).
2. Row 0 mask: `matcher.fill_mask(mask_row_0)`
   (`grammar/matcher.mojo:66-69`) at the pre-window state.
3. For each draft step `j` in `1..k` (the loop at `serve/window.mojo:917-926`):
   draft using `mask_row_(j-1)` (masked draft head, question 1/2), then
   `matcher.accept(drafted_id_j)` (`grammar/matcher.mojo:44-64`) with the
   *actual* chosen id (always legal, so this always succeeds), then
   `matcher.fill_mask(mask_row_j)` for the next row.
4. Upload all `k+1` rows in one buffer (`MROWS x words`, `serve/registry.mojo:58`
   for `MROWS`/`KMAX` sizing) with one `enqueue_copy`, replacing the current
   single-row `gmask_h`/`gmask_d` (`serve/window.mojo:624-625`).
5. Verify (question 1/2's masked target argmax or masked `Pt`) determines the
   real accepted count `n_acc` (`serve/window.mojo:1497-1499` greedy shape,
   `1496-1499` sampled shape).
6. If `n_acc == k`: the matcher from step 3 is already exactly where it
   should be (every drafted id it walked was in fact accepted); accept the
   bonus token (`chosen_tok`, same as the non-spec path's post-choice
   `mm2.accept(chosen_tok)` at `serve/window.mojo:1684`) and skip rollback.
   If `n_acc < k`: `matcher.rollback(snap)`
   (`grammar/matcher.mojo:37-42`, proven to reproduce identical mask
   signatures across a rollback+replay in `grammar/test_rollback.mojo:53-93`),
   then `accept()` only the first `n_acc` real ids plus whatever token verify
   actually emitted at the rejection point.

**Round trips per window, before vs after.** `fill_mask` calls stay
`O(tokens emitted)` either way (one walk per token, spec or not: today it is
one call per plain-decoded token, after item 3 it is one call per drafted
row plus the bonus row) -- the CPU-side cost is unchanged in aggregate.
What changes is the device upload: today, one `enqueue_copy` per token
(`serve/window.mojo:1631`); after item 3, one `enqueue_copy` of a `k+1`-row
buffer per *window*, i.e. up to `k+1` tokens' worth of masks in one round
trip instead of `k+1` separate ones. At `k=2` that is up to a 3x reduction in
mask-upload round trips, at the cost of the same total `fill_mask` CPU time
either way.

## 4. Megakernel

`amar_mega_token`/`amar_mega_window`'s shared body, `mega_body`
(`kernels/mega.mojo:985-1219`), fuses the head GEMM and the argmax into one
distributed, grid-barrier-synced reduction: each block scans a slice of
`VOCAB` rows (`kernels/mega.mojo:1172-1181`), partial winners are reduced
through shared memory (`kernels/mega.mojo:1182-1200`), and after a
`grid_barrier` (`kernels/mega.mojo:1201`) block 0 picks the global winner and
writes it straight into `Toks_`/`Dtok_` (`kernels/mega.mojo:1204-1218`).
There is no mask parameter anywhere in this path, and none of the intervening
launches materialize a full logits row the host or a later kernel could mask
independently (unlike the launch path's `Logitsm`, which does exist as a
buffer and is what `sample_row_masked_k` masks today). `engine.mojo` already
forces both `mega_req` and `mega_win_req` off whenever `want_grammar`
(`serve/engine.mojo:682-684, 692, 698`), which is why a grammar request never
reaches this kernel today.

**A masked megakernel head is possible**, in principle cheaply: the
per-candidate compare at `kernels/mega.mojo:1178`
(`if t[r] > bv[r] or (t[r] == bv[r] and Int32(row) < bi[r])`) is one more
scalar compare per row; gating it on a mask bit is the same shape as
`greedy_tok`'s masked reduction (`kernels/sample.mojo:265-272`) and should
add negligible device time, since the measured 1.3% mask tax on the launch
path (`bench/grammar-protocol.md`'s Gate 5 diagnostic, below) is almost
entirely host `fill_mask` + upload, not device compute. But it means
threading a new mask pointer and stride through `mega_body`'s signature
(`kernels/mega.mojo:985-1030`) and both its callers
(`amar_mega_token`, `amar_mega_window`, `kernels/mega.mojo:1223-1316`) -- a
change to the shared, `grid_barrier`-synced kernel that serves essentially
all default (non-grammar, T=0, no-penalty) decode traffic, not a change
scoped to the grammar path. It is also a kernel file under the "zero
comments" rule (`CLAUDE.md`), owned by the kernel coordinator, not the JSON
lane.

**Cost estimate, masked megakernel.** From `bench/grammar-protocol.md`'s Gate
5 numbers: plain arm with the megakernel, no grammar, no spec: 6.682 ms/token
= 149.65 tok/s. Diagnostic arm, both sides forced to the launch path
(`BARO_MEGA=0`): grammar 8.392 ms/token vs plain 8.287 ms/token, ratio
**1.013x** -- this isolates the mask fill/upload/matcher-advance tax at
1.3%, on the launch path. If that same 1.3% tax applies to a masked
megakernel (a reasonable assumption since the tax is host-side, not a
property of which kernel does the reduction), the estimate is
`149.65 / 1.013 ~= 147.7 tok/s`, i.e. **~1.3% cost against the true best
baseline**, recovering essentially all of today's 1.253x gap.

**Cost estimate, spec composition (item 3) instead.** Champion numbers at
`k=2` (`docs/BASELINE.md:76-91`, `bench/mtp-protocol.md:321,326`): spec on
150.96 tok/s vs spec off 137.02 tok/s on the same (non-mega) launch path,
a 1.102x gain. Applying that gain on top of today's masked-launch-path
number (8.371 ms/token = 119.46 tok/s, `bench/grammar-protocol.md`'s Gate 5
frozen result) gives an optimistic ceiling of
`119.46 * 1.102 ~= 131.6 tok/s`. This is optimistic because grammar masking
can only lower acceptance versus the unconstrained 0.660-0.691 rate measured
in `bench/spec-sample-protocol.md:116-121` (a forbidden draft is now
correctly rejected instead of silently accepted), so the realistic range is
`~122-131 tok/s`, i.e. **cost ~1.14x-1.22x** against the 149.65 tok/s
mega/no-grammar ceiling -- better than today's 1.253x, but item 3 can never
close the gap fully, because `engine.mojo` keeps the megakernel and
grammar mutually exclusive regardless (`serve/engine.mojo:682-698`): a
grammar+spec request still runs on the launch path, just with speculation
turned back on.

**Which is the better recovery.** Numerically the masked megakernel wins
(~1.01x vs ~1.14-1.22x estimated). Practically, it is the riskier, larger
change: a new parameter threaded through a shared, grid-synced kernel that
every non-grammar T=0 decode already depends on, owned outside the JSON
lane's current file scope (`serve/window.mojo`, `serve/engine.mojo`,
`serve/serve_proto.mojo` per `bench/grammar-protocol.md:33-36`). Item 3 stays
inside `serve/window.mojo` (the self-optimizing candidate file, already the
JSON lane's file) plus glue in `kernels/sample.mojo` reusing kernels that
already exist and are already gated. See item 6 for the recommendation.

## 5. Gates

Reuse, unmodified:

- `bench/grammar-gate.py` (gates 1, 2, 4): corpus validity, one live HTTP
  round trip, and the masked-draws-equals-accepted-equals-completion-tokens
  receipt (`serve/engine.mojo:848-852` prints
  `"grammar masked draws:" ... "accepted:" ... "terminated:"`). Under item 3
  this receipt needs to keep meaning the same thing: masked draws must equal
  accepted draws must equal completion tokens, now counted across spec
  windows instead of one-at-a-time -- the print site and its MISMATCH check
  (`serve/engine.mojo:852`) do not need to change if `st.grammar_masked_draws`
  and `st.grammar_accepted` are incremented once per emitted token regardless
  of whether it came from a draft or a plain decode step.
- `bench/grammar-cost.py` (gate 5): 20-schema median ms/token, grammar vs
  plain, same stint. Re-run after item 3 lands; the target is materially
  below 1.253x (see the estimate in question 4), not necessarily under the
  original <5% prediction, which was already shown wrong for a reason
  (megakernel loss) orthogonal to the mask cost itself.
- `bench/ab-prompts.sh` (gate 3 shape): 20-prompt forced identity,
  **no `response_format` in any request**, main vs lane build -- item 3 must
  not touch the no-grammar spec path's output, same as
  `bench/spec-sample-protocol.md`'s own Gate 1 ("T=0 unchanged, byte for
  byte").
- `grammar/test_rollback.mojo` (already passing, `PASS` at
  `grammar/test_rollback.mojo:104`): proves `snapshot`/`rollback` reproduce
  identical mask signatures and matcher state across a replay. Extend it (or
  add a sibling test) to cover the walk-forward-then-partial-rollback shape
  item 3 actually uses (accept `k` times, roll back to a snapshot taken
  before any of them, re-accept only a prefix) -- the existing test only
  exercises snapshot-at-a-point / replay-the-same-prefix-twice, not
  accept-further-then-partially-undo.

**New gate needed**, not reused: a spec-composition identity/exactness check
matching `bench/spec-sample-protocol.md`'s Gate 2 shape (device accept equals
host reference per draw, exact equality) but for the grammar-masked rows: for
fixed seeds and a real logits/mask pair, every accept/reject decision under a
masked `Pt` must match a host reference that applies the same "mask target
only" rule from question 2. This is what would catch a masking-order bug
(masking after truncation instead of before, mismatched mask row per verify
position, stale matcher state) that gate 5's throughput number cannot
distinguish from "working but slow."

**Kill line.** If the new exactness gate fails, or if `bench/ab-prompts.sh`'s
no-grammar identity regresses even by one token, revert item 3 and fall back
to today's forced-spec-off grammar path (`serve/engine.mojo:682-684`) rather
than debug live -- the grammar-off path is the one every non-grammar request
depends on and must stay untouched (`bench/PROTOCOL-RULES.md` P6, harness
before kernel).

**Effort, LOC (XS<20, S<60, M<150, L<400):**

- `grammar/matcher.mojo`: none needed, `snapshot`/`rollback`/`accept`/
  `fill_mask` already exist and are already proven correct in isolation
  (`grammar/test_rollback.mojo`).
- `serve/window.mojo`: the per-window mask-build loop (question 3, steps
  1-6), widening `gmask_h`/`gmask_d` from one row to `MROWS x words`
  (`serve/window.mojo:624-625` and the `WindowBufs` field list,
  `serve/window.mojo:502-631`), and wiring masked draft/verify calls into
  both the greedy (`serve/window.mojo:1512-1535`) and sampled
  (`serve/window.mojo:1454-1511`) branches. **M (60-150 LOC)**.
- `kernels/sample.mojo`: a masked-probability-row path for `Pt` (question 2)
  if `amar_sample_probs` needs a mask variant rather than masking `Logitsm`
  before it runs -- check whether masking `Logitsm` in place (one
  `apply` pass before `sample_probs_k`, same shape as
  `amar_apply_penalties`, `kernels/sample.mojo:1036-1062`) avoids a new
  kernel entirely; if so this item drops to **XS**, otherwise **S (20-60
  LOC)** for a masked `amar_sample_probs` variant plus a masked draft-side
  argmax wrapper for greedy (question 1).
- `serve/engine.mojo`: drop the blanket `spec = False` on `want_grammar`
  (`serve/engine.mojo:683-684`), keep `mega_req`/`mega_win_req` forced off
  (question 4: composition, not the megakernel). **XS (<20 LOC)**.
- New exactness test (sibling of `kernels/test_sample_device.mojo`'s shape,
  `bench/spec-sample-protocol.md:57-61`): **S (20-60 LOC)**.

Total: roughly M-L across the round (150-250 LOC), before the masked
megakernel option (not scoped here, see question 4 and 6).

## 6. Risks and recommendation

**Risks:**

- **Matcher-walk ordering bugs.** The host loop's correctness depends on
  accepting the *actual* drafted id at each step (question 1/3); an
  off-by-one (accepting the previous row's id when building the current
  row's mask, or filling a mask before the preceding accept lands) produces
  a mask that looks plausible but is wrong, silently changing which tokens
  get accepted rather than crashing -- exactly the class of bug
  `bench/grammar-protocol.md`'s Gate 4 receipt (masked draws == accepted)
  exists to catch, but only if that receipt still increments per-token
  under spec (question 5).
- **Rollback-after-partial-accept is untested today.** `grammar/test_rollback.mojo`
  proves snapshot/replay is stable, not the walk-forward-then-roll-back-to-
  a-shorter-prefix shape item 3 needs (question 5's new gate item).
- **Acceptance rate drop.** A grammar can force `n_acc` down independent of
  model quality (a drafted token the model liked but the schema forbids), so
  the throughput gain from composition will land below the unconstrained
  0.660-0.691 acceptance measured in `bench/spec-sample-protocol.md:116-121`
  -- report the real acceptance number per question 5's cost gate, do not
  assume the champion's 1.10x transfers.
- **Buffer sizing.** Widening `gmask_h`/`gmask_d` from a single row to
  `MROWS x words` touches shared buffer-allocation code
  (`serve/window.mojo:502-631`); get the read-back receipt (CLAUDE.md's "read
  every arm-defining parameter back from the running system") on the actual
  mask-row count used per window before trusting any throughput number.
- **Megakernel option's blast radius** (question 4): any defect in a masked
  `mega_body` risks the default non-grammar decode path used by effectively
  all traffic, not just grammar requests -- much higher cost of a mistake
  than item 3, which is scoped to a path only grammar requests take.

**Recommendation: build item 3 (spec composition) first**, not the masked
megakernel head. It stays inside the JSON lane's existing file ownership
(`serve/window.mojo`, `serve/engine.mojo`), reuses kernels already built and
already gated (`amar_spec_accept`, `amar_sample_row_masked`,
`amar_sample_probs`) with the "mask target only" argument from question 2
meaning no kernel-file edits are even required there, and its failure mode is
contained to grammar requests. It does not close Gate 5 to under 5% (the
megakernel's 1.24x advantage is structurally unavailable to any grammar
request per `serve/engine.mojo:682-698`), but the estimate in question 4
(~1.14x-1.22x, down from 1.253x) is a real improvement and the honest ceiling
for this shape of change. Treat a masked megakernel head as a distinct,
separately-gated round owned by the kernel coordinator, only after item 3's
number is in and the remaining gap is quantified against a real (not
estimated) composition result.
