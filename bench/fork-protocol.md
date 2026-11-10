# B5 fork protocol: forking a conversation into parallel branches

`docs/NEXT-PLAN.md` B5, `briefs/2026-09-15-p2-b5-fork.md`. Frozen before any
gate runs.

## Endpoint shape, decided before the build

`POST /v1/fork`, not `n` on `/v1/completions`. `n` in the real OpenAI API
means N independent completions of the *same* request (same sampler
params); B5's own claim needs branches with *different* per-branch sampler
state (seed, temperature, ...), which `n` cannot express without inventing
a parallel-array convention on top of it. A dedicated endpoint also gives
each branch its own `max_tokens`/`stop`, useful for a tree-search demo
later (B5's own listed use case). Takes `prompt` (token ids or a string,
the same shape `/v1/completions` already accepts) plus a `branches` array;
chat-message rendering (`tools`, `chat_template_kwargs`) is not in this
round -- "fork at any token" is a token-count claim, not a chat-message
one, and reusing A5's chat machinery here would just be surface area this
protocol's three gates do not need.

## Why no engine change is needed (read from source, not assumed)

`serve/engine.mojo`'s prefix-checkpoint chain (`serve/prefix.mojo`, M1a/M1b)
is already engine-global, always on under `--serve` (`BARO_CKPT` defaults
to `8` slots), and fully automatic: `chain.lookup`/`chain.restore` run on
**every** request regardless of any client-visible flag, matched by a
SHA-256 hash of `salt || tokens[0:pos]` (`serve/prefix.mojo:230-241`), and
a checkpoint is saved unconditionally at the prompt's own end position
(`wst.pos == len(prompt) - 1`, `serve/engine.mojo:593-599`) whenever
`ckpt_cap > 0`. Sending the *same* prompt token array in two separate
requests therefore already gets the second one served from a restored
checkpoint, with zero API surface built for it -- this is exactly forking's
"no recompute of the prefix" requirement, already true today for any
repeated prompt. B5's endpoint is a convenience wrapper around this
existing mechanism (N branches from one shared prompt, one call), not new
plumbing.

**Restore cost does not scale with prefix length**, also read from source,
not assumed: `CKPT_BYTES = (CONV_SLOT + SSM_SLOT) * 4` is a fixed constant
(`serve/prefix.mojo:34`) -- the SSM/conv ring state Chain.restore() copies
is the model's own constant-size recurrent state, independent of how many
tokens produced it. The KV cache itself is never copied by a checkpoint
restore at all: it is position-addressed and, per `serve/prefix.mojo`'s own
docstring, "still in place unless a later replay overwrote them." A
checkpoint restore at position 32000 costs the same as one at position
1000.

**One risk this protocol measures, not assumes away**: `ckpt_cap = 8`
total slots, and a fork-point checkpoint (saved via the plain prompt-end
path, `pinned=False, boundary=False`) sits at the *lowest* eviction
priority among the three classes `serve/prefix.mojo:284-286` defines
(periodic-class evicted before boundary-class, pinned never evicted). If
enough other periodic-grid checkpoints get created by a branch's own
decode crossing a later 1024-token boundary, the fork point's own
checkpoint could in principle be evicted before a later branch restores
it -- `cached_tokens` in each branch's own response is the check, not an
assumption.

## Gate 1: identity

Each branch's continuation at T=0 is token-for-token equal to the same
branch run as an ordinary, from-scratch `/v1/completions` request (no
fork, no checkpoint hit -- a fresh prompt of the same content forces a
cache miss the first time it is sent). Two branches at the ~1k prefix
length, both T=0 (deterministic, no seed needed to make this a fair
identity check): compare their `tokens` array byte-for-byte against a
plain `/v1/completions` call on the identical prompt+`max_tokens`. **This
is the claim; anything else in this protocol is a performance note.**

**Predicted, before running**: exact match. Nothing about MTP/spec path,
sampler path or window/prefill code differs between "restored via a
checkpoint hit" and "prefilled fresh" -- the restored state is defined to
be bit-identical to the state after a fresh prefill to the same position
(that is what M1a's own gate already proved, `bench/chat-protocol.md`
M1a); a fork request differs from an ordinary one only in *how many other
requests shared its prompt first*, which the decode path cannot see.
Falsifier: any token mismatch at T=0.

## Gate 2: cost

Wall clock of N forks (1 real prefill + N-1 restores, one `/v1/fork` call)
against N from-scratch recomputes (N independent prefills of distinct,
same-length prompts, N separate `/v1/completions` calls), at prefix
lengths ~1k, ~8k and ~32k tokens (RULER `niah_single` prompts,
`bench/ruler/gen.py`, real text, not synthetic padding), N = 4. Every
branch's own `max_tokens` kept small (8-16) so decode cost is a minor,
roughly constant term at every prefix length and does not confound the
prefill-vs-restore comparison this gate is actually about.

**Predicted shape, before measuring**: per-fork `restore_s`/`prefill_s`
in the fork arm's branches 2-4 stays flat across all three prefix
lengths, in the same rough range as M1a's already-measured 7.6 ms at 1088
tokens (`exchange/lane-chat-report.md`) -- because, per the source read
above, restore cost is O(1) in prefix length. Each arm's from-scratch
`prefill_s` (fork arm's branch 1; every branch of the recompute arm)
grows with prefix length, tracking the model's own known prefill cost at
that length; the exact current numbers are not predicted here (the
prefill-long lane's own 8k/32k figures are from an unmerged branch,
`main` may differ) -- measuring them *is* this gate. The total-wall-clock
saving from forking should grow with both N and prefix length, since it
is `(N-1) * avoided_prefill_s` and `avoided_prefill_s` itself grows with
prefix length.

**Void check**: a fork-arm branch whose `cached_tokens` (in its own
response) is not within a few tokens of the shared prompt's length is not
a restore, it is a hidden re-prefill -- reported as a failure of the cost
claim for that length, not silently averaged in.

**Sizing, the 32k arm specifically**: `BARO_TMAX` must exceed the longest
prompt used across all three lengths (set once, for the whole session,
to comfortably clear ~32636 + max_tokens); a from-scratch 32k prefill is
expensive (tens of seconds by the historical, unmerged prefill-long
lane's own before-number, `docs/prefill-long-ctx-2026-09-11.md`) -- the
32k arm runs as its own `gpu-wait` job, separate from the 1k/8k arms, so
a slow or failed run at 32k does not block reporting the other two, and
so its own GPU minutes are visible on their own line.

## Gate 3: sampler isolation

At T=0.7, four branches from one shared prompt: two with seed=1, two with
seed=2. Predicted (grounded in `kernels/test_sample_ref.mojo`'s own
reproducibility gate, run this session under `run-tests.sh`: 2000/2000
same-seed reproduced, 1780/2000 changed under a different seed): the two
seed=1 branches produce byte-identical token sequences, the two seed=2
branches produce byte-identical token sequences, and the seed=1 pair
differs from the seed=2 pair. Shown on a real `/v1/fork` request, bodies
pasted in the report, not a unit test.

## Standing gates

`./run-tests.sh` and `tools/ci-checks.sh` green at every commit; also
`tools/test_server.sh` (this lane is in its HTTP-surface territory).
`kernels/*.mojo` never edited by this lane; `serve/window.mojo` and the
spec/window config in `serve/engine.mojo` are A1's this leg, not touched.
