# A3: continuous batching, the plan (2026-09-15)

Plan only. Nothing here is built, and the first item is not started until A2
lands, for the reason in "Why A2 is a blocker" below.

This plan is written in the shape the builder asked for. Before it existed,
a lane pane (who built B5's fork endpoint and A5's host half, and who would build
the scheduler half of this) was asked four questions: how it wants items
shaped, what would make it fail, what `docs/NEXT-PLAN.md` A3 gets wrong, and
what the first gate should be. Its answers are
`exchange/2026-09-15-p2-a3-conference.md`, they are grounded in direct reads
of `serve/src/engine.rs`, `serve/PROTOCOL.md` and `serve/prefix.mojo` rather
than in the plan's own summary of them, and the four rules it insisted on are
adopted here verbatim rather than paraphrased. Three of its corrections
changed this plan's content, not just its wording.

## Four corrections that changed the plan

1. **The Rust server is not the scheduler and must not be called one.**
   `EnginePool::submit` picks the pool member with the shortest queue and
   hands over the whole request (`serve/src/engine.rs:236-252`); the worker
   then runs one job start to finish before the next (`engine.rs:281-284`).
   There is no point in the Rust code where a batching decision could be
   inserted, because two requests are never in flight on one engine. **The
   seam has to be built in the Mojo decode loop.** A plan that says "extend
   the queue to batch requests" sends the builder hunting for a seam that
   does not exist.
2. **`BARO_POOL` is not a step toward this on this box.** One engine's MAX
   runtime reserves 23.5 to 24.8 GB of a 25.75 GB card, and the second pool
   member failed with `hipErrorOutOfMemory` on all three attempts
   (`bench/chat-protocol.md` P-J2/P-J3). `BARO_POOL` buys concurrency across
   GPUs, and this box has one. A3's concurrency must come from batching rows
   inside the single engine process.
3. **`mrows up to 8` is not request batching.** `serve/PROTOCOL.md` defines
   `mrows` as the prefill chunk and `kmax`/`spec_k` as draft width: both are
   properties of one request's own window. Reusing that machinery for rows
   from different requests means threading a request id through every row,
   which today's buffers, KV pool and sampler call sites do not carry.
   `docs/NEXT-PLAN.md`'s phrasing ("the engine already runs m up to 8 rows
   per launch") reads like a running start and is not one.
4. **The checkpoint chain is sequential-reuse, not concurrent-hold.**
   `serve/prefix.mojo`'s `Chain.restore`/`Chain.save` take a fixed slot, and
   every call site passes one active sequence's ring index. B5's forking
   works precisely because branches run sequentially through that one
   conv/ssm region. N simultaneously live regions is a memory-layout change
   inside A2, not something A3 can defer.

## Why A2 is a blocker, not a dependency

The KV pool today is a single position-addressed region for one sequence. N
concurrent sequences cannot share it without paging or a hard cap that
defeats the point of batching. `docs/NEXT-PLAN.md` says A3 "needs A2 (block
tables let different lengths share a batch)", which understates it: **without
A2 there is nothing for item (b) below to put two sequences into.** A3 starts
when A2's block tables and per-request conv/ssm regions exist.

## Items, one per wire-protocol invariant

Each item states the invariant it proves rather than the feature it ships,
names the files it touches **and the files it does not**, carries its own
protocol note written before any code, has a real-request check rather than a
unit test, and has its `ENGINE:`-shaped stopping point written in advance
rather than discovered mid-build. That last one is not ceremony: B3 and A5
both hit a real boundary this session, and the lanes that had the stopping
point written in advance lost no time to it.

### (a) The wire can carry more than one request in flight

Invariant: the engine can be told about a second request before the first has
finished, and can address its output lines to the right one, with no batching
behind it yet.

- Touches: `serve/PROTOCOL.md` (the wire delta, written first),
  `serve/serve_proto.mojo`, `serve/src/engine.rs`, `serve/src/main.rs`.
- Does not touch: `kernels/*.mojo`, `serve/window.mojo`, the sampler.
- Protocol note states the delta against today's contract in its own words:
  one request line in, `tok`/`done` lines out, strictly serial
  (`serve/PROTOCOL.md` lines 30 to 32).
- Check: a fake multi-id echo through a running `baro-serve`. Two requests
  submitted, both id streams interleaved on the wire, both complete, neither
  line misaddressed. Bodies pasted.
- Stopping point: if the engine's reader cannot accept a line while a
  request is decoding without a redesign of the request loop, stop and send
  `ENGINE:` with the exact reader change needed.

### (b) Two sequences' state coexists in VRAM without corruption

Invariant: two requests' KV, conv and SSM state live at once and neither
corrupts the other, while decoding still happens one at a time.

- Touches: A2's block tables and per-request regions, `serve/harness.mojo`
  allocation, `serve/prefix.mojo` slot handling.
- Does not touch: the decode loop's batching (there is none yet).
- Check: request A decodes 64 tokens, request B decodes 64 interleaved at
  the window boundary, and each output is byte-identical to that request run
  alone at T=0. This is the memory-layout gate; it says nothing about speed.
- Prediction to freeze before running: restore-based preemption beats
  swap-based preemption on this engine, because prefix checkpoint restore is
  **O(1) in prefix length, measured flat at 2.2 to 4.3 ms from 869 to 32,636
  tokens** (B5, `exchange/2026-09-15-p2-b5-report.md`). `docs/NEXT-PLAN.md`
  leaves preemption as an open design question citing vLLM's ablation; on
  this engine the answer is predictable in advance, so it is predicted here
  and gated rather than left open.

### (c) One decode launch advances two requests

Invariant: a single decode step produces the next token for two different
requests, each from its own state and its own sampler parameters.

- Touches: `serve/window.mojo`'s decode path and the row bookkeeping,
  per-row `SampleParams` threading.
- Does not touch: `kernels/*.mojo`. This is still a decode-loop change of the
  same class the kernel-file rule exists for, so it is coordinated with
  whoever owns the window at the time (A1 owned it this leg).
- Check, and this is the first gate that actually tests the feature: **two
  requests advanced by one decode step together, each reading back the token
  its own single-request run produces at that position, plus an explicit
  receipt that the step ran with both request ids present.** The receipt
  matters as much as the identity: two launches serialized inside one call
  are indistinguishable from today's behaviour from the outside, and would
  pass an identity-only gate. The engine prints the ids for the launch, the
  way `mega fail word` and `tok_s_gen` are already printed and read back
  rather than assumed (P1).
- **The plan's own proposed first gate is not used as the first gate.** "N=4
  concurrent clients, each output identical to its single-request run at
  T=0" already passes today with zero batching: `tools/test_server.sh`'s
  queue case is that claim at N=2 on unchanged code, and it is green. A gate
  a no-op implementation satisfies is not a gate for this feature. It stays
  as a regression check at the end, not as the gate that proves the work.

### (d) Admission and preemption

Invariant: the engine decides which in-flight requests advance in a given
step, and a preempted request resumes with identical output.

- Touches: the scheduling policy in the decode loop, and the queue on the
  Rust side only where it must expose the request's state.
- Check: N clients through `baro-serve` with mixed prompt lengths, every
  output identical to its single-request run at T=0, plus aggregate tok/s
  against the row-scaling receipt named below.

## The throughput gate, corrected

`docs/NEXT-PLAN.md` says "aggregate tok/s at m=4 against the P5 row-scaling
receipt". **P5's receipt is for the ffn GEMM shape and does not describe this
model's delta phase.** `bench/ssm-mrow-protocol.md`'s own measured table puts
the delta stage at 1.46x cost at m=2 (0.7074 to 1.0331 ms per window), the
worst-scaling stage in the sub-block, while conv, l2 and the gates sit near
1.05 to 1.06x and the profiled total comes in at 1.15x. So either the m=4
gate predicts a specific sub-linear number grounded in the delta row of that
table, or it states plainly that **m=4 needs a fresh row-scaling receipt on
the delta phase before any number can be predicted at all**. This plan takes
the second option: the receipt comes first, the prediction second, in that
order, per P5 and P2.

## What is already done and needs no design

Per-request sampler state. C3 landed `temperature`, `top_p`, `top_k`,
`min_p` and `seed` per request, A1 now runs the speculative rule under them,
and B5 exercised independent seeds live through a real fork today. The work
left is threading the existing `SampleParams` through a multi-request step so
row i gets request i's parameters, which is an implementation detail of item
(c), not an open question.

## One measurement to take before any target N is fixed

Whether MAX's roughly 24 GB per-process VRAM reservation is fixed per process
or scales with the number of tracked sequences. Nobody has measured it, and
it bounds every later gate in this plan: if it scales, the achievable N is a
function of it, and items (b) and (d) need their targets rewritten around the
measured number.
