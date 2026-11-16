# Plan: shared state store and agent features (#8 #9 #4 #5 #7 #11), 2026-11-16

Ideas: `docs/ideas/2026-11-16-innovation-ideas.md`. Inventory: survey 2026-11-16 (sources cited per row below).

## Ground truth (what the inventory changed)
- KV/SSM state handoff WORKS: E12 equals re-prefill at 8k-32k; IPC ingest 8 ms, TTFT 88 vs 121 ms; kv_ipc_bench
  40 MiB in 0.51 ms, 16/16 equal (docs/CAPABILITIES.md:40). Routes live: /v1/checkpoints*, /v1/ipc/fork.
- Hidden-state handoff is VERIFIED NEGATIVE: E8 120 items, text 79 vs latent 24-27 (Brain 2026-09-09-e8).
  miniagi latent no gain over text CoT. Bridge trainer: synthetic data only.
- Broken edges: E14 identity FAIL at N=10 (9/10); co-resident engines need BARO_MEGA=0; cross-host untested;
  /v1/latent/fork|resume return 410 (serve/src/main.rs:2274); no input-side hidden API (`hidden:true` is output only).

Consequence: #8 ships as KV-state handoff (proven), not hidden-vector handoff. Hidden vectors stay a research arm
(R1) and only re-enter the product after a latent arm beats its text control on a held-out set.

## Architecture: one store, six features
```
              StateStore  (pages: KV/SSM, refcount, COW, content hash, owner handle)
   /v1/state/{id}: fork | resume | snapshot | drop | share
     |            |             |             |              |
  #8 handoff   #9 fork/rewind  #4 prefix    #5 confidence  #7 steering  -> #11 budget planner
  (agent->agent) (branch at K)  (hash dedup)  (side channel)  (layer hook)   (picks handoff vs text)
```

## Phases (each ends on a gate a user would meet; effort in LOC)
P0 Fix the floor (M). E14 N=10 failure root-caused and fixed; BARO_MEGA co-residence or a named refusal.
   Gate: E14 identity 10/10 at N=10 and N=32, two engines co-resident.
P1 StateStore (L). Refcounted pages + COW on write + handle table, over the existing checkpoint/ipc code (reuse, not
   rewrite). Gate: reverse-arm test (store disabled must fail), 1000 fork/drop cycles, VRAM returns to baseline.
P2 #9 fork/rewind (S on P1). POST /v1/state/{id}/fork?at=K. Gate: two branches from K continue bit-equal to two fresh
   prefills of the same prefixes.
P3 #4 prefix share (M). Hash of (pack, tokens[0:n]) per page block; second agent reuses. Gate: 3 agents, same 20k
   prefix -> one copy in VRAM (measured bytes), outputs equal to no-share run; TTFT before/after in the commit.
P4 #8 agent handoff (M). Agent A hands B a state id through the HQ proxy; B continues without re-prefill.
   Replace the 410 routes. Gate: Claude-Code-style client via HQ :9015 -> :8082, two-agent task, teacher-forced
   agreement vs text handoff, TTFT and bytes logged. Same pack + same host only (named refusal otherwise).
P5 #5 confidence (S). Per-token entropy + top-2 margin in the stream (logits_topk exists); handoff carries the
   low-confidence spans. Gate: values equal an offline recompute from dumped logits.
P6 #7 steering (M). Per-request vector added at layer L via the same hook as state ingest. Gate: zero vector bit-equal
   to off; a known vector moves a held-out probe metric past a preregistered bar.
P7 #11 budget planner (M). Request field budget_ms|budget_j; planner chooses handoff vs text, quant, context trim,
   using mojo-amdgpu power. Gate: on 50 requests, met-budget rate and quality vs the always-text baseline.
R1 Research arm (parallel, no product claim): hidden-vector handoff retry on Flash (prereg exists, DRAFT) and
   bridge on real data. Promote only on a held-out win over text.

## Order and lanes
P0 -> P1 -> {P2, P3, P5} in parallel lanes -> P4 -> {P6, P7}. R1 independent, GPU-budgeted.
After gemma4 port lands (GPU contention). Lanes: Sonnet builders, Opus review on P1 and P4, codex audit per phase.

## Risks
- COW on the megakernel's KV layout: page granularity vs dkv_tab_off strides (two pools after gemma4).
- Prefix hash must include pack hash + rope/window config, or a shared page is silently wrong.
- Security: state ids are capabilities; another session must not read a page it did not get a handle for.
