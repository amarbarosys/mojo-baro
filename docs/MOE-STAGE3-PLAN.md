# MoE stage 3: the host-resident expert tier without its per-miss stall (plan, 2026-09-16)

Shape and rules come from the conference's round 2 (three sonnet framings, all three retracted the
"transfer is 0.33%" arithmetic once the denominator was corrected, and converged on item 0 first).

**Lane item `MOE3` = items 0, 1 and 2 below, built and committed in that order on one branch.**

Builder: sonnet, host code only (`serve/expert_tier.mojo`, `serve/window.mojo`, `bench/`). Any edit
under `kernels/` is a request to the coordinator (fable), written to the report, not built here.
Conference: `exchange/conference/moe-stage3/` (19-builder, 12-engineer, 18-skeptic, 00-coordinator).
Report: `exchange/lane-MOE3-report.md`. Protocol: `bench/moe-tier-protocol.md`, new section "Stage 3",
frozen by commit before each item's timed run. Every item has Files / Change / Check / Receipt / Done.

Packs and receipts: split pack `.work/moe-tier` (trunk `pack.bin` + `experts.bin`, the `BARO_PACK` of
every tier run), stage-2b receipts and the full-pack reference ids under `.work/b4/g23main/`, run
commands in `bench/moe-tier-protocol.md` and `exchange/lane-B4-stage2b-report.md`. The worktree gets
`.venv`, `.work/moe-tier`, `.work/b4` and the shim build linked or built by the lane prep script.

## What the record and the code say (read before item 0)

- Full-pack MoE, experts in VRAM: 111.89 tok/s = 8.94 ms per token. Tier at cap 64: 39.06 tok/s =
  25.6 ms. **The tier adds 16.7 ms per token.** That is the number to decompose. `fetch_s` (1.74 of
  1.75 s) is wall time: `prepare()` synchronizes the stream and parks the host while the GPU finishes
  the layer, so it also contains compute the full-pack engine pays.
- The miss path is synchronous: three blocking `pread` calls per miss (about 0.59 MB each) from the
  page cache into a staging buffer, then an async copy, inside the per-layer loop, GPU idle meanwhile.
  About 217 preads per token at cap 64. Measured CPU-only on `.work/moe-tier/experts.bin`: 16.0 ms
  per token page-cache hot (8 GB/s), 145 ms cold. Cap 128 (23% fewer misses) gave +20% tok/s, the
  signature of a per-miss cost.
- `BARO_TIER_PINNED=1` already exists: pinned host store, direct H2D, no pread. Never timed.
- The previous-token repeat (39.3%) is already the LRU's hit mechanism (12-engineer): it is not a
  prefetch source. Genuine misses are unknown until the layer's own router runs. A cross-layer
  predictor has no measurement behind it and is not built in this lane.
- A device-side residency check (no host sync on a hit layer) removes the 40 per-layer sync bubbles.
  It is kernel work: coordinator item, sized after item 1's numbers.

## Item 0: pinned store A/B, zero code (S)

Files: none changed; `bench/ab-prompts.sh` (env A/B on one engine) or the served harness the
  stage-2b report used, whichever prints identity and per-prompt tok/s.
Change: none. Arm A `BARO_TIER_PINNED=0`, arm B `BARO_TIER_PINNED=1`, cap 64, 20 prompts, same
  stint, `bench/clock-probe.sh` around it.
Check: both arms print `BARO_TIER_PINNED:` (add the echo if the engine does not print it: that is
  the P1 read-back and it is a one-line harness change, allowed), hit rate and bytes per token equal
  between arms to 3 digits (same LRU, same requests), identity 20/20 between arms and against the
  stage-2b full-pack reference ids.
Receipt: `.work/moe3/item0/results.txt` plus arm.txt (power cap, vddgfx, sclk median), pasted.
Done: arm B's 20-prompt median tok/s, read as a two-way falsifier, not a target (the conference
  refused any band derived from arithmetic alone, and this one is derived from the measured pread
  cost): **arm B at or above 55** means the blocking pread is the term (model: 8.94 ms compute +
  5.81 ms PCIe still on the critical path + 40 sync bubbles of 50 to 150 us = 17 to 21 ms, 48 to 58
  tok/s, with the 12-engineer fit of the same model at cap 128 landing within 2% of the measured
  46.75); **arm B near 39 (below 43)** means it is not, and item 1 decides what is. Either way item 0
  ships nothing: the docstring of `expert_tier.mojo` already rejected holding 18 GB pinned on a box
  that runs other work, so a green arm B names the term and item 2 picks the shape.

## Item 1: stamp timeline of one tier token (S)

Files: `serve/expert_tier.mojo` (`prepare()`, `_fetch_piece()`, `report()`), the rocprofv3 pattern in
  `bench/moe-launch-count.sh`, a new parser `moe-tier-stamp.py` under `bench/` (parse and sum;
  ci-checks flags a path to a file that does not exist yet, so it is named this way here).
Change: `perf_counter_ns()` buckets accumulated per layer inside `prepare()`: open/close, readback
  enqueue + synchronize wait, LRU touch loop, pread (sum over pieces), copy enqueue, id writeback.
  Printed by `report()` as a per-layer table (40 rows) and a per-token sum, behind
  `BARO_TIER_STAMP=1`, off by default, no new side tool. Next to it, one rocprofv3 kernel trace of the
  same request (the `moe-launch-count.sh` pattern) so GPU busy time per token is known.
Check: the buckets sum to `fetch_ns` within 5% on the same run; GPU idle per token = decode wall
  minus kernel busy, and the host buckets that overlap GPU idle (pread, LRU, enqueue, open/close)
  must account for it within 20%. `bench/clock-probe.sh` sampled during the run (18-skeptic: 40 idle
  bubbles per token can deboost the clock; sclk min/median/max in the receipt).
Receipt: `.work/moe3/item1/stamps.txt` (per-layer table, both arms of item 0), the trace summary,
  the clock probe line. Pasted into the report.
Done: the dominant term of the 16.7 ms named with its number, and the per-layer sync bubble in us.

## Item 2: the miss path off the critical path, host only (M)

Built only against item 1's dominant term. Shapes, in order of what the numbers allow:
(a) If item 0 confirmed the pread: take it off the critical path without pinning the whole store.
    Two shapes, measure both on the quick subset, keep the shorter one that passes: a partial pin of
    the hot set (about 100 of 256 experts per layer are touched in a request, `1aa06d5` trace; pin
    those, page-cache the rest, so most misses copy directly), or a staging thread that preads the
    misses while the GPU runs (the layer loop enqueues the copy when the piece is ready). With either,
    12-engineer's W2(a) is back in play: stage the previous token's repeat set (39.3%) into the
    staging buffer before the round trip; a wrong guess costs a pread, never correctness, and it never
    calls `LayerLru.touch` (rule below). Files: `serve/expert_tier.mojo`.
(b) If the H2D copies dominate after (a): issue the misses' copies on a second stream
    (`DeviceContext.create_stream`, events; 19-builder verified the API exists) so they overlap the
    layer's shared-expert and attention kernels, with an event wait before the routed kernels.
    Files: `serve/expert_tier.mojo`, `serve/window.mojo` (the enqueue order around line 786).
(c) If the sync bubbles dominate: stop; write the coordinator request for the device-side residency
    check with item 1's numbers attached. Nothing host-only moves that term.
Check: identity 20/20 vs the full-pack reference at cap 64 and 128, AND three repeated 20-prompt runs
  at cap 64 bit-identical to each other (12-engineer: an in-flight overwrite race that fires one run
  in five passes a single 20/20). Hit rate per layer, bytes per token, wasted bytes per layer (a copy
  that completed after use is a miss, not a hit).
Receipt: `.work/moe3/item2/` results and stamps (item 1's table re-run on the new build).
Done: 20-prompt median tok/s at cap 64 with the band frozen from item 1's arithmetic before the run,
  and item 1's dominant bucket shrunk by at least 50% in the re-run stamps (a tok/s gain without the
  matching bucket shrink is not this item's gain: re-read the arm file). Kill line: below the frozen
  band's floor, or any identity miss.

## Gates that close the lane

- `run-tests.sh` exit 0, `tools/ci-checks.sh` exit 0, `bench/preflight.sh` before every GPU job.
- Full-pack reference ids come from the refcache key (pack sha, engine sha, T=0), never re-run.
- `gpu-wait stats --days 1` line and every job's timeout in the report.

## Rules (verbatim from the conference, binding)

- No commit touching `expert_tier.mojo`'s fetch path before item 1's table exists and is read.
- The 39.06 baseline is re-measured in the same stint as any arm it is compared with.
- No speculative `LayerLru.touch` on an id no kernel used (the replay gate assumes every touch is a
  real reference).
- No cross-layer predictor without an offline replay of its accuracy on the `1aa06d5` trace.
- Per-layer receipts, never token totals alone; clock sampled inside the run.
- Every GPU job via `gpu-wait run --timeout`; 20-prompt medians; arm parameters read back from the
  running engine; commits by pathspec; no em dashes; deliverables are files.
