# B4 stage 2b: the host-resident expert tier

Preregistered 2026-09-15, before a line of the tier was written, at commit
`c52ed6b`. Binds `bench/PROTOCOL-RULES.md`. Decision to build: the maintainer, on the
numbers in `bench/moe-locality-protocol.md` (`1aa06d5`).

## What is being built

Routed expert weights leave VRAM and live in host RAM. The trunk (attention,
SSM, norms, router, shared expert, embeddings, head) stays in VRAM. A VRAM
cache of `cap` experts per layer, laid out exactly as the pack lays out its
256, serves the top-8 gather; an LRU per layer decides what stays; a miss is
fetched over PCIe before the gate/up launch.

**No kernel changes, by construction.** `moe_gate_up_q4k_pack` indexes expert
`e` at `e * FFN * row_bytes` from a base pointer it is handed, with the gate
to up distance passed as a runtime argument, and `amar_moe_down_q4k` indexes
`e * N * row_bytes` from its own base. So a cache of `cap` experts with the
same inner layout, a base pointer into it, a gate-to-up distance of
`cap * 589,824`, and an index array holding **slots instead of expert ids**
runs the existing kernels unchanged. `serve/window.mojo` is fable's this leg
(R6.0b), so the wiring is delivered as a patch under `.work/b4/` and applied
by fable.

Sizes, from the pack index rather than from arithmetic on paper: per expert
per layer, gate 589,824 B, up 589,824 B, down 589,824 B (q4_k) or 860,160 B
(q6_k, layers 34, 38 and 39). At `cap` 64 that is 113 MB of cache per layer
and **4.53 GB over 40 layers**; the expert store in host RAM is 18.1 GB.

## P1 read-back, before any timed run

- engine sha256, built in the same command as the run.
- the tier's own startup print: capacity, cache bytes, host store bytes, and
  whether the host store is pinned or read through the page cache.
- `pack loaded in ... bytes`: with the tier on, this number must be the
  **trunk only** (about 2.9 GB, not 21 GB). If it is not, the tier pack was
  not what the engine loaded and every later number is void.
- per-run counters printed by the engine: expert references, hits, misses,
  bytes fetched over PCIe.
- power cap, vddgfx, sclk from `bench/clock-probe.sh` for any timed arm.

## Frozen predictions

1. **Live hit rate at capacity 64 reproduces the offline replay within 2
   points.** The replay says 76.7% on these 20 prompts with an LRU reset per
   prompt (`bench/moe-locality-protocol.md`). Same policy, same prompts, so
   the live number is 74.7% to 78.7%. Outside that band, one of the two is
   wrong and the report says which, with the trace to prove it: the replay
   reads `.work/b1/experts.txt`, and the live tier can emit the same rows.
2. **Bytes over PCIe per token at capacity 64: about 0.18 GB.** 0.78 GB of
   routed plus shared plus router bytes per token, of which the routed 0.573
   GB is what the cache serves, times the 23.3% miss rate, is 0.134 GB;
   adding the q6_k layers' larger experts puts it near 0.14 to 0.18 GB. At
   stage 1's measured 28.6 GB/s that is **4.7 to 6.4 ms per token** of
   transfer.
3. **Throughput, stated but not gated.** The current MoE decode is 107.27
   tok/s (R6.0, `1e270b5`), i.e. 9.3 ms per token. Adding 4.7 to 6.4 ms of
   demand-fetch transfer that nothing overlaps yet predicts **62 to 71 tok/s**
   with the tier at capacity 64, and about 107 at capacity 256 (no misses
   after warm-up). This stage does not gate on that: demand fetching without
   prefetch is stage 3's problem, and a per-layer host round trip is inherent
   to demand fetching, since the top-8 is only known after the router runs.

## Kill line

**Live hit rate below 70% at capacity 64.** Not tok/s: this stage proves the
mechanism and the locality, and a slower engine with a correct tier is a
result, not a failure. Below 70% the replay was optimistic about live
behaviour and B4's build order is rewritten around prefetch instead.

## Gates

1. **`test_expert_tier` (CPU).** The tier module's own LRU, replayed over the
   `1aa06d5` trace rows, reproduces 76.7% at capacity 64 and the same hit
   rate at 256 that the Python replay reports. This is what makes the live
   number comparable: the same code decides residency in both.
2. **Identity, 20 prompts, twice (GPU).** `GENERATED` bit-identical against
   `.work/moe-perf/engine-r60` with the tier at capacity 256 and at capacity
   64. Two capacities because a gather that depended on residency would pass
   at 256 and fail at 64, and because at 256 nothing is ever evicted.
3. **Live counters, 20 prompts (GPU).** Hit rate and bytes fetched per token
   from the engine's own counters, against predictions 1 and 2.
4. **`./run-tests.sh`, `tools/ci-checks.sh`, the kernel census**, on the tree
   being committed.

## Falsifiers

- Gate 2 failing at 256 but passing at 64, or vice versa: the slot remapping
  is wrong in a way that residency hides; report which capacity and stop.
- Gate 3's hit rate matching the replay while gate 2 fails: the LRU is right
  and the fetch is wrong, which is the more likely defect of the two.
- The pack load printing 21 GB with the tier on: the engine loaded the full
  pack and the tier is decorative. Every number in the run is void.

## Result (2026-09-15): the tier works, and every prediction held

Engine `.work/b4/engine-main`, sha256 `af50a7357db6a3f2`, built from a clean
`git archive` of `f798ed4` (the wiring as it landed on main, not the scratch
copy it was developed in). Reference `.work/b1/engine-r4base`, pack
`.work/moe-tier` (trunk 2.68 GB, experts.bin 18.33 GB). Receipts in
`.work/b4/g23main/`.

**The pack load is the first read-back and it is the one that could have
voided everything:** `loading pack: 2678180352 bytes  tier tensors: 120`. The
engine loaded the trunk only, 2.68 GB against 21 GB, so 18.3 GB of expert
weights left VRAM.

| capacity | cache VRAM | identity | hit rate | bytes/token | ms/token at 28.5 GB/s | median tok/s_gen |
|---|---|---|---|---|---|---|
| 64 | 5.22 GB | **20/20** | **0.7743** | 0.1656 GB | 5.81 | 39.06 |
| 128 | 10.4 GB | **20/20** | 0.8274 | 0.1266 GB | 4.44 | 46.75 |
| 256 | 20.9 GB | does not fit: `hipErrorOutOfMemory`, `oom log request=19.45GB` | | | | |

- **Prediction 1 held.** Live hit rate at capacity 64 is 0.7743 against the
  replay's 0.767, inside the frozen 0.747 to 0.787 band and 0.7 points above
  the replay rather than below it. The live tier sees the prefill replay steps
  as well as the 64 decode steps, which is the difference.
- **Prediction 2 held.** 0.1656 GB per token against "about 0.18", and 5.81 ms
  of transfer against the frozen 4.7 to 6.4 ms.
- **Prediction 3, stated and not gated, was pessimistic in one direction and
  optimistic in the other.** It predicted 62 to 71 tok/s from transfer cost
  alone; measured 39.06. The missing term is the one the protocol named as
  inherent: 40 host round trips per token to read the router's top-8 back
  before the fetches can be issued. `fetch_s` is 1.74 s of a 1.75 s decode on p01, so the tier's own path, not the kernels, is what the engine now waits
  on.
- **Kill line not reached.** It was below 70% live at capacity 64; measured
  77.4%.

**Gate 2 is the result worth keeping.** `GENERATED` is bit-identical to the
full-pack engine on 20 of 20 prompts at BOTH capacities. At 128 nothing is
evicted (the trace shows about 100 distinct experts per layer per request), at
64 roughly a quarter of references miss and are refetched: the gather produces
the same tokens either way, which is what proves the slot remapping is right
rather than accidentally masked by residency.

Deviation: the brief's second capacity was 256, which does not fit on this
card. 128 replaces it as the no-eviction arm, and its hit rate (0.8274) sits
just above the offline ceiling for an unlimited cache (0.805), which is the
consistency check that it really evicted nothing.


Gate 4 on `f798ed4`: `tools/ci-checks.sh` all non-GPU checks passed;
`./run-tests.sh` exit 0, 104 PASS, census `97 kernels, 52 in registry, 0
orphans`.

## Stage 3

Lane `MOE3`, plan `docs/MOE-STAGE3-PLAN.md`. Frozen before item 0's run.

### Item 0: pinned store A/B, zero code

Arm A `BARO_TIER_PINNED=0` (page-cache, the current default), arm B
`BARO_TIER_PINNED=1` (pinned host store), same engine binary, cap 64, the 20
`bench/mtp-prompts/`, one stint, `bench/clock-probe.sh` around the pair. No
file changes: the tier already prints `mode pinned|page-cache` at load
(`serve/expert_tier.mojo:218-222`), which is the P1 read-back, and per-request
`refs`, `hits`, `bytes_fetched`, `bytes_per_token` (`:355-361`), so the item's
own "add the echo if the engine does not print it" clause does not apply.

Check: both arms' load line, hit rate and bytes/token equal to 3 digits (same
LRU, same requests), identity 20/20 both arms and against the stage-2b
full-pack reference ids.

Frozen two-way falsifier (arithmetic from the measured pread cost, not a
target): **arm B at or above 55 tok/s** means the blocking pread is the
dominant term (model: 8.94 ms compute + 5.81 ms PCIe still on the critical
path + 40 sync bubbles of 50-150 us = 17-21 ms, 48-58 tok/s). **Arm B near 39
(below 43)** means it is not, and item 1's stamp table decides what is.
Either result, item 0 ships no code.

### Item 1: stamp timeline of one tier token

`BARO_TIER_STAMP=1` buckets (open/close, readback+sync, LRU touch, pread,
copy enqueue, id writeback) accumulated per layer in `ExpertTier`, printed by
`report()` as a 40-row table plus a run sum. `bench/moe-tier-stamp.py` parses
and sums a log. Check: buckets sum to `fetch_ns` within 5% on the same run;
one `rocprofv3` kernel trace of the same request (`bench/moe-launch-count.sh`
pattern) gives GPU busy time per token, and decode wall minus kernel busy
(GPU idle) must be accounted for within 20% by the host buckets that overlap
GPU idle (pread, LRU, enqueue, open/close). `bench/clock-probe.sh` sampled
during the run.

### Item 2(a): partial pin, request-scoped hot store

Item 0/1 confirmed pread dominant (arm B 67.12 tok/s >= 55; pread 54.6% of
fetch_ns, readback/sync 42.6%, cap64 vs cap256's offline ceiling only 4
points apart). Shape: a per-request, non-evicting host-pinned cache
(`BARO_TIER_HOT=1`, `BARO_TIER_HOTCAP=128` default) that remembers, per
layer, every distinct expert already fetched this request; a re-reference to
one (evicted from the 64-slot VRAM LRU, referenced again) copies directly
from it instead of paying another `pread`. First touches always `pread`
(and populate the hot store for later reuse); never touches `LayerLru`.

**Frozen prediction, from a deterministic replay of the real `1aa06d5`-style
trace** (`.work/moe3/item2/expert-trace.txt`, 20 prompts, this tree,
cross-checked 20/20 against the full-pack reference before use): of the
23.27% of references that miss the 64-slot LRU, only 16.4% (3.81% of all
references) are repeats of an expert already seen this request past HOT_CAP
eviction-free tracking; 81.9% of misses are genuine first touches no
request-scoped cache can avoid. **Predicted pread-bucket shrink: about 16%,
not the item's 50% bar.** This is a below-the-bar prediction, frozen before
the timed run specifically so a live number near it is not later read as a
surprise. Kill line unchanged: any identity miss, or a live shrink worse
than the offline number (would mean the hot store is not being reached,
P8).

## Pinned store by default (2026-09-17, the maintainer's decision, `5ad2e2d`)

`BARO_TIER_PINNED` defaults to 1. Receipt on the merged main (`.work/tier-pinned-default/`, engine
`-D BARO_MODEL=qwen35moe`, cap 64, `BARO_MEGA=0 BARO_SPEC=0`, 20 prompts, one stint under
`bench/clock-probe.sh`, sclk median 3279 MHz): page-cache arm (`BARO_TIER_PINNED=0`) 48.99 tok/s_gen,
default arm 67.54, ratio 1.379, identity 20/20, each arm's own start-up line reading `mode page-cache`
and `mode pinned` respectively (P1). Matches stage 3 item 0 (48.57 / 67.12). Cost: 18.3 GB of locked
host RAM for the engine's lifetime; `=0` restores the page cache.

## Zero-copy misses (2026-09-17, frozen before the timed run)

**Probe first** (`bench/tier_zerocopy_probe.mojo`, `.work/tier-zc/`, checksums equal across arms, medians of 11):
a kernel reading pinned host memory over PCIe with 16-byte loads reaches 24.6 GB/s at 8 pieces of
589,824 bytes and 27.2 GB/s at 24 pieces, against 21.9 GB/s for today's per-piece `enqueue_copy` path,
and 14 to 15 GB/s at 2 pieces where too few loads are in flight (the same at 32 KB chunks, so it is
parallelism, not chunking). Kernel-after-DMA (today's serialized miss cost) is 82 / 243 / 666 us at
2 / 8 / 24 pieces; the zero-copy kernel is 84 / 191 / 520 us.

**Design.** `BARO_TIER_ZC=1` (opt-in, pinned store only; the start-up line reads `mode pinned+zc`, P1).
On a miss, `prepare()` no longer enqueues the DMA for the q4_k pieces: it writes the expert's original
id into a per-launch `hoste` table (-1 for hits, `slots` still carries the cache slot) and the gate/up
and down kernels (`moe_gate_up_q4k_zc`, `amar_moe_down_q4k_zc`) read that expert's rows straight from
the pinned store while each lane stores the bytes it loaded into the assigned cache slot (read-through
fill, `q4k_dot_blocks_fill`). No second stream, no extra PCIe bytes: the fill is a VRAM write of what
the wave already holds. The three q6_k down layers (34, 38, 39) keep the DMA path, so their kernel is
unchanged. The hit path is byte-identical to today.

**Predictions.** (1) Identity: 20/20 generated sequences equal to the `BARO_TIER_ZC=0` arm, because
the fill writes the same bytes the DMA wrote and the dot reads the same values; any miss is a defect.
(2) The tier's `bytes_fetched` per token is unchanged (same misses), `copy_us` in the stamp table drops
to the q6_k share (about 3/40). (3) Decode: the miss transfer per token is about 140 MB (20-prompt
median `bytes_per_token` from the pinned receipt); at 21.9 vs 26 GB/s that is 6.4 vs 5.4 ms of a
14.8 ms token, so **predicted 67.5 to about 72 tok/s_gen, +7%, band +4% to +10%**. Below +4% means
the GEMV's read pattern does not reach the probe's parallelism regime and the round closes at the
number. **Kill line:** any identity miss, or a ratio under 1.00.

**Gate.** `bench/ab-prompts.sh` on `.work/engine-moe`, arms `BARO_PACK=.work/moe-tier BARO_TIER=64
BARO_MEGA=0 BARO_SPEC=0` with `BARO_TIER_ZC=0` (A) and `=1` (B), 20 prompts one stint under
`bench/clock-probe.sh`, identity per prompt, both start-up lines read back. Dry-run on CPU first.

**Result (2026-09-17, `.work/tier-zc/`, one stint under `bench/clock-probe.sh`, sclk median 3276 MHz,
engA = engB sha `b9f5b2f11819bdfc`, start-up lines `mode pinned` and `mode pinned+zc` read back):**
zc0 median **67.71** tok/s_gen, zc1 median **72.02**, ratio **1.064**, identity 20/20 (`ab.log`). Inside the
frozen band (+4% to +10%), below the point prediction (+7%). Parity (`kernels/test_moe_block.mojo`, zc arm):
routed output bit-identical to the device-resident q4k arm, every filled slot byte-equal to its source,
hit-only relaunch from the filled cache identical. A first, inadmissible run happened bare on the GPU
under another lane's job (the dry-run tool ran the harness for real; fixed in iTools `dcd1ecb`): it read
1.068 with identity clean, consistent with the queued run. `BARO_TIER_ZC` stays opt-in until decided.
