# BARO_TIER_CPU: tier misses on the CPU, hits on the GPU

Preregistered 2026-10-09, after the CPU kernel (`6341fc8`) and BEFORE the engine
wiring or any GPU run. Binds `bench/PROTOCOL-RULES.md` and
`bench/moe-tier-protocol.md` (same pack, cap, prompts). Plan:
`~/.claude/plans/zany-napping-zebra.md`.

## What is being built

At cap 64 a tier miss today is read over PCIe by the zero-copy kernels
(`BARO_TIER_ZC`, ON by default whenever the store is pinned,
`serve/expert_tier.mojo:240`). With `BARO_TIER_CPU=1` the miss is computed on
the CPU from the pinned store (`serve/expert_cpu.mojo`) while the GPU runs the
hits; the CPU partial (H f32) is copied up and added to `routed`. q6_k down
layers (34, 38, 39), prefill and m>1 rows keep the ZC path.

## Arms

- **A (champion/control):** `BARO_TIER=64`, pinned, ZC default. Same binary as B.
- **B:** A plus `BARO_TIER_CPU=1`.
Same engine sha in both arms; the flag is the only difference. A vs A is the
same-arm control for agreement.

## P1 read-back, every timed run

- engine sha256, built in the same command.
- tier startup line: mode must read `pinned+cpu` in B and `pinned+zc` in A.
- tier report: B must show `cpu_misses > 0` and `bytes_zc` only from q6_k
  layers; A must show `cpu_misses 0`. A run missing these lines is void.
- clock probe (`bench/clock-probe.sh`), `gpu-wait stats`, CPU governor.

## Predictions (frozen)

Basis: cap 64 hit 0.7743, 0.1656 GB of misses per token; ZC moves them at
~26 GB/s (6.4 ms of a ~13.9 ms token); the CPU kernel does ~1.8 misses per
layer in ~84 us (3.4 ms per token) and overlaps the GPU's hit kernels; one
extra 8 KB H2D + add per layer.

1. 20-prompt median tok/s_gen, B / A: **1.25x**, band **1.10x to 1.45x**.
2. B's CPU time per layer (stamp) at or under 100 us median.
3. Under 1.10x is a TARGET MISS, not a kill: the stamps name the limiter (CPU
   time vs GPU idle at the join, sync cost, thread wake) and the next round
   optimizes it.

## Identity gate (bar frozen here)

The CPU path is not bit-exact (int8 activations, sum order), so greedy
equality is not the gate. `bench/moe-gate2-force.sh` teacher-forces each arm
on the llama.cpp reference ids over `bench/mtp-prompts` (20 prompts):

- **Control:** A's own agreement, mean and min over 20.
- **Bar:** B mean agreement >= A mean - 0.5 pp AND B min over 20 >= A min - 1.0 pp.

Synthetic parity (`serve/test_expert_cpu.mojo`) decides nothing here; it only
gates the kernel build (rel 4.7e-7 vs a same-int8 reference).

## Order

1. `bench/preflight.sh --check` green with the wiring (CPU only).
2. `pack-fit` for cap 64; P1 lines read back on a 3 prompt x 32 token smoke.
3. Identity gate (both arms + control), then the 20-prompt A/B.
4. Report against predictions 1 and 2 in this file, numbers in the commit.

## Results 2026-10-09 (engine ec06673566609745, commit 746e631, pack tier-e5c7fc0d q8d)

Receipts: `exchange/receipts/tier-cpu/`.

- Smoke PASS: A `pinned+zc` cpu_misses 0, B `pinned+cpu` cpu_misses > 0.
- **Identity gate PASS:** A mean 83.05% min 62.50%, B mean 82.89% min 62.50% (bar B mean >= A - 0.5,
  B min >= A - 1.0). B equals A on 17 of 20 prompts.
- **Prediction 1 MISSED:** speed B/A **0.997** (20-prompt median 64.85 vs 65.03 tok/s_gen, resident),
  band 1.10 to 1.45. B wins only on cold-cache prompts (p02 2.39x, p03 1.76x); warm prompts 0.96-0.99x.
- **Prediction 2 HELD (per step):** CPU expert time 5.92 s over the run, about 3.6 ms per forward step
  (counters are cumulative and include prompt-replay steps, so per generated token reads higher).
  In-engine CPU rate: misses x 1,769,472 B / cpu_s = 30.9 GB/s on the stamped run (an earlier
  "about 26 GB/s" divided by generated tokens only; retracted).
- Same hit rate in both arms (0.7932 vs 0.7931), and B still moved 194.2 GB over PCIe (side-stream slot
  refills) against A's 194.3 GB. Target miss, not a kill: next is the limiter (stamps).

## Round 2 (preregistered 2026-10-09, before any of the code below exists)

Consult: `.work/tiercpu/codex-answer.md` (codex, read-only). Changes, one engine:
1. **Deferred, event-ordered refill.** prepare() records a CPU miss's slot refill instead of enqueuing it;
   after cpu_compute the main stream records an event, the refill stream waits on it, copies, records
   fill_ev. Removes the current layer's refill from competing with the CPU's RAM reads.
2. **q6_k layers keep ZC** in CPU mode (round 1 silently fell back to blocking DMA: B bytes_zc 0).
3. **Missed experts skipped on the GPU:** HE = -2 marks a CPU miss; the ZC gate/up kernel writes 0 for
   its rows and the ZC down kernel skips it. Fixes round 1's unordered read of a refilling slot. Arm A
   never writes -2, so A's kernels are bit-identical (byte test: `kernels/test_moe_block.mojo` zc arm).
4. CPU-only diagnostics of the in-engine CPU rate (30.9 GB/s vs 42-47 standalone): gapped calls, a
   competing streaming reader. Inform round 3; no engine claim.

Predictions (same metric, 20-prompt resident median, unchanged A):
- B/A **1.08**, band **1.03 to 1.15** (codex's 1.05 for change 1, plus about +2% from 2 and 3).
- B in-engine CPU rate >= 36 GB/s (from 30.9), computed as cpu_misses x 1,769,472 B / cpu_s.
- B bytes_zc > 0 (q6 layers), cpu_misses > 0; A unchanged from round 1 within 3%.
- Identity: same frozen bar as round 1.
Under band is a target miss: the stamps and step-4 diagnostics name the next limiter.

## Round 2 results 2026-10-09 (engine c12bd96acd2c68cd, commit 96d89e6, pack tier-e5c7fc0d q8d)

Receipts: `exchange/receipts/tier-cpu/r2/`.

- Smoke PASS (read-back): A `pinned+zc` cpu_misses 0; B `pinned+cpu` cpu_misses 19873.
- **Identity gate PASS:** A mean 83.05% min 62.50%, B mean 82.81% min 62.50%.
- **Prediction 1 MISSED, and worse than round 1:** B/A **0.806** (52.95 vs 65.70 tok/s_gen, 20-prompt
  resident median), band 1.03 to 1.15. B is slower on all 20 prompts (0.656 to 0.913).
- **Prediction 2 MISSED:** in-engine CPU rate 99679 x 1,769,472 B / 5.147 s = **34.3 GB/s** (target >= 36,
  round 1 30.9).
- **Prediction 3 HELD:** B bytes_zc 9.91 GB (> 0, q6 layers on ZC), cpu_misses > 0; A 65.70 vs round 1
  65.03 (+1.0%, within 3%).
- **Limiter (GPU busy per decode token, `bench/tier-gpu-busy.sh`, p01, rocprofv3):** A kernels 15.29 ms,
  copies 0.17, wall 19.17. B kernels **9.81 ms**, copies 4.10 (side-stream refills), wall **20.12**. The
  CPU path removed 5.5 ms of GPU work per token, and the wall grew: the GPU sits idle about 10 ms per
  token in B against about 4 ms in A. CPU expert time is about 3.1 ms per step (cpu_s / steps), so
  roughly 3 ms more of host time per token is outside cpu_compute. The host thread is the limiter, not
  the CPU kernel's rate and not PCIe. Ceiling if the host gap returned to A's: about 9.8 + 3.9 = 13.7 ms,
  1.4x A. Target miss, not a kill: the stamp run (b_copy, refill enqueue) names the next lever.

## Round 3 (preregistered 2026-10-09, before any GPU run of the code below)

Change: **split the misses.** `BARO_TIER_CPU_MAX=K` (default 8 = round 2) sends at most K misses per
q4_k layer to the CPU; the rest stay on the GPU's ZC path in the same kernels (HE = expert id, weight
kept). K = 0 is arm A's work with B's code path. Round 2 put about 61 misses per token on the CPU
(~1.5 per layer, up to 8), so the CPU sat on the critical path while the GPU idled about 10 ms per token.

Order: sweep K in {1, 2, 3}, resident, 3 prompts x 32 tokens (smoke shape, no claim), then the
20-prompt speed and force phases on the best K only.

Predictions (20-prompt resident median, best K):
- B/A **1.06**, band **1.02 to 1.15**. Floor reasoning: at K small the CPU's per-layer work (~50 us per
  expert) fits under the GPU's hit kernels, so B should not fall below A.
- Sweep shape: K=1 > K=3 (fewer CPU experts per layer, less exposed host time).
- B cpu_misses > 0 and bytes_zc greater than round 2's (overflow misses on ZC).
- Identity: same frozen bar as rounds 1 and 2.
Under band is a target miss: the GPU-busy profile names the next lever (persistent CPU worker, refill
enqueue cost).

## Memory receipt 2026-10-09 (engine 5fbc636 source, 3 prompts x 32 tokens resident, one process per arm)

`exchange/receipts/tier-cpu/mem/` (`run.sh`, `/usr/bin/time`, rocm-smi sampled every 0.5 s):

| arm | wall incl. load | peak RSS | peak VRAM (card) |
|---|---|---|---|
| A (ZC) | 14.62 s | 19.43 GB | 13297 MB |
| B (CPU, all misses) | 9.55 s | 19.49 GB | 13337 MB |

The CPU path adds 58 MB RSS and 40 MB VRAM; its own buffers are about 50 KB (`TierCpu`), the rest is
the per-layer event list, the side stream and allocator slack. The RSS is the 18.3 GB pinned expert
store, shared by both arms. Bytes are not the CPU path's problem; host time is (round 2). Wall
includes pack load and is one run per arm: no speed claim.

## Round 4: doorbell decode + byte-sized slots (Strata port P2 + P3, preregistered 2026-10-09 before any GPU run)

Plan: `~/.claude/plans/zany-napping-zebra.md`; Strata notes: Brain `mojo/mojo-baro/2026-10-09-strata-knowledge.md`.
P0 (`ca2f01b`): coherent `hipHostMalloc` memory hands off GPU<->host in 1.95 us vs 23.0 us for kernel+copy+sync.

Changes (one binary, arms by env):
- **P2 doorbell (`BARO_TIER_BELL`, default 1 under ZC):** per MoE layer the GPU writes ids, weights and x into
  coherent pinned memory and a 1-wave kernel spins until the host's plan is back; the host enqueues 4 layers ahead
  (`BARO_TIER_BELL_LAG`) and answers each doorbell with the existing LRU plan; no per-layer `synchronize`, no per-layer
  H2D copies. q6_k down DMAs go to a side stream the host waits on. CPU arm: the host computes misses after raising
  the plan flag, a spin kernel adds the partial; CPU-miss slots are not refilled (LRU forgets them).
- **P3 byte-sized slots (`BARO_TIER_BYTES`, default 1):** same VRAM as cap x the largest layer; each layer holds as
  many slots as fit its own expert bytes (q4_k layers more than 64).

Arms (resident, `bench/ab-resident.sh`, pack tier-e5c7fc0d):
- A0 = `BARO_TIER=64 BARO_TIER_PINNED=1 BARO_TIER_BELL=0 BARO_TIER_BYTES=0` (today's ZC champion code path)
- A1 = `BARO_TIER=64 BARO_TIER_PINNED=1` (bell + bytes)
- B1 = A1 + `BARO_TIER_CPU=1`

Predictions (20-prompt resident median, GEN 64):
- **A1 tokens byte-identical to A0** on every prompt (same kernels, same slots-to-bytes mapping; only where bytes
  come from and when the host plans changes).
- **A1/A0 = 1.12, band 1.05 to 1.25.**
- **B1/A1 = 1.10, band 0.95 to 1.25**; B1 teacher-forced agreement against A1 under the rounds 1-2 bar.
- A1 hit rate above A0's (P3); GPU idle per token (`bench/tier-gpu-busy.sh`) below A0's ~4 ms.
Under band: target miss, the idle profile and the doorbell stamps (readback_us = host spin, lru_us = plan) name the
next lever.
## Round 4 results 2026-10-09 (one engine, arms by env; receipts `exchange/receipts/tier-cpu/r4/`)

Load gate: runs waited for 1-min load < 3.5 (other sessions on the box); load sampled once a second per arm.

- **A1/A0 = 1.056** (67.34 vs 63.76 tok/s_gen, 20-prompt resident median, GEN 64): **inside the band 1.05-1.25, at
  its floor.** A1 tokens byte-identical to A0 on 20/20 prompts.
- A1 wins 16/20 prompts (+7 to +33%). p04-p07 lost (0.85-0.90) in that run only: those prompts ran at the run's load
  peak (3.72-3.79 vs ~2.85 later); the same binary in the second job ran them at 70.6/72.7/79.2/68.1, above A0's
  58.6/66.3/74.0/62.9. Load is the cause by correlation, not by a controlled rerun. The doorbell's host thread spins
  and plans every layer, so host CPU contention lands on decode directly.
- **P3 alone:** hit rate 0.765 -> 0.779 (q4_k layers 73 slots, q6_k 64, same VRAM); tokens identical.
- **B1/A1 = 0.878** (61.85 vs 70.45): **below the band 0.95-1.25, target miss.** Agreement gate not yet run; 4/20
  greedy prompts differ (expected: CPU int8 sums).
- Lag sweep (3 prompts, stamped): host enqueue lag 0/1/2/4/8 flat (69-70/66/75 tok/s).

Limiter for B (hypothesis, to measure): the host thread computes the CPU misses itself (~78 us per layer) after
raising the plan flag, so it cannot answer the next doorbell or enqueue ahead meanwhile; the GPU then waits on both
the flag2 add and the next plan. Strata runs misses on a worker pool while its host thread only plans.

## Round 5: CPU misses on a worker thread (preregistered 2026-10-09, before any timed run of the code below)

Probe (`serve/test_cpu_worker.mojo`, CPU only, `0ca3770`, `d4891e2`): a pthread (`pthread_create` through
`external_call`, entry = a Mojo `abi("C")` function) runs `expert_ffn_cpu` with 8 `parallelize` workers.
Bitwise equal to the direct call on 40 layers; n=0 job round trip 0.62 us; submit 0.02 us; n=2 job 60.5 us
via the worker against 60.5 us direct; wake from the idle sleep 20 us median.

Change (`serve/cpu_worker.mojo`, `ExpertTier.serve`): `serve()` plans, hands (layer, ids, w, n, store
offsets) to the worker and raises the plan flag; the worker converts x, computes into the coherent `part`
buffer and raises flag2. The host reaps the job before its next plan (spin on the doorbell, then collect):
only then it enqueues the CPU-miss refills on the doorbell stream and records `fill_ev`, so LRU state and
fill lists stay single-owner. `drain()` reaps the last job. Worker spins while jobs arrive, `usleep(100)`
after 2 ms idle. A1 (CPU off) takes none of these code paths.

Order of arms: A1 = `BARO_TIER=64 BARO_TIER_PINNED=1`, B1 = A1 + `BARO_TIER_CPU=1`, one engine, GEN 64,
`bench/ab-resident.sh` 20 prompts (1-min load < 3.5 before the run, as Round 4), then
`bench/tier-cpu.sh ENGINE OUT force`.

Predictions (frozen):
1. **A1 unchanged:** ids md5 `92381a907c63` on the 3-prompt smoke (`bench/tier-smoke.sh`), as before the change.
2. **B1 tokens equal Round 4's B1 tokens** (the worker is bitwise equal to the direct call): the 20 B1 ids
   match `exchange/receipts/tier-cpu/r4/speed-b-B1.gen` on every prompt (the run's A1 must match `speed-b-A1.gen` too).
3. **B1/A1 = 1.00, band 0.95 to 1.15** (20-prompt resident median; Round 4 was 0.878). Reason: 37 CPU
   layers x 78 us = 2.9 ms of host time per ~14.2 ms token leaves the host thread; the CPU work (60 us) fits
   under the GPU's hit kernels.
4. **Host plan time per CPU layer** (`BARO_TIER_STAMP=1`, 3 prompts x 32, `lru_us` summed over layers / steps)
   drops by at least 50% against the Round 4 engine (`.work/p2/eng`) on the same smoke; `cpu_wait_s`
   (host blocked on the worker) is under 2% of B1's decode wall.
5. **Agreement:** the frozen Round 1 bar (B mean >= A mean - 0.5 pp AND B min >= A min - 1.0 pp, teacher-forced).
Under the band is a target miss, not a kill: the stamp split (readback_us, lru_us, cpu_wait_s) names the
next lever.

## Round 5 results 2026-10-09 (engine 8da0eca0493ea94a, commit 977d81e, pack tier-e5c7fc0d q8d; receipts `exchange/receipts/tier-cpu/r5/`)

One gpu-wait job: 20-prompt A1/B1 (`bench/tier-cpu.sh ... speed`) then the agreement gate (`... force`), load_start 3.46, **load_max 3.86 during A and 6.05 during B** (other sessions' jobs; B is the CPU-heavy arm, so the contamination lands on it).
`bench/preflight.sh` PASS 5b602be095d5. run-tests: deferred to boss on main.

- **Prediction 1 HELD:** A1 ids md5 `92381a907c63` on the 3-prompt smoke (`bench/tier-smoke.sh`, engine 8da0eca0493ea94a); B1 smoke md5 equal too.
- **Prediction 2 HELD:** B1 ids equal Round 4's B1 on 20/20 prompts, A1 equal on 20/20 (`diff` against `speed-b-B1.gen` / `speed-b-A1.gen`): the worker is bitwise the same math.
- **Prediction 3 MISSED:** B1/A1 **0.842** (58.88 vs 69.94 tok/s_gen, 20-prompt resident median), band 0.95 to 1.15; Round 4 was 0.878. Not clean: B ran at load up to 6.05, B in-engine CPU time 6.12 s over 1639 steps against Round 4's 4.62 s (+33%, contention).
  Smoke at load 3.4 (3 prompts x 32, same box state, Round 4 engine vs the first worker build): B1 61.4/56.7/68.1 vs 60.3/55.7/67.4 tok/s, i.e. **no change from moving the compute off the host thread**.
- **Prediction 4 split:** host plan time (`lru_us` per layer per step, BARO_TIER_STAMP, smoke) 105.8 us (Round 4 engine) -> 32.2 (worker) -> 18.5 (job handed over before the fill_ev sync): HELD (-83%). `cpu_wait_s` 0.0 in every run: HELD. But the host's doorbell spin (`readback_us`) grew by the same amount (11.6 -> 16.1 -> 19.8 ms per step): the host was waiting for the GPU, so it was never the limiter.
- **Prediction 5 HELD:** teacher-forced agreement A mean 83.05% min 62.50%, B mean 82.89% min 62.50% (n=20), bar passed; identical to Round 4.

Limiter (stamps, smoke at load about 5, so magnitudes are inflated): per layer the cycle from plan-raised to the next doorbell is 519 us in B1, of which the CPU job is 180 us (submit to done seen; 172 us of it is worker compute, 60 us standalone) and the GPU's work after flag2 is 316 us (`done_seen_to_bell_us`). The GPU's own hit kernels (gate/up, down, shared expert) finish in well under the CPU job, so the CPU job, not the host thread, is on the critical path: only the GPU kernels that follow the plan can hide it, and they are shorter than the job. A worker thread cannot fix that; the lever is the job's own latency (two `parallelize` barriers at about 9 us each, int8 quantize of x on one thread, cold experts) or giving the GPU more work to overlap (not in this round).
Round 4's hypothesis that the host thread computing inline blocked the next doorbell is **retracted**: the host finished its 78 us well before the next doorbell (the cycle is 300+ us).
