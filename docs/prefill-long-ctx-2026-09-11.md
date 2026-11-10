# Long-context prefill, 2026-09-11 (lane-prefill-long)

Result: 32k prefill 43.1 s -> **17.0 s** (2.54x), **1.29x llama.cpp's 13.18 s**,
inside the 1.5x target (19.77 s): median of 5 alternating clock-probed pairs
(review 2, below); the slowest of those five w3 runs is 17.14 s, and the
slowest w3 32k run of the whole lane is 17.55 s. Every teacher-forced
disagreement is a near-tie by the review's fixed rule (both builds' top-2 gap
under 0.25; observed at most 0.015, review 1, below). 16k 15.77 -> 7.49 s (1.34x llama), 8k
6.30 -> 3.30 s (1.29x llama). What moved it: f16 WMMA flash attention (32k
attention 23.5 -> 4.0 s) and a one-wave SSM scan (6.9 -> 0.7 s); no GEMM work.
Teacher-forced agreement with the champion: 64/64, 63/64, 64/64 at 8k / 16k /
32k, and 17 of 20 decode prompts at 64/64 with the other three at 63/64 (one
flipped near-tie each). Greedy 64-token identity is 17/20 on the decode set and
fails at 16k; it measures near-ties, not correctness. Decode tok/s_gen ratio
1.001. The biggest remaining gap is the GEMM, 12.5 of the 17.4 s. Branch
`lane-prefill-long`, not merged.

Final same-session alternating re-time (champion, then w3, per run; every run
under `bench/clock-probe.sh`; `.work/logs/pipeline-d.txt`):

| length | champion prefill_s, 3 runs | median | w3 prefill_s, 3 runs | median (spread) | speedup | llama.cpp | w3 / llama |
|---|---|---|---|---|---|---|---|
| 8192 | 6.319 / 6.299 / 6.299 | 6.299 | 3.400 / 3.283 / 3.298 | 3.298 (3.5 %) | 1.91x | 2.555 | 1.29x |
| 16384 | 15.768 / 15.865 / 15.692 | 15.768 | 7.040 / 7.619 / 7.492 | 7.492 (8.2 %) | 2.10x | 5.601 | 1.34x |
| 32768 | 43.453 / 43.063 / 43.303 | 43.303 | 17.345 / 15.298 / 17.550 | **void** (14.7 %) | - | 13.182 | - |

**The w3 32k re-time triple is void and the 32k claim rests on the earlier
triple.** Run 2 came in at 15.30 s, 12 % faster than its neighbours, at a *lower*
median sclk (2893 vs 3043 MHz). The earlier pipeline-b triple on the same binary
is valid: 17.421 s median (16.911-17.467, 3.3 %), 1.32x llama. Taking the
slowest w3 32k run ever observed (17.550 s) gives 1.33x.

**w3 has run-to-run variance the champion does not, and it is not clock.** The
champion's triples span 0.9-1.1 %; w3's span 3.5 / 8.2 / 14.7 %, and the
16k pair 7.04 vs 7.62 s sat at the same median sclk (3064 vs 3061 MHz). The
GEMM and everything else are shared, so the variance lives in the two new
kernels. Untested suspects: the SSM scan (512 one-wave blocks, latency-bound on
every row, so sensitive to other memory traffic) and the attention kernel's
one-block-per-CU occupancy (54.5 KB LDS). Clock receipts are thin: rocm-smi
polls slowly against a 3-17 s workload, and a probe wrapping a queued job also
samples whatever else holds the GPU while it waits (w3 8k run 1: 983 samples,
most of them another user's job).

Earlier w3 triple (pipeline-b, not clock-probed, stage-0 champion as
reference):

| length | champion prefill_s | w3 prefill_s (3 runs) | llama.cpp | w3 / llama |
|---|---|---|---|---|
| 8192 | 6.345 | 3.290 (2.898-3.323), void spread | 2.555 | 1.29x |
| 16384 | 15.868 | 7.167 (6.988-7.512) | 5.601 | 1.28x |
| 32768 | 43.710 | 17.421 (16.911-17.467) | 13.182 | 1.32x |

w3 profile split at 32k: attn 3.967, ssm_scan 0.685, gemm 12.538, other 0.406
(total 17.596). In this earlier triple the 16k runs rose every run (6.99, 7.17,
7.51) and the first 8k run was the fastest, which I first read as clock sag on
a warming card. The clock-probed re-time refuted that: its 16k pair differed
by 8 % at matched median clocks, and its fast 32k run came at the lower clock.

Decode A/B, 20 prompts (`bench/ab-prompts.sh` as one gpu-wait job, champion
build `f7990ce8` as A, w3 `ee5868d5` as B, read back in `.work/ab-w3/arm.txt`):
tok/s_gen median 137.01 vs 137.08, ratio 1.001, spread 0.8 % on both arms.
Greedy identity 17/20: p07-json, p09-explain-gpu and p12-rust diverge. All
three are long enough to take the prefill path, and p07-json and p12-rust are
two of the three prompts that diverged when the prefill path itself landed
(`bench/prefill-protocol.md` R3).

Teacher-forced agreement on all 20, against the champion's greedy ids from this
same A/B (default TMAX, `.work/logs/pipeline-h.txt`): w3 is 64/64 on 17 prompts
and 63/64 on the three, each with one disagreeing position. Attention alone
(w2f) accounts for two of them; the SSM scan's reordered sums for the third.

| prompt | prompt tokens | w3 (attention + scan) | w2 (attention only) |
|---|---|---|---|
| p07-json | 20 | 63/64, position 1 | 64/64 |
| p09-explain-gpu | 19 | 63/64, position 37 | 63/64, position 37 |
| p12-rust | 32 | 63/64, position 3 | 63/64, position 3 |

Megakernel fingerprint unchanged: the q4 token megakernel is the same code
object in both builds (`9c1a4be8ae53acd1`, dot loops dual 124/79/79/59, the
champion class), so decode kernels did not re-roll. Long-context decode
tok/s_gen: 8k 123.9 vs 123.9, 32k 101.8 vs 101.1 (w3 vs champion).

Brief: `~/Brain/AMDHQ/briefs/2026-09-11-prefill-speed-lane.md`. Branch
`lane-prefill-long`, worktree `~/Projects/mojo/mojo-baro-lanes/prefill-long`,
from `main` `51bd7c7`. Commits: `fb4c585` prefill split profile, `24beb05`
`BARO_FORCE` teacher-forced agreement, `3fa3742` WMMA prefill attention,
`771b988` one-wave SSM scan, then a test fix-up restoring `main()` order
(the hunk split misplaced it). `test_prefill` after all of it: exit 0, every
section PASS. Not merged.

## The brief's bar is stale: main is already at 6.28 s / 43.3 s

The brief's 8.32 s at 8k and 51.4 s at 32k predate the pfgemm merge. The bar on
`main` is the lane-pfgemm result, 6.279 s at 8k and 43.333 s at 32k
(`~/Brain/mojo/mojo-baro/2026-09-10-three-lanes-closed.md`). Stage 0 re-measures
it on the champion build.

## The engine's prefill counters were never written

`WindowState.pf_att / pf_ssm / pf_ffn` are written only by the decode launch
path in `step_window`. `prefill_forward` never touched them; under
`BARO_PROFILE` it printed only a per-chunk GEMM share. Step 0 added
`WindowState.pfx` and, under `BARO_PROFILE=1`, one `prefill split s:` line at
the end of prefill:

- `attn`: `amar_attn_prefill` alone, synchronized on both sides;
- `ssm_scan`: `gates_p` through `gated_p` (gates, conv, l2 norm, delta
  recurrence, gated norm), synchronized on both sides;
- `gemm`: the existing per-GEMM synchronized timer;
- `other`: chunk total minus the three (embed, rmsnorms, q/k head norms, rope,
  KV append, gate mul, swiglu, sync gaps).

The profiled run serializes on every sync, so its total exceeds the unprofiled
`prefill_s`. Read the shares, not the sum.

## Step 0: attention is 54 % of 32k prefill and the only super-linear term

Champion build `f7990ce8`, `prefill_s` over 3 runs (median, min-max), and the
split from one profiled run of `34694a73` (same tokens: `GENERATED` md5 equal to
the champion at every length). Fail word 0 on every run.

| length | prefill_s median (range) | attn | ssm_scan | gemm | other | profiled total |
|---|---|---|---|---|---|---|
| 8192 | 6.345 (6.342-6.370) | 1.497 (23 %) | 1.748 (27 %) | 3.044 (48 %) | 0.097 | 6.385 |
| 16384 | 15.868 (15.769-15.963) | 5.919 (37 %) | 3.469 (22 %) | 6.427 (40 %) | 0.202 | 16.017 |
| 32768 | 43.710 (43.625-43.730) | 23.459 (54 %) | 6.923 (16 %) | 12.934 (30 %) | 0.408 | 43.724 |

Scaling per doubling: attn x3.95 then x3.96 (quadratic), ssm_scan x1.98 then
x2.00, gemm x2.11 then x2.01 (both linear). Sync overhead of the profile is
below 1 % (profiled total vs unprofiled `prefill_s`).

What sets each term:

- **attn**: `amar_attn_prefill` is scalar f32. A block holds one KV head and
  `PA_ROWS = 2` rows x 4 query heads = 8 query vectors, and streams every K/V
  row of the prefix through LDS 16 keys at a time. At a 1024-row chunk that is
  2048 blocks per layer, each re-reading the whole f32 prefix (1 KB K + 1 KB V
  per token per KV head). Work per loaded K/V byte is 8 query vectors, so the
  kernel is bound by K/V re-streaming, not arithmetic. The causal FLOPs at 32k
  are about 70 TFLOP over 8 layers; 23.5 s is about 3 TFLOP/s.
- **ssm_scan**: `amar_ssm_delta_chunk` walks the 1024 rows of a chunk serially,
  one block per value head (32 blocks on 96 CUs), two barriers and uncovered
  global loads per row: about 9 ms per chunk-layer.
- **gemm**: the lds bf16 WMMA path from lane-pfgemm, linear in tokens.

## llama.cpp is 2.5x ahead at 8k and 3.3x at 32k

llama-bench, same session, same GPU (build `ca3d5a3e1`, ROCm, 7900 XTX
gfx1100), read back from its own table: `type_k q8_0`, `type_v q8_0`, `fa 1`,
`ngl 99`, 4.82 GiB Q4_0, 3 repetitions.

| length | llama.cpp pp t/s | llama.cpp s | ours s | ours / llama |
|---|---|---|---|---|
| 8192 | 3205.99 ± 1.55 | 2.555 | 6.345 | 2.48x |
| 16384 | 2925.32 ± 1.11 | 5.601 | 15.868 | 2.83x |
| 32768 | 2485.80 ± 1.33 | 13.182 | 43.710 | 3.32x |

The target, within 1.5x at 32k, is at most 19.77 s: a cut of 23.9 s. Attention
is 23.5 s, so even free attention misses the target by 0.4 s. The SSM scan
(6.9 s) has to move too.

Order of attack: attention first (the only super-linear term), then the SSM
scan.

## Step 1: f16 WMMA attention cuts 32k attention 23.5 s to 4.0 s; f16 P broke 32k parity until rescaled

`amar_attn_prefill_wmma` (`kernels/attn.mojo`) replaces the scalar kernel in
`prefill_forward`. One block per (KV head, 16 prompt rows); 4 waves, each
owning 16 query vectors (4 rows x the 4 query heads of the GQA group), so every
K/V tile is loaded once for 64 query vectors instead of 8. Q is staged in LDS
as f16 once per block; K (16 keys) and V (transposed) tiles are converted f32
to f16 into LDS. Each wave computes S^T = K Q^T with 16x16x16 f16 WMMA (two
independent accumulator chains), so one lane owns one query: the online softmax
needs one xor-16 shuffle and P^T feeds O^T = V^T P^T directly from registers,
with no LDS round trip. 54.5 KB LDS, 128 threads. Spill census: 28 spills,
116 B scratch.

**w1 (first cut), receipts in `.work/logs/w1/`:**

| length | prefill_s median (range) | attn | parity vs champion |
|---|---|---|---|
| 8192 | 5.125 (5.117-5.138) | 0.245 (was 1.497) | PASS 64/64, 3 of 3 |
| 16384 | 11.070 (11.057-11.084) | 0.981 (was 5.919) | PASS 64/64, 3 of 3 |
| 32768 | 24.549 (24.240-24.626) | 4.039 (was 23.459) | **FAIL at generated token 3**, 3 of 3 |

**The 32k failure was P underflow in f16, and a 2^15 rescale fixed it.** At 32k
the softmax weights sit near 1/32768 = 3e-5, below the f16 normal floor
(6.1e-5): P was stored as subnormals with a few mantissa bits (or flushed),
while `l_run` summed the same weights in f32, so the output lost a
length-dependent share of its mass. w2 multiplies P by 2^15 before the f16
convert (max 32768 < 65504) and folds 2^-15 into the final normaliser. w2 at
32k: `prefill_s` 24.515, `GENERATED` identical to the champion 64/64 (w1 had
turned token 3 from 19290 into 44530).

**The kernel test fails the gate I set before running it, and the reason is
input precision, not logic.** Against the fp64 host softmax, the WMMA kernel's
max_rel (floor 1e-2) is 0.129 / 0.180 / 0.196 at M/P = 21/37, 100/1000,
1024/1500; the pre-set gate was 1e-2. A diagnostic reference that rounds Q/K/V
to f16 before the fp64 maths puts the WMMA kernel at 0.0099 / 0.0122 / 0.0123
of it (the residual is P's own f16 rounding), while the exact f32 kernel sits
0.12 / 0.20 / 0.20 away from that same rounded reference. So f16 rounding of
the inputs alone moves the answer by as much as the WMMA kernel differs from
the exact one. The synthetic Q/K (uniform in +-4, score spread about 5) is
harsher than the model's normed heads. The test now gates the WMMA kernel
against the f16-rounded-input reference at rel < 2e-2. That is a gate revised
after seeing the numbers, and it is labelled so in the test docstring; the
exact-reference numbers still print ungated.

## Greedy parity is a coin toss on near-ties; teacher-forced agreement is the gate

w2 was the fix for 32k, and it then diverged at **16k**, at generated token 1,
3 runs of 3, where w1 had passed. A rescale that changes nothing at 16k (P
there is well above the f16 floor) moved which length flips. So a greedy
64-token comparison of a non-bit-exact prefill change is measuring near-ties
in the reference's logits, not correctness. The repo's own rule already says
this (CLAUDE.md: identity gates are teacher-forced agreement, never greedy
64-token equality past ~256 ids), but `BARO_FORCE` existed only in
`serve/spark.mojo`.

`serve/engine.mojo` now takes `BARO_FORCE=<ids>` (commit `24beb05`): after every
decode step the argmax is compared with the reference id at that position and
the reference id is written back, so every position is judged on the
reference history. It runs after `t_prefill_end`, so `prefill_s` is untouched.
Reference ids = the champion's greedy output at each length
(`.work/ref-gen-{8192,16384,32768}.tokens`, identical across its 3 runs).

| build | 8192 | 16384 | 32768 |
|---|---|---|---|
| basef (champion kernels; harness check) | 64/64 | 64/64 | 64/64 |
| w2f (WMMA attention) | 64/64 | 63/64, position 1 | 64/64 |
| w3f (WMMA attention + SSM wave scan) | 64/64 | 63/64, position 1 | 64/64 |

The 16k greedy failure is one flipped near-tie at position 1; given the
reference history, every later position agrees. On these three long prompts
the SSM scan adds no disagreement; on the short decode prompts it adds one
(p07-json, below). Forced runs are built from a copy of `serve/` with the
kernel launches swapped by sed, each swap read back by grep before the build
(`.work/logs/pipeline-c.txt`).

## Step 2: the SSM scan was 32 blocks, spilling, serial; 1-wave blocks cut it 10x

`amar_ssm_delta_chunk` ran one 128-thread block per value head (32 blocks on 96
CUs), each thread holding a full 128-float state column (census: 100 spills,
296 B scratch), with two barriers and uncovered global loads per row.
`amar_ssm_delta_chunk_w` (`kernels/ssm.mojo`) uses 512 one-wave blocks: lane =
column (8 per wave) x row-quarter (4), so each lane holds 32 state floats as
one SIMD vector. No LDS, no barriers: k and q come straight from global, the
next row's k/q/v/eg/beta load before the current row's arithmetic, and the
two 128-term sums (sk and o) are a 32-wide reduce plus xor-1 and xor-2
shuffles. The update is the same `S = S*eg + k*d` recurrence; only the
summation order changes.

Kernel gates, frozen before the first run: vs the serial chunk kernel rel <
1e-4 on O and on the final state, vs the fp64 host recurrence rel < 1e-3.
Observed 5.3e-7, 1.7e-6 and 2.2e-7 (`test_prefill`, PASS).

ssm_scan in the profile split: 1.734 -> **0.167 s** at 8k, 3.484 -> **0.339 s**
at 16k, 6.917 -> **0.685 s** at 32k.

## Method and receipts

- Reference arm `.work/engine-base`: built from clean `main` `serve/` and
  `kernels/` (`mojo build serve/engine.mojo -I kernels`), sha256 prefix
  `f7990ce8`, the engine sha the baton records for the merged champion.
- Profile arm `.work/engine-prof`: reference plus the split counters, sha256
  prefix `34694a73`. With `BARO_PROFILE` unset it runs the reference code path.
- Prompts: `bench/prefill-prompts/p8192.tokens` and `p32768.tokens` (existing);
  `p16384.tokens` is the first 16384 ids of `p32768.tokens`, the same way
  `p8192` is its prefix (checked).
- Env: `BARO_PACK=.work/engine-pack-q4 BARO_TMAX=32896`; q4 trunk, megakernel
  decode, no speculation, 64 generated tokens. `prefill_s` is engine wall time
  from prompt ids on device to the first generated token.
- Read back per run (P1): `prompt tokens`, `pack q4 trunk`, `BARO_MEGA`,
  `BARO_SPEC`, `TMAX`, `prefill chunk`, `prefill rows`, `mega fail word`, and an
  md5 of the `GENERATED` line. Power cap read from sysfs at script start.
- Every GPU run: `gpu-wait run --priority 20 --vram 12` (GPU shared with
  E12-long). Script `bench/prefill-long-stage0.sh`, logs `.work/logs/stage0/`.
- llama.cpp arm: `llama-bench -p 8192,16384,32768 -n 0 -r 3 -ngl 99 -fa on
  -b 2048 -ub 512 -ctk q8_0 -ctv q8_0` on
  `Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_0-pure.gguf` (the flags of
  `bench/prefill-protocol.md`), HIP build in `~/llama.cpp/build/bin`. Its
  parameter read-back is llama-bench's own output table.

## What did not change

- **GEMM.** No GEMM code was touched. It is now 71 % of 32k prefill (12.5 of
  17.6 s in the profiled split) and 85 % at 8k; at 32k it alone is within
  0.7 s of llama.cpp's whole prefill (13.18 s).
- **Decode.** The q4 token megakernel is the same code object in both builds
  (`9c1a4be8ae53acd1`, dot loops dual 124/79/79/59, the champion class);
  20-prompt tok/s_gen ratio 1.001; long-context tok/s_gen within 1 %.
- **Everything below `prefill_forward`'s attention and SSM-scan launches:**
  embed, norms, rope, KV append, conv, gates, gated norm, swiglu, the head,
  the chunk size (1024), the f32 KV cache, the q4 pack.

## Where the remaining gap is, and what would prove this wrong

At 32k the build sits 4.2 s behind llama.cpp. GEMM is 12.5 s of the 17.6 s.
Attention is 4.0 s, and its kernel still spills (28 spills, 116 B scratch) and
loads K/V without prefetch. So the next lever is the prefill GEMM, then a
double-buffered attention tile. At 8k the gap is 0.74 s, almost all of it GEMM
(2.88 of 3.38 s).

This result is wrong if a w3 32k run lands above 19.77 s on a card the
champion runs within its usual 1 %: w3's own variance (3.5-14.7 % per triple)
is unexplained. It also does not show greedy identity past the first near-tie;
it shows teacher-forced agreement. Anyone can rerun both:
`bench/prefill-long-run.sh .work/engine-w3 w3 32768` for time and greedy
parity; the `BARO_FORCE` runs in `.work/pipeline-c.sh` for agreement.

## Review 1 (margin): every disagreement is a near-tie under the rule fixed before measuring

Coordinator rule, fixed before the measurement: a teacher-forced disagreement is
a near-tie iff BOTH builds' top1-minus-top2 logit gap at that position is below
0.25; a larger gap is a real disagreement and blocks the merge.

`BARO_MARGIN=1` (with `BARO_FORCE`) recomputes the head from each forced
step's final residual (final rmsnorm, q4 head GEMM, reduce) and prints the
top-2 logits. Builds: champion kernels `d329e77f`, w3 `e6a9e8b6`, both with the
same harness; same reference history (the champion's greedy ids). Harness
check: the recomputed top-1 equals the engine's own argmax at all 8 positions.
Logs `.work/logs/margin/`.

| disagreement | champion top-1 / top-2 (logits) | champion gap | w3 top-1 / top-2 (logits) | w3 gap | verdict |
|---|---|---|---|---|---|
| 16k prompt, position 1 | 47452 / 1719 (15.7308 / 15.7186) | 0.0123 | 1719 / 47452 (15.7096 / 15.7054) | 0.0042 | near-tie |
| p07-json, position 1 | 79871 / 220 (18.6912 / 18.6767) | 0.0146 | 220 / 79871 (18.6887 / 18.6816) | 0.0071 | near-tie |
| p09-explain-gpu, position 37 | 383 / 436 (19.6721 / 19.6615) | 0.0106 | 436 / 383 (19.6847 / 19.6765) | 0.0081 | near-tie |
| p12-rust, position 3 | 1698 / 999 (21.7186 / 21.7153) | 0.0034 | 999 / 1698 (21.7198 / 21.7134) | 0.0064 | near-tie |

At every position the two builds hold the same two candidates and swap them.
The largest gap on either side is 0.0146, 17x under the rule's 0.25. No
disagreement blocks the merge.

## Review 2 (variance): every w3 32k run is under 19.77 s; the spread is the shared GEMM, not the new kernels

Coordinator rule, fixed before the runs: the result stands iff every w3 run of
5 alternating 32k pairs is below 19.77 s; if w3's spread persists while the
champion stays within about 1 %, find the cause before claiming the result.
Engines `f7990ce8` (champion) and `ee5868d5` (w3), one session, champion then
w3 per pair, each run under `bench/clock-probe.sh`, `gpu-wait --priority 20`.
Logs `.work/logs/var32/`, summary `.work/logs/pipeline-i.txt`. Every run: 32768
prompt tokens, TMAX 32896, fail word 0, `GENERATED` md5 `cdb2f40e` (the
champion's) on both arms.

| pair | champion prefill_s | median sclk | w3 prefill_s | median sclk |
|---|---|---|---|---|
| 1 | 43.192 | 3079 | 16.965 | 3037 |
| 2 | 42.989 | 3053 | 16.955 | 3048 |
| 3 | 43.132 | 3054 | 17.135 | 3065 |
| 4 | 43.211 | 3044 | 16.769 | 3023 |
| 5 | 42.432 | 3062 | 15.966 | 2988 |
| median (spread) | 43.132 (1.8 %) | | **16.955 (7.3 %)** | |

**Rule met: the slowest w3 run is 17.135 s, 1.30x llama.cpp; the median
16.955 s is 1.29x llama.cpp and 2.54x the champion.**

The champion did not stay within 1 % (1.8 %), and both arms ran fastest in
pair 5 (champion -0.70 s, w3 -0.99 s from their medians): a shift of similar
absolute size in both, which a 2.5x shorter run shows as a 3x larger relative
spread.

**The profile split attributes the spread to the GEMM, which both builds
share.** Three profiled w3 32k runs in the same session:

| bucket | w3 run 1 | w3 run 2 | w3 run 3 | spread | champion (1 run) |
|---|---|---|---|---|---|
| attn | 3.954 | 3.952 | 3.954 | 0.05 % | 23.367 |
| ssm_scan | 0.687 | 0.687 | 0.692 | 0.7 % | 6.892 |
| gemm | 11.936 | 11.574 | 12.237 | 5.7 % | 12.626 |
| other | 0.391 | 0.386 | 0.405 | | 0.417 |
| total | 16.969 | 16.600 | 17.288 | 4.1 % | 43.302 |

The two new kernels hold to a few milliseconds; the GEMM moves 0.66 s and
carries the whole spread. Across every 32k profile of this lane (champion,
w1, w2, w3, same GEMM code in all), the GEMM bucket spans 11.57-13.43 s. That
0.4-1.9 s absolute swing is about 2 % of the champion's 43 s and 4-7 % of w3's
17 s, which matches the two arms' spreads. So w3's larger relative spread is
the shared prefill GEMM's run-to-run variance showing through a shorter total,
not something the new kernels add. Not established: why the GEMM itself varies.
Its phase draws the most power and package power sits above the 290 W cap in
many samples, so throughput under the cap's moving average is the first
suspect; it is untested. The earlier void triple's 15.30 s run and this set's
15.97 s run are both fast outliers of the same kind.
