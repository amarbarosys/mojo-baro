# Prefill protocol — prompt ingestion and TTFT

Binds to `bench/PROTOCOL-RULES.md` P1-P6. Question: how long from prompt token
ids to the first generated token (TTFT), for prompt lengths 32 / 128 / 512 /
1024 on the q4 pack, ours vs llama.cpp Q4_0-pure on the same token ids, and
does the prefill path change any generated token.

## Arms

- **ours/chunk8** — the pre-round engine: prompt rows through the decode
  window path, `m = min(8, remaining)` per window, `amar_matmul_skinny_q4rowb`
  MR=8, last prompt token through the megakernel. TMAX=128 admits only the
  32-token prompt; TMAX is raised (step 2 of the lane) before 128/512/1024
  can run on this arm.
- **ours/prefill** — the same engine with the prefill path: chunk C of prompt
  rows through `kernels/matmul_prefill.mojo` (bf16 WMMA over the Q4_0
  nibbles, scale applied per 32-block on the accumulator; exact products),
  causal prefill attention with online softmax (`amar_attn_prefill`), SSM
  conv + delta recurrence batched per chunk (`amar_ssm_conv_chunk`,
  `amar_ssm_delta_chunk`). v1 (R3): the last `(L-1) mod 8` rows (8 when
  that is 0) still go through the decode window path so the MTP draft sees
  the same final window as before. v2 (R4): every prompt row but the last
  goes through the prefill path; the last chunk also writes the
  post-final-norm hidden rows of that same window into `hn` and `pos_prev`
  is set so the draft's process step sees the identical window. In both
  the last prompt token still goes through the megakernel, so the first
  generated token comes from the unchanged decode kernels.
- **llama.cpp** — `llama-server` on `Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_0-pure.gguf`
  with the flags of `tools/llama-mtp-prompts.sh` (`-c 8192 -ngl 99 -fa on
  -b 2048 -ub 512 -t 8 -ctk q8_0 -ctv q8_0`), no speculation,
  `/completion` with the token ids, `n_predict 64`, `cache_prompt false`.

## Measurement

- Prompt ids: `bench/prefill-prompts/p{0032,0128,0512,1024}.tokens`, the
  first N ids of `llama-tokenize` (Q4_0-pure gguf, same tokenizer as the
  packs; `p01-water` re-tokenised identically to its `.tokens`) over the
  concatenation of the 20 `bench/mtp-prompts/*.txt` plus two docs files.
- ours: `prefill_s` printed by the engine = wall time from the prompt ids on
  device to the first argmax synchronised on the host (includes the last
  prompt token's megakernel and the head GEMM). 3 runs per length, median,
  min-max beside it. Run: `BARO_PROMPT=bench/prefill-prompts/pNNNN.tokens
  gpu-wait run --priority 60 --timeout 1800 -- .work/engine`.
- llama.cpp: `timings.prompt_ms` of the response (prompt ingestion wall
  time), `timings.prompt_n` must equal N. 3 runs per length, median.
- Both engines alone on the GPU (llama-server down for ours; ours not
  running for llama.cpp); every run through `gpu-wait run`.

## P1 receipts (read back, never assumed)

- ours: `prompt tokens: N` (N == length), `pack q4 trunk: True`,
  `BARO_MEGA: True`, `BARO_SPEC: False`, `TMAX:` and `prefill chunk:` lines
  printed by the binary (added with the integration), `mega fail word: 0`,
  and the binary rebuilt in the same command as the timed run for any
  comptime change (TMAX, C).
- llama.cpp: `GET /props` (`n_ctx`, `n_batch`, `n_ubatch`, flash_attn) saved
  next to the results; per-request `timings.prompt_n == N`, `draft_n`
  absent/0.
- kernel bench: grid/block dims and template params printed by
  `bench/bench_prefill.mojo`.

## Frozen predictions (commit before the first timed run)

TTFT in ms:

| length | ours/chunk8 | ours/prefill (C=1024) | llama.cpp Q4_0-pure |
|---|---|---|---|
| 32 | 57 | 30 | 40 |
| 128 | 200 | 55 | 80 |
| 512 | 780 | 170 | 250 |
| 1024 | 1550 | 330 | 500 |

Reasoning: chunk8 streams the 3.75 GB q4 trunk once per 8 rows at ~2.3x the
m=1 window time (mrow receipt), ~12 ms/window; prefill is compute-bound above
~64 rows at 13.8 GFLOP/token on a matrix path measured at 70% of 512
FLOP/clk/CU (`wmma-fp16-protocol.md`), taken at ~50 TFLOP/s effective after
dequant and scale epilogue; llama.cpp from its MMQ prompt rate on this card
(~2000 tok/s at Q4_0, ub 512).

Kernel arm on the ffn shape (N=12288, K=4096, q4, us per GEMM, NBUF=8
rotation so the weight stream is cold):

| n | WMMA (predicted) | wave-per-row n/8 passes (predicted) |
|---|---|---|
| 16 | 60 | 270 |
| 64 | 130 | 1100 |
| 256 | 500 | 4300 |
| 1024 | 1900 | 17000 |

Prediction: WMMA wins at every n >= 16; the wave-per-row kernel cannot hold
more than 8 rows of accumulators per lane (MR=8 is its register ceiling,
`mrow-gemm-protocol.md`), so its n-row variant IS the n/8-pass loop.

Decode must not move: no-spec 20-prompt median tok/s_gen before and after,
same session, within 1%; `bench/mtp-prompts/` identity 20/20 and
`ref-tokens-64` 64/64 on the q4 pack with the prefill path active.

## Results

### R1. Baseline receipts (chunk8 arm), 2026-09-06

Binary `.work/engine-base` (source of `751bc3c`, TMAX=128) and
`.work/engine-tmax` (same source, TMAX=1088 / MAX_T=1088 only). Read-back
per run: `pack q4 trunk: True`, `BARO_MEGA: True`, `prompt tokens: N`,
`mega fail word: 0`. Identity: `tools/check-tokens.sh` on the default prompt
PASS 64/64 for both binaries; decode `tok/s_gen` on the default prompt
130.9 (tmax) — the TMAX raise moves nothing (20-prompt no-spec median with
`engine-base`: 131.3, min 126.4, max 131.6; logs `.work/prefill-base/`).

TTFT = `prefill_s`, 3 runs each (ms):

| length | TMAX=128 | TMAX=1088 (chunk8) | predicted |
|---|---|---|---|
| 32 | 214.5 / 217.2 / 215.9 | 216.7 / 214.4 / 216.5 | 57 |
| 128 | blocked | 831.5 / 834.2 / 833.2 | 200 |
| 512 | blocked | 3306 / 3326 / 3333 | 780 |
| 1024 | blocked | 6701 / 6675 / 6686 | 1550 |

The chunk8 prediction was 4x optimistic: the m=8 window on the q4 row
kernel costs ~50 ms, not ~12 ms (the q4rowb MR=8 instantiation streams the
28 MB ffn weight in 293 us = 97 GB/s, R2 below; the m=1 figure of 59 us
does not carry to MR=8). Logs `.work/prefill-tmax/`.

### R2. Kernel arm, ffn shape (`bench/bench_prefill.mojo`, `.work/stage1.log`)

N=12288, K=4096, Q4_0 `blk.0.ffn_gate.weight`, NBUF=8 rotation, ITERS=16,
median of 5 (us per GEMM); grid/block echoed by the bench. Correctness
gate (WMMA vs row loop rel < 1e-4 at every n): true.

| n | wmma 16x256 | wmma 32x256 | wmma 64x128 | wmma 128x128 | rowloop q4rowb MR8 x n/8 | best wmma TFLOP/s | predicted wmma | predicted rowloop |
|---|---|---|---|---|---|---|---|---|
| 16 | 355 | 339 | 333 | 443 | 587 | 4.8 | 60 | 270 |
| 32 | 656 | 548 | 533 | 535 | 1126 | 6.0 | - | - |
| 64 | 1008 | 729 | 637 | 621 | 2242 | 10.4 | 130 | 1100 |
| 128 | 2175 | 1384 | 1269 | 677 | 4480 | 19.0 | - | - |
| 256 | 3862 | 2393 | 2137 | 1651 | 8937 | 15.6 | 500 | 4300 |
| 512 | 6214 | 4499 | 4045 | 2597 | 17881 | 19.8 | - | - |
| 1024 | 10297 | 7112 | 7473 | 4522 | 35934 | 22.8 | 1900 | 17000 |

Verdict: WMMA wins at every n (1.8x at 16, 7.9x at 1024) — the direction of
the prediction holds, the magnitude does not: the prefill kernel reaches
22.8 TFLOP/s at n=1024 (predicted ~50) and only 85 GB/s of weight stream at
n=16 (48 blocks for 12288 columns: too few waves in flight for a
bandwidth-bound shape). The row loop is 2x slower than predicted for the
same MR=8 reason as R1. Dispatch adopted from this table: 16x256 is never
the best; 64x128 for n <= 64, 128x128 above.

### R3. Engine v1 (prefill chunk C=1024, tail window on the decode path), 2026-09-06

Binary `.work/engine` built from the integrated source (TMAX=1088, CP=1024,
PF_MIN=16); `.work/stage2.log`, logs `.work/prefill-new/`. Read-back per
run: `prompt tokens: N`, `TMAX: 1088  prefill chunk: 1024  prefill rows: R`
(R = N-1 minus the tail window), `mega fail word: 0`, `tokens: 64`.
`kernels/test_prefill.mojo` PASS in the same job (first line of the log).

TTFT = `prefill_s`, 3 runs each (ms):

| length | prefill rows | ours/prefill v1 | ours/chunk8 (R1) | llama.cpp Q4_0-pure `prompt_ms` | predicted prefill | predicted llama |
|---|---|---|---|---|---|---|
| 32 | 24 | 125.2 / 125.0 / 124.1 | 216 | 77.7 / 56.9 / 52.4 | 30 | 40 |
| 128 | 120 | 185.3 / 183.1 / 184.3 | 833 | 73.3 / 73.4 / 73.3 | 55 | 80 |
| 512 | 504 | 488.1 / 489.0 / 490.7 | 3326 | 165.5 / 167.5 / 166.8 | 170 | 250 |
| 1024 | 1016 | 853.2 / 865.2 / 857.3 | 6686 | 315.3 / 313.8 / 315.1 | 330 | 500 |

llama.cpp arm: `.work/run-llama-ttft.sh` (server flags as frozen, port
8097), `.work/prefill-llama/props.json` (`n_ctx 8192, n_batch 2048,
n_ubatch 512, flash_attn on`), per run `prompt_n == N`, `draft_n 0`,
`.work/prefill-new/llama-ttft.txt`; ours was not running during the llama
arm and vice versa.

Verdict on the frozen predictions: direction right, magnitude wrong on
both sides. ours/prefill is 1.7x (32) to 7.8x (1024) faster than chunk8 but
2.6-4.2x slower than the predicted TTFT: (a) the GEMM reaches 22.8 not 50
TFLOP/s (R2); (b) below 64 rows the path is weight-stream-bound and the
kernel only reaches 85 GB/s of the 3.75 GB trunk (R2 n=16); (c) the tail
window on the decode path (q4rowb MR=8, ~50 ms, R1) is paid by every
prompt — 40% of the 32-token TTFT. llama.cpp is 2.3x faster than
predicted at every length (its MMQ prompt path runs ~3300 tok/s here, not
~2000). Net: ours/prefill v1 = 2.7x (1024) to 2.3x (32) slower than
llama.cpp.

Identity (`.work/stage2.log`): first generated token equals the chunk8
arm at all four lengths (264 / 2469 / 16 / 3706); 64-token stream equals
the chunk8 arm at all four lengths; default prompt 64/64 vs
`ref-tokens-64.txt` (5 prompt tokens: below PF_MIN, the prefill path is not
exercised by this gate).

Decode (P4, 20 prompts, same session, `.work/prefill-new/decode.txt`):
no-spec median 130.7 tok/s_gen (min 129.8, max 131.1) vs 131.3 before
(-0.5%, within the 1% bound); spec k=2 median 151.6, `identity_B` 20/20.
`identity_vs_base` (64-token stream vs `engine-base`) 17/20: p07-json
(n=20, diverges at generated token 1), p12-rust (n=32, token 3), p15-bash
(n=18, token 14) differ; every one of the 20 first tokens is unchanged.
Nine of the 20 prompts have >= 17 tokens and take the prefill path; all
three divergences are among those nine, the other six match 64/64. The
prefill GEMM sums each row in a different order (WMMA 16-wide products,
scale on the block accumulator) from `q4rowb`, so the token stream after
a near-tie is not expected to be bit-identical; R4 settles whether these
three are near-ties (numpy reference `tools/model-ref.py` on the same
prompts, plus a chunk-size invariance check) or a defect.

### R4. Engine v2 (every prompt row but the last through the prefill path; f32-cvt nibble dequant), 2026-09-06

Two changes over v1, both in `.work/stage3.log` (logs `.work/prefill-v2/`):
(1) the tail window no longer goes through the decode path — the last
chunk writes the post-final-norm hidden rows of the last `(L-1) mod 8`
rows (8 when 0) into `hn` and sets `pos_prev` so the MTP draft's process
step sees exactly the window it saw before; (2) `f32x16_to_bf16_trunc` in
`kernels/matmul_prefill.mojo`: the nibble goes uint8 -> f32 -> bf16 by
`llvm.amdgcn.perm` truncation (exact for integers in [-8, 7] and
[-128, 127]) instead of the int8 -> bf16 cast chain. Same read-back per run
as R3 (`prefill rows: N-1` now).

TTFT = `prefill_s`, 3 runs each (ms):

| length | prefill rows | ours/prefill v2 | ours/prefill v1 (R3) | ours/chunk8 (R1) | llama.cpp (R3) | predicted prefill |
|---|---|---|---|---|---|---|
| 32 | 31 | 65.3 / 66.0 / 66.8 | 125 | 216 | 56.9 | 30 |
| 128 | 127 | 124.7 / 124.6 / 124.3 | 184 | 833 | 73.3 | 55 |
| 512 | 511 | 415.4 / 418.0 / 414.6 | 489 | 3326 | 166.8 | 170 |
| 1024 | 1023 | 761.4 / 759.7 / 762.7 | 857 | 6686 | 315.1 | 330 |

v2 vs chunk8: 3.3x (32) to 8.8x (1024). v2 vs llama.cpp: 1.16x slower at
32, 1.7x at 128, 2.5x at 512, 2.4x at 1024 — the gap is the GEMM's 24.6
TFLOP/s (kernel arm below) against llama.cpp's MMQ path.

Kernel arm re-run with the f32-cvt dequant (`.work/prefill-v2/bench.txt`,
same setup as R2, `correct: true`), us per GEMM, 128x128 config: n=128
645 (R2 677), 256 1629 (1651), 512 2451 (2597), 1024 4183 (4522) = 24.6
TFLOP/s (+8%); 64x128 at n=64 512 (637). At n=16 every config read
435-470 this run against 333-443 in R2 — the n=16 shape is latency-bound
and moves between sessions; it is not resolved by this round.

Identity (`.work/stage3.log`): first token and 64-token stream equal the
chunk8 arm at all four lengths; default prompt 64/64; `BARO_PREFILL=0` on
p07-json and p12-rust reproduces `engine-base` 64/64 (the decode path is
untouched); chunk-size invariance `BARO_PREFILL_C=16` vs 1024 on p12-rust
(24 rows -> 16+8) and p1024 (64 chunks) 64/64 (chunk boundaries carry the
conv/SSM/KV state exactly).

Decode (P4, 20 prompts, `.work/prefill-v2/decode.txt`): no-spec median
130.7 tok/s_gen (min 129.9, max 131.6) vs 131.3 before (-0.5%); spec k=2
median 151.0, `identity_B` 20/20. `identity_vs_base` 18/20: p07-json
(token 1) and p09-explain-gpu (token 37) differ; p12-rust and p15-bash,
which differed in v1, match again in v2, and all 20 first tokens match.
The numpy fp32 reference (`tools/model-ref.py decode`, run on the same
prompt ids from `.work/ref-p07-json/`) sides with the prefill path on
p07-json: reference `198 79871 763 328 760 8252 ...` == prefill v1/v2,
while `engine-base` produced `198 220 328 2034 ...` — the divergence is a
near-tie at generated token 1 that the decode path resolves the other
way. Same on p09-explain-gpu (`.work/ref-p09-explain-gpu.log`, 40
tokens): the reference equals the prefill path on all 40 and differs from
`engine-base` at token 37 (383 vs 436). Both divergent streams are the
prefill path being right where the decode path resolves a near-tie the
other way. The identity gate for a non-bit-exact path is therefore the
model-ref agreement (as for the q8 and q4 rounds): 2/2 checked, plus
first-token identity 20/20 and the 64-token identity on the four protocol
prompts.

### R5. Int8 MMQ prefill GEMM on an LDS-pipelined schedule, bf16 on the same schedule (frozen 2026-09-08, before any build or timed run)

Lane int8 (`~/Brain/mojo-baro/briefs/2026-09-08-lane-int8.md`). Two new
kernels in `kernels/matmul_mmq.mojo`, one schedule: 8 waves 4x2, wave tile
2x4 (128x128 block) and 2x2 over 2x4 waves (64x128), BLK_K 32 = one q4
block per K-step, two LDS buffers, register-staged global prefetch one
K-step ahead, XOR-swizzled LDS rows, one barrier per K-step -- the
`kernels/matmul_wmma_pipe.mojo` structure (wmma-fp16-protocol R3).

- **mmq** (`amar_quant_q8` + `amar_matmul_mmq_q4q8`): activations
  quantised once per input to int8 per 32-block (`d8 = amax/127`,
  round-to-nearest), stored with `d8` (f32) and `nu = -8 * sum(q)` (i32)
  in accumulator order; the GEMM stages int8 activation rows and the q4
  nibbles unpacked to unsigned int8 in LDS, two
  `v_wmma_i32_16x16x16_iu8` per 16x16 tile per block with `nu` as the
  first C-input, epilogue `acc += f32(t) * (d8 * d4)`.
- **bf16-lds** (`amar_matmul_prefill_q4_lds`): the R4 kernel's maths
  (nibble -> bf16 by `f32x16_to_bf16_trunc`, `acc = fma(t, d4, acc)`) on
  the same schedule; dequant done by the loader into LDS.

Bench: `bench/bench_prefill.mojo`, same shape / NBUF=8 / ITERS=16 / median
of 5 as R2-R4; correctness before timing: bf16-lds vs row loop rel < 1e-4
(floored metric), mmq vs bf16-lds relative Frobenius < 1e-2 and
max|a-b|/rms(b) < 5e-2. Parity (`kernels/test_mmq.mojo`): quantiser
bit-exact vs host; mmq vs fp64 host dot over the same q8 codes < 1e-3
floored (predicted < 1e-4); bf16-lds vs `amar_matmul_prefill_q4` < 1e-4.

Predicted, us per GEMM (first build; R4 bf16 column is the measured bar):

| n | bf16 R4 | bf16-lds | mmq | mmq vs R4 | mmq TOPS |
|---|---|---|---|---|---|
| 16 | 435-470 | 440 (R2 cfg) | 400 | 1.1x | 4.0 |
| 64 | 512 | 380 | 320 | 1.6x | 16 |
| 128 | 645 | 390 | 300 | 2.15x | 43 |
| 256 | 1629 | 600 | 470 | 3.5x | 55 |
| 512 | 2451 | 1100 | 850 | 2.9x | 61 |
| 1024 | 4183 | 2100 | 1600 | 2.6x | 64 |

Issue model behind the table, per K-step per wave, wave tile 2x4: bf16-lds
16 WMMA x 32 clk + ~214 VALU = ~726 clk; mmq 16 x 16 + ~312 VALU = ~570
clk, so mmq / bf16-lds = 1.27x on the same schedule; the schedule itself is
predicted to recover most of R4's 2.9x stall residual (R4's ISA receipt:
378 VALU + 512 WMMA clk per K-block against 4183 us measured = 2.9x over
issue). Gate: mmq <= 1226 us at n=512 and <= 2092 us at n=1024.
Falsifiers: bf16-lds / R4 < 1.3x at n=1024 = the schedule did not transfer
(check bank conflicts / VGPR / blocks per WGP before touching the dtype);
mmq / bf16-lds < 1.1x = the f32 epilogue is the bound; > 1.6x = the bf16
side of the issue model is undercounted. Quantiser cost reported beside
the GEMM (predicted < 15 us at n=1024), not added: one quantise serves
every GEMM on that layer input.

Sub-rounds, each preregistered below before its build: R5a two-deep
prefetch (PGR2; +11% on the dense kernel), R5b BLK_K 64 (two q4 blocks per
barrier), R5c small-n shapes. Stop after two consecutive sub-rounds with
no gain (driver ruling, status file).

### R5a. Uniform loader (frozen 2026-09-08, before its build; R5 attempt 1 was void -- shared GPU -- and is being re-run first)

ISA receipt of the R5 kernels (`.work/isa-r5`, status file): the int8
128x128 K-loop is ~350 instructions of which ~120 are `v_cndmask` /
`s_and_saveexec` / `s_cbranch_exec*` from per-thread loader guards
(`gr < M`, `tid < BN`, the two scale-thread ranges); bf16-lds carries the
same guards. R5a removes them: A rows clamped to `M-1` (padded rows compute
garbage that the epilogue's `r < M` never stores), the B column's 16 nibble
bytes split over two threads (8 B each: all 256 threads load, unpack half a
block, store 8 B int8 / 16 B bf16 per half), one 4-byte scale word per
thread (waves 0-3 `d8`, waves 4-7 `nu`, wave-uniform). Same LDS layout,
same maths, parity gates unchanged (bf16-lds must stay bit-exact with the
R4 kernel; mmq vs fp64 unchanged).

Predicted: **+5% on both mmq and bf16-lds at n >= 512** (the dense fp16
kernel's ALIGNED step, which removed its edge branches, gave +6%);
falsifier: < +2% on both = the exec-mask ops were hidden behind the WMMA
issue and the loop is bound elsewhere (LDS or the epilogue). Measured in
the same quiet window as the R5 re-run, same bench binary layout
(`.work/bench_prefill_r5a`), R5 binary interleaved.

### R5-abl. Where the K-loop time goes (receipt, not an arm; frozen 2026-09-08 before its run)

`bench/bench_prefill_abl.mojo` built six times with `ABL` in
`kernels/matmul_mmq.mojo` = 0 (none), 1 (B global load replaced by a
per-step constant), 2 (A global load replaced), 3 (per-block epilogue
replaced by one packed add: int8 loses cvt + mul + fma, bf16 loses the d4
fma), 4 (no barrier), 5 (int8 d8/nu LDS staging replaced by constants).
128x128 config, n = 256 / 512 / 1024, same rotation. Output is wrong by
construction; only the time is read. rocprofv3 hangs under this runtime
(wmma-fp16-protocol R2 step P), so this is the profile.

Predicted, us relative to ABL 0 at n=1024: no-B -10% (bf16) / -10% (mmq);
no-A -5% / -5%; no-epilogue -3% (bf16) / **-25% (mmq)**; no-barrier
-8% / -8%; no-scales (mmq) -5%. Reading rule: the largest drop names the
next sub-round's lever; a drop < 3% retires that lever. If no-epilogue on
mmq is < 10%, the int8 kernel is bound by its loads/LDS, not by the
epilogue, and the WMMA-rate receipt alone explains the loss.

#### R5 / R5a / R5-abl Result (recorded 2026-09-08, exclusive GPU, driver `.work/r5-all.sh`)

**Spread rule deviation, recorded.** Six R5 tries and four R5a tries, every
one on an exclusive GPU (`gpu-wait run --priority 90 --vram 23`, queue
empty, the orphaned ruler llama-server cancelled first), and not one run met
the 5 % in-run spread rule on the n >= 256 rows. The n <= 128 rows are clean
(<= 3 %) in every run. The spread is the bench's own at n >= 256 (bf16-lds
n=512 ranges 1578-2280 us across runs, bimodal, not a co-runner), so the
table below is the **median across tries of each row's in-run median**, with
the cross-try min beside it. Numbers are receipts of that construction, not
a P4-clean arm; the verdicts below do not depend on which of the two is read.

us per GEMM, ffn shape (N=12288, K=4096), 128x128 config, NBUF 8, ITERS 16:

| n | R4 bf16 (frozen) | wmma | bf16-lds med (min) | mmq med (min) | bf16-lds / R4 | mmq / bf16-lds | gate |
|---|---|---|---|---|---|---|---|
| 16 | 435-470 | 672 | 589 (582) | 535 (534) | 0.77 | 1.10 | - |
| 64 | 512 | 605 | 631 (626) | 588 (576) | 0.81 | 1.07 | - |
| 128 | 645 | 642 | 634 (629) | 600 (596) | 1.02 | 1.06 | - |
| 256 | 1629 | 1650 | 932 (926) | 1040 (1011) | 1.75 | 0.90 | - |
| 512 | 2451 | 3234 | 2175 (1578) | 2163 (2097) | 1.13-1.55 | 1.01 | int8 <= 1226: **FAIL** |
| 1024 | 4183 | 4552 | 2250 (2219) | 2486 (2448) | 1.86 | 0.91 | int8 <= 2092: **FAIL** |

Quantiser alone (`quant`, reported beside, not added): 10.8 us at n=512,
15.4 us at n=1024.

Verdict against the frozen predictions:
- bf16-lds / R4 at n=1024 = 1.86x (predicted 2.0x, falsifier < 1.3x not
  triggered): the LDS schedule transferred.
- int8 / bf16-lds at n=1024 = 0.91x (predicted 1.31x; falsifier "< 1.1x =
  epilogue VALU is the bound"). The int8 arm is **slower** than bf16 on the
  same schedule. The gate is not met; R5 is closed negative.
- The 07:39 WMMA issue-rate receipt (`.work/logs/peak-i8.txt`: iu8 133.4
  TOP/s vs bf16 132.4 TFLOP/s, ratio 1.007) is the mechanism: gfx1100 issues
  int8 WMMA at the bf16 rate, so int8 buys only half the operand bytes and
  pays a per-block epilogue (cvt + mul + fma per 32-block d4) that bf16 does
  not. Already in the vault as a standing rule; this round confirms it on the
  real shape.

R5a (uniform loader, `8347c55`): mmq n=1024 2511 (2357) vs R5 2486 (2448);
bf16-lds 2520 (2396) vs 2250 (2219). No gain on mmq, a loss on bf16-lds
inside the spread. Closed, no arm landed. The R5 adjacent control did not
run (the driver only runs it after a clean R5a; none was).

R5-abl (one run each, ABL 0..5, median of 5 repeats, us; bf16-lds / mmq):

| ABL | n=256 | n=512 | n=1024 | drop at 1024 vs ABL 0 (bf16 / mmq) | predicted |
|---|---|---|---|---|---|
| 0 none | 1450 / 1566 | 2158 / 2201 | 2697 / 2890 | - | - |
| 1 no-B | 342 / 394 | 704 / 737 | 1449 / 1488 | **-46 % / -49 %** | -10 % / -10 % |
| 2 no-A | 1183 / 1426 | 1646 / 1854 | 2314 / 2645 | -14 % / -8 % | -5 % / -5 % |
| 3 no-epilogue | 1341 / 1458 | 1727 / 1922 | 2485 / 2512 | -8 % / -13 % | -3 % / -25 % |
| 4 no-barrier | 1401 / 1291 | 2082 / 1957 | 2910 / 3170 | +8 % / +10 % | -8 % / -8 % |
| 5 no-scales | 1465 / 1338 | 2113 / 1818 | 2619 / 2506 | -3 % / -13 % | - / -5 % |

Reading (rule: largest drop names the lever, < 3 % retires it):
- The B (weight) global load is half the K-loop on both arms, 4-5x the
  prediction. The kernel is bound by weight bytes in flight, not by issue.
  That is the lever, and it is the same lever for bf16 and int8, which is
  why int8 cannot win here: it halves A, not B, and B is q4 in both arms.
- no-epilogue on mmq is -13 % (< the 25 % predicted, > the 10 % that would
  have said "loads only"): the epilogue is a real but secondary cost.
- no-barrier is slower on both arms: removing the barrier breaks the
  double-buffer handoff, so this row measures a broken pipeline, not the
  barrier's cost. Retired as a receipt, not as a lever.
- ABL 0 is 10-20 % slower than the R5 rows for the same config (2697 vs
  2250 at n=1024): the ablation binary carries the ABL branches and its own
  rotation; compare rows within this table only.

**Round verdict.** Int8 MMQ prefill closed negative on gfx1100 (int8 WMMA =
bf16 rate; the kernel is B-load bound). The bf16-lds schedule is the win:
1.86x over R4 at n=1024, 1.75x at n=256, bit-exact gate vs
`amar_matmul_prefill_q4` held (test_mmq). Next round, if any, is the B-load
path (bytes in flight per CU: wider loads, deeper NBUF, or K-tile 64), not
the dtype. Item 5 (integration) stays dropped for int8; landing bf16-lds as
the prefill GEMM is a separate preregistration on the prefill protocol.
