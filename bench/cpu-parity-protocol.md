# cpu-parity protocol (lane-cpu-parity, item 2 step 0c, frozen before any GPU run)

Claim under test: one q4 row dot, CPU emulation of the gfx1100 m=1 kernel (`q4_dot_blocks` via
`amar_matmul_skinny_q4rowb[2,1]`) is bitwise equal on real pack rows.

Arm identity (P1 read-back): GPU arm = `amar_matmul_skinny_q4rowb[UNROLL=2, MR=1]`, grid ceildiv(N, ROW_WAVES),
block ROW_THREADS, no env knobs. CPU arm = `cpu/ops.mojo` `q4_row_dot` (std only, explicit fma, no `reduce_add`).
The test prints both arms and the order flags it was given; the gate greps them.

Data: `.work/engine-pack-q4` (Qwythos 9B Q4_0, sha in `identity.json`), layer 3 `attn_output` (K 4096),
`ffn_gate` (K 4096), `ffn_down` (K 12288); 512 rows from the head and 512 from the tail of each tensor;
activations = bf16 rows of `token_embd.weight` for 4 token ids (real values, not post-norm: stated limit).
12288 dots per run.

Order candidates (the unknown the step settles):
- RA: reduction order of the 16-element per-lane accumulator. 1 = adjacent pairs, 0 = halving.
- WA: reduction order across the 32 lanes. 1 = adjacent pairs (what the DPP chain in the stdlib source gives:
  quad_perm 1032, 2301, row_half_mirror, row_ror 8, xor 16), 0 = halving (xor 16 first).

Predictions (frozen):
1. RA=1, WA=1 gives 0 mismatches of 12288 (confidence 60%).
2. Exactly one of the four (RA, WA) combinations gives 0 mismatches; if none does, the dot drops to the agreement
   gate and the lane reports it (design risk 1 and 2 of `docs/design/CPU-BACKEND.md`).
3. The planted mutant (the other WA value than the passing one) gives mismatches > 0 on at least 90% of rows.

Gate: `bench/check-cpu-parity.sh` (CPU build, GPU run under gpu-wait, mutant run must FAIL, fail word read on
every run). Exploratory runs of the four-way scan exit 3.

## Result, step 0c (2026-10-21; written after the data, the predictions above stay as frozen)

Exploratory four-way scan (flags in the scan binary, since removed; `exchange/receipts/cpu-parity/0c-scan-exploratory.txt`,
the `exit=` lines there are `tail`'s, not the test's): RA=1 WA=1 1744 mismatches, RA=1 WA=0 7729, RA=0 WA=1 **0**,
RA=0 WA=0 7668 (of 12288).
- Prediction 1 MISS: the passing combination is RA=0 (halving fold of the 16-element accumulator), not RA=1.
- Prediction 2 HOLDS: exactly one combination passes.
- Prediction 3 MISS on the number: the mutant (WA=0) mismatches 7668 of 12288 dots (62%), not >= 90%; it fails the gate.
- Pinned in `cpu/ops.mojo`: lane accumulator folds by halves, the 32 lanes fold by adjacent pairs (the DPP chain).
Gate `bench/check-cpu-parity.sh`: PASS, 0 of 12288 mismatches, mutant caught (`0c-run.txt`, `0c-run-mut.txt`).

# Step 1, one dense layer (frozen before the first GPU run of `kernels/test_cpu_layer.mojo`)

Claim: every op of the launch-path attention layer (`serve/window.mojo` `blk32_forward` lines 231-293 shape, layer 3
of the Qwythos pack, real weights, `BARO_SHARED_EXP=1`) has a CPU form (`cpu/ops.mojo`) that is bitwise equal to its
GPU kernel on the GPU's own input for that op. Each stage is compared on the GPU's input, so one wrong op cannot hide
or inflate another. Positions 0..5, 300 and 5000 (page table all zero: the KV rows alias, GPU and CPU read the same
slots), activations = bf16 embedding rows as the f32 residual, 8 steps, f32 KV (`BARO_KVQ=f32`, the dense default).
Arm identity: the test prints the GPU kernel list and the CPU functions; the gate greps the arm line and the
`-D BARO_SHARED_EXP=1` build flag, and reads every stage line.

Stages: rms1, q/k/v GEMV, qgate split (copy), head rmsnorm q/k, rope q/k, attention, gate-mul, o GEMV + residual add,
rms2, gate/up GEMV, swiglu, down GEMV + residual add. 8 steps x 17 stages.

Predictions (frozen):
1. GEMV stages (q, k, v, o, gate, up, down): 0 mismatches (step 0c; confidence 90%).
2. rmsnorm (rms1, rms2, head q/k): mismatches > 0 with std `rsqrt` against CPU `1 / sqrt`; 0 after the shared `rsq`
   (`1 / sqrt(x)`, two correctly rounded ops) lands in `kernels/bexp.mojo` under `BARO_SHARED_EXP` (confidence 70%).
3. rope q/k: mismatches > 0 (std `sin` / `cos` differ CPU vs GPU); 0 after a shared f64 sincos (confidence 80%).
4. gate-mul and swiglu: 0 mismatches with `BARO_SHARED_EXP=1` (IEEE divide on both; confidence 75%).
5. attention: 0 mismatches if the GPU does not contract `acc += q*k` into fma (confidence 55%); the scan runs both
   CPU forms and reports which one matches.
6. Final: the layer is bitwise on all 8 steps after the shared helpers land; the GPU default build is byte-identical to
   main (helpers are `comptime if SHARED_EXP` only; checked by comparing the default-flag test output).

Gate: `bench/check-cpu-layer.sh`: PASS only if all stages 0 mismatches; planted mutant (rope sincos argument
perturbed by one ulp on the CPU side) must fail the rope stage and no other.

## Result, step 1 (2026-10-21; written after the data, predictions above stay as frozen)

Scans (receipts `exchange/receipts/cpu-parity/1-scan-*.txt`, mismatching elements of 8 steps):
1. kernels unpatched, `BARO_SHARED_EXP=1`: GEMV q/k/v/o/gate/up/down 0, rms1/rms2 0, attn 0, gate-mul 0, swiglu 0,
   hrms_q 10291, hrms_k 2566, rope_q 2641, rope_k 658.
2. kernels on the shared `rsqrt`/`sin`/`cos` of `bexp.mojo`: rope 0 / 0, hrms_q 1554, hrms_k 199, rest 0.
3. CPU `head_rmsnorm` models the GPU wave sum lane by lane (`dpp_sum`: xor 1, 2, 7, 8, 16 chain, first level
   `fma(v, v, partner * partner)` because the compiler contracts `warp.sum(v * v)`): all 14 stages 0.
Verdicts on the frozen predictions:
- P1 GEMV 0: HOLDS (five matrices, 8 steps).
- P2 rmsnorm: PARTLY. The f32 head norm mismatched with std `rsqrt` (10291) as predicted, but the shared `rsq` alone
  did not reach 0: the sum of squares needed the contracted-first-level model (scan 3). rms1/rms2 gave 0 at every
  stage because the output is bf16 (a 1 ulp f32 scale rarely flips a bf16 rounding): they are weak evidence, stated.
- P3 rope: HOLDS (2641/658 before, 0 after the shared f64 sincos).
- P4 gate-mul and swiglu 0: HOLDS.
- P5 attention: MISS as framed. Both CPU forms (mul+add and explicit fma) gave 0, because the CPU backend contracts
  `acc + q * k` too. Pinned: explicit `fma` in `cpu/ops.mojo` for rms partial, attention score, `l_run`, PV. Rope
  `x0 * c - x1 * s` still relies on the CPU backend contracting the same way as the GPU (x86 needs FMA3); the gate
  would catch a CPU without it.
- P6 HOLDS: gate `bench/check-cpu-layer.sh` PASS, mutant (sincos argument x 1.0000001) fails rope only
  (rope_q 1430, rope_k 374), default-flag GPU digest equals the one built from main's kernels.
Limits: layer 3 only (full attention); SSM layers, embedding, final norm, head and the megakernel path
(`mega.mojo` has its own `warp.sum(v * v)` rmsnorm sites that need the same `dpp_sum` model) are not covered.

# Item 1, per-token certificate (frozen before the first run of `kernels/test_certificate.mojo`)

Design call (the plan's "exact mod-p vs float with a bound"): **float with a calibrated bound**. The production q4
GEMV takes bf16/f32 activations and dequantizes in f32; the integer part (nibbles x int8) never exists as a stored
value (the q8dot kernel scales per 32-block before it accumulates), so an exact mod-p check has no value to check.
`kernels/freivalds.mojo` stays the integer-GEMM tool; this item does not use it.

Mechanism. Per certified matrix W (N x K, q4): seeded r in {-1,+1}^N (splitmix64 of seed and row), u = r^T W in f64
from the dequantized weights exactly as the GPU decodes them (`fma(nibble, d, -8d)`), computed once on the host
(`cpu/certify.mojo` `Cert`). Per token: the host receives x (bf16, 2K bytes) and the GEMV output y (f32, 4N bytes)
of the sampled matrix and computes z = |sum_i r_i y_i - sum_j u_j x_j| in f64. Scale s = ||y||_2 * sqrt(K) * 2^-24,
ratio rho = z / s. PASS iff rho <= KAPPA. Memory delta: u is 8 bytes x K per matrix (the 7 layer-3 matrices:
see the arm line `u-vector bytes`); the 9B trunk is about 11 MB, stated not measured.
Soundness: one wrong output element is always seen once |delta| > KAPPA * s / 1 (r_i is never 0); several wrong
elements that cancel under r escape with probability 1/2 per matrix.

Procedure. Calibrate on SEED 1 (48 steps, 336 checks, KAPPA 0 = measure only), then freeze KAPPA = 2 x the largest
rho seen, in a commit, then confirm on SEED 2 (other tokens, other r) with that KAPPA. Mutants are kernel mutants:
a sed copy of `amar_matmul_skinny_q4rowb` that adds DELTA to one output (row 5) of every GEMV (built under `.work`,
never in the tree), DELTA in {0.05, 0.005}.

Predictions (frozen):
1. Honest rho: median <= 1, max <= 4 over the 336 SEED-1 checks (confidence 55%).
2. SEED-2 confirm: 0 false alarms of 336 at the frozen KAPPA (confidence 80%).
3. Mutant DELTA 0.05: caught on 7 of 7 matrices; DELTA 0.005: caught on at least 4 of 7 (confidence 60%).
4. CPU check cost per matrix <= 100 us (u.x over K <= 12288 plus a sum over N <= 12288; confidence 70%).
5. Identity, performance (decode B/A <= 1.01) and the `check-fuse-mutants` rerun need the engine hook and are NOT
   claimed by this harness; the report names them UNVERIFIED until the hook lands.

## Result, item 1 (2026-10-21; written after the data, predictions above stay as frozen)

SEED 1 calibration (`2-cert-calibration-seed1.txt`): rho median 0.010, p99 0.041, max 0.057 over 336 checks; KAPPA
frozen at 0.125 (`bench/cert-kappa.txt`, commit 998b56f) before anything below ran.
- P1 HOLDS (median <= 1, max <= 4; the scale is a pessimistic bound, honest rho is 20x under it).
- P2 HOLDS: SEED 2 (other tokens, other r) 336 of 336 checks <= KAPPA, max rho 0.064.
- P3 HOLDS and exceeds: kernel mutant +0.05 at row 5 of every GEMV caught on 7 of 7 matrices at every step
  (rho 102 on q), +0.005 caught on 7 of 7 (rho 33 to 53). Detection floor is about KAPPA * s = 6e-5 absolute on these y.
- P4 HOLDS: 15.9 us per matrix check on the host (u.x over K and the sum over N), 294,912 bytes of u for 7 matrices.
- P5: engine identity, decode B/A and `check-fuse-mutants` rerun NOT done: no engine hook exists. UNVERIFIED.
Gate `bench/check-certificate.sh`: PASS. Both gates were first run with a build that silently used the unmutated
`matmul_skinny.mojo` (the test file's own directory wins over `-I`): the first certificate run read "caught on 0 of 7".
Mutant and main-kernel builds now compile from a copied tree. The earlier layer-gate digest comparison was vacuous for
the same reason and was re-run; it still PASSes with the same digest.

# Step 1b, one SSM layer and the head argmax (lane cpu-tier, frozen before the first GPU run, 2026-11-10)

Claim: every op of the dense m=1 SSM sub-block (`serve/window.mojo` `step_window`, `MEGA_ALLOWED` branch: `gemm_w` q4,
`amar_skinny_reduce`, `amar_ssm_conv_l2g`, `amar_ssm_delta_step[1]`, `amar_ssm_gated_out_bf16`, out GEMV + residual)
and the greedy pick (`amar_argmax_pos`) has a CPU form that is bitwise equal on the GPU's own input. Layer 0 of the
Qwythos pack (`.work/engine-pack-q4`), real weights, `-D BARO_SHARED_EXP=1`, the same 8 tokens/positions as step 1;
conv and delta state carried across the 8 steps in a 2-slot ring (each step reads the GPU's slot, writes the other).
CPU forms: `cpu/ops.mojo` `ssm_conv` (explicit fma chain over the 4 taps, wave-model `dpp_sum` l2 norm, `frsq32`),
`ssm_gates`, `ssm_delta` (t = s*eg, `fma(t, k, sk)`, `fma(k, d, t)`, `fma(s, q, o)`, kernel-parity rule 2),
`ssm_gated` (reuses `head_rmsnorm`), `swiglu`; greedy pick = `serve/sample_ref.mojo` `sample_row_ref` at T = 0
(reused, no new argmax). Argmax input: 3 synthetic vocab rows with planted ties (first index of the max must win).

Stages: rms1, gemv_qkv, gemv_z, gemv_alpha, gemv_beta, conv (out + written state), gates (eg, beta), delta (o + written
state), gated (bf16), out_add, argmax.

Predictions (frozen):
1. GEMV stages and rms1: 0 mismatches (step 0c / step 1; confidence 90%).
2. conv: mismatches > 0 on the q/k channels with the kernel's std `rsqrt` against CPU `frsq32` (v channels and the
   written state 0); 0 everywhere once `kernels/ssm.mojo` takes `rsqrt` from `bexp` (a kernel change: QUESTION to the
   boss before it lands; confidence 70%).
3. gated: same as 2 (std `rsqrt` in the kernel); bf16 output makes the count small (confidence 60% for > 0 unpatched).
4. gates: eg mismatches > 0 from `log1p` (std on both sides, CPU libm vs GPU lowering; confidence 50%); beta 0.
5. delta: 0 mismatches on o and state with the explicit fma form (confidence 75%).
6. argmax: 3 of 3 rows equal (confidence 95%).
Gate: `bench/check-cpu-layer.sh` extended: PASS only with 0 on every stage of both layers; the rope mutant still fails
rope only; a second mutant (delta `d` scaled by 1.0000001 on the CPU side) must fail delta and no other stage (each
stage reads the GPU's input, so nothing downstream inherits it).

# Step 2, pack (frozen before the first GPU run of the `pack` stage, 2026-11-10)

Claim: `cpu/pack.mojo` (index.txt parsed once, pack.bin `mmap` read-only) yields the same bytes per tensor as the GPU
loader (`serve/harness.mojo` `load_pack`, device buffer `wbuf` at `off[i]`) for 5 named tensors: `token_embd.weight`
(bf16, first), `blk.0.ssm_a` (f32, 32 elements), `blk.3.attn_q.weight` (q4), `output_norm.weight` (f32),
`output.weight` (q4, 1 GB). Compared byte by byte (stronger than the plan's sha256; an FNV digest is printed per tensor).
Prediction: 0 mismatching bytes on 5 of 5 (confidence 95%). Stage `pack` of `kernels/test_cpu_layer.mojo`.

## Result, steps 1b and 2 (2026-11-10; written after the data, predictions above stay as frozen)

Receipts `exchange/receipts/cpu-parity/1b-*`. Scan 1 (kernels unpatched, `1b-scan-unpatched.txt`): every stage 0 but conv
(8932 of 65536). Fix 1, `kernels/ssm.mojo` takes `rsqrt` from `bexp` (commit cc52b85; default-flag engine ISA 288 of 288
code objects byte-identical to main, `1b-ssm-import-isa-identical.txt`): conv 3209, all from step 3 on, the first step
with four nonzero taps. Fix 2, the tap chain read from the kernel ISA (`1b-conv_l2g-default.s`: `t = w1 * cw1`,
`fma(w0, cw0, t)`, then w2, w3) and pinned in `ssm_conv`: conv 0. Gate `bench/check-cpu-layer.sh` PASS
(`1b-gate-run.txt`): 27 stages 0 mismatches, mutant fails rope_q 1430, rope_k 374, delta_o 21810, delta_s 2919427
and nothing else, default-flag GPU digest equal to main's attn/elementwise/ssm (16977346893223634253).
- P1 GEMV and rms1 0: HOLDS.
- P2 conv: HOLDS on the rsqrt part (q/k only), MISS on "0 once rsqrt is shared": the 4-tap fma grouping was a second
  cause, hidden until the conv state filled (kernel-parity rule 2: read the ISA first; it would have saved one run).
- P3 gated > 0 unpatched: MISS. 0 with std `rsqrt` in the kernel; the bf16 output hid the scale difference, as for
  rms1/rms2 in step 1 (weak evidence, now moot: the kernel uses the shared `rsq` under the flag).
- P4 gates: eg MISS (0, `log1p` agrees CPU vs GPU on these 256 inputs), beta 0 HOLDS.
- P5 delta 0: HOLDS (o and the full 32 x 128 x 128 state, 8 steps).
- P6 argmax 3 of 3: HOLDS (planted ties: first index wins on both).
- Step 2 pack: HOLDS, 5 of 5 tensors 0 mismatching bytes (2.6 GB compared).
Limits: one SSM layer (layer 0), states from zero over 8 steps; the FFN of an SSM layer is the same code as the
attention layer's (not re-run); `log1p` agreement is measured on 256 values only.

# Step 3, engine loop on one thread (2026-11-10; no prediction was frozen for it beyond the plan's done line)

`cpu/engine_cpu.mojo`: the dense m=1 launch path op for op (q4 GEMVs with the NSPLIT=1 reduce `0 + dot`, rmsnorm_cast,
q/gate split, head rmsnorm, rope, f32 KV in one allocation, attention head, gate mul, SSM gates/conv/delta/gated, swiglu,
final norm, head, first-max argmax), prompt replayed through the decode step, `"force"` ids as the GPU engine reads them.
The ops moved from List results to caller buffers (pointer form) so the token loop allocates nothing; the layer gate was
re-run on the pointer form: PASS, 27 stages 0 (`.work/cpu-tier-gate.txt`).
Gate `bench/check-cpu-engine.sh`, QUICK=3 NTOK=1 (receipts `exchange/receipts/cpu-parity/3/`): GPU reference built with
`-D BARO_SHARED_EXP=1` (ISA read-back: conv_l2g has 0 `v_rsq_f32`) and run BARO_SERVE=1 BARO_MEGA=0 BARO_PREFILL=0
BARO_REPLAY_M=1 BARO_SPEC=0 BARO_CKPT=0 (echoes read back); control (GPU forced on its own ids) 3/3 at 100%; CPU 3/3
first tokens equal; mutant (gate and up swapped) 0/3. Allocations (heaptide, same prompt, n = 2 vs n = 6): 1935 and 1935
allocations, 463 and 463 temporaries: four more decode tokens, zero more allocations (`3-heaptide-n2.txt`, `-n6.txt`).
Memory at done: VmRSS 4.43 GB of which RssAnon 61 MB; the rest is the mapped pack (file pages). Speed, one thread: 0.9 s per
token (smoke run, not a gate number).

# Step 4, thread pool (frozen before the first pooled run, 2026-11-10)

Host (7800X3D, DDR5): `bench/cpu-gemv.mojo roof` over 2.4 GB: 1 thread 44.2 GB/s, 8 threads 55.0, 16 threads 54.2 (load
2.24). Bytes per token: 4.47 GB (trunk q4 + f32 norms + head, one bf16 embedding row; index.txt sum).
Roof: 55.0 / 4.47 = 12.3 tok/s. One thread measured 0.9 s/token = 5.0 GB/s, so the one-thread engine is issue-bound
(the q4 dot keeps the GPU's 32-lane sum order, 16-wide, no int8 path), not bandwidth-bound.
Predictions (frozen):
1. Ids: identical to step 3 at every thread count (rows are independent; confidence 95%).
2. Speed at 16 threads: 4 to 8 tok/s decode (confidence 60%); scaling 1 -> 8 threads at least 5x.
3. RssAnon at done: under 100 MB above step 3 (pool state and 16 stacks; confidence 80%).
The plan wanted this prediction before the engine first ran; the one-thread smoke run came first (0.9 s/token), stated.

## Result, step 4 (2026-11-10; written after the data, predictions above stay as frozen)

`cpu/pool.mojo`: persistent pthread workers on a generation word, rows from one atomic counter; every q4 GEMV goes
through it. p01, n = 8, same binary, load 3-5 (`exchange/receipts/cpu-parity/4/threads-*.txt`): threads 1 / 4 / 6 / 8 /
12 / 16 = 1.29 / 4.39 / 5.51 / 7.27 / 7.20 / 3.71 tok/s, ids identical at every count. Default BARO_THREADS = online CPUs
/ 2 (8 here). Gate `bench/check-cpu-engine.sh` QUICK=3 NTOK=8 PASS at 8 threads (control 100%, CPU 24/24, mutant 1/24).
- P1 ids identical at every thread count: HOLDS.
- P2 4 to 8 tok/s at 16 threads: MISS, 3.71 (workers spinning on SMT siblings); at 8 threads 7.27 tok/s, inside the
  band; 1 -> 8 threads 5.6x: HOLDS. 7.27 tok/s = 32.5 GB/s, 59% of the 55 GB/s roof.
- P3 RssAnon under +100 MB: HOLDS, 61.7 MB at one thread, 61.8 MB at eight.

# Step 5, engine gate (prediction copied from docs/lanes/2026-11-10-cpu-tier.md, frozen there before the run)

`bench/check-cpu-engine.sh` defaults: 20 prompts (bench/mtp-prompts), 64 forced ids each, bar 100% per prompt, control and
mutant arms as in step 3. Plan predictions: agreement 100% on 20 of 20 (confidence 60%); speed bound = bytes per token
over DRAM bandwidth (12.3 tok/s here, step 4); RSS above the page cache under 1 GB at BARO_TMAX=16384 with f32 KV.
Added before the run: decode 6 to 7.5 tok/s per arm alone at 8 threads (two CPU arms share the 16 hardware threads in
the gate, so the gate's own tok/s is not this number); RssAnon at done under 300 MB (KV pages touched for up to 123
positions plus 50 MB of SSM state).

## Result, step 5 (2026-11-10; written after the data, predictions above stay as frozen)

First run (`exchange/receipts/cpu-parity/5-miss/`): FAIL, CPU 1279 / 1280 (p01 step 30: CPU 42900, GPU 36405, GPU gap
0.0029), control 1280 / 1280, mutant 62 / 1280. MARGIN lines on both engines showed the logits apart from the first token
(21.0307 vs 21.0333). Dump diff (BARO_DUMP, 1-token prompt, 30 forced tokens, every position): first divergence token 0,
layer 7, FFN half; the FFN ops of layer 7 on that very residual row were bitwise in the harness (probe stage), so the
engine, not an op, differed. Cause: `cpu/ops.mojo` took `exp` and `log` from `bexp`, whose meaning follows
`-D BARO_SHARED_EXP`; the harness is built with the flag, `engine_cpu` was not, so the engine ran std exp/log. Fix: the CPU
tier imports `fexp32` / `flog32` directly (one rule now in `tools/ci-checks.sh`). After it the two dumps are equal in
every element of 30 tokens x 32 layers x 2 halves.
Rerun on the fixed tree (`5/`): layer gate PASS 28 stages; engine gate PASS: CPU 1280 / 1280 on 20 of 20 prompts at 100%,
control 1280 / 1280, mutant 62 / 1280 (every prompt under the bar).
- Agreement 100% on 20 of 20: HOLDS on the rerun (MISS on the first run, 19 of 20, from the build-flag defect above).
- Speed: p17 (59-token prompt, 64 tokens) alone at 8 threads, load 15: decode 6.18 tok/s, inside 6 to 7.5; the roof is 12.3.
- Memory: RssAnon at done 68.5 MB (VmRSS 4.45 GB is the mapped pack): HOLDS against both the plan's 1 GB and the 300 MB band.
Steps 3 and 4 passed with the std-exp engine: their first tokens and 8 ids matched, their logits were not bitwise (stated).
