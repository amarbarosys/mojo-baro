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
