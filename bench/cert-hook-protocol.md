# Item 1 follow-up: per-token certificate engine hook and decode B/A (frozen before the first timed run)

Hook (lane `lane-cert-hook`): `BARO_CERT=SEED` turns it on; unset (default) the engine issues the same GPU commands as before.
Launch path only (`BARO_MEGA=0`, one row per call, no spec window). At pack load the host copies the q4 weights of the 7
GEMVs of layer 3 (q, k, v, o, gate, up, down) and computes u = r^T W in f64 (`serve/certify.mojo` `Cert`, the math of
`kernels/test_certificate.mojo`, seed*1000+matrix). In `step_window`, right after each of those `gemm_w` calls, a
stream-ordered async copy moves y (f32, the P partial) into a pinned host buffer, and x (bf16) once per distinct input
(4 copies for 7 matrices). After the token's host sync the host checks z = |r.y - u.x| against KAPPA * s (KAPPA 0.125
from `bench/cert-kappa.txt`, s = ||y||2 sqrt(K) 2^-24) and prints on stderr one line per token:
`cert tok N seed S layer 3 pass P/7 maxrho R`. The check adds one `ctx.synchronize()` at the end of `step_window` while the
flag is on (a no-op when serve mode already synced).

Costs, stated before the run. Device to host per token: x 2*(2 H + ATT + FFN) = 49,152 B plus y 4*(QF + 2 KV + 2 H + 2 FFN)
= 172,032 B (H = ATT = 4096, FFN 12288, QF 8192, KV 1024), 221,184 B in 11 async copies. Host pinned buffers the same size.
u: 294,912 B for the 7 matrices (the `cert on` load line prints it).
Host check: 15.9 us per matrix measured by the harness, 7 matrices about 111 us serial after the token sync.

Gates, each committed as it passes: (1) token identity cert on vs off, 5 prompts (`bench/check-cert-hook.sh identity`);
(2) planted mutant, +0.05 on row 5 of every q4 GEMV output, engine built from a kernel copy (`... mutant`), cert lines must
print pass 0/7; (3) `bench/check-fuse-mutants.sh` PASS; (4) `./run-tests.sh` exit 0.

## Timed A/B (decode B/A)

Harness: `bench/cert-ab.sh`, a wrapper over `bench/decode-arms.sh` (champion method: one engine process per prompt, the 20
`bench/mtp-prompts`, `BARO_MEGA=0 BARO_SPEC=0 BARO_TMAX=16384`, dense q4 pack `.work/engine-pack-q4`). One binary; the arms
differ only in the environment. Per prompt the order is A1 B1 B2 A2 (ABBA), A = `BARO_CERT` unset, B = `BARO_CERT=1`.
Metric: tok/s_gen per run; B/A = median over the 40 A runs of tok/s divided by the median over the 40 B runs (time ratio,
>1 means the certificate costs time). Receipt (P1): `current_link_speed` of the root port 0000:00:01.1 and the GPU at job
start and end (must read 16.0 GT/s, else the job is void), the arm env per run, the engine sha256, the `cert on seed 1` load
line present in every B log and absent in every A log, a `cert tok` line per generated token in B and none in A,
GENERATED md5 equal between A and B on every prompt (identity on every timed run), the `mega fail word` read on every run.

Kill line (the maintainer's brief): B/A > 1.01 kills the hook as shipped (a result, not a pause: the optimisation named below
follows). No other threshold.

Predictions (frozen):
1. Identity: GENERATED equal for cert on and off on 20 of 20 prompts (confidence 95%).
2. Honest cert lines: all pass 7/7 on every token of every B run (SEED 1 at the frozen KAPPA 0.125; harness max rho was
   0.064 on SEED 2; confidence 85%).
3. B/A median 1.013, range of per-prompt medians 1.003 to 1.03: the 111 us host check plus a sync sits serial between two
   tokens of about 10 ms (confidence that B/A <= 1.01: 35%).
4. If P3 misses the kill line the cause is the serial host check, not the copies: the D2H bytes per token are under 0.3 MB
   (about 10 us at 16 GT/s x16). The optimisation then is to run the check of token N while the GPU executes token N+1
   (two-slot ring), not part of this verdict.

## Result (2026-10-21; written after the data, predictions above stay as frozen)

Receipts `exchange/receipts/cert-hook/ab1/` (engine sha256 b64ebc09dab326a2, checkout 19f9c6a, link 16.0 GT/s root and GPU at start and end).
- P1 HOLDS: GENERATED md5 equal on 80 of 80 timed runs (and 5 of 5 prompts in the gate).
- P2 HOLDS: 2560 of 2560 cert lines pass 7/7 in the 40 B runs; gate max rho 0.063.
- P3 MISSES HIGH: B/A 1.0229 (A 98.53, B 96.32 tok/s median of 40 runs each), per-prompt median ratios 1.0103 to 1.0346, outside
  the frozen 1.003 to 1.03 range at the top. A1/A2 0.9975, B1/B2 1.0044: the ruler is steady. Kill line B/A > 1.01 HIT:
  the hook as shipped (check serial after the token sync) is killed for default-on. Cost 0.23 ms per token against 0.11 ms predicted.
- P4 NOT TESTED: the split of the 0.23 ms between the host check (about 0.11 ms by the harness figure), the end-of-step sync,
  the 11 small copies and the stderr line was not isolated. The named follow-up (check of token N while the GPU runs token N+1)
  is untried.
