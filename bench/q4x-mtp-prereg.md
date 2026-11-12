# q4x MTP verify window: preregistration (lane q4x-mtp, plan-1234 item 2), frozen by this commit

Question: what does `Q4X_MTP` + `Q4X_SPEC=K` give today on the Strata Q2_0 pack at the 16K window?

## Priors (read, not measured here)
- `exchange/lane-Q4X-R5-report.md`: K=3, 20 prompts x 32 ids, 20.61 -> 20.21 ms/token (1.02x), 2.82 tokens per window, ids 20/20 equal.
- `exchange/q4x-gemm-alu-diagnosis-2026-10-12.md`: multi-row q4x GEMMs are issue-bound, R=4 costs 1.4-2.3x the VALU of R=1 and
  always computes 4 rows, so a 2-3 token window pays for 4.
- 3-prompt probe on this tree (NG 32, Q4X_SPEC_P=0): tokens/round K=2 1.9-2.8, K=4 2.8-3.9 (`.work/mtp/repro-head`, not a claim).

## Arms (bench/q4x-mtp-ab.sh, one gpu-wait job, one binary, order rotated per prompt)
A = `Q4X_SPEC` off. B2 = `Q4X_MTP=<mtp-Qwen3.8-Flash-Next-Q8_0.gguf> Q4X_SPEC=2`. B4 = same with `Q4X_SPEC=4`. `Q4X_SPEC_P` 0.5 (default).
`Q4X_TMAX=16384`, 20 prompts `bench/q4x-prompts`, NG = 64, warm list `exchange/receipts/q4x-strata/r3a-warm-l01.txt`.
Read-back per run: `kv 16384`, expert-cache slots (B loads 1350 MB of MTP experts, so B has fewer cache slots: part of the cost), `spec rounds ... k K pmin P`.

## Predictions (A about 20 ms/token)
| arm | B/A (median of per-prompt medians) | tokens/round | acceptance accepted/drafted |
|---|---|---|---|
| B2 | 1.00 (0.95 to 1.05) | 2.4 | 0.70 |
| B4 | 1.03 (0.95 to 1.10) | 3.1 | 0.55 |

Mechanism: tokens/round gain (1.4x to 1.9x fewer rounds) is eaten by the R-row window cost (m=3 to 5 rows at 2.2-3.6x one row) plus the draft passes.

## Rule (set before the run, not edited after)
- IDENTITY must PASS on every run (B ids equal A ids). A DIFF voids the arm: verdict INVALID, no speed claim.
- Verdict per arm against B/A < 1.0: WIN (same-job, identity kept, any size lands) or MISS.
- If either arm reaches B/A < 0.9: tune the window (new prereg first). Otherwise stop and report why (acceptance, verify cost per round).
- Budget: harness 50 LOC, engine +0 LOC (one print extended). Bytes: no buffers added.
