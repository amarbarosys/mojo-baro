# Speculation default: MTP k=7 on the FEWROW build vs today's k=2 (2026-11-16)

Candidate from the dflash2 lane D1 (same job): MTP k=7 on `-D BARO_FEWROW=1` 76.72 tok/s vs k=2 70.94; k=7 on the
default build 50.15. ids vs m=1 differed on 6/20 (k7 fw1) vs 2/20 (k2).

## Gate (frozen before the run): `bench/kdefault-gate.sh`, nerkyor 27B q4 pack, 20 mtp-prompts, GEN 256, TMAX 16384
- G1 (re-frozen before any measurement: BARO_FORCE refuses spec): prompts whose 256 greedy ids differ from the m=1
  launch-path reference, b <= max(a, a2) + 2; a2 (repeat of a) shows the run-to-run floor.
- G2: live median tok/s_gen b > a, same job.
Both pass -> FEWROW=1 becomes the build default and the nerkyor pack gets `spec-k.txt` = 7.

## Result (tree 2392665, `.work/kgate/run/summary.txt`): G1 FAIL, G2 PASS -> default NOT changed

| arm | ids != m=1 ref (of 20) | tok/s_gen median |
|---|---|---|
| a: FEWROW=0, k=2 (default) | 8 | 70.52 |
| a2: same, repeat | 8 | 70.60 |
| b: FEWROW=1, k=7 | 12 | 76.08 (1.079x) |

b is 7.9% faster but differs from the reference on 4 more prompts than the default (bar: at most 2 more). The
8-row WMMA verify route is not bit-exact (dflash2 D1 E2). It stays opt-in (`-D BARO_FEWROW=1`, `BARO_SPEC_K=7`).
Side finding: the default plain megakernel hit NOT-RESIDENT on request 1 of a resident run three times today
(fail word 1, invalid tokens); the gate's reference arm runs on the launch path for that reason.

## First-request NOT-RESIDENT on the plain megakernel (2026-11-16)
Seen 3 times today, always request 1 of a resident run inside a multi-arm script (gap-ab plain1 twice, right after the
plain0 engine exited; kdefault ref once, first arm of the job). Repro attempt `bench/mega-first-req.sh` (8 fresh
starts, 4 default + 4 HSA_NO_SCRATCH_RECLAIM=1, back to back): 0/8 (`.work/mfr/summary.txt`). Cause UNRESOLVED.
Lead not yet tested: ROCr on-demand scratch on the first dispatch (the window kernel spills to scratch; docs:
rocr-runtime env_variables HSA_NO_SCRATCH_RECLAIM / HSA_SCRATCH_SINGLE_LIMIT). The invalid tokens still return
with status 200; the fail word is the only signal.

## dot2 GEMV for the MTP window (lane verify8 3d9d1cc + -D BARO_TURBO_DOT2=1), same gate with B_K=2
`.work/dot2gate/summary.txt`: a 8/20 ids != ref at 70.41 tok/s, a2 8/20 at 71.38, b (dot2) 11/20 at **86.04 (1.222x)**.
G1 (<= 10) FAIL by one prompt, G2 PASS. the maintainer's call: decide on task quality instead of greedy divergence, so the
120-task eval (nerkyor-vs-forge `.work/nerkyor/quality.sh`, cap 16000, thinking off) runs on dot2 vs the same-tree
default; adopt if tasks are not worse. Result below when it lands.

## dot2 default: decided by the 120-task eval (2026-11-16)
Same tree (lane-verify8 3d9d1cc + wiring), engines `.work/v8/eng-dot2` (`-D BARO_TURBO_DOT2=1`) and `.work/v8/eng-a`, nerkyor 27B q4 pack,
thinking off, cap 16000, MTP k=2, `nerkyor-vs-forge/.work/nerkyor/quality.sh`. Rule frozen before the run: dot2 becomes the default if its task
score is not worse than arm a's.

| arm | ours /120 | llama.cpp /120 (same run) | truncated |
|---|---|---|---|
| dot2 | 87 (math 85, json 2) | 90 | 0 |
| a (exact turbo) | 79 (math 77, json 2) | 91 | 1 |

Result: dot2 is not worse (87 vs 79), so `BARO_TURBO_DOT2` defaults to 1 (`serve/window.mojo`). The 8-point gap is larger than the change
can explain on its own: the earlier default engine scored 83 on the same tasks, so run-to-run spread under MTP is several points; read it as
"no task loss", not as a quality gain. `-D BARO_TURBO_DOT2=0` restores the exact window path. Receipts: `nerkyor-vs-forge/.work/q-dot2/SUMMARY.txt`,
`.work/q-a/SUMMARY.txt`.
