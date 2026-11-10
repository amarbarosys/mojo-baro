# lane-moeprefill report

Next kernel lever on the MoE and prefill paths (`bench/moeprefill-protocol.md`). Traces: the MoE tier prefill chunk is bound by the
18.3 GB expert staging copy (about 1.5 s of 2.3 s, kernels 14%); the kq dense prefill is the K-quant GEMM (84% of kernel time). Lever
landed `652d4c7`: `moe_down_q6k_grouped` loads each 256-block's bytes once per wave32 lane and keeps the m=1 `q6k_row_dot` order:
20856 -> 4674 us per launch at the chunk shape, 0 differing bytes against `amar_moe_down_q6k`; engine A/B in tier mode on 20 prompts
of 768 tokens, same job, ABBA: prefill_s median 0.9500 -> 0.9246 s (B/A 0.9732, B faster 20/20), 16 ids equal on all four runs of all
20 prompts, in-engine receipt 45571 -> 12878 us per 3 launches. Commits: 6ddf0c2 (bench + frozen predictions), 652d4c7 (kernel),
81b9254 543be6e 0cc8754 (A/B harness `bench/moe-prefill-ab.sh`), 689e9a4 (census), 79ece97 (docs). Decode m=1 kernels untouched.

run-tests: `RUN_ALL=1 ./run-tests.sh` under gpu-wait (timeout 5400 s), rc=0, log .work/moeprefill/run-tests.log (last line rc=0, before
it "PASS run-tests: all tests, pack .work/engine-pack-q4", 137 PASS lines, "FAIL: 0 (none)"). Toolchain: worktree-local `uv sync` venv,
Mojo 1.2.0 (2016d0cd), max 26.7.0; the linked 1.3 nightly does not build the engine. State layer ran with `.work/engine` built from
the lane tree and `serve/target/release/baro-serve` built in the worktree; the dense pack is a link to the primary's
.work/engine-pack-q4.

Receipts on the lane worktree: .work/moeprefill/ab/summary.tsv, .work/moeprefill/ab/arm.txt (arm shas 585935a8 / 9d1f6750, read-back
tmax 16384, prefill_rows 767, cached 0), .work/moeprefill/bench-old.txt, .work/moeprefill/bench-new3.txt, .work/moeprefill/test_moe_rows.txt,
traces .work/moeprefill/tr-moe-wmma, .work/moeprefill/tr-moe-exact, .work/moeprefill/tr-kq. ci-checks: 23 OK, 0 FAIL on 79ece97.

Unverified: no resident-mode MoE prefill number (the 21 GB pack does not fit next to the desktop); 8k and 32k prompt lengths not run;
the kq prefill path got a trace and a named limiter, no lever. Open: the staging copy rate (12 GB/s), the exact delta scan
(`amar_ssm_delta_rows`, 24% of MoE prefill kernel time), the kq GEMM's Q6_K instances.
