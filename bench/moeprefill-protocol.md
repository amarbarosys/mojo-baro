# Lane moeprefill: next kernel lever on the MoE and prefill paths (preregistered 2026-11-02)

Brief `.work/briefs/moeprefill.md`. Frozen by this commit before any timed run of the lever.

## Step 1, traces (`bench/prefill-trace.sh`, rocprofv3 kernel trace, LONG p1024 minus SHORT p0128, one gpu-wait job)

MoE path: qwen35moe, tier-split pack (`.work/latent-spec/pack-tier`, same 733-tensor table as `.work/moe-ngram/pack`),
`BARO_TIER=64`, TMAX 16384, int8 KV, one 1023-row chunk. Default arm (`BARO_PF_GEMM=wmma`): prefill_s 2.306 s LONG,
1.636 s SHORT (127 rows), so about 1.5 s per chunk is row-independent: the 18.3 GB expert staging copy
(`expert_tier.stage_layer`, about 12 GB/s). Prefill kernels for the 896 extra rows sum to 275 ms, about 0.31 s per chunk,
14% of the wall. Top prefill kernels, us per 896 rows: `amar_ssm_delta_rows` 65007 (30 launches), `moe_down_q6k_grouped`
38506 (3 launches, the three q6_k down layers), trunk `matmul_prefill_lds` 52k (4 instances), `moe_wmma_q4k_grouped` gate/up
24923 + down 13841 (37 launches), shared expert rows 24476 + 18126, `moe_pairs_group` 22971, router `moe_skinny_f32_rows` 11224.
Exact arm (`BARO_PF_GEMM=exact`): 1328 ms per 896 rows, `moe_gate_up_q4k_grouped` 540 ms, `moe_matmul_q8d_rows_shared` 403 ms,
`moe_down_q4k_grouped` 197 ms.

Prefill path: Ornith-1.5-9B Q4_K_M kq pack, dense engine, 1023 rows: prefill_s 0.252 s; `amar_matmul_prefill_kq` 139 ms of a
165 ms prefill kernel sum (84%), 87 ms of the wall is outside kernels.

Top limiter named: MoE tier prefill = the staging copy (not a kernel); MoE kernel side = the exact delta scan (24% of prefill
kernel time) then the q6_k down grouped kernel (14% from 3 of 40 layers: 12.8 ms per layer against 0.37 ms for the wmma q4_k
down and 5.3 ms for the exact q4_k down grouped). kq prefill = the K-quant WMMA GEMM (84%), B-load-issue bound per run 10.

## Step 2, lever: `moe_down_q6k_grouped` with hoisted per-block loads

Design: the kernel keeps the m=1 `q6k_row_dot` element order (lane L owns k = L + 64 i, accumulates in k order, `warp.sum`
after), so it stays byte-exact to `amar_moe_down_q6k` by construction and the decode m=1 kernel is untouched. The change: the
per-element `q6k_value` call (5 scalar byte loads and the index math per element, 4 elements per 256-block per lane) becomes
per-block work with lane-constant offsets: 2 ql bytes, 2 qh bytes, 4 scale bytes, one d per block per lane, the block loop
comptime-unrolled (E_FFN // 256 = 2), activations for the MR rows loaded before the FMAs, `flat_work_group_size` annotated.
Value per element stays `d * sc * (q - 32)`, accumulation stays `acc += h * v`.

Checks: `bench/bench_moe_q6k_rows.mojo` (byte-exact against the m=1 kernel over 1023 tokens on random bytes, then us per
launch at the engine shape, HEAD before and after), `kernels/test_moe_rows.mojo` (run-tests arm), then `bench/moe-prefill-ab.sh`:
one gpu-wait job, two binaries (sha printed from the run, equal shas refused), tier mode, 20 prompts of 768 tokens, ABBA,
every request's 16 generated ids compared between arms, prefill_s median ratio, read-back of TMAX, prefill rows, cached, tier line.

Predictions (frozen):
- P1 bench us per launch, old kernel: 11000 to 16000 (the trace says 12.8 ms per layer at 896 rows).
- P2 bench us per launch, new kernel: 4000 to 8000, ratio new/old 0.3 to 0.6.
- P3 byte test: 0 differing bytes against the m=1 kernel, both in the bench and in test_moe_rows.
- P4 engine A/B, tier mode, 768-token prompts: prefill_s median ratio B/A 0.98 to 1.00; the chunk is copy-bound, so the
  kernel gain (about 25 ms of 310 ms kernel time per chunk) is predicted NOT to show on the wall. Identity 20/20.
- P5 rocprofv3 per-kernel receipt inside the A/B job (p1024, one run per arm): `moe_down_q6k_grouped` us per 3 launches drops
  by the P2 ratio.
Falsifiers: P2 ratio above 0.8 means the loads were not the cost (then wavewait the kernel before any second attempt); any
differing byte is a bug in the rewrite, not a tolerance question; P4 outside 0.96 to 1.02 means something else moved.
Adoption: the kernel lands if P3 holds and P2 or P5 shows a gain; P4 is reported as measured (no minimum-gain bar).

## Verdict (2026-11-02, after the runs; receipts `.work/moeprefill/` on the lane)

| prediction | band | measured | verdict |
|---|---|---|---|
| P1 old kernel us/launch | 11000 to 16000 | 20856 (median of 11 x 20) | MISS high: standalone cold weights, 1023 rows vs 896 in the trace |
| P2 new kernel us/launch, ratio | 4000 to 8000, 0.3 to 0.6 | 4674, 0.224 | absolute HIT, ratio MISS low |
| P3 byte test | 0 differing bytes | bench 0 of 8380416 bytes; test_moe_rows 18 outputs bit-exact | HIT |
| P4 engine A/B tier 20 x 768 | B/A 0.98 to 1.00, ids 20/20 | median 0.9500 -> 0.9246 s, B/A 0.9732, paired 0.9715, B faster 20/20, ids equal 20/20 | ratio MISS low (gain shows), identity HIT |
| P5 in-engine kernel receipt | drops by the P2 ratio | 3 launches 45571 -> 12878 us (0.283), p1024 GENERATED equal | HIT |

Adopted: the kernel lands (`652d4c7`). The copy-bound reading of the chunk was too strong: 25 ms of kernel time removed bought
25 ms of wall, so at 767 rows the staging copy does not fully hide the kernels. The remaining MoE prefill wall is still about
0.9 s per chunk of which the kernels are now about 0.29 s. Arm A sha 585935a85bb3f3c4, arm B 9d1f6750676687a7, same job.
