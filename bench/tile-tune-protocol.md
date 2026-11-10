# Tile tune protocol (lane tile-tune, 2026-10-09)

Honest version of the CAPABILITIES "GGUF kernel JIT hot-patching" claim, which
was a dead env flag (`BARO_JIT_ASSEMBLY`, read and printed in `serve/spark.mojo`,
never used). Idea kept: tile choice per weight shape, measured once on this card,
stored in the GGUF, read at boot to pick among kernels that are already compiled.
No runtime compile: Mojo specializes tiles at build time.

## Stage 0: is there anything to tune?

The prefill GEMM dispatch (`serve/registry.mojo` `gemm_prefill_q4`) uses one rule
for every weight shape: m <= 64 wmma 64x128, m <= 128 wmma 128x128, else bf16-lds
128x128. The rule was tuned on the ffn shape only (`bench/pfgemm-protocol.md`).

Bench: `bench/bench_tile_sweep.mojo`, one build per shape (`-D TS_N -D TS_K
-D TS_TENSOR`), real q4 weights from the pack, six compiled variants plus the rule
arm at m = 16..1024, weight copies rotated over >= 256 MB (cold), every variant
bit-exact vs the rule arm before timing (all six share the K-order, so a mismatch
is a FAIL, not noise).

Shapes (Qwythos dense q4 pack): attn_q 8192x4096, attn_output/ssm_out/attn_gate
4096x4096, attn_k/v 1024x4096, ffn gate/up 12288x4096, ffn_down 4096x12288.

### Frozen predictions (before any GPU run)

Mechanism: the rule picks a 128-row tile at m > 64, so narrow-N shapes launch
fewer workgroups than the 96 CUs.

- P1 kv 1024x4096: at m = 256 the rule launches 8x2 = 16 workgroups; some
  variant beats it by >= 1.3x at m = 256 and m = 512.
- P2 attn_output 4096x4096 and ffn_down 4096x12288: best variant beats the rule
  at m = 128..256 by >= 1.1x (rule grid 32x1..32x2).
- P3 ffn 12288x4096: rule within 3% of the best variant at every m (it was tuned here).
- P4 at m = 1024 the rule (lds128x128) is best or within 3% on every shape.

A miss means optimize further, not drop (no minimum-gain bar).

## Stage 1 (after stage 0 numbers)

Winners per (shape, m-bucket) baked as `baro.tune.pf.<N>x<K>` KVs; registry reads
the table at boot into a small host array; dispatch indexes it. Gate: engine
prefill identity (bit-exact by construction, verified) + same-job prefill A/B.

### Stage 1 engine A/B predictions (frozen after the A/B job was queued, before any result was read)

`bench/tune-ab.sh`, same binary, BARO_TUNE=0 vs 1, prompts 128/256/512/1024/8192.
kv GEMMs are 2 of the 12 trunk GEMMs on attention layers only (1 in 4), so the
kernel-level 1.7-2x shrinks a lot end to end.

- E1 identity: GENERATED equal across all arms and runs at every length.
- E2 prefill_s A/B: 1.00-1.05 at 128..512, 1.00-1.02 at 1024 and 8192; no length below 0.99.

### Stage 1 results (2026-10-09, receipts `exchange/receipts/tile-tune/ab1`, `ab2`)

E1 HIT: GENERATED identical on every run, every length; TUNE read-back correct on every run.
E2 first run MISS: 1.004 / 1.009 / 1.013 at 128 / 256 / 512, but **0.927 at 1024, 0.945 at
8192**. Cause: the table took the fastest NON-rule variant even where the rule arm itself was
fastest (ffn_down m>=1024 picked lds64x128, 4174 us vs rule 4110 us; noise pick, and the
engine runs down with ACC=True, which the sweep never timed). `tools/tune-table.sh` now keeps
the rule's variant unless a variant beats the rule arm's median (one row changed). Rerun:
0.998 / 1.001 / 1.000 at 512 / 1024 / 8192. End to end the gain is inside run-to-run noise
(~1%): the kv GEMMs are too small a share of prefill for their 1.7-2x to show.
Verdict: lands, no regression, table opt-in per pack (no tune.txt = old rule, unchanged).

### Memory pass (2026-10-09, memory-optimization skill): no win, reverted

Engine side cold: the table is 35 x 4 Int (1.1 KB), read once at boot, borrowed by
`prefill_forward`/`gemm_pw`, scanned ~35 rows per GEMM launch. Sweep side candidates:
direct read into the pinned weight buffer (drops a 28 MB List + byte loop) and m*N-only
gate readback (~2.4 GB less D2H on ffn). Same-job, ffn shape, `/usr/bin/time`:
before 9.40 s / 1546 MB, after 9.78 s / 1525 MB (-1.3% RSS, wall inside noise). The
1.5 GB peak is the MAX runtime, not the bench's buffers. Under the skill's 5% floor:
reverted, nothing to tune here.

### Stage 1b: tighter A/B at 128/256 (frozen before the run, after pane a lane pane's review)

Review `exchange/review-lane-tile-tune-2026-10-09.md`: 128/256 gains (1.004/1.009) are inside the
1.5% swing seen at 512 on identical rows. Rerun `REPS=10 bench/tune-ab.sh ... 0128 0256`, fixed
table, strict-parser engine. Bar, set now: WIN at a length iff median A/B > 1.000 AND B faster in
>= 8 of 10 adjacent pairs (binomial p ~ 0.055 under no effect); otherwise NULL at that length.
Prediction: 256 WIN, 128 NULL.
Result (`exchange/receipts/tile-tune/ab3`, load 3.69, identity + read-back PASS): 128 A/B 1.004,
B faster 9/10 -> **WIN**; 256 A/B 1.006, B faster 7/10 -> **NULL** by the frozen bar. The
prediction (256 WIN, 128 NULL) was inverted on both lengths. A 0.4% prefill gain at 128 tokens.
