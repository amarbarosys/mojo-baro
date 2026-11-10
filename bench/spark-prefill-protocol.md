# Spark X2.5 4B prefill: prompt ingestion tok/s, ours vs llama.cpp master

Preregistered 2026-09-08 ~22:30, engine 5f5297d, before any timed run. Binds to
`bench/PROTOCOL-RULES.md` P1-P6. Question: prompt tokens per second at 64 / 256 / 1024 /
2048 ids, ours (token-by-token today; chunked prefill after the round) vs llama.cpp master
b10860 fast config, and does the prefill path change any generated token.

## Arms
- **ours/tbt**: `.work/spark/spark-engine`, prompt rows through the decode kernels one token
  at a time (`BARO_PF_CHUNK=0` once the flag exists). `prefill_s` = wall from the first
  prompt-token enqueue to the sync after the last prompt token (printed by the engine),
  `prefill rows` = n_prompt - 1 (the last prompt token is timed with generation).
- **ours/prefill**: same binary after steps 2-4, `BARO_PF_CHUNK=64` (and 128), rows 0..L-2
  in chunks through `amar_matmul_prefill_q8` + the SWA/gated prefill attention.
- **llama.cpp fast**: `~/llama.cpp-master/build/bin/llama-server`, `-fa on -ctk f16 -ctv f16
  -b 2048 -ub 512 -c 4096 -ngl 99 -t 8`, `/completion` on the ids, `n_predict 1`,
  `cache_prompt false`, `timings.prompt_ms` median of 3.
- **llama.cpp f32** (identity reference only): `-fa off -ctk f32 -ctv f32`, `n_predict 64`.
- Prompt set: `bench/spark-prefill-prompts/p{0064,0256,1024,2048}.txt` from
  `tools/spark-prefill-prompts.py` (mtp-prompts + docs text, sliced to exactly 64/256/1024/2048
  Mojo-tokenizer ids). Each arm tokenizes its own text; id-list identity is part of the gate.
- Read-back before any number is read (P1): engine sha256, power cap, vddgfx, `prefill rows`,
  chunk size; llama `props.json` per pass (fa, KV types, n_batch, n_ubatch), `prompt_n == N`.
  No receipt, no arm. Never both arms on the GPU at once. The RegesCore :8099 server is down
  for every timed run.

## Frozen prediction
- llama.cpp fast prompt tok/s: 64: 1500-3000 (launch-bound), 256: 3500-5500, 1024 and 2048:
  **5000-8000** (a 9B Q4 ran 3300 here; 4B Q8 has 2.2x fewer FLOPs per token).
- ours/tbt: **130-145 tok/s** at every length (decode path, 7 ms/token; attention cost at
  2048 is a few percent).
- ours/prefill after step 4: 64: 800-1500 (two chunks below the GEMM knee), 256: 1500-2500,
  1024: **2500-3500**, 2048: 2500-3500. Derivation: 8.2 GFLOP/token (4.1 B params x 2), GEMM at
  the Qwythos prefill kernel's 22.8 TFLOP/s = 2780 tok/s GEMM-bound; attention + elementwise
  + the last-token decode step eat the rest. Below 64 rows the path is weight-stream-bound
  (Qwythos R2), so chunk 64 is the floor.
- Ratio ours/prefill vs llama fast at 1024: **0.35-0.7**. Claim only if the band is cleared
  from below; the honest expected outcome is "3x faster than today, still behind llama.cpp".
- Identity (amended after step 1, see Result): greedy 64-token identity vs llama f32 is
  required at 64 and 256 ids only. Beyond that llama.cpp's own fast config fails it against
  its own f32 reference, so the gate is **teacher-forced agreement** (`BARO_FORCE=<ref ids>`,
  argmax per position vs the f32 reference stream, 64 positions): ours must be >= llama fast's
  own agreement at the same length minus 1, at every length. ours/prefill must additionally
  equal ours/tbt under teacher forcing at >= 63/64 (same numerics, only accumulation order).
- Decode must not move: `.work/spark/gate.sh` tok/s within +-2 % of 144 one-shot.

## Gates for step 1 (this commit)
G1 this file; G2 prompt files at exactly 64/256/1024/2048 ids (tools/spark-prefill-prompts.py:
64/256/1024/2048, all OK); G3 `tools/spark-prefill-ref.sh` end to end with id identity between
passes; G4 engine prints `prefill_s` + `prefill rows`, gate.sh 64/64 + 43/43; G5 committed
before any timing is read.

## Result — step 1 (2026-09-08 ~23:40, engine 245d31b, 290 W / -100 mV, llama.cpp b10860)

Read-back: engine sha 028d47d6…→245d31b after the fix, `prefill rows: N-1`, `chunk: 0`;
llama `props-fast.json` n_ctx 4096, `prompt_n == N` every run; id lists identical across engine,
llama fast and llama f32 at every length (`.work/spark/prefill*/`).

| ids | llama fast prompt_ms (median 3) | llama fast prompt tok/s | ours/tbt prefill_s (3 runs) | ours/tbt tok/s |
|---|---|---|---|---|
| 64 | 29.5 | 2171 | 0.426 / 0.434 | ~148 |
| 256 | 56.3 | 4546 | 1.700 / 1.699 / 1.703 | 150 |
| 1024 | 193.7 | 5288 | 7.353 / 7.364 / 7.368 | 139 |
| 2048 | 400.1 | 5119 | 15.41 / 15.39 / 15.37 | 133 |

Both arms inside their frozen bands (llama 5288 at 1024 in 5000-8000; ours 133-150 in/near 130-145,
the 256 row is 3 % above the band top). Ours is 35x slower than llama.cpp on prompt ingestion today.

**Identity finding, two parts.**
1. Greedy 64-token identity is an identity lottery past ~256 ids. llama.cpp fast (f16 KV, fa on)
   vs its own f32 reference: PASS at 64/257, FAIL at 256 (pos 58), 400 (pos 3), 512 (46), 1024 (8),
   2048 (49). Ours failed 400 at pos 3 too, where llama f32's top-1 has logprob -1.38 vs -1.82 —
   f16 KV rounding is enough to flip it. Token identity cannot be the gate at these lengths.
2. A real bug on top of it: `attn_head_span` addressed V rows from the page of the span start, wrong
   whenever the span is not page-aligned — i.e. on every sliding-window layer once T > 512. Fixed in
   1e7ab91 (Qwythos `test_attn_block` still PASS). Before: 768/1024/2048 diverged at position 1-2;
   after: 768 and 2048 are 64/64, 1024 flips at 54.

Teacher-forced agreement (argmax per position vs the f32 reference stream, 64 positions):

| ids | llama fast | ours/tbt (245d31b) |
|---|---|---|
| 64 | 64 | 64 |
| 256 | 64 | 64 |
| 400 | 63 | 62 |
| 768 | 64 | 64 |
| 1024 | 63 | 63 |
| 2048 | 64 | 64 |

Ours >= llama fast - 1 everywhere: gate PASS under the amended rule. G1-G5 all pass.
Ledger: `m.ledger/tooling.md` 2026-09-08 (the oracle's first f32 pass ran on an orphaned fast server).
