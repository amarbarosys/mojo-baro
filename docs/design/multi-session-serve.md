# Multi-session serve (street-sim AI lab), 2026-11-06

One Ornith-1.5-9B kq engine serves 32 sessions at 1.71x the aggregate tok/s of the serial server and 2.6x
llama-server `--parallel 32` (fresh game states), with each session's greedy ids equal to the single-session engine's at the same
restore point. Brief: `briefs/2026-11-06-multi-session-serve.md`; report
`briefs/2026-11-06-multi-session-serve-report.md`.

## Use it

```
mojo build serve/engine.mojo -I . -I kernels -D BARO_SEQ_CAP=32 -D BARO_SLOTS=2 -o .work/engine-seqs32
BARO_SPEC=0 serve/target/release/baro-serve --engine .work/engine-seqs32 --pack PACK-KQ --parallel 32 --kv-pool 3 \
  --engine-env BARO_TMAX=20480 --engine-env BARO_SPEC=0
```

`--parallel N` sets `BARO_SEQS=N` and `BARO_KV_SHARE=1`; `--kv-pool P` is the KV pool in TMAX units
(P = 3 at TMAX 20480: 61,440 tokens, 4.0 GB f32). VRAM loaded at 32 slots: 17.1 GB. TMAX must hold
prompt + `max_tokens` (the repo floor is `max_tokens` 16000; game prompts are ~1.8K tokens).

## What existed before (read 2026-11-06)

| piece | state |
|---|---|
| `baro-serve` admission | A3(a): N request lines in flight on the wire, ids routed. Engine still serial. |
| per-slot state | A3(b): KV pool and rings sized `SEQ_CAP` slots, page table. |
| one-launch rows over distinct requests | NOT BUILT. `exchange/lane-A3-c-kernel-request.md` (five `RowDesc` kernels) never started. |
| pair engine `BARO_SEQS=2` | MoE megakernel only, R = 2, `KV_SHARE` needs `SEQ_CAP <= 2`. Not the dense kq path. |
| kq decode m = 1 | launch path 98.3 tok/s (Mojo 1.2), megakernel same ids. |
| kq prefill | WMMA GEMM, 0.97x llama.cpp time at 1K. Not bit-exact with replay. |
| prefix reuse | `serve/prefix.mojo` checkpoints (KV + SSM state snapshot), pinned system checkpoint. Serial. |
| tools | `tools` reach the template, `tool_calls` parsed. `tool_choice` not read. |

## Design

```
HTTP x N --> baro-serve (--parallel N, 429 past slots/KV pool) --lines--> engine BARO_SEQS=N
engine loop:  admit -> prefill one request (restore shared prefix: copy SSM state, map prefix KV pages read-only)
              -> one batched decode step over every active slot -> emit tok lines -> free slot on done
```

- Slot = conv ring + SSM state + page list. Prefix pages are refcounted and shared read-only; a session's own
  pages start after the prefix end (the partial tail page is copied, never shared writable).
- Batched step: every GEMM reads the weights once for R rows; each row dots in the m = 1 per-element order
  (`gemv_kq.kq_lane_part`), so a row's ids do not depend on N. Attention, conv, delta, gated-out, rope, norm,
  argmax: per-row descriptor `(pos, page base, state base)`, row = `block_idx.y`, m = 1 arithmetic.
- P6 harness before kernel: first the batched loop with N serial m = 1 launches per step (identity and
  scheduler), then the row kernels one at a time, each with a byte test against m = 1.
- Memory: the pool is `--kv-pool P` x TMAX tokens (size it from measured free VRAM); a slot takes pages for its
  prompt plus a 512-token decode reserve at admission and one more page whenever it reaches one. A request that
  does not fit waits in the engine until pages free; `--max-inflight` (64) is the hard bound, 429 past it, and a
  request that cannot fit even alone is refused with 429. Pages are freed at `done`. Each slot's SSM ring has 2 entries (`-D BARO_SLOTS=2`, 100 MB per slot) instead of 9 (453 MB).

## Measured (Ornith-1.5-9B Q4_K_M kq pack, 20 game states, lockstep bursts, `bench/ailab/`)

Fresh state per request (the game never repeats one), fresh server per N:

| N | serial baro-serve tok/s (p50 / p95 s) | multi-session (p50 / p95) | llama-server --parallel N (p50 / p95) |
|---|---|---|---|
| 1 | 63.8 (0.51 / 0.66) | 61.3 (0.53 / 0.69) | 70.6 (0.76 / 0.99) |
| 4 | 65.7 (1.06 / 2.15) | 99.3 (1.25 / 1.44) | 52.5 (3.86 / 4.58) |
| 16 | 67.0 (4.52 / 8.04) | 118.4 (4.59 / 4.78) | 52.4 (17.03 / 18.43) |
| 32 | 65.9 (8.69 / 15.58) | 112.5 (9.32 / 9.80) | 42.5 (39.97 / 41.72) |

Decode step time is linear in rows: 12.0 ms at 1 row, 34.6 at 8, 69.1 at 16, 138.3 at 32 (about 3.1 ms per
extra row; groups of 8 because the step scratch holds `MROWS` = 8 rows). The rows GEMV reads the weights once
per group but is far from weight-bound: the next lever (an extra row costs 26% of an m = 1 step).
With fresh states prefill is 58% of engine time at N = 32 (5.2 s of 9.0 s: one WMMA prefill of ~270 rows per
request after the shared-prefix restore): batching a burst's prefills is the larger lever.

## Phases and checks

| # | item | check | LOC |
|---|---|---|---|
| 0 | 20 states `bench/ailab/states/*.json`, load driver, baselines (current serve, llama-server `--parallel N`) | driver runs N = 1, 4, 16, 32 on both | M |
| 1 | `tool_choice` (none / auto / required / named -> grammar), `--parallel`, 429 | cargo test + HTTP | S |
| 2 | engine `BARO_SEQS=N` dense: slots, scheduler, prefix share, serial-row step | N = 16 ids == N = 1 ids, 20/20 | L |
| 3 | row kernels: gemv_kq rows, attn rows, ssm rows, elementwise rows | byte test vs m = 1 per kernel | L-XL |
| 4 | gates: parity, throughput table, 1,000-burst stability | `bench/ailab/*.sh` | M |

## Update 2026-11-07: the street-sim runner's real requests (`docs/plan-ailab-server.md`)

- Tool prompts now render like HF and llama.cpp (`9c18bc4`): before, sorted compact `tojson` made every tool
  prompt differ from llama-server at token 20 (0/268 runner requests). Gate: `bench/ailab/render-parity.sh` 272/272.
- Real runner prompts are 4.0-4.6K tokens (manual + 18 tools = 3,569 shared). With the repo's `max_tokens` floor of
  16000, run at TMAX 24576; at 20480 the server answers 400 and `ask_openai` silently re-asks without tools.
- `bench/ailab/serve.sh up` warms through the runner twice and fails unless the second ask restores >= 3500 cached
  tokens (`9e61b00`); without it a cold burst of 20 overflowed the KV pool (429 `kv pool full`).
- Measured on the runner's own loop (`bench/ailab/step.sh`, N bodies, 3 steps, timed): N=20 step 34.8 s (llama-server
  `--parallel 20` 89.3 s), N=50 84.0 s (llama-server: 117/150 asks time out at the runner's 120 s). 2.67 rounds per
  ask; prefill 56% of engine time. Soak N=50 x 30 steps: 3,992 posts, 0 errors, engine VRAM slope 0, card 24.2/24.5 GB.
- Identity holds per checkpoint STATE, which depends on the request that created it (arrival order): single vs
  multi-session with a pinned creator 0/19 mismatches (P4d); identity gates pin the creator in every arm.
- Parity vs llama-server (P4, f32 KV both, 63 requests): FAIL on the frozen bar, forced agreement 86.3% vs llama's
  own 92.7% p20-vs-p1; cause not established (exact-model arm P4b blocked on its self-check).
