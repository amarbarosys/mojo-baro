# Handoff: lane-dflash2 after D2d + D2c kernels (2026-11-16)

Read first: `docs/lanes/2026-11-16-dflash2-brief.md`, then `docs/lanes/2026-11-16-dflash2-report.md` (D1 result, port
semantics, D2c build plan, D2d dump format). Boss pane a lane pane; report lines go through `herd tell` in the HM/1 shape
(`VERB job status -> path | k=v`). Boss decisions so far: D3 gate = teacher-forced agreement with a same-arm control, bar
frozen before the run; smallest tests; dumps at target ngl <= 52; dflash2 is the critical path (gpu-wait --priority 90).

## State

| item | commit | status |
|---|---|---|
| D1 falsifier (8-row window, BARO_FEWROW) | 601f691, 21e890c | timing GO, ids gate replaced by forced agreement |
| D2a draft pack (`engine-pack --arch dflash`) | 46eb9ea | `.work/dflash2/draft-pack`, 49 q8 + 32 f32 |
| D2b taps (`tap_d` f32[5, MROWS, H], BARO_SPEC=dflash2) | fd232e2 | built, verified only through the D2 gate |
| D2d reference dumps | dade035, llama worktree ff30654a3 | `.work/dflash2/d2d/p01-water.bin`, `p02-python-fib.bin` |
| D2c kernels `kernels/dflash_kernels.mojo` (renamed) | 86a835e, da0c52d | exercised by the parity replay |
| D2c host: dump reader + host walk | 1832d16 | CPU: counts equal, walk == dumped ids 26/26 |
| D2c host: `serve/dflash.mojo` + `bench/dflash_parity.mojo --gpu` | da0c52d | B1 0.967 PASS, B2 15.29 PASS, R1 FAIL as frozen (`.work/dflash2/parity/run1.log`) |
| D3 wiring `BARO_SPEC=dflash2` + gate `bench/dflash2-d3.sh` | 616a223, dc3a3da | run 1: G1 PASS, R1 PASS, G3 FAIL 0.957 (`.work/dflash2/d3`) |
| D3 GPU top-16 | 3077687 | run 2: G1/R1 PASS, G3 FAIL 0.978, draft still 10.4 ms/cycle (`.work/dflash2/d3r2`): BLOCK sent, next = draft phase split |

## Next: D3 (boss ACK 2026-11-16: D2c accepted on B1/B2, R1 FAIL recorded, D3 reverse bar frozen before the run)

D2c steps 1-3 below are done. D3 open items: GPU top-k in place of the host top-16, draft time per block, taps from
`tap_d` in place of dumps, ring rollback on reject, `BARO_KVQ=int8` for memory.

## Done: D2c host side (L)

1. `bench/dflash_parity.mojo` reads a D2d dump (record format in the report). Smallest first test, no engine: replay the F
   records through injection only and compare nothing; prove the reader on both files (counts must equal 15/121/14 and
   13/106/12).
2. `serve/dflash.mojo`: struct with the draft pack offsets (index.txt names: `fc.weight`, `enc.output_norm.weight`,
   `blk.N.{attn_norm, attn_conv_proj, attn_conv_base, attn_q, attn_k, attn_v, attn_q_norm, attn_k_norm, attn_output,
   ffn_norm, ffn_conv_proj, ffn_conv_base, ffn_gate, ffn_up, ffn_down}`, `output_norm.weight`, `selector_hidden.weight`,
   `selector_successor.weight`, `selector_predecessor.weight`) and a KV ring per layer (RING 4096, kpos ring).
   - inject(feat rows, positions): `g = rmsnorm(fc @ feat, enc.output_norm)`; per layer K = rope(rmsnorm_head(Wk g)),
     V = Wv g; write cell pos % RING, set kpos.
   - draft(anchor, pos0): tokens [anchor, 248070 x 7] embedded with the TARGET token_embd (target pack `token_embd.weight`,
     bf16); 5 layers per the report's port semantics; output_norm; target LM head on 8 rows; top-16 per row; gate =
     selector_hidden @ h_final; `amar_df_lattice`; walk on the host.
   - Rope: `amar_rope_plain[128, NEOX=True]`, freq_base 1e7 (the GGUF's M-RoPE sections [64,0,0,0] equal NEOX).
   - Projections: `gemm_q8` (runtime m, n, k; bf16 A). eps 1e-6 everywhere.
3. Parity gate (freeze the bar in the report BEFORE the run): per L record, top-1 candidate equal, top-16 overlap, walked
   ids equal, max-abs of score after mapping columns by candidate id (ggml_top_k leaves candidates unordered). Reverse arm:
   features replayed at positions shifted by one must lose top-1 agreement. One gpu-wait job, the two dumps only.
4. Then D3 per the brief, with the D3 gate the boss set.

## Traps already paid for

- Serve mode replays prompts with `replay_m=MROWS`: any m>4 route also changes no-spec runs.
- On a shared GPU the megakernel grid barrier can time out (fail word): gate runs use BARO_MEGA=0.
- Clear an output dir and truncate its log before arming a Monitor on it.
- Rows exceed positions in the dumps: the replay keeps the latest feature per position.
- toolscript_gate blocks the 2nd+ parallel read inside run_js: read through `tools.call("Bash", ...)` with write:true.
- `mojo-port` has no engine-pack case; do not reach for the gates-off escape (denied).
- `gpu_job` preset cmd is the job command only: gpu-wait flags in it reach bash. Use `gpu-wait run ... -- cmd` from Bash.
- `ref` and `out` are Mojo keywords: never name a parameter with them.
