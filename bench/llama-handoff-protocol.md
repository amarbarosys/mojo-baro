# llama.cpp prefill, mojo-baro decode (LatentOS use 1)

Frozen 2026-09-11 before the first run. Question: can llama.cpp's fast prefill
replace ours on a long prompt, by saving its slot (`/slots/0?action=save`),
converting the state (`tools/llama-slot-to-state.mojo`, byte layout in
`.work/latent-llama/spec.md`) and loading it into our engine
(`BARO_STATE_LOAD`, landed in `c6d69a8`)?

## Instrument (`bench/llama-handoff.sh`)

Qwythos-9B, 8,000-token prompt P (essays) and P+Q (+17-token question), the
same token files as `bench/state-roundtrip.sh`. One gpu-wait job:
1. llama-server on `Qwythos-9B-...-Q4_0-pure.gguf`, `-np 1`, flash-attn on,
   `/completion` over P's token ids with `n_predict=1`, then save slot 0.
2. Convert the slot to a BAROST01 state for `.work/engine-pack-q4`.
3. Our engine, cold, on P + first token of Q, saving its own state at the same
   position (reference state).
4. Our engine, cold, on P+Q (reference continuation).
5. Our engine loading the converted state, P+Q, greedy, and again with
   `BARO_FORCE` = step 4's continuation (teacher-forced agreement).
6. `tools/state_diff.py`: converted state vs step 3's, per section.

Receipts: llama `timings.prompt_ms`, slot save ms, convert s, `state loaded ... in`,
`cached:` and `replay rows:` from our log, `BARO_FORCE` agreement line.

## Predictions

- P-LH1 layout: relative L2 error, converted vs ours, below 2e-2 for conv and
  SSM state and below 5e-2 for K and V (f16 KV, different kernels and weight
  quantisation paths). A transpose or ordering error gives O(1): the falsifier.
- P-LH2 the loaded run reuses the whole prefix: `cached: 8000`, 17 replay rows.
- P-LH3 teacher-forced agreement of the loaded run vs our cold continuation:
  >= 90% of the 64 steps (ORNITH measured 98.4% median between the two engines
  on whole prompts).
- P-LH4 time to first token, llama prefill + save + convert + load: 2.5 to 5 s,
  against our cold prefill of 6.21 s on P+Q.

Falsifier: any section's relative error above 0.3, or agreement below 50%:
the state does not map, whatever the timing.

### Run 1 (2026-09-11): VOID, instrument bug, no result recorded

The slot's token list is `server_tokens::serialize()` output, framed as
`[-1, 1, n, ids..., 0]` (8004 entries for the 8000-token prompt; the spec's open
item 5 had assumed a flat list). The converter wrote the frame header into the
state's token prefix, the prefix hash missed, and the engine prefilled cold
(`cached: 0`, 8016 replay rows). Its "greedy identical" and "64/64" compare a cold
run with a cold run and say nothing about the handoff. Fix: the converter reads
the framed ids and refuses any other framing. Predictions unchanged; run 2 next.
