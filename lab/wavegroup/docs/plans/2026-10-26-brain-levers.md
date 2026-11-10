# Levers from the Brain network for wavegroup (2026-10-26)

Thesis: the next 2x on decode comes from fewer dependent dispatches, not from faster kernels; Brain already holds the
measured recipe (mojo-baro) for each step, and two tempting levers are already known dead.

Baseline today (Llama 3.2 1B Q4_K_M, 7900 XTX): wavegroup decode ~470 tok/s, prefill 511 tokens 24.4 ms, 64/64 tokens
equal to llama.cpp. tinygrad (same GGUF, examples/llama3.py model, greedy): 45.5 tok/s default, 133.6 tok/s with
JITBEAM=2, also 64/64 (`tools/tinygrad_decode.py`). llama.cpp tg64 441 tok/s.

## Ranked by expected gain, each with its Brain source

| # | Lever | Evidence | Expected | Size |
|---|---|---|---|---|
| 1 | Cut launches per layer: rope + KV append in one kernel, q/k/v in one GEMV (concatenate at load), norm folded into the next GEMV's prologue | mojo-baro launch-count round: 117.5 -> 142 tok/s bit-identical (`mojo/mojo-baro/2026-09-08-spark-engine-gate.md`); rmsnorm fold +2.6% (`2026-09-08-rmsnorm-fold-round.md`) | +15-25% | M |
| 2 | Persistent decode kernel: one dispatch per token, grid barriers between phases | barrier 0.6 us vs launch 2.4 us on gfx1100 (`mojo/mojo-baro/2026-09-05-grid-barrier-spike.md`); skill `persistent-kernel-gfx11`; Luce megakernel 1.55x over llama.cpp | up to ~1.5x | L |
| 3 | GEMV unpack cost: 60-85% of loop VALU is unpack, not FMA | `mojo/mojo-baro/2026-10-12-q4x-r5-strata-rest.md` (GEMM ALU diagnosis); wave-per-row weight-native layout reaches 96% of HBM on fp32 (`2026-09-04-qingming-gemv.md`) | small shapes up to 2x kernel time | M |
| 4 | Text in/out: GGUF BPE tokenizer + sampler already in Mojo | mojo-baro `serve/tokenizer.mojo`, `serve/sample_ref.mojo`; skill `gguf-tokenizer-spec` (llama-bpe defaults) | product usability | S (import) |
| 5 | Fair llama.cpp comparison: same-stint script, 20 prompts, ignore_eos, drift control | `mojo/mojo-baro/2026-10-12-llama-stint-bar.md` (`bench/llama-stint.sh`); skill `llm-benchmark-method` | trustworthy claim | S |
| 6 | KV cache int8 for long context | mojo-baro attn has `BARO_KVQ=int8`; briefs on KV quantisation | only past ~4k context | S (flag) |

## Known dead, do not retry

- Graph capture (hipStreamBeginCapture): removes host enqueue only (5.1 -> 3.5 us/launch); the ~3.5 us GPU-side dispatch
  floor stays, and a 540-launch replay was speed-null (`2026-10-12-q4x-r5-strata-rest.md`).
- GEMV rows per wave: R1 12.8, R2 13.2, R4 15.2, R8 18.5 us; more rows is slower (`wavewait/2026-10-22-gemv-rows-rejected.md`).
- int8 WMMA for speed: same issue rate as bf16 on gfx1100, MMQ measured 0.91x (skill `inference-kernels` section 5).

## Measuring traps that apply to our benches

- Memory clock steps up after ~1 s of load: GEMV numbers need >= 1 s warm-up (ours: 2000 launches, about 10 ms: too short).
- Interleaving variants per weight copy lets later variants read from the 96 MB Infinity Cache.
- Kernel-name hash order is not launch order in traces: map by start time.

## Measured after the launch cuts (2026-10-26, wavegroup 61f72c2)

- GEMV kernels are near bandwidth: gate+up 23.85 us for 18.9 MB = 82% of 960 GB/s. Cheaper unpack has little left to win.
- The default path is idle 31% of the span (dispatch gaps, ~148 dispatches/token); rmsnorm is 33 x 4.3 us per token.
- Folding the ffn norm into gateup (every one of 1024 blocks recomputes the rms scale): 549-552 tok/s vs 555-560 without,
  400 tokens, 64/64 both ways. Rejected: the redundant x reads cost more than the launch they save.
