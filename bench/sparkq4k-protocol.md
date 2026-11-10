# sparkq4k: native K-quant weights in the spark engine (preregistration)

Brief: `exchange/lane-sparkq4k-brief.md`. Rules: `bench/PROTOCOL-RULES.md` (P1, P2, P4, P11, P14, P18).
Frozen BEFORE the gate job runs (this file's commit precedes the receipts in `exchange/receipts/sparkq4k/`).

## Change under test

`tools/engine-pack.{py,mojo} --dense` writes Q4_K / Q6_K matrices as their gguf blocks (dt `q4k`, `q6k`,
`q4k/q6k/<A elements>` for a fused QKV whose V rows are Q6_K); `serve/spark.mojo` runs them through
`amar_gemv_kq` (`kernels/gemv_kq.mojo`, byte test `kernels/test_gemv_kq.mojo`). q8 and f32 entries are
unchanged, so an existing q8 pack runs the old path (control arm).

## Arms (same job, `bench/sparkq4k-ab.sh`)

| model | gguf | profile | q8 pack (control) | K-quant pack |
|---|---|---|---|---|
| vibethinker-1.5b Q4_K_M | `~/Models/vibethinker-1.5b-gguf/vibethinker-1.5b-Q4_K_M.gguf` | `~/Models/vibethinker-1.5b-gguf/profile` | `~/Models/vibethinker-1.5b-gguf/pack` (import 2026-10-18) | `.work/sparkq4k/packs/vibe-kq` |
| MiniCPM5-2B Q4_K_M | `~/Models/minicpm5-2b/MiniCPM5-2B-Q4_K_M.gguf` | `~/Models/minicpm5-2b/profile` | `~/Models/minicpm5-2b/pack` (import 2026-10-18) | `.work/sparkq4k/packs/minicpm5-kq` |
| Qwen2.5-7B-Instruct Q4_K_M | `~/Models/qwen2.5-7b-instruct-import/qwen2.5-7b-instruct-q4_K_M-BARO-3c6c25c.gguf` | `~/Models/qwen2.5-7b-instruct-import/profile` | `.work/sparkq4k/packs/qwen7b-q8` (rebuilt with the pre-change tool, commit 8e8f853) | `.work/sparkq4k/packs/qwen7b-kq` |

Gate per arm: `bench/dense-run.sh` (llama.cpp CPU reference, f32 KV, 16K window, 20 prompts, `MIN_PCT=90`,
`MIN_REF=16`): teacher-forced agreement per prompt, plus no-spec `tok/s_gen` per prompt (64 generated tokens).
Read-back (P1): the engine's `pack loaded ... matrices: q8 N kquant M` line per run (q8 arm must read `kquant 0`,
K-quant arm `q8 0` on vibethinker and MiniCPM5, whose matrices are all Q4_K / Q6_K), the arm file's pack
index sha256 and pack.bin size, the engine sha256, `BARO_SPEC=0`, `BARO_GEN=64`, `TMAX` from the profile.
Fail words grepped on every engine log: `NOT-RESIDENT`, `error`, `Error`, `FAIL`.

## Bars (frozen)

1. vibethinker Q4_K_M K-quant pack: `dense-run` PASS (min >= 90% over the exercised prompts, >= 15 exercised)
   AND p04-list-planets >= 58/64. The q8 control on the same job must reproduce the known FAIL (p04 51/64).
2. No regression: MiniCPM5-2B K-quant pack PASS at MIN_PCT 90; Qwen2.5-7B K-quant pack PASS at MIN_PCT 90.
3. Speed lands at any ratio if 1 and 2 hold (no minimum-gain bar, the maintainer 2026-10-09); the number is reported
   either way as the 20-prompt median of `tok/s_gen`, K-quant over q8, min-max beside it.

## Predictions (frozen)

| quantity | point | band | falsifier |
|---|---|---|---|
| vibethinker p04 forced agreement | 62/64 | 58-64 | < 58: the native path still loses the first token; per-layer oracle (`tools/llama-oracle.py`) before any kernel change |
| vibethinker gate mean | 97% | 94-100 | |
| MiniCPM5 gate min / mean | 91 / 98 | min >= 90 | < 90 with q8 control >= 90.6: a K-quant defect on H=2048 (NKVH 2, V rows Q6_K in 21 layers) |
| Qwen2.5-7B gate min / mean | 97 / 99 | min >= 90 | |
| tok/s ratio K-quant / q8, vibethinker | 1.10 | 0.90-1.40 | below 0.90: the lane decoder is issue-bound (Q4_K header reloaded per lane); report, do not block the land |
| tok/s ratio, MiniCPM5 | 1.10 | 0.90-1.40 | |
| tok/s ratio, Qwen2.5-7B | 1.25 | 1.00-1.50 | bytes per token 9.0 -> 6.1 GB, bandwidth-bound at 7B |

Pack bytes (measured before this freeze, not predicted): vibethinker 2454.9 -> 1824.9 MB, MiniCPM5 3299.7 -> 2360.6 MB,
Qwen2.5-7B 9244.4 -> 6247.1 MB (`pack.bin`; token_embd stays exact f32 in both).

## Repairs

Three per failure (brief). A repair that changes a kernel reruns `kernels/test_gemv_kq` first and reruns the whole
job, never one arm. Receipts: `exchange/receipts/sparkq4k/` (arm file, SUMMARY and results per arm, medians).

## Amendment A1 (2026-10-19, written AFTER the data; P14: it annotates the frozen verdict above, it does not flip it)

Finding: the frozen bar 1 is unreachable. An f64 forward of the gguf (`exchange/receipts/sparkq4k/p04-force-emu.py`)
teacher-forced on llama.cpp CPU's p04 tokens reads 52/64 (`p04-emu-f64.txt`); the reference disagrees with the exact
model on 12 positions, so no correct engine reaches 58/64 against it. Verdict of the frozen run stays FAIL (53/64).

From this amendment on, the vibethinker Q4_K_M gate is the exact-model arm, `bench/sparkq4k-truth.sh`:
- reference: the f64 model's own 64 greedy tokens per prompt (`truth-gen.py`, f64 everywhere, gguf-py dequant), 20 prompts,
  token files checksummed in the receipts (`truth/truth-tokens.md5`);
- bar: teacher-forced agreement min >= 90% over the 20 prompts, p04-list-planets >= 58/64, q8 control in the same job;
- llama.cpp CPU (`bench/dense-run.sh`) stays as a reported control row for every arm, never a bar;
- the control that keeps the arm honest: the q8 pack must read below the K-quant pack on p04 (it reads 41/64 vs 56/64).

Measured against this arm (2026-10-19, `exchange/receipts/sparkq4k/truth/`): K-quant 87.5 min / 99.2 mean, p04 56/64:
FAIL on the new bar too, by p04 alone. Falsifier for the next lever: the bf16-input emulation forced on the exact model's
p04 tokens reads 56/64 (`p04-emu-bf16act-vs-truth.txt`), the engine's number, so the remaining misses are GEMV input
rounding. Next lane, frozen here: f32 GEMV inputs on the K-quant path, prediction p04 64/64 (band 62-64), gate min 90.

## Amendment A2 (2026-10-19, frozen BEFORE its runs): f32 GEMV activations

Change: the spark engine's activation dtype `ADT` (rmsnorm output, attention output, gate*up epilogue; the inputs of
every GEMV) is f32 by default; `-D BARO_ACT_BF16=1` rebuilds the previous bf16 numerics as the control arm. Kernels
`amar_rmsnorm_cast`, `amar_attn_decode_swa_gated`, `amar_gemv_q8`, `amar_gemv_kq`, `gemv_epi` take the dtype as a
trailing defaulted parameter (bf16), so every other caller is unchanged. Read-back (P1): the engine's `act: float32`
word on the pack-loaded line, grepped per run; `kernels/test_gemv_kq` runs its 4 tensors x 4 epilogues in both dtypes.

Arms: `bench/sparkq4k-truth.sh` (exact-model arm) once per dtype; `bench/sparkf32-ab.sh` (llama.cpp CPU control row and
no-spec tok/s_gen, 20 prompts) vibethinker bf16 then f32, MiniCPM5 f32 only (its bf16 row is the sparkq4k ab1 receipt on
the same kernels). Qwen2.5-7B is NOT re-run in this amendment (45 GPU minutes for a row predicted unchanged); stated in
the report as not exercised.

| quantity | point | band | falsifier |
|---|---|---|---|
| vibethinker p04, exact-model arm, K-quant pack, f32 | 64/64 | 62-64 | < 62: the emulation (`p04-emu-bf16act-vs-truth.txt`, 56/64) does not model the engine; stop, per-layer oracle |
| exact-model arm K-quant f32, min / mean over 20 | 98 / 99.8 | min >= 90 | |
| exact-model arm q8 pack f32 (q8 weights, f32 inputs) | p04 >= 41/64, mean >= 97.3 | no regression | |
| llama.cpp control row, vibethinker f32, min / mean | 85 / 97 | min >= 82.8 (bf16 row) | below the bf16 row: regression |
| llama.cpp control row, MiniCPM5 f32, min | >= 92.2 | min >= 90 | |
| tok/s_gen f32 over bf16, vibethinker K-quant, 20-prompt median | 0.97 | 0.94-1.02 | below 0.94: the doubled L2 re-read of X per wave shows; report, land anyway (no minimum-gain bar) |

Bars: exact-model arm min >= 90 AND p04 >= 58/64 (A1); llama control rows no worse than the bf16 rows; speed reported.
