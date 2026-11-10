# Family generator protocol (preregistered 2026-10-26, before any GPU run)

Question: can a new dense model family get a correct, tuned spark engine from one command, with the
kernels chosen and tuned from what llama.cpp itself builds for the arch?

Pipeline (`bench/family-gate.sh GGUF PACK OUT`):
1. `llama-eval-callback` on 1 token on the CPU: the op graph llama.cpp builds.
2. `tools/gen-profile.mojo --recipe`: activation from the ops (SWIGLU/SILU vs GEGLU/GELU), and a coverage
   check. Any op or tensor stem spark has no template for prints `FAIL coverage:` and stops the run. Rope type
   still comes from the table, because eval-callback does not print the mode.
3. `tools/tune-family.sh`: split-K knobs on the kq GEMVs (`BARO_SP_QKV_S`, `_O_S`, `_DOWN_S`, `_HEAD_S`),
   one binary per arm.
   - Stage 1 moves one knob at a time against base.
   - Stage 2 races the combined winners against base.
   - A knob wins on the median per-prompt ratio, with ids equal on every prompt. Spilling builds are dropped.
   - Packs without K-quant matrices skip the tune.
   - Each log is read back through its `spark knobs:` line.
4. `bench/dense-run.sh` with the flags: forced agreement against llama.cpp CPU f32 on the 20 prompts, bar min
   >= 90% per prompt, and tok/s_gen. Then `llama-bench tg64` for the speed ratio (reported, not gated).

CPU receipts taken before this prereg (not predictions):
- `--recipe` profiles are byte-equal to the table path for Llama-3.2-1B and granite-4.2-3b (`8e4bd7f`).
- gemma-3-270m (gemma3) fails coverage with 5 named features: Qcur_normed, Kcur_normed, attn_post_norm,
  ffn_post_norm, and SCALE (`inp_scaled`).

## Predictions

| arm | check | prediction |
|---|---|---|
| Llama-3.2-1B Q4_K_M | recipe | PASS, no WARN |
| | tune s2 tuned/base | 1.00 - 1.05x, ids equal 12/12 (or flags empty when no knob beats base) |
| | agreement | PASS, min >= 90 (prior spark sweeps 95-98) |
| | speed vs llama-bench tg64 | reported; prior same-stint ratio vs llama-server 1.38x |
| granite-4.2-3b bf16 (q8 pack) | recipe | PASS |
| | tune | SKIP (no K-quant matrices) |
| | agreement | PASS, min >= 90 (prior 98.4 - 100) |

A miss is reported as a miss and means optimize further. The gemma3 templates are a follow-up, sized from the
coverage list.
