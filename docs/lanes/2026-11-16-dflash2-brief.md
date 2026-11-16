# Brief: DFlash2 speculative decoding in mojo-baro (lane-dflash2, 2026-11-16)

Boss pane: a lane pane. Report: `docs/lanes/2026-11-16-dflash2-report.md`.
Per milestone: `~/iTools/bin/herd tell a lane pane "DONE dflash2 <milestone> -> <report> | ..."`.

## Goal
Target nerkyor Qwen3.8-27B (q4 pack) decodes with its DFlash2 block-diffusion drafter: greedy output identical to
no-spec, tok/s above our MTP path, measured against llama.cpp's own DFlash2 on this card (the boss runs that).

## Load first
Skills: `qwen38-nerkyor-coder` (DFlash2 section: config, taps, FP8 layout), `gpu-kernels`, `inference-kernels`,
`kernel-parity`, `bench-gates`, `mojo-lean`, `mojo-nightly-lane-builder`, `memory-optimization`, `rdna3-isa`, `lanes`.
Read: `docs/plans/2026-11-16-decode-speed-levers.md` (P3), `.work/consult/astra.md` section 4 (integration
constraints: SM=8 caps rows, SLOTS >= k+2, 144 MiB per SSM slot, fewrow_wmma needs D=4 at 27B shapes),
`.work/consult/agy.md` (DFlash2 risks), `docs/lanes/2026-11-15-fewrow-report.md`.

## Semantics reference (port from, do not guess)
`~/Projects/imports/llama.cpp`: `src/models/dflash.cpp` (draft graph), `common/speculative.cpp` (dflash block,
accept), `conversion/qwen.py` DFlashModel (tensor names, block_size, selector, conv, target_layer_ids + 1).
Weights: `~/Models/qwen38-27b-nerkyor-coder-q4lynn/dflash2-fp8/` (FP8 E4M3, 128x128 weight_scale_inv; q/k/v, fc,
selector, convs, norms BF16). Target config + tokenizer: `.../hf-target/`. Target GGUF: same folder.

## Milestones (each ends on a gate; commit as each lands)
D1 Falsifier first (S-M): 9-row verify window at 27B shapes. `amar_fewrow_wmma` D=4 into `gemm_w` for m > 4
   (keep m <= 4 on turbo), scoped verify capacity instead of raising SM globally, SLOTS >= 10 memory measured.
   Gate: target-only 9-row window time with fixed tokens on real weights + ids equal to m=1 replay on 20 prompts.
   Report the break-even acceptance tau = 68.72 x (T_verify9 + T_draft_est). If tau > 6, stop and report.
D2 Drafter (M-L): pack the DFlash2 weights (FP8 -> q8 or bf16, smallest that keeps parity), hidden-state taps at
   target layers 5/19/33/47/61 in the window path, Mojo forward of the 5 non-causal SWA layers + fc + selector +
   two-tap convs. Gate: per-position draft logits vs llama.cpp's dflash on dumped inputs (top-1 equal, top-16 overlap
   reported), reverse arm (taps shifted one layer must drop acceptance).
D3 Integrate (M): draft block -> verify window -> longest accepted prefix + bonus -> ring rollback (existing).
   Gate: greedy ids equal to no-spec on 20 held-out prompts at TMAX >= 16384, accepted-prefix histogram, 20-prompt
   median tok/s vs MTP k=2 on the same pack, same job.

## Rules
gpu-wait for every GPU job. No run under 16k. Read every bench script's Usage line. Kernel change -> wavewait
capture (bench/ww.sh). Do not change MTP default behavior; DFlash2 is opt-in (BARO_SPEC=dflash2) until it wins.
Stuck twice on one milestone -> report.
