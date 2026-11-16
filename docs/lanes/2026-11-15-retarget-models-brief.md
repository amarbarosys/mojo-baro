# Brief: retarget tests after the ~/Models trim (2026-11-15)

Boss pane: a lane pane. Report: `docs/lanes/2026-11-15-retarget-models-report.md`. Per job end:
`~/iTools/bin/herd tell a lane pane 'DONE retarget -> docs/lanes/2026-11-15-retarget-models-report.md'` (HM/1 form).

## What changed
the maintainer deleted every small model except `~/Models/llama-3.2-1b-instruct-q4_K_M` (the spark-engine rig). Gone: qwen2.5-7b-instruct-import,
qwen2.5-0.5b, qwen2.5-7b-instruct-gguf, minicpm5-2b, granite-4.2-3b-bf16, vibethinker-1.5b(-gguf), spark-x2.5-4b, spark-x2.6, lily-7b,
nemotron-3.5-bolt, lfm3, gemma-3-270m, distilgpt2. Kept big models: Qwythos-9B, Ornith-9B, Qwen3.8 27B (Turbo, OBLITERATED, Orca), Flash-Next, RegesCore.

## Job
Every live script/config that names a deleted model gets retargeted, then the check it belongs to is run once and passes.
Find them in one pass (rule: never loop recursive greps):
`rg -j16 -l -F -f PATTERNS -g '*.{sh,py,toml,json,mojo,rs,yaml}' ~/Projects/mojo/mojo-baro ~/Projects/mojo/mojo-neuron ~/Projects/street-sim`
(PATTERNS = the deleted folder names). Skip archives, exports, `*-public`, lane worktrees and street-sim clones (`street-sim-*`): list them in the report only.
- spark-engine tests (llama/qwen2/granite/spark2_5 arch) -> Llama-3.2-1B.
- qwen35-shaped needs -> Qwythos-9B q4 (`.work/engine-pack-q4`) or Ornith.
- `bench/quality-models.json`, `quality-bands.json`: drop deleted entries; `docs/BASELINE.md` rows stay as history, add one line "model deleted 2026-11-15".
- multi-model matrices (mojo-neuron lens/steer/fingerprint, street-sim casts): reduce to the models that exist; say what coverage was lost.
- A test that needed a second spark model (cross-model) and has no stand-in: mark it PARKED with the reason, never silently skip.

## Rules
- Load skills first: mojo-lean, gpu-lane-budget, mojo-nightly-lane-builder. Haiku sub-assistant allowed for reads.
- Worktree per repo (`git worktree add -b lane-retarget ... && lane-prep ...` for mojo-baro; plain branch elsewhere). Commit per repo, no push.
- GPU only via `gpu-wait run --timeout S`, env inside `bash -c`; BARO_TMAX >= 16384; the active LACT profile is AI-2500-uv125, leave it.
- Every mistake -> `docs/rules.md` + SINDEX lesson-entry (Edit tool). Board card ticked at the end. Housekeeping at the end.
- Merge to main only after `lane-merge` READY; if a merge is denied, report READY and stop.

## Addendum (2026-11-15, later): also deleted
OrcaSAQ-2-27B, Qwen3.8-27B-OBLITERATED, Qwen3.8-Flash-Next-GSQ Q2_0. Retarget: any 27B qwen35 use -> `~/Models/qwen3.8-27b-turbo-fcf-mtp-q4_K_M/Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M-BARO-a4eb549.gguf`; any Flash-Next GGUF -> `~/Models/qwen3.8-flash-next/UD-IQ1_M/Qwen3.8-Flash-Next-UD-IQ1_M-BARO-d37ec15-00001-of-00003.gguf` (a q4x pack built from GSQ stays valid only if its source sha is the IQ1_M one; otherwise rebuild with tools/baro and say so). Find: `rg -j16 -l -F -e OrcaSAQ -e OBLITERATED -e flash-next-gsq -e GSQ-RCO`.
