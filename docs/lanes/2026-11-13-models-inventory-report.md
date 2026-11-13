# ~/Models inventory, duplicates and ComfyUI usage (2026-11-13)

Read-only survey, no file was deleted, moved or rewritten, no GPU used. Brief: `docs/lanes/2026-11-13-models-inventory-brief.md`.

Disk: `/home` 1.5 TB, 99% used, 22 GB free. `~/Models` is 837 GB apparent but **622.7 GB real** (btrfs reflinks: bakes share their vendor file). `image.video` is 189.4 GB of it.

## How the numbers were made

- **exclusive GB** = bytes a delete of that file alone frees. **freed GB** = bytes a delete of a whole set frees (an extent counts only when every file that references it is in the set). Both come from btrfs physical extents (`filefrag -v`, 313 files of 20 MB or more under `~/Models` and `~/.cache/baro`), cross-checked against `btrfs filesystem du -s --raw` (image.video: 189.42 vs 189.48 GB).
- A reflinked twin shows exclusive about 0 GB: deleting one of a pair frees nothing, only deleting both does. `ls -l` sizes overstate every bake family.
- Logical bytes. The mount is `compress=zstd:3`; `compsize` needs root and was not run. GGUF and safetensors barely compress, so on-disk freed is at most these figures and close to them.
- Reflinks from outside `~/Models` and `~/.cache/baro` (lane worktrees, `.work/`) were not scanned. A file kept alive there frees less than listed.
- Status words. **USED**: a script, gate, tool config or output file loads it (path or basename hit in mojo-baro `bench/ tools/ face/ serve/ tests/ ncm/ lab/`, `~/iTools` code, other `~/Projects` code, or ComfyUI output metadata). **CITED**: named only in docs, Brain notes, lane reports or sha-cache rows. **UNUSED**: no hit anywhere. Hits inside `.work/`, `.archive/`, `logs/`, `library-old/`, generic listing scripts (`brain-laptop-sync.sh`, `draft-head-protocol.md`) do not count.
- Newest BARO bake per model = newest sha by mtime and commit date, checked against `~/.cache/baro/sha-cache.tsv` (which pack was built from which bake).
- Two downloads landed during the survey (`lfm3-1.7b-q4_k_m`, `spark-x2.6-4b-q4_k_m`, both 2026-11-13). Another session may be fetching into `~/Models`; sizes drift.

## Totals

| scenario | freed GB |
|---|---:|
| Only superseded BARO bakes (11 files, 82.0 GB apparent) | **0.11** |
| Only twin variants (fclones groups: qwen2.5-0.5b hf copy, flash-next BARO shards, RealESRGAN copy) | **0.07** |
| Both of the above | **0.18** |
| Superseded packs in `~/.cache/baro` (llama 919b9b.. and 6e62e4..) | 3.70 |
| All UNUSED (non-image.video 83.77 + image.video 71.23) | **155.00** |
| UNUSED + CITED-only (docs name it, nothing loads it) | **235.15** |
| UNUSED + CITED + vidcode-only z_image_turbo_bf16 | 247.46 |

The superseded bakes are worthless to delete: every bake is a btrfs reflink of its vendor file or of the previous bake, so the old shas hold 0.01 GB of private extents each. The space is in the vendor files and in unused models, not in bake versions.

`fclones group ~/Models` (min 50 MB) found 4 groups and claims 75.6 GB redundant. All four are already reflink-shared on btrfs, so the true redundancy is 0.07 GB (the RealESRGAN copy, which is a 67 MB plain duplicate).

## Candidates, sorted by freed GB

Each row is independent; rows overlap only where the label says so. Rebuild notes name the source.

| # | candidate | freed GB | status | risk of deleting, and how to rebuild |
|---:|---|---:|---|---|
| 1 | qwen3.8-flash-next UD-IQ1_M shards 00002 and 00003 (vendor; BARO shards are reflinks) | 74.53 | USED | bench/q4x-gate.sh default GGUF; docs/BIG-MODELS.md. Biggest single item. Delete only if the q4x gate moves to GSQ; HF unsloth UD-IQ1_M re-download about 75 GB. |
| 2 | qwen3.8-flash-next-gsq (Q2_0, 2 shards) | 66.42 | USED | Default GGUF of most q4x gates. Custom GSQ-RCO quant: only source copy, HF repo not recorded in this survey. |
| 3 | image.video/kandinsky-6.0-lite-distill-5s (whole dir) | 26.64 | UNUSED | Nothing loads it; re-download from HF (Kandinsky 6.0 lite distill 5s, ~27 GB). Licence unchecked in Brain index. |
| 4 | image.video/ltx-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors | 26.26 | UNUSED by owner workflows | the ComfyUI/models/text_encoders link would dangle; upstream stock LTX workflows in ltx-2.5/workflows/ name it. The int8-convrot encoder used by .work/ltx stays. Re-download from HF Lightricks/LTX-2.5. |
| 5 | qwen3.8-27b-turbo-fcf-mtp-q4_K_M (vendor + 2 bakes) | 18.52 | CITED | Lane docs (turbokernel, rowbw, turbobake). Live pack 57a7bf.. (17.2 GB) in ~/.cache/baro was built from it, so the gguf is needed only to rebuild the pack. Turbo Q4_K_M from HF. |
| 6 | qwen3.8-27b-obliterated-q4_K_M (vendor + bake + mmproj) | 17.75 | USED | tools/llama-chat-27b.sh, tools/deerflow-smoke.py, docs/models/qwen38-27b-obliterated.md. The mmproj (0.93) alone has no reader. |
| 7 | OrcaSAQ-2-Cyber-27B-Uncensored-GGUF (vendor + bake) | 15.69 | USED | bench/naked.sh, serve prompts, docs/models/orcasaq-2-27b.md. |
| 8 | image.video/comfy ACE-Step 1.5 stack (acestep + 2 qwen encoders + vae) | 14.70 | USED | 41 flac in ComfyUI/output/ace (2026-11-06), skill sellable-audio-assets. HF Comfy-Org repack. |
| 9 | image.video/comfy/wan2.2_i2v_low_noise_14B_fp8_scaled_lightx2v4step.safetensors | 14.30 | CITED | comfy-ops i2v default (baked low expert), ComfyUI .work/prs experiments. term2leg.py and comfy_pipeline.py use the non-baked low UNET kept in ComfyUI/models/diffusion_models instead. Rebuild: tools/amd-perf/bake_lora.py. |
| 10 | image.video/comfy/wan22RemixT2VI2V_i2vHighV30_lightx2v4step.safetensors | 14.29 | USED | Projects/gpu-media/vidcode/make/renderer/term2leg.py HIGH_UNET. Civitai Remix, baked LoRA: not rebuildable from HF. |
| 11 | image.video/comfy/z_image_turbo_bf16.safetensors | 12.31 | USED (vidcode only) | Projects/gpu-media/vidcode/make/renderer/comfy_pipeline.py hard-codes ZIMAGE_UNET = z_image_turbo_bf16 (174 PNGs, last 2026-09-18); comfy-ops says the bf16 is redundant. Point vidcode at the int8 first, then delete. |
| 12 | image.video/comfy/mopMixtureOfPerverts_v71Flexible (+_int8) | 11.69 | UNUSED | Last output 2026-07-17; no workflow, no script. Civitai download; Brain index: not for sale. int8 is a local quantisation of the bf16 (bf16 = 6.94, int8 = 4.75). |
| 13 | hunyuan3d-2 (two fp16 DiTs) | 9.86 | CITED | Symlinked from ComfyUI/models/checkpoints; no 3D output made with it (Brain index: no sale in the EU). HF tencent/Hunyuan3D-2. |
| 14 | tts/ (f5-tts-romanian 5.4 + chatterbox 3.2) | 8.58 | CITED | docs/PLATFORM-PLAN.md, MoneyPrinterTurbo config.example. f5-tts-romanian checkpoint (model_235000.pt) is a trained step checkpoint: may not be on HF. |
| 15 | vibethinker-1.5b (HF safetensors) + bf16 gguf + q4_0-pure | 8.12 | CITED | Only sources of the packs; rebuild needs the bf16 gguf or the HF dir (one of them). HF WeiboAI/VibeThinker-1.5B. |
| 16 | ncm-checkpoints except sudoku/easy | 8.12 | CITED | ncm_pack.py lists them, check.sh uses sudoku/easy only. Refetch with ncm-checkpoints/.fetch.sh. |
| 17 | trm-arc2-8gpu + styledem + IFM + lgm | 7.96 | UNUSED | No reader of any of them. |
| 18 | phi-4-mini-reasoning | 7.67 | UNUSED | No reader. HF microsoft/Phi-4-mini-reasoning. |
| 19 | Ternary-Bonsai-2-27B-Uncensored-Heretic-PQ2_0.gguf | 7.21 | UNUSED | Only Brain brief briefs/2026-09-05-tern-E-real.md names it. Third-party gguf. |
| 20 | deep-live-cam-models (47 onnx/pth) | 6.84 | CITED | Real-ESRGAN-x4plus.onnx (stock-prep), RealESRGAN_x4plus.pth (stock-kit), sam2 + buffalo_l (pixelshaper) are read; the rest is not. |
| 21 | deepseek-ocr-2 | 6.78 | UNUSED | No reader (OCR runs through img-read/tesseract). HF deepseek-ai/DeepSeek-OCR-2. |
| 22 | xlam-2-3b-fc-r | 6.17 | UNUSED | No reader. HF Salesforce/xLAM-2-3b-fc-r. |
| 23 | qwythos-9b-claude-mythos-5-1m-mtp-q4_k_m | 5.89 | UNUSED | Zero hits. The Q4_0-pure sibling in the bf16 dir is the one gates load. |
| 24 | snowflake-arctic-embed-m-v2.0 (safetensors + 7 onnx variants) | 5.50 | UNUSED | No loader; brain-ask uses bge-small + ms-marco. HF Snowflake/snowflake-arctic-embed-m-v2.0. |
| 25 | modernbert-embed-base + nomic-embed-text-v1.5 + jina-embeddings-v3 | 5.44 | UNUSED | No loader. |
| 26 | nemotron-3.5-bolt-e4b-q4_k_xl | 5.06 | UNUSED | No reader. HF unsloth. |
| 27 | bitnet-b1.58-2b-4t-bf16 | 4.83 | UNUSED | No reader. HF microsoft/bitnet-b1.58-2B-4T-bf16. |
| 28 | misc small: RVC zips, all-MiniLM, bert-turkish, TriLM, gemma-3-270m weights, distilgpt2, bge-m3-latest, llama.cpp-prism, vocos | 4.81 | UNUSED | No reader (bge-m3-latest: a gguf-kv test case; gemma-3-270m tokenizer.json is used by run-tests.sh and stays). |
| 29 | spark-x2.6-4b-q4_k_m + lfm3-1.7b-q4_k_m (downloads of 2026-11-13) | 4.64 | NEW | Fetched today, no consumer yet; may belong to an in-flight session. Leave. |
| 30 | bge-m3-safetensors (pytorch_model.bin + onnx data) | 4.54 | UNUSED | No loader. bge-m3-latest gguf (1.2) is a separate, test-only entry. |
| 31 | siglip-so400m-patch14-384 | 3.51 | UNUSED | Only the archived photo-tag.py names it. |
| 32 | dinov3 | 3.36 | CITED | gpu-media/Blender/kit/asset_dino.py path; no run evidence. Meta licence, gated HF. |
| 33 | MiniCPM-V-4.7-1B | 3.11 | UNUSED | No reader. |
| 34 | minicpm5-2b/MiniCPM5-2B-Q8_0.gguf | 2.68 | UNUSED | Only source of minicpm5-2b/pack (kept). Delete only together with the pack or if the pack is never rebuilt. |
| 35 | image.video/comfy/snofs_krea_v1 + klein_snofs_v1_1 | 2.65 | UNUSED | snofs_krea was a Krea 2 LoRA (Krea 2 removed 2026-11-04); only a vidcode log of 2026-08-27 names them. Civitai. |
| 36 | image.video/comfy/wan2.2_i2v_lightx2v_4steps LoRAs (high + low) | 2.45 | CITED | Needed only for the non-baked Wan path (comfy_pipeline.py WAN_LORA high; term2leg.py comment). Baked unets make them redundant. HF Comfy-Org / lightx2v. |
| 37 | whisper-large-v3-turbo-hf | 1.62 | UNUSED | No reader; the ggml q5_0 in whisper/ and the ct2 copy are the used ones. |
| 38 | image.video/ltx-2.5/vae/ltx-2.5-video-vae-bf16.safetensors | 1.47 | UNUSED by owner workflows | .work/ltx/*.json use video-vae-conv-bf16; plain vae only in stock workflows and ComfyUI/models/vae link. |
| 39 | image.video/ltx-2.5/.cache (two .incomplete downloads) | 1.10 | UNUSED | Partial HF downloads, never finished. |
| 40 | qwen2.5-0.5b-instruct-import/qwen2.5-0.5b-instruct-bf16.gguf | 0.99 | UNUSED | No hit; regenerable from the hf copy. |
| 41 | image.video/comfy NSFW Wan/SDXL LoRAs (MysticXXX, sloppy, jfj, Stable Yogi) | 0.98 | UNUSED | Last output 2026-07-17 (MysticXXX). Civitai; Brain index: not for sale. |
| 42 | image.video/idol-test (bear .ply and png) | 0.43 | UNUSED | Test outputs of 2026-11-03; receipt in Brain/ComfyUI-v0.43.0/2026-11-03-idol-bear-receipt.md. |
## Families (every file of 80 MB or more outside image.video)

Exclusive GB is per file alone: a reflinked bake or vendor twin reads about 0 even though deleting both frees the full size. Directory totals are in the candidates table.

### Qwen3.8 Flash-Next (80B-class MoE, q4x path)

| path | apparent GB | exclusive GB | role | status | evidence |
|---|---:|---:|---|---|---|
| `qwen3.8-flash-next-gsq/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf` | 37.62 | 37.62 | variant (GSQ Q2_0) | USED | bench/q4x-vec-gate.sh, amdgpu-tune-ab.sh, latent-bridge-*.sh default GGUF; docs/KERNELS.md |
| `qwen3.8-flash-next-gsq/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf` | 28.80 | 28.80 | variant (GSQ Q2_0) | USED | bench/q4x-vec-gate.sh, amdgpu-tune-ab.sh, latent-bridge-*.sh default GGUF; docs/KERNELS.md |
| `qwen3.8-flash-next/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf` | 4.14 | 4.14 | variant (MTP Q8_0 draft) | USED | bench/q4x-mtp-ab.sh, q4x-rows-trace.sh MTP_PATH default |
| `qwen3.8-flash-next/UD-IQ1_M/Qwen3.8-Flash-Next-UD-IQ1_M-00002-of-00003.gguf` | 49.99 | 0.00 | variant (UD-IQ1_M, vendor) | USED | bench/q4x-gate.sh Q4X_GGUF default; iTools/llm/gguf-experts/test.sh |
| `qwen3.8-flash-next/UD-IQ1_M/Qwen3.8-Flash-Next-UD-IQ1_M-00003-of-00003.gguf` | 24.54 | 0.00 | variant (UD-IQ1_M, vendor) | USED | bench/q4x-gate.sh Q4X_GGUF default; iTools/llm/gguf-experts/test.sh |
| `qwen3.8-flash-next/UD-IQ1_M/Qwen3.8-Flash-Next-UD-IQ1_M-BARO-d37ec15-00002-of-00003.gguf` | 49.99 | 0.00 | bake d37ec15 (reflink of vendor shards) | CITED | no script loads the BARO shards; bench/q4x-gate.sh loads vendor shard 00001; docs/BIG-MODELS.md |
| `qwen3.8-flash-next/UD-IQ1_M/Qwen3.8-Flash-Next-UD-IQ1_M-BARO-d37ec15-00003-of-00003.gguf` | 24.54 | 0.00 | bake d37ec15 (reflink of vendor shards) | CITED | no script loads the BARO shards; bench/q4x-gate.sh loads vendor shard 00001; docs/BIG-MODELS.md |

### Qwen3.8 27B (Turbo FCFusion, OBLITERATED)

| path | apparent GB | exclusive GB | role | status | evidence |
|---|---:|---:|---|---|---|
| `qwen3.8-27b-obliterated-q4_K_M/Qwen3.8-27B-OBLITERATED.Q4_K_M-BARO-36e440b.gguf` | 16.81 | 0.01 | bake 36e440b | CITED | docs/models/qwen38-27b-obliterated.md, docs/ARTIFACTS.md (rebuild source of the verify pack) |
| `qwen3.8-27b-obliterated-q4_K_M/Qwen3.8-27B-OBLITERATED.Q4_K_M.gguf` | 16.81 | 0.01 | source (vendor q4_K_M) | USED | tools/llama-chat-27b.sh default model; iTools/llm/debug-mine/debug-mine-run |
| `qwen3.8-27b-obliterated-q4_K_M/Qwen3.8-27B-OBLITERATED.mmproj-f16.gguf` | 0.93 | 0.93 | vision encoder | UNUSED | no hit in code or docs |
| `qwen3.8-27b-turbo-fcf-mtp-q4_K_M/Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M-BARO-a4eb549.gguf` | 18.50 | 0.01 | bake (newest by commit date) | CITED | docs/lanes/turbokernel-report.md; no code path; its pack is not in ~/.cache/baro |
| `qwen3.8-27b-turbo-fcf-mtp-q4_K_M/Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M-BARO-c71dbdc.gguf` | 18.50 | 0.01 | bake (older; the cached pack 57a7bf.. was built from it) | CITED | sha-cache.tsv row; pack qwen35-eec459.. live |
| `qwen3.8-27b-turbo-fcf-mtp-q4_K_M/Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf` | 18.50 | 0.01 | source (vendor q4_K_M) | CITED | docs/lanes/turbobake-report.md, rowbw-report.md |

### RegesCore-1.0-35B (qwen35moe)

| path | apparent GB | exclusive GB | role | status | evidence |
|---|---:|---:|---|---|---|
| `RegesCore-1.0-35/RegesCore-1.0-35B-UD-Q4_K_S-BARO-3c6c25c.gguf` | 20.89 | 0.01 | bake (superseded by cf78732) | CITED | bench/3-prereg.md, docs/ARTIFACTS.md name it (moe-ngram pack was built from it) |
| `RegesCore-1.0-35/RegesCore-1.0-35B-UD-Q4_K_S-BARO-cf78732.gguf` | 20.89 | 0.01 | bake (newest) | USED | sha-cache/current; bench/moe-*.sh, docs/BASELINE.md resolve the newest bake |
| `RegesCore-1.0-35/mmproj-F16.gguf` | 0.90 | 0.90 | vision encoder | USED | bench/vision-smoke.sh, vision-ref.sh, iTools with-llama regescore-judge.sh |

### OrcaSAQ-2 27B

| path | apparent GB | exclusive GB | role | status | evidence |
|---|---:|---:|---|---|---|
| `OrcaSAQ-2-Cyber-27B-Uncensored-GGUF/OrcaSAQ-2-27B-Uncensored-BARO-e658f85.gguf` | 15.68 | 0.01 | bake e658f85 | USED | bench/naked.sh, serve/prompts/orcasaq-system.md, docs/models/orcasaq-2-27b.md |
| `OrcaSAQ-2-Cyber-27B-Uncensored-GGUF/OrcaSAQ-2-27B-Uncensored.gguf` | 15.68 | 0.01 | source (vendor) | CITED | docs/models/orcasaq-2-27b.md, docs/CAPABILITIES.md |

### Qwythos 9B (bf16 source, v2, Q4_K_M)

| path | apparent GB | exclusive GB | role | status | evidence |
|---|---:|---:|---|---|---|
| `qwythos-9b-claude-mythos-5-1m-mtp-bf16/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_0-pure.gguf` | 5.19 | 5.19 | variant (Q4_0 pure) | USED | bench/e14-handoff.mojo, e13_engine_dump.mojo, a2-bar-llama.sh |
| `qwythos-9b-claude-mythos-5-1m-mtp-bf16/hf/model-00001-of-00004.safetensors` | 4.97 | 4.97 | source (bf16 HF safetensors) | USED | tools/mtp_train.py, mtp_head.py, amdhq e13_train.py: --hf-dir default |
| `qwythos-9b-claude-mythos-5-1m-mtp-bf16/hf/model-00002-of-00004.safetensors` | 4.99 | 4.99 | source (bf16 HF safetensors) | USED | tools/mtp_train.py, mtp_head.py, amdhq e13_train.py: --hf-dir default |
| `qwythos-9b-claude-mythos-5-1m-mtp-bf16/hf/model-00003-of-00004.safetensors` | 4.95 | 4.95 | source (bf16 HF safetensors) | USED | tools/mtp_train.py, mtp_head.py, amdhq e13_train.py: --hf-dir default |
| `qwythos-9b-claude-mythos-5-1m-mtp-bf16/hf/model-00004-of-00004.safetensors` | 2.99 | 2.99 | source (bf16 HF safetensors) | USED | tools/mtp_train.py, mtp_head.py, amdhq e13_train.py: --hf-dir default |
| `qwythos-9b-claude-mythos-5-1m-mtp-q4_k_m/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_K_M.gguf` | 5.89 | 5.89 | variant (Q4_K_M) | UNUSED | zero hits in mojo-baro, docs, Brain/mojo, iTools, Projects |
| `qwythos-9b-v2-mtp-q6_k/Qwythos-9B-v2-MTP-Q4_0-BARO-3c6c25c.gguf` | 5.19 | 0.01 | bake (superseded) | UNUSED | no script names this sha |
| `qwythos-9b-v2-mtp-q6_k/Qwythos-9B-v2-MTP-Q4_0-BARO-eeb082e.gguf` | 5.19 | 0.01 | bake (newest) | USED | bench/qwythos-v2-run.sh; pack 5657ea.. built from it |
| `qwythos-9b-v2-mtp-q6_k/mmproj-Qwythos-9B-v2-BF16.gguf` | 0.92 | 0.92 | vision encoder | CITED | docs only |

### Ornith 1.5 9B

| path | apparent GB | exclusive GB | role | status | evidence |
|---|---:|---:|---|---|---|
| `ornith-1.5-9b-q4_K_M/Ornith-1.5-9B-Q4_K_M-BARO-3c6c25c.gguf` | 5.78 | 0.01 | bake (superseded) | UNUSED | no script names this sha |
| `ornith-1.5-9b-q4_K_M/Ornith-1.5-9B-Q4_K_M-BARO-cf78732.gguf` | 5.78 | 0.01 | bake (newest) | USED | bench/ornith-run.sh, p1-bridge-gate.sh; pack e86ba3.. built from it |
| `ornith-1.5-9b-q4_K_M/Ornith-1.5-9B-Q4_K_M-stock.gguf` | 5.78 | 5.78 | source (stock llama.cpp q4_K_M) | USED | bench/ornith-run.sh, ailab/llama.sh (llama.cpp reference arm) |
| `ornith-1.5-9b-q4_K_M/mmproj-Ornith-1.5-9B-BF16.gguf` | 0.92 | 0.92 | vision encoder | USED | iTools/llm/with-llama/profiles/ornith-judge.sh, mmproj-info |

### Granite 4.2 3B

| path | apparent GB | exclusive GB | role | status | evidence |
|---|---:|---:|---|---|---|
| `granite-4.2-3b-bf16/granite-4.2-3b-BF16-BARO-1d6e2dc.gguf` | 7.32 | 0.00 | bake (newest) | USED | bench/quality-models.json, semantic-latent-*.sh; pack 491c4b.. built from it |
| `granite-4.2-3b-bf16/granite-4.2-3b-BF16-BARO-3c6c25c.gguf` | 7.32 | 0.00 | bake (superseded) | UNUSED | no script names this sha |
| `granite-4.2-3b-bf16/granite-4.2-3b-BF16-BARO-cf78732.gguf` | 7.32 | 0.00 | bake (superseded) | UNUSED | no script names this sha |

### Dense bench set (Qwen2.5, Llama 3.2 1B, Spark X2.5, MiniCPM5, VibeThinker, Lily)

| path | apparent GB | exclusive GB | role | status | evidence |
|---|---:|---:|---|---|---|
| `lily-cybersecurity-7b-v0.2-q6_k/lily-cybersecurity-7b-v0.2-IQ1_S-Q4_0mix.gguf` | 1.85 | 1.84 | variant (IQ1_S) | CITED | bench/quality-models.json names a q6_k BARO-04867e2 file that is not on disk |
| `lily-cybersecurity-7b-v0.2-q6_k/lily-cybersecurity-7b-v0.2-IQ1_S.gguf` | 1.61 | 1.61 | variant (IQ1_S) | CITED | bench/quality-models.json names a q6_k BARO-04867e2 file that is not on disk |
| `llama-3.2-1b-instruct-q4_K_M/Llama-3.2-1B-Instruct-Q4_K_M-BARO-1d6e2dc.gguf` | 0.81 | 0.01 | bake (newest) | USED | pack e8e60e.. built from it |
| `llama-3.2-1b-instruct-q4_K_M/Llama-3.2-1B-Instruct-Q4_K_M-BARO-3c6c25c.gguf` | 0.81 | 0.01 | bake (superseded) | CITED | tools/test-gguf-check.sh names 3c6c25c |
| `llama-3.2-1b-instruct-q4_K_M/Llama-3.2-1B-Instruct-Q4_K_M-BARO-cf78732.gguf` | 0.81 | 0.01 | bake (superseded) | CITED | tools/test-gguf-check.sh names 3c6c25c |
| `llama-3.2-1b-instruct-q4_K_M/Llama-3.2-1B-Instruct-Q4_K_M-BARO-cf78732fix.gguf` | 0.81 | 0.01 | bake (pack 919b9b.. built from it) | CITED | sha-cache.tsv row only |
| `llama-3.2-1b-instruct-q4_K_M/Llama-3.2-1B-Instruct-Q4_K_M.gguf` | 0.81 | 0.01 | source (vendor) | USED | lab/wavegroup/tools/stint.sh, llama_self.sh, gate.sh; bench/p1-bridge-gate.sh |
| `minicpm5-2b/MiniCPM5-2B-Q4_K_M.gguf` | 1.56 | 0.00 | source (vendor q4_K_M) | USED | bench/sparkq4k-ab.sh, sparkf32-ab.sh |
| `minicpm5-2b/MiniCPM5-2B-Q8_0.gguf` | 2.68 | 2.68 | source of the q8 pack | UNUSED | no script names the Q8_0 gguf; the pack is built from it (rebuild needs it) |
| `minicpm5-2b/minicpm5-2b-q4_K_M-BARO-b2d0d6e.gguf` | 1.56 | 0.01 | bake | CITED | docs/ARTIFACTS.md; reflink twin of Q4_K_M |
| `minicpm5-2b/pack/pack.bin` | 3.46 | 3.46 | pack (q8) | USED | bench/sparkq4k-ab.sh pack(); run-tests.sh; face/presets_test.mojo; serve/harness.mojo |
| `qwen2.5-0.5b-instruct-hf/model.safetensors` | 0.99 | 0.00 | source (HF safetensors, byte twin) | CITED | tools/jlens_multihop.py reads the -hf copy; import/hf is a reflink twin |
| `qwen2.5-0.5b-instruct-import/hf/model.safetensors` | 0.99 | 0.00 | source (HF safetensors, byte twin) | CITED | tools/jlens_multihop.py reads the -hf copy; import/hf is a reflink twin |
| `qwen2.5-0.5b-instruct-import/qwen2.5-0.5b-instruct-bf16.gguf` | 0.99 | 0.99 | variant (bf16) | UNUSED | no hit |
| `qwen2.5-0.5b-instruct-import/qwen2.5-0.5b-instruct-q4_0-pure.gguf` | 0.28 | 0.28 | variant | USED | tools/spirv-probe/time.sh |
| `qwen2.5-0.5b-instruct-import/qwen2.5-0.5b-instruct-q4_K_M.gguf` | 0.40 | 0.40 | variant | USED | iTools/llm/latent-wa/tests/test.sh, hidden-probe/tests/smoke.sh |
| `qwen2.5-7b-instruct-import/qwen2.5-7b-instruct-q4_K_M-BARO-1d6e2dc.gguf` | 4.68 | 0.01 | bake (newest) | USED | bench/quick-speed.sh, sparkq4k-ab.sh; pack 47c8e9.. built from it |
| `qwen2.5-7b-instruct-import/qwen2.5-7b-instruct-q4_K_M-BARO-3c6c25c.gguf` | 4.68 | 0.01 | bake (superseded) | CITED | bench/sparkq4k-protocol.md names 3c6c25c |
| `qwen2.5-7b-instruct-import/qwen2.5-7b-instruct-q4_K_M-BARO-cf78732.gguf` | 4.68 | 0.01 | bake (superseded) | CITED | bench/sparkq4k-protocol.md names 3c6c25c |
| `spark-x2.5-4b/Spark-X2.5-4B-Q4_K_M-BARO-1d6e2dc.gguf` | 2.60 | 0.01 | bake (newest) | USED | bench/quality-models.json, minja corpus; pack 26d23a.. built from it |
| `spark-x2.5-4b/Spark-X2.5-4B-Q4_K_M-BARO-3c6c25c.gguf` | 2.60 | 0.01 | bake (superseded) | UNUSED | no script names this sha |
| `spark-x2.5-4b/Spark-X2.5-4B-Q4_K_M-BARO-cf78732.gguf` | 2.60 | 0.01 | bake (superseded) | UNUSED | no script names this sha |
| `spark-x2.5-4b/Spark-X2.5-4B-Q4_K_M.gguf` | 2.60 | 0.01 | source (vendor) | CITED | docs/ARTIFACTS.md, release notes; reflink twin of the bakes |
| `vibethinker-1.5b-gguf/bf16/pack/pack.bin` | 2.57 | 2.57 | pack | USED | run-tests.sh, serve/harness.mojo, serve/spark.mojo, tools/bytes-per-token.mojo |
| `vibethinker-1.5b-gguf/pack/pack.bin` | 2.57 | 2.57 | pack | USED | run-tests.sh, serve/harness.mojo, serve/spark.mojo, tools/bytes-per-token.mojo |
| `vibethinker-1.5b-gguf/vibethinker-1.5b-Q4_K_M.gguf` | 1.12 | 1.12 | variant (Q4_K_M) | USED | bench/sparkq4k-ab.sh, sparkf32-ab.sh, sparkq4k-truth.sh |
| `vibethinker-1.5b-gguf/vibethinker-1.5b-bf16.gguf` | 3.56 | 3.56 | variant (bf16 gguf, pack source) | CITED | docs only (12 files); the pack is built from it |
| `vibethinker-1.5b-gguf/vibethinker-1.5b-q4_0-pure.gguf` | 1.01 | 1.01 | variant (q4_0 pure) | CITED | 1 doc |
| `vibethinker-1.5b/model.safetensors` | 3.55 | 3.55 | source (HF safetensors) | CITED | bench/import3-protocol.md; rebuild source for the ggufs |

### Other LLMs (no bench set)

| path | apparent GB | exclusive GB | role | status | evidence |
|---|---:|---:|---|---|---|
| `Ternary-Bonsai-2-27B-Uncensored-Heretic-PQ2_0.gguf` | 7.21 | 7.21 | model | UNUSED | zero hits in mojo-baro code and docs, iTools, Projects scripts |
| `TriLM-1.5B-TQ2_0.gguf` | 0.48 | 0.48 | model | UNUSED | zero hits in mojo-baro code and docs, iTools, Projects scripts |
| `bitnet-b1.58-2b-4t-bf16/model.safetensors` | 4.83 | 4.83 | model | UNUSED | zero hits in mojo-baro code and docs, iTools, Projects scripts |
| `distilgpt2/model.safetensors` | 0.35 | 0.35 | tiny HF model | CITED | tokenizer.json read by run-tests.sh; safetensors unused |
| `gemma-3-270m/model.safetensors` | 0.54 | 0.54 | tiny HF model | CITED | tokenizer.json read by run-tests.sh; safetensors unused |
| `lfm3-1.7b-q4_k_m/LFM3-1.7B-Q4_K_M.gguf` | 1.10 | 1.10 | variant (Q4_K_M), download dated 2026-11-13 | UNUSED | no consumer yet; another session may be fetching it |
| `nemotron-3.5-bolt-e4b-q4_k_xl/NVIDIA-Nemotron-3.5-Bolt-E4B-UD-Q4_K_XL.gguf` | 5.06 | 5.06 | model | UNUSED | zero hits in mojo-baro code and docs, iTools, Projects scripts |
| `phi-4-mini-reasoning/model-00001-of-00002.safetensors` | 4.90 | 4.90 | model | UNUSED | zero hits in mojo-baro code and docs, iTools, Projects scripts |
| `phi-4-mini-reasoning/model-00002-of-00002.safetensors` | 2.77 | 2.77 | model | UNUSED | zero hits in mojo-baro code and docs, iTools, Projects scripts |
| `spark-x2.6-4b-q4_k_m/Spark-X2.6-4B-Q4_K_M.gguf` | 3.54 | 3.54 | variant (Q4_K_M), download dated 2026-11-13 | UNUSED | no consumer yet; another session may be fetching it |
| `xlam-2-3b-fc-r/model-00001-of-00002.safetensors` | 4.96 | 4.96 | model | UNUSED | zero hits in mojo-baro code and docs, iTools, Projects scripts |
| `xlam-2-3b-fc-r/model-00002-of-00002.safetensors` | 1.21 | 1.21 | model | UNUSED | zero hits in mojo-baro code and docs, iTools, Projects scripts |

### Embedding and rerank models

| path | apparent GB | exclusive GB | role | status | evidence |
|---|---:|---:|---|---|---|
| `all-MiniLM-L6-v2/model.safetensors` | 0.09 | 0.09 | embedding/vision model | UNUSED | only generic listings (brain-laptop-sync.sh, draft-head-protocol.md); no loader |
| `all-MiniLM-L6-v2/onnx/model.onnx` | 0.09 | 0.09 | embedding/vision model | UNUSED | only generic listings (brain-laptop-sync.sh, draft-head-protocol.md); no loader |
| `all-MiniLM-L6-v2/onnx/model_O1.onnx` | 0.09 | 0.09 | embedding/vision model | UNUSED | only generic listings (brain-laptop-sync.sh, draft-head-protocol.md); no loader |
| `all-MiniLM-L6-v2/onnx/model_O2.onnx` | 0.09 | 0.09 | embedding/vision model | UNUSED | only generic listings (brain-laptop-sync.sh, draft-head-protocol.md); no loader |
| `all-MiniLM-L6-v2/onnx/model_O3.onnx` | 0.09 | 0.09 | embedding/vision model | UNUSED | only generic listings (brain-laptop-sync.sh, draft-head-protocol.md); no loader |
| `all-MiniLM-L6-v2/openvino/openvino_model.bin` | 0.09 | 0.09 | embedding/vision model | UNUSED | only generic listings (brain-laptop-sync.sh, draft-head-protocol.md); no loader |
| `all-MiniLM-L6-v2/pytorch_model.bin` | 0.09 | 0.09 | embedding/vision model | UNUSED | only generic listings (brain-laptop-sync.sh, draft-head-protocol.md); no loader |
| `bert-base-turkish-nli-stsb/pytorch_model.bin` | 0.44 | 0.44 | model | UNUSED | zero hits in mojo-baro code and docs, iTools, Projects scripts |
| `bge-m3-latest/bge-m3-latest.gguf` | 1.16 | 1.16 | embedding gguf | CITED | iTools/llm/gguf-kv/tests/cases.txt only |
| `bge-m3-safetensors/onnx/model.onnx_data` | 2.27 | 2.27 | embedding/vision model | UNUSED | only generic listings (brain-laptop-sync.sh, draft-head-protocol.md); no loader |
| `bge-m3-safetensors/pytorch_model.bin` | 2.27 | 2.27 | embedding/vision model | UNUSED | only generic listings (brain-laptop-sync.sh, draft-head-protocol.md); no loader |
| `bge-small-en-v1.5/model.safetensors` | 0.13 | 0.13 | embedding/rerank model | USED | iTools/brain/brain-ask/embed.py/.mojo, rerank.mojo |
| `bge-small-en-v1.5/onnx/model.onnx` | 0.13 | 0.13 | embedding/rerank model | USED | iTools/brain/brain-ask/embed.py/.mojo, rerank.mojo |
| `bge-small-en-v1.5/pytorch_model.bin` | 0.13 | 0.13 | embedding/rerank model | USED | iTools/brain/brain-ask/embed.py/.mojo, rerank.mojo |
| `jina-embeddings-v3/model.safetensors` | 1.14 | 1.14 | embedding/vision model | UNUSED | only generic listings (brain-laptop-sync.sh, draft-head-protocol.md); no loader |
| `modernbert-embed-base/model.safetensors` | 0.60 | 0.60 | embedding/vision model | UNUSED | only generic listings (brain-laptop-sync.sh, draft-head-protocol.md); no loader |
| `modernbert-embed-base/onnx/model.onnx` | 0.60 | 0.60 | embedding/vision model | UNUSED | only generic listings (brain-laptop-sync.sh, draft-head-protocol.md); no loader |
| `modernbert-embed-base/onnx/model_bnb4.onnx` | 0.22 | 0.22 | embedding/vision model | UNUSED | only generic listings (brain-laptop-sync.sh, draft-head-protocol.md); no loader |
| `modernbert-embed-base/onnx/model_fp16.onnx` | 0.30 | 0.30 | embedding/vision model | UNUSED | only generic listings (brain-laptop-sync.sh, draft-head-protocol.md); no loader |
| `modernbert-embed-base/onnx/model_int8.onnx` | 0.15 | 0.15 | embedding/vision model | UNUSED | only generic listings (brain-laptop-sync.sh, draft-head-protocol.md); no loader |
| `modernbert-embed-base/onnx/model_q4.onnx` | 0.22 | 0.22 | embedding/vision model | UNUSED | only generic listings (brain-laptop-sync.sh, draft-head-protocol.md); no loader |
| `modernbert-embed-base/onnx/model_q4f16.onnx` | 0.14 | 0.14 | embedding/vision model | UNUSED | only generic listings (brain-laptop-sync.sh, draft-head-protocol.md); no loader |
| `ms-marco-MiniLM-L-6-v2/model.safetensors` | 0.09 | 0.09 | embedding/rerank model | USED | iTools/brain/brain-ask/embed.py/.mojo, rerank.mojo |
| `nomic-embed-text-v1.5/model.safetensors` | 0.55 | 0.55 | embedding/vision model | UNUSED | only generic listings (brain-laptop-sync.sh, draft-head-protocol.md); no loader |
| `nomic-embed-text-v1.5/onnx/model.onnx` | 0.55 | 0.55 | embedding/vision model | UNUSED | only generic listings (brain-laptop-sync.sh, draft-head-protocol.md); no loader |
| `nomic-embed-text-v1.5/onnx/model_bnb4.onnx` | 0.16 | 0.16 | embedding/vision model | UNUSED | only generic listings (brain-laptop-sync.sh, draft-head-protocol.md); no loader |
| `nomic-embed-text-v1.5/onnx/model_fp16.onnx` | 0.27 | 0.27 | embedding/vision model | UNUSED | only generic listings (brain-laptop-sync.sh, draft-head-protocol.md); no loader |
| `nomic-embed-text-v1.5/onnx/model_int8.onnx` | 0.14 | 0.14 | embedding/vision model | UNUSED | only generic listings (brain-laptop-sync.sh, draft-head-protocol.md); no loader |
| `nomic-embed-text-v1.5/onnx/model_q4.onnx` | 0.17 | 0.17 | embedding/vision model | UNUSED | only generic listings (brain-laptop-sync.sh, draft-head-protocol.md); no loader |
| `nomic-embed-text-v1.5/onnx/model_q4f16.onnx` | 0.11 | 0.11 | embedding/vision model | UNUSED | only generic listings (brain-laptop-sync.sh, draft-head-protocol.md); no loader |
| `nomic-embed-text-v1.5/onnx/model_uint8.onnx` | 0.14 | 0.14 | embedding/vision model | UNUSED | only generic listings (brain-laptop-sync.sh, draft-head-protocol.md); no loader |
| `snowflake-arctic-embed-m-v2.0/model.safetensors` | 1.22 | 1.22 | embedding/vision model | UNUSED | only generic listings (brain-laptop-sync.sh, draft-head-protocol.md); no loader |
| `snowflake-arctic-embed-m-v2.0/onnx/model.onnx` | 1.23 | 1.23 | embedding/vision model | UNUSED | only generic listings (brain-laptop-sync.sh, draft-head-protocol.md); no loader |
| `snowflake-arctic-embed-m-v2.0/onnx/model_bnb4.onnx` | 0.84 | 0.84 | embedding/vision model | UNUSED | only generic listings (brain-laptop-sync.sh, draft-head-protocol.md); no loader |
| `snowflake-arctic-embed-m-v2.0/onnx/model_fp16.onnx` | 0.61 | 0.61 | embedding/vision model | UNUSED | only generic listings (brain-laptop-sync.sh, draft-head-protocol.md); no loader |
| `snowflake-arctic-embed-m-v2.0/onnx/model_int8.onnx` | 0.31 | 0.31 | embedding/vision model | UNUSED | only generic listings (brain-laptop-sync.sh, draft-head-protocol.md); no loader |
| `snowflake-arctic-embed-m-v2.0/onnx/model_q4.onnx` | 0.84 | 0.84 | embedding/vision model | UNUSED | only generic listings (brain-laptop-sync.sh, draft-head-protocol.md); no loader |
| `snowflake-arctic-embed-m-v2.0/onnx/model_q4f16.onnx` | 0.45 | 0.45 | embedding/vision model | UNUSED | only generic listings (brain-laptop-sync.sh, draft-head-protocol.md); no loader |

### Vision, OCR, speech, TTS, 3D, face

| path | apparent GB | exclusive GB | role | status | evidence |
|---|---:|---:|---|---|---|
| `IFM/K2-Type-0.9B/model.safetensors` | 2.16 | 2.16 | model | UNUSED | zero hits in mojo-baro code and docs, iTools, Projects scripts |
| `MiniCPM-V-4.7-1B/model.safetensors` | 3.11 | 3.11 | model | UNUSED | zero hits in mojo-baro code and docs, iTools, Projects scripts |
| `deep-live-cam-models/4xFFHQDAT.pth` | 0.15 | 0.15 | face/upscale onnx set | CITED | iTools/media/stock-prep, design/stock-kit, pixelshaper read a few files; most files unreferenced |
| `deep-live-cam-models/GFPGANv1.4.onnx` | 0.34 | 0.34 | face/upscale onnx set | CITED | iTools/media/stock-prep, design/stock-kit, pixelshaper read a few files; most files unreferenced |
| `deep-live-cam-models/GFPGANv1.4.pth` | 0.35 | 0.35 | face/upscale onnx set | CITED | iTools/media/stock-prep, design/stock-kit, pixelshaper read a few files; most files unreferenced |
| `deep-live-cam-models/GPEN-BFR-512.onnx` | 0.28 | 0.28 | face/upscale onnx set | CITED | iTools/media/stock-prep, design/stock-kit, pixelshaper read a few files; most files unreferenced |
| `deep-live-cam-models/GPEN-BFR-512.pth` | 0.28 | 0.28 | face/upscale onnx set | CITED | iTools/media/stock-prep, design/stock-kit, pixelshaper read a few files; most files unreferenced |
| `deep-live-cam-models/buffalo_l/buffalo_l/1k3d68.onnx` | 0.14 | 0.14 | face/upscale onnx set | CITED | iTools/media/stock-prep, design/stock-kit, pixelshaper read a few files; most files unreferenced |
| `deep-live-cam-models/buffalo_l/buffalo_l/w600k_r50.onnx` | 0.17 | 0.17 | face/upscale onnx set | CITED | iTools/media/stock-prep, design/stock-kit, pixelshaper read a few files; most files unreferenced |
| `deep-live-cam-models/codeformer.onnx` | 0.38 | 0.38 | face/upscale onnx set | CITED | iTools/media/stock-prep, design/stock-kit, pixelshaper read a few files; most files unreferenced |
| `deep-live-cam-models/dw-ll_ucoco_384.onnx` | 0.13 | 0.13 | face/upscale onnx set | CITED | iTools/media/stock-prep, design/stock-kit, pixelshaper read a few files; most files unreferenced |
| `deep-live-cam-models/gfpgan-1024.onnx` | 0.37 | 0.37 | face/upscale onnx set | CITED | iTools/media/stock-prep, design/stock-kit, pixelshaper read a few files; most files unreferenced |
| `deep-live-cam-models/hyperswap_1c_256.onnx` | 0.40 | 0.40 | face/upscale onnx set | CITED | iTools/media/stock-prep, design/stock-kit, pixelshaper read a few files; most files unreferenced |
| `deep-live-cam-models/inswapper_128.onnx` | 0.55 | 0.55 | face/upscale onnx set | CITED | iTools/media/stock-prep, design/stock-kit, pixelshaper read a few files; most files unreferenced |
| `deep-live-cam-models/inswapper_128_fp16.onnx` | 0.28 | 0.28 | face/upscale onnx set | CITED | iTools/media/stock-prep, design/stock-kit, pixelshaper read a few files; most files unreferenced |
| `deep-live-cam-models/liveportrait/live_portrait_generator.onnx` | 0.22 | 0.22 | face/upscale onnx set | CITED | iTools/media/stock-prep, design/stock-kit, pixelshaper read a few files; most files unreferenced |
| `deep-live-cam-models/liveportrait/live_portrait_motion_extractor.onnx` | 0.11 | 0.11 | face/upscale onnx set | CITED | iTools/media/stock-prep, design/stock-kit, pixelshaper read a few files; most files unreferenced |
| `deep-live-cam-models/motion_extractor.safetensors` | 0.11 | 0.11 | face/upscale onnx set | CITED | iTools/media/stock-prep, design/stock-kit, pixelshaper read a few files; most files unreferenced |
| `deep-live-cam-models/restoreformer_plus_plus.onnx` | 0.29 | 0.29 | face/upscale onnx set | CITED | iTools/media/stock-prep, design/stock-kit, pixelshaper read a few files; most files unreferenced |
| `deep-live-cam-models/reswapper_256.onnx` | 0.55 | 0.55 | face/upscale onnx set | CITED | iTools/media/stock-prep, design/stock-kit, pixelshaper read a few files; most files unreferenced |
| `deep-live-cam-models/sam2/sam2_hiera_tiny.encoder.onnx` | 0.13 | 0.13 | face/upscale onnx set | CITED | iTools/media/stock-prep, design/stock-kit, pixelshaper read a few files; most files unreferenced |
| `deep-live-cam-models/simswap_256.onnx` | 0.22 | 0.22 | face/upscale onnx set | CITED | iTools/media/stock-prep, design/stock-kit, pixelshaper read a few files; most files unreferenced |
| `deep-live-cam-models/simswap_512.onnx` | 0.22 | 0.22 | face/upscale onnx set | CITED | iTools/media/stock-prep, design/stock-kit, pixelshaper read a few files; most files unreferenced |
| `deep-live-cam-models/spade_generator.safetensors` | 0.22 | 0.22 | face/upscale onnx set | CITED | iTools/media/stock-prep, design/stock-kit, pixelshaper read a few files; most files unreferenced |
| `deep-live-cam-models/warping_module.safetensors` | 0.18 | 0.18 | face/upscale onnx set | CITED | iTools/media/stock-prep, design/stock-kit, pixelshaper read a few files; most files unreferenced |
| `deepseek-ocr-2/model-00001-of-000001.safetensors` | 6.78 | 6.78 | model | UNUSED | zero hits in mojo-baro code and docs, iTools, Projects scripts |
| `dinov3/dinov3-vith16plus-pretrain-lvd1689m/model.safetensors` | 3.36 | 3.36 | vision model | CITED | Projects/gpu-media/Blender/kit/asset_dino.py path; no run evidence |
| `faster-whisper-large-v3-turbo-ct2/model.bin` | 1.62 | 1.62 | ASR ct2 | USED | iTools/video/vidtext/test.sh |
| `flux2/FLUX.2-dev/vae/diffusion_pytorch_model.safetensors` | 0.34 | 0.34 | data/aux | USED | see usage notes |
| `hunyuan3d-2/hunyuan3d-dit-v2-mv_fp16.safetensors` | 4.93 | 4.93 | 3D (image to mesh) | CITED | symlinked in ComfyUI/models/checkpoints; no 3D workflow output uses it (outputs/mesh = TRELLIS/pixal3d) |
| `hunyuan3d-2/hunyuan3d-dit-v2_fp16.safetensors` | 4.93 | 4.93 | 3D (image to mesh) | CITED | symlinked in ComfyUI/models/checkpoints; no 3D workflow output uses it (outputs/mesh = TRELLIS/pixal3d) |
| `lgm/model_fp16_fixrot.safetensors` | 0.83 | 0.83 | model | UNUSED | zero hits in mojo-baro code and docs, iTools, Projects scripts |
| `sdxl-vae/sdxl_vae.safetensors` | 0.33 | 0.33 | data/aux | USED | see usage notes |
| `siglip-so400m-patch14-384/model.safetensors` | 3.51 | 3.51 | embedding/vision model | UNUSED | only generic listings (brain-laptop-sync.sh, draft-head-protocol.md); no loader |
| `styledem/styledem_weights/styledem_30m.pt` | 1.25 | 1.25 | model | UNUSED | zero hits in mojo-baro code and docs, iTools, Projects scripts |
| `styledem/styledem_weights/styledem_5m.pt` | 1.25 | 1.25 | model | UNUSED | zero hits in mojo-baro code and docs, iTools, Projects scripts |
| `trm-arc2-8gpu/model.ckpt` | 2.47 | 2.47 | model | UNUSED | zero hits in mojo-baro code and docs, iTools, Projects scripts |
| `tts/chatterbox/s3gen.safetensors` | 1.06 | 1.06 | TTS (f5 romanian, chatterbox) | CITED | docs/PLATFORM-PLAN.md; MoneyPrinterTurbo config.example |
| `tts/chatterbox/t3_cfg.safetensors` | 2.13 | 2.13 | TTS (f5 romanian, chatterbox) | CITED | docs/PLATFORM-PLAN.md; MoneyPrinterTurbo config.example |
| `tts/f5-tts-romanian/model_235000.pt` | 5.39 | 5.39 | TTS (f5 romanian, chatterbox) | CITED | docs/PLATFORM-PLAN.md; MoneyPrinterTurbo config.example |
| `whisper-large-v3-turbo-hf/model.safetensors` | 1.62 | 1.62 | model | UNUSED | zero hits in mojo-baro code and docs, iTools, Projects scripts |
| `whisper.cpp/build/ggml/src/ggml-vulkan/mul_mm.comp.cpp` | 0.13 | 0.13 | build tree | USED | llama.cpp: reference arm everywhere; whisper.cpp: audio.rs; prism: no hit |
| `whisper/ggml-base.bin` | 0.15 | 0.15 | ASR ggml | USED | serve/src/audio.rs default model (large-v3-turbo-q5_0); bench/p3a-gate.sh |
| `whisper/ggml-large-v3-turbo-q5_0.bin` | 0.57 | 0.57 | ASR ggml | USED | serve/src/audio.rs default model (large-v3-turbo-q5_0); bench/p3a-gate.sh |

### Checkpoints, data, assets, build trees, archives

| path | apparent GB | exclusive GB | role | status | evidence |
|---|---:|---:|---|---|---|
| `ChelooRVCv2.zip` | 0.28 | 0.28 | voice-model archive | UNUSED | no references anywhere |
| `Nicusor Dan (Romanian President) - Weights Model.zip` | 0.13 | 0.13 | voice-model archive | UNUSED | no references anywhere |
| `Puya (Romanian Rapper) - Weights Model.zip` | 0.18 | 0.18 | voice-model archive | UNUSED | no references anywhere |
| `Training/devdocs-offline-2026-09-27 (1).json` | 0.41 | 0.41 | data/aux | USED | see usage notes |
| `becaliRVC.rar` | 0.10 | 0.10 | voice-model archive | UNUSED | no references anywhere |
| `blender-library/blender-assets__nature.blend` | 0.10 | 0.10 | Blender assets | USED | iTools/media/blender-library, street-sim forge |
| `blender-library/street-sim__forge__interior__clean.blend` | 0.09 | 0.09 | Blender assets | USED | iTools/media/blender-library, street-sim forge |
| `datasets/graphs/rgg_n_2_20_s0.edges` | 0.10 | 0.10 | data/aux | USED | see usage notes |
| `e-girl.zip` | 0.17 | 0.17 | voice-model archive | UNUSED | no references anywhere |
| `llama.cpp/.git/objects/pack/pack-2b059472c6ee4fdca8e6dd197c48ac3bd3dca2ff.pack` | 0.45 | 0.45 | build tree | USED | llama.cpp: reference arm everywhere; whisper.cpp: audio.rs; prism: no hit |
| `llama.cpp/build/bin/libggml-hip.so.0.23.0` | 0.12 | 0.12 | build tree | USED | llama.cpp: reference arm everywhere; whisper.cpp: audio.rs; prism: no hit |
| `llama.cpp/build/common/CMakeFiles/llama-common.dir/cmake_pch.hxx.gch` | 0.15 | 0.15 | build tree | USED | llama.cpp: reference arm everywhere; whisper.cpp: audio.rs; prism: no hit |
| `llama.cpp/build/src/CMakeFiles/llama.dir/cmake_pch.hxx.gch` | 0.11 | 0.11 | build tree | USED | llama.cpp: reference arm everywhere; whisper.cpp: audio.rs; prism: no hit |
| `llama.cpp/build/tests/CMakeFiles/test-peg-parser.dir/cmake_pch.hxx.gch` | 0.16 | 0.16 | build tree | USED | llama.cpp: reference arm everywhere; whisper.cpp: audio.rs; prism: no hit |
| `llama.cpp/build/tools/mtmd/CMakeFiles/mtmd.dir/cmake_pch.hxx.gch` | 0.10 | 0.10 | build tree | USED | llama.cpp: reference arm everywhere; whisper.cpp: audio.rs; prism: no hit |
| `ncm-checkpoints/lm1b/large/generator.pt` | 2.90 | 2.90 | checkpoint | CITED | ncm/tools/ncm_pack.py lists them; ncm/check.sh uses sudoku/easy only |
| `ncm-checkpoints/lm1b/medium/generator.pt` | 1.27 | 1.27 | checkpoint | CITED | ncm/tools/ncm_pack.py lists them; ncm/check.sh uses sudoku/easy only |
| `ncm-checkpoints/lm1b/small/generator.pt` | 0.39 | 0.39 | checkpoint | CITED | ncm/tools/ncm_pack.py lists them; ncm/check.sh uses sudoku/easy only |
| `ncm-checkpoints/lm1b/tokenizer.pt` | 0.60 | 0.60 | checkpoint | CITED | ncm/tools/ncm_pack.py lists them; ncm/check.sh uses sudoku/easy only |
| `ncm-checkpoints/sudoku/easy/generator.pt` | 0.10 | 0.10 | checkpoint + bake | USED | ncm/check.sh, ncm/tools/ncm_pack.py |
| `ncm-checkpoints/sudoku/easy/mojo/ncm-sudoku-easy-BARO-7fbaba2.gguf` | 0.14 | 0.13 | checkpoint + bake | USED | ncm/check.sh, ncm/tools/ncm_pack.py |
| `ncm-checkpoints/sudoku/easy/mojo/ncm-sudoku-easy.gguf` | 0.14 | 0.13 | checkpoint + bake | USED | ncm/check.sh, ncm/tools/ncm_pack.py |
| `ncm-checkpoints/sudoku/easy/mojo/weights.f32` | 0.14 | 0.14 | checkpoint + bake | USED | ncm/check.sh, ncm/tools/ncm_pack.py |
| `ncm-checkpoints/sudoku/hard/generator.pt` | 0.10 | 0.10 | checkpoint | CITED | ncm/tools/ncm_pack.py lists them; ncm/check.sh uses sudoku/easy only |
| `ncm-checkpoints/sudoku/hard/mojo/ncm-sudoku-easy.gguf` | 0.14 | 0.14 | checkpoint | CITED | ncm/tools/ncm_pack.py lists them; ncm/check.sh uses sudoku/easy only |
| `ncm-checkpoints/sudoku/hard/mojo/weights.f32` | 0.14 | 0.14 | checkpoint | CITED | ncm/tools/ncm_pack.py lists them; ncm/check.sh uses sudoku/easy only |
| `ncm-checkpoints/sudoku/medium/generator.pt` | 0.10 | 0.10 | checkpoint | CITED | ncm/tools/ncm_pack.py lists them; ncm/check.sh uses sudoku/easy only |
| `ncm-checkpoints/tinystories/generator.pt` | 0.36 | 0.36 | checkpoint | CITED | ncm/tools/ncm_pack.py lists them; ncm/check.sh uses sudoku/easy only |
| `ncm-checkpoints/tinystories/tokenizer.pt` | 0.22 | 0.22 | checkpoint | CITED | ncm/tools/ncm_pack.py lists them; ncm/check.sh uses sudoku/easy only |
| `ncm-checkpoints/wmt14/generator.pt` | 0.48 | 0.48 | checkpoint | CITED | ncm/tools/ncm_pack.py lists them; ncm/check.sh uses sudoku/easy only |
| `ncm-checkpoints/wmt14/tokenizer.pt` | 0.65 | 0.65 | checkpoint | CITED | ncm/tools/ncm_pack.py lists them; ncm/check.sh uses sudoku/easy only |
| `ncm-checkpoints/xsum/generator.pt` | 0.35 | 0.35 | checkpoint | CITED | ncm/tools/ncm_pack.py lists them; ncm/check.sh uses sudoku/easy only |
| `ncm-checkpoints/xsum/tokenizer.pt` | 0.30 | 0.30 | checkpoint | CITED | ncm/tools/ncm_pack.py lists them; ncm/check.sh uses sudoku/easy only |
| `quant-lab/wikitext-103-raw/wiki.train.raw` | 0.54 | 0.54 | data/aux | USED | see usage notes |


## image.video by workflow (189.4 GB)

### Where ComfyUI reads models

- Launcher: `~/.local/bin/comfyui` (ComfyUI-v0.43.0, torch ROCm), started inside a GPU slot by `~/iTools/bin/comfy-up`; batch use through `with-comfy`; Idol profile through `comfy-idol` (`~/.config/comfyui-profiles/idol.yaml`).
- `~/Projects/imports/ComfyUI-v0.43.0/extra_model_paths.yaml` maps four roots: `image.video/xxx/` (empty), `image.video/comfy/` (checkpoints, diffusion_models, text_encoders, vae, loras), `image.video/upscale/` (upscale_models), and the legacy `~/Projects/imports/ComfyUI/models/`. A copy of the first two sections is in `~/iTools/design/comfy-provision/extra_model_paths.yaml`.
- `ComfyUI-v0.43.0/models/` also holds symlinks into `image.video/ltx-2.5`, `wan-vace`, `controlnet-union-sdxl`, `upscale`, and local files not in `~/Models` (RealVisXL_V5.0_fp16 6.6 GB, wan2.2_i2v_low_noise_14B_fp8_scaled 13.6 GB, umt5_xxl_fp8 6.4 GB, trellis_2_int8 5.0 GB, pixal3d_int8 5.3 GB, wan_2.1_vae).
- No saved UI workflow exists (`ComfyUI-v0.43.0/user/default/workflows` is absent). The workflows the owner runs are code and JSON in the repos below. `~/Projects/imports/comfy-workflows/templates` (92 upstream template JSONs) and `image.video/ltx-2.5/workflows/*.json` (upstream LTX examples) are stock downloads, not runs; they are listed as "stock workflows only".
- Run evidence: 3867 PNGs in `ComfyUI-v0.43.0/output` were parsed for the loader file names in their `prompt` metadata (counts below); video and audio outputs were matched by directory and date (`output/ltx` 5 mp4 2026-11-02, `output/ace` 41 flac 2026-11-06).

### Workflows (pics, textures, video, audio) and the model files each loads

| id | workflow (owner's) | models loaded from image.video | evidence |
|---|---|---|---|
| W1 | `~/iTools/design/comfy-run/comfy-run.mojo` kind `zimage` (pictures, default 4 steps 768) | comfy/z_image_turbo_int8, comfy/qwen_3_4b, comfy/ae | file defaults lines 57-59; Brain ComfyUI 2026-11-04-zimage-2s |
| W2 | `~/Projects/street-sim/tools/texgen.py` (textures): Z-Image graph and SDXL graph | Z-Image: z_image_turbo_int8, qwen_3_4b, ae, upscale/4x-UltraSharp. SDXL: RealVisXL_V5.0_fp16 (not in Models), comfy/sdxl_lightning_4step_lora, 4x-UltraSharp | texgen.py lines 141-143, 199-232; 680 PNGs with RealVisXL, 579 with 4x-UltraSharp |
| W3 | stock pipeline: `~/iTools/design/stock-kit/render.sh`, ComfyUI `.work/stock/office100/jobs.json` | Z-Image int8 stack + upscale/RealESRGAN_x4plus | 55 PNGs 2026-11-07 |
| W4 | SDXL restyle with ControlNet (skill comfy-ops) | controlnet-union-sdxl/..promax (+ RealVisXL, IPAdapter from legacy root) | 26 PNGs restyle_*, newest 2026-09-24 |
| W5 | `~/Projects/gpu-media/vidcode/make/renderer/comfy_pipeline.py` stills | comfy/z_image_turbo_bf16, qwen_3_4b, ae, 4x-UltraSharp, wan lightx2v high LoRA (+ RealVisXL, ip-adapter, minimax files not on disk) | constants lines 47-59; 174 PNGs with bf16, newest 2026-09-18 |
| W6 | vidcode `term2leg.py` (Wan 2.2 image to video) | comfy/wan22RemixT2VI2V_i2vHighV30_lightx2v4step (baked), non-baked low UNET from ComfyUI/models | lines 22-25 |
| W7 | comfy-run kind `i2v` / ComfyUI `.work/prs` experiments (comfy-ops: baked 4-step Wan) | comfy/wan2.2_i2v_low_noise_14B_fp8_scaled_lightx2v4step | `.work/prs/out/*/wana.md` |
| W8 | LTX 2.5 video: ComfyUI `.work/ltx/{api,base,m}.json` | ltx-2.5/LTX-2.5-Distilled-Q5_K_M.gguf, text_encoders/..int8-convrot, vae/..video-vae-conv-bf16, vae/..audio-vae-bf16, latent_upscale_models/..x2 | JSON loader fields; 5 mp4 in output/ltx 2026-11-02 |
| W9 | ACE-Step 1.5 music (skill sellable-audio-assets, blueprint "Text to Audio (ACE-Step 1.5)") | comfy/acestep_v1.5_turbo, qwen_4b_ace15, qwen_0.6b_ace15, ace_1.5_vae | 41 flac `mus_bar_*` 2026-11-06 |
| W10 | Wan VACE 1.3B (comfy-ops VACE section) | wan-vace/wan2.1_vace_1.3B_fp16 (+ umt5, wan_2.1_vae from ComfyUI/models) | 112 PNGs vace_*, newest 2026-09-24 |
| W11 | Idol node (`comfy-idol`, bear receipt 2026-11-03) | idol/denoiser, decoder, decoder_gs, birefnet | `iTools/design/comfy-idol/nodes/comfy_idol/__init__.py` |
| none | kandinsky-6.0-lite-distill-5s, mop*, snofs*, klein_snofs, the four NSFW LoRAs, gemma4 bf16 text encoder, plain ltx video VAE, idol-test | | no loader name, no output; mop and MysticXXX last appear in outputs of 2026-07-17 |

Loader names counted in output PNG metadata: ae 2406, qwen_3_4b 2404, z_image_turbo_int8 2232, RealVisXL 680, 4x-UltraSharp 579, mvadapter_i2mv_sdxl 228 (not in Models), sdxl_vae 224, z_image_turbo_bf16 174, wan2.1_vace 112, RealESRGAN_x4plus 55, controlnet-union-promax 26, mopMixtureOfPerverts 9 (2026-07-17), sdxl_lightning 6.

### Every image.video file of 30 MB or more

| file under ~/Models/image.video | apparent GB | exclusive GB | status | evidence |
|---|---:|---:|---|---|
| `ltx-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` | 26.26 | 26.26 | UNUSED (stock workflows only) | named only by upstream workflows in ltx-2.5/workflows/*.json and the ComfyUI/models/text_encoders link |
| `ltx-2.5/LTX-2.5-Distilled-Q5_K_M.gguf` | 16.82 | 16.82 | USED-BY W8 LTX 2.5 | ComfyUI/.work/ltx/{api,base,m}.json UnetLoaderGGUF; ComfyUI/models/unet link |
| `ltx-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` | 15.37 | 15.37 | USED-BY W8 LTX 2.5 | same three workflow JSONs |
| `comfy/wan2.2_i2v_low_noise_14B_fp8_scaled_lightx2v4step.safetensors` | 14.30 | 14.30 | CITED W7 (comfy-run i2v default, .work/prs) | vidcode loads the non-baked low UNET from ComfyUI/models instead |
| `comfy/wan22RemixT2VI2V_i2vHighV30_lightx2v4step.safetensors` | 14.29 | 14.29 | USED-BY W6 vidcode term2leg | term2leg.py HIGH_UNET; no PNG evidence (video output) |
| `comfy/z_image_turbo_bf16.safetensors` | 12.31 | 12.31 | USED-BY W5 vidcode renderer only | comfy_pipeline.py ZIMAGE_UNET; 174 PNGs, newest 2026-09-18; comfy-ops calls it redundant |
| `comfy/qwen_4b_ace15.safetensors` | 8.38 | 8.38 | USED-BY W9 ACE-Step 1.5 | 41 flac in ComfyUI/output/ace dated 2026-11-06 |
| `comfy/qwen_3_4b.safetensors` | 8.04 | 8.04 | USED-BY W1, W2, W3, W5 | text encoder of the Z-Image graph; 2404 PNGs, newest 2026-11-12 |
| `comfy/mopMixtureOfPerverts_v71Flexible.safetensors` | 6.94 | 6.94 | UNUSED | last output 2026-07-17; no script or workflow names it |
| `kandinsky-6.0-lite-distill-5s/transformer/diffusion_pytorch_model.safetensors` | 6.38 | 6.38 | UNUSED | no workflow, script or output; Brain index lists it as "unchecked licence" |
| `comfy/z_image_turbo_int8.safetensors` | 6.29 | 6.29 | USED-BY W1 comfy-run zimage, W2 texgen (Z-Image graph), W3 stock | 2232 output PNGs carry it, newest 2026-11-12; comfy-run.mojo default unet |
| `comfy/acestep_v1.5_turbo.safetensors` | 4.79 | 4.79 | USED-BY W9 ACE-Step 1.5 | 41 flac in ComfyUI/output/ace dated 2026-11-06 |
| `comfy/mopMixtureOfPerverts_v71Flexible_int8.safetensors` | 4.75 | 4.75 | UNUSED | last output 2026-07-17; no script or workflow names it |
| `wan-vace/wan2.1_vace_1.3B_fp16.safetensors` | 4.31 | 4.31 | USED-BY W10 Wan VACE | 112 PNGs vace_*, newest 2026-09-24; comfy-ops section on VACE |
| `kandinsky-6.0-lite-distill-5s/text_encoder/model-00001-of-00005.safetensors` | 3.90 | 3.90 | UNUSED | no workflow, script or output; Brain index lists it as "unchecked licence" |
| `kandinsky-6.0-lite-distill-5s/text_encoder/model-00004-of-00005.safetensors` | 3.86 | 3.86 | UNUSED | no workflow, script or output; Brain index lists it as "unchecked licence" |
| `kandinsky-6.0-lite-distill-5s/text_encoder/model-00002-of-00005.safetensors` | 3.86 | 3.86 | UNUSED | no workflow, script or output; Brain index lists it as "unchecked licence" |
| `kandinsky-6.0-lite-distill-5s/text_encoder/model-00003-of-00005.safetensors` | 3.86 | 3.86 | UNUSED | no workflow, script or output; Brain index lists it as "unchecked licence" |
| `controlnet-union-sdxl/diffusion_pytorch_model_promax.safetensors` | 2.51 | 2.51 | USED-BY W4 SDXL restyle (ControlNet) | 26 PNGs restyle_*, newest 2026-09-24; comfy-ops |
| `kandinsky-6.0-lite-distill-5s/audio_vae/diffusion_pytorch_model.safetensors` | 1.71 | 1.71 | UNUSED | no workflow, script or output; Brain index lists it as "unchecked licence" |
| `comfy/snofs_krea_v1.safetensors` | 1.56 | 1.56 | UNUSED | Krea 2 era LoRAs (Krea 2 removed 2026-11-04); only vidcode log 2026-08-27 |
| `idol/denoiser.safetensors` | 1.48 | 1.48 | USED-BY W11 Idol node (comfy-idol) | iTools/design/comfy-idol launcher + node; bear receipt 2026-11-03 |
| `ltx-2.5/vae/ltx-2.5-video-vae-bf16.safetensors` | 1.47 | 1.47 | UNUSED (stock workflows only) | .work/ltx uses the conv vae; plain vae only in upstream workflows and ComfyUI/models/vae link |
| `ltx-2.5/vae/ltx-2.5-video-vae-conv-bf16.safetensors` | 1.45 | 1.45 | USED-BY W8 LTX 2.5 | .work/ltx JSONs |
| `comfy/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors` | 1.23 | 1.23 | CITED W6/W7 | comfy_pipeline.py WAN_LORA; OpenMontage |
| `comfy/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors` | 1.23 | 1.23 | CITED W6/W7 | term2leg.py comment; OpenMontage |
| `comfy/qwen_0.6b_ace15.safetensors` | 1.19 | 1.19 | USED-BY W9 ACE-Step 1.5 | 41 flac in ComfyUI/output/ace dated 2026-11-06 |
| `comfy/klein_snofs_v1_1.safetensors` | 1.09 | 1.09 | UNUSED | Krea 2 era LoRAs (Krea 2 removed 2026-11-04); only vidcode log 2026-08-27 |
| `kandinsky-6.0-lite-distill-5s/text_encoder/model-00005-of-00005.safetensors` | 1.09 | 1.09 | UNUSED | no workflow, script or output; Brain index lists it as "unchecked licence" |
| `ltx-2.5/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | 1.00 | 1.00 | USED-BY W8 LTX 2.5 | .work/ltx JSONs |
| `kandinsky-6.0-lite-distill-5s/vae/diffusion_pytorch_model.safetensors` | 0.99 | 0.99 | UNUSED | no workflow, script or output; Brain index lists it as "unchecked licence" |
| `ltx-2.5/.cache/huggingface/download/text_encoders/2sKFbFgEyLO8lcXc6bTV9P5Yphk=.6ce688a0aa98a5fa36a9f1e6c3f42152a498cc2b53ee8c15674c64244f91487f.b9d33539.incomplete` | 0.96 | 0.96 | UNUSED | partial HF downloads (.incomplete) |
| `idol/decoder_gs.safetensors` | 0.95 | 0.95 | USED-BY W11 Idol node (comfy-idol) | iTools/design/comfy-idol launcher + node; bear receipt 2026-11-03 |
| `kandinsky-6.0-lite-distill-5s/text_encoder_2/model.safetensors` | 0.49 | 0.49 | UNUSED | no workflow, script or output; Brain index lists it as "unchecked licence" |
| `kandinsky-6.0-lite-distill-5s/vocoder/diffusion_pytorch_model.safetensors` | 0.49 | 0.49 | UNUSED | no workflow, script or output; Brain index lists it as "unchecked licence" |
| `idol/birefnet.safetensors` | 0.44 | 0.44 | USED-BY W11 Idol node (comfy-idol) | iTools/design/comfy-idol launcher + node; bear receipt 2026-11-03 |
| `comfy/sdxl_lightning_4step_lora.safetensors` | 0.39 | 0.39 | USED-BY W2 texgen SDXL graph | tools/texgen.py LoraLoaderModelOnly; 6 PNGs 2026-11-04 |
| `ltx-2.5/vae/ltx-2.5-audio-vae-bf16.safetensors` | 0.36 | 0.36 | USED-BY W8 LTX 2.5 | .work/ltx JSONs |
| `comfy/ace_1.5_vae.safetensors` | 0.34 | 0.34 | USED-BY W9 ACE-Step 1.5 | 41 flac in ComfyUI/output/ace dated 2026-11-06 |
| `comfy/ae.safetensors` | 0.34 | 0.34 | USED-BY W1, W2, W3, W5 | Z-Image VAE; 2406 PNGs |
| `comfy/blowjobs_sloppy_000008000_high_noise.safetensors` | 0.31 | 0.31 | UNUSED | NSFW LoRAs; last output 2026-07-17 |
| `comfy/Wan2.2-Low-MysticXXX-v2.safetensors` | 0.31 | 0.31 | UNUSED | NSFW LoRAs; last output 2026-07-17 |
| `comfy/jfj-deepthroat-W22-T2V-HN-v1.safetensors` | 0.23 | 0.23 | UNUSED | NSFW LoRAs; last output 2026-07-17 |
| `idol/decoder.safetensors` | 0.20 | 0.20 | USED-BY W11 Idol node (comfy-idol) | iTools/design/comfy-idol launcher + node; bear receipt 2026-11-03 |
| `comfy/Demo AI Influencer 014 By Stable Yogi.safetensors` | 0.14 | 0.14 | UNUSED | NSFW LoRAs; last output 2026-07-17 |
| `ltx-2.5/.cache/huggingface/download/rPB56h2H4FwPTTTdnd74fZsZa7A=.ba24225c48052bb6d19a356ce48e746f627f34c1595b49a1c2f7af10c210cda9.32988905.incomplete` | 0.13 | 0.13 | UNUSED | partial HF downloads (.incomplete) |
| `idol-test/bear_front.ply` | 0.07 | 0.07 | UNUSED | test outputs 2026-11-03 |
| `idol-test/bear_solid.ply` | 0.07 | 0.07 | UNUSED | test outputs 2026-11-03 |
| `idol-test/bear_s7.ply` | 0.07 | 0.07 | UNUSED | test outputs 2026-11-03 |
| `idol-test/bear_s42.ply` | 0.07 | 0.07 | UNUSED | test outputs 2026-11-03 |
| `idol-test/bear_s123.ply` | 0.07 | 0.07 | UNUSED | test outputs 2026-11-03 |
| `idol-test/bear_s2026.ply` | 0.07 | 0.07 | UNUSED | test outputs 2026-11-03 |
| `upscale/RealESRGAN_x4plus.pth` | 0.07 | 0.07 | USED-BY W3 stock | 55 PNGs 2026-11-07; stock-kit render.sh; twin of deep-live-cam-models copy |
| `upscale/4x-UltraSharp.pth` | 0.07 | 0.07 | USED-BY W2 texgen, W5 vidcode | 579 PNGs, newest 2026-11-10; CC BY-NC-SA, not for sale |
| `idol/assets/reel.mp4` | 0.03 | 0.03 | USED-BY W11 Idol node (comfy-idol) | iTools/design/comfy-idol launcher + node; bear receipt 2026-11-03 |
Totals inside image.video (exclusive): USED-BY 101.44 GB, CITED 16.75 GB, UNUSED 71.23 GB. Small files (workflows, docs, assets, `Qwen-Image-main` clone, `idol/assets`) add under 0.1 GB. `image.video/xxx` and `image.video/shared` are empty.

## Packs in ~/.cache/baro (about 62 GB, rebuildable by `tools/baro` from the bakes)

| pack | apparent GB | exclusive GB | built from (sha-cache.tsv) | status |
|---|---:|---:|---|---|
| `~/.cache/baro/qwen35-eec459db896e8bc6/packs/57a7bf8249da85a3cea8bea985b064/pack.bin` | 17.20 | 17.20 | Turbo-FCF bake c71dbdc (sha-cache row) | built 2026-11-10; no script reads it, only lane docs |
| `~/.cache/baro/qwen35-c19706672fb25b4a/packs/5657eb0f4ce2e32245090603d4e45e/pack.bin` | 10.73 | 10.73 | Qwythos-v2 bake eeb082e | current |
| `~/.cache/baro/qwen35-821da0e0fb3ffe87/packs/e86ba300a85d14a688af7bfd80aa72/pack.bin` | 10.73 | 10.73 | Ornith bake cf78732 | current |
| `~/.cache/baro/qwen2-9cb3d655831d981e/packs/47c8e9944221be4e9f6df580b160757/pack.bin` | 6.55 | 6.55 | qwen2.5-7b bake 1d6e2dc | current |
| `~/.cache/baro/spark2_5-6be66f945d465e82/packs/26d23ab64a37d1a93f5ddb286b9d/pack.bin` | 5.71 | 5.71 | Spark-X2.5-4B bake 1d6e2dc | current |
| `~/.cache/baro/granite-d8ea7d76a98aa5af/packs/491c4b79f9a72ae1c3dd165988048/pack.bin` | 4.64 | 4.64 | granite bake 1d6e2dc | current |
| `~/.cache/baro/llama-531b581265e3c68f/packs/6e62e4066f951208db9e718f48e0dec/pack.bin` | 1.85 | 1.85 | .work/auto-llama bake 93a8b08 (2026-11-02, outside Models) | superseded: candidate |
| `~/.cache/baro/llama-531b581265e3c68f/packs/919b9b98b7e9023b1ec828235fff73c/pack.bin` | 1.85 | 1.85 | Llama-3.2-1B bake cf78732fix | superseded: candidate |
| `~/.cache/baro/llama-531b581265e3c68f/packs/e8e60e774180545f8c4dfa06f66729f/pack.bin` | 1.85 | 1.85 | Llama-3.2-1B bake 1d6e2dc | current |
Packs of superseded bake shas (llama 919b9b.. and 6e62e4..) free 3.70 GB and are rebuilt by `tools/baro` from the current bake. The qwen35-821d.. and qwen35-c197.. packs share no extents.

## Findings outside the brief (one line each)

- `ComfyUI-v0.43.0/models/{checkpoints/flux1-dev-fp8, clip_vision/sigclip.., controlnet/flux1-controlnet-union.., style_models/flux1-redux..}` and five `image.video/comfy/*krea2*`/`qwen3vl*`/`qwen_image_vae` links point at deleted dirs (Flux 2026-09-22, Krea 2 2026-11-04). Dangling, 0 bytes; remove the links.
- `bench/quality-models.json` names `lily-cybersecurity-7b-v0.2-q6_k-BARO-04867e2.gguf`, which is not on disk (only the two IQ1_S ggufs are).
- `qwen35-eec459..` pack (17.2 GB, Turbo-FCF) is the newest pack in the cache (2026-11-10) yet no script reads it; its bake `c71dbdc` is older than `a4eb549` by date. Decide which Turbo bake is current before any delete there.
- RealVisXL_V5.0_fp16 (6.6 GB, used by texgen and vidcode, 680 PNGs) lives only in `ComfyUI-v0.43.0/models/checkpoints`, not in `~/Models`. It has no twin and no backup path.
- `docs/ARTIFACTS.md` says the RegesCore resident pack was rebuilt from `RegesCore-..-BARO-3c6c25c.gguf` and equals the pack of the old e340ee1 bake. The newer `cf78732` bake is the current one; dropping 3c6c25c frees only 0.01 GB, so there is no reason to.

## Cleanup done 2026-11-13 (owner ran each line; free space 22 GB -> 97 GB)
- 7 embedding models deleted after bge-small became the standard embedder (baro-embed.service, :8081, merge 278726d). Kept: bge-small-en-v1.5, ms-marco-MiniLM-L-6-v2 (brain-ask).
- image.video: 13 UNUSED paths (kandinsky-6.0, gemma4-12b bf16 encoder, plain LTX VAE, partial HF downloads, mop*, snofs/klein LoRAs, 4 NSFW LoRAs, idol-test) and the 2 ComfyUI links to them; list `docs/receipts/models-cleanup/image-video-delete.txt`.
- 29 old/unused models (`docs/receipts/models-cleanup/models-old-delete.txt`): phi-4-mini, Ternary-Bonsai, deepseek-ocr-2, xlam-2, qwythos q4_k_m, bitnet, siglip, MiniCPM-V, styledem, trm-arc2, IFM, lgm, TriLM, RVC voice archives, qwen2.5-0.5b imports, 3c6c25c bakes except RegesCore.
- Superseded spark bakes (cf78732, 1d6e2dc) deleted; current bakes 38e5113 (Llama-1B, granite, Spark-X2.5, qwen2.5-7B, Ornith-9B), eeb082e (Qwythos-9B v2). RegesCore-35B has only its 3c6c25c bake (no buildable bake yet).
