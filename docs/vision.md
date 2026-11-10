# Vision input (images in chat requests)

Status 2026-09-24: sidecar route, MoE engine (`qwen35moe`) only. Served check
`bench/vision-smoke.sh` (stub dry run PASS on the CPU; the real-engine run is
recorded below when it lands). Dense engines keep text only.

```
image_url part ─► baro-serve (vision.rs) ─► tools/mtmd-embed (libmtmd, CPU) ─► .bemb file
                       │                                                          │
                       └─ text gets <|vision_start|><|image_pad|>×N<|vision_end|>  │
                                                                                  ▼
engine request {"prompt":[ids],"embeds":["/abs/x.bemb"]} ─► serve/engine.mojo stage_vision
   pos3_d   3-axis rope positions per token (t,h,w), text (p,p,p), image (p+t, p+y, p+x), p += n_pos
   imgmap_d token -> image row (-1 = text)      img_d image rows (f32, IMG_CAP 4096 rows)
   prefill: moe_embed then amar_inject_rows; attention rope: amar_head_rmsnorm_mrope / amar_rope_mrope
```

## Pieces
- `tools/mtmd-embed.c` (+ `tools/mtmd-embed-build.sh`): links llama.cpp's `libmtmd`/`libllama`
  from `~/Projects/imports/llama.cpp/build/bin`; opens the text gguf vocab-only, reads the
  projector width from the mmproj header, encodes one image, writes `BEMB` v1: u32 n_tokens,
  n_embd, n_pos, then n_tokens × (t, x, y) i32 with pos_0 = 0, then n_tokens × n_embd f32.
  `--max-tokens N` caps the dynamic-resolution token count (`image_max_tokens`); `--gpu` uses the
  ggml HIP backend (about 1 GB VRAM, not used next to the resident 35B pack).
- `serve/src/vision.rs`: OpenAI content parts; `image_url` as a `data:` base64 URL or an absolute
  path. Env: `BARO_MTMD_EMBED`, `BARO_MMPROJ`, `BARO_TEXT_GGUF`, `BARO_IMAGE_MAX_TOKENS` (576),
  `BARO_VISION_DIR` (cache `.work/vision/cache`, keyed by content hash and cap). Unset => 400.
- `kernels/vision.mojo`: `amar_rope_mrope`, `amar_head_rmsnorm_mrope` (interleaved mrope: pair j
  takes axis j % 3 of (t, h, w), Qwen3.5 `mrope_section [11, 11, 10]`, `mrope_interleaved`),
  `amar_inject_rows`. Text-only requests never call them: the text path is the old kernels.
- `serve/engine.mojo`: `"embeds"` request field, `stage_vision`, `IMAGE_PAD = 248056`. A request
  with images runs the launch path (the MoE megakernel ropes inline without mrope), with
  speculation off and prefix checkpoints off (image pads share one id, so a prefix hash cannot
  tell two images apart; `invalidate_above(0)` also drops earlier checkpoints, the next text
  request re-prefills).

## Costs and limits
- Encode on the CPU: 922×2048 render at the default cap 576 → 560 tokens in about 60 s; at 256 →
  252 tokens in 18 s (7800X3D, 8 threads). GPU encode needs about 1 GB VRAM the resident profile
  does not have; tier mode could host it.
- One image per `<|image_pad|>` run, files in prompt order; the engine refuses a run whose length
  differs from the file's token count, a width other than H, or more than `IMG_CAP` rows.
- Not done: dense engines, video, Ollama `images`, the MoE megakernel with images, checkpoint reuse
  across image requests.

## Checks
- `bench/vision-smoke.sh IMAGE OUT` (stub engine on the CPU, or the real engine under `gpu-wait`):
  `/tokenize` on the three specials, one image request, one text request after it.
- Text identity: `bench/ab-prompts.sh OLD OUT ENV ENV old new` with `AB_ENGINE_B=NEW` (the same
  env on both arms, kill line on).
- Reference agreement against llama.cpp (`llama-server --mmproj --image-max-tokens`): planned, not
  run.
