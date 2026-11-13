# Brief: ~/Models inventory, duplicates and ComfyUI usage (sonnet, effort medium, M) - READ ONLY

Goal: a report the owner uses to decide what to delete. **Delete, move or rewrite nothing.** No GPU.
Disk is at 99% (about 21 GB free); every number you report is bytes that a delete would really free.

## Do
1. Inventory every top-level entry of `~/Models`: apparent size and btrfs exclusive size
   (`btrfs filesystem du -s --raw`). Exclusive is what a delete frees; reflinked bakes share their vendor file.
2. Group into families (same base model): source weights (bf16/safetensors), quant variants (q4/q5/q6/q8, IQ, UD),
   BARO bakes (`*-BARO-<sha>.gguf`: one per commit; the newest per model is current), packs in `~/.cache/baro/<id>/packs/`,
   embedding models, vision encoders, TTS, image/video. Find exact twins with `fclones group ~/Models` (read-only mode).
3. Mark each file USED / UNUSED with evidence: grep mojo-baro (`bench/`, `tools/`, `face/`, `docs/`, `sha-cache.tsv`,
   `.lane-prep`), `~/iTools`, other `~/Projects/*` for its path or name. A name in a doc only = "cited", not "used".
   Superseded BARO bakes (an older sha of the same model when a newer one exists) are candidates; say which.
4. `~/Models/image.video` (189 GB exclusive): find the ComfyUI install and its workflows (start at
   `~/Projects/imports/comfy-workflows`, `~/iTools` comfy tools such as comfy-up / with-comfy, the comfyui-baro pack;
   locate `extra_model_paths.yaml`). Parse every workflow JSON the owner runs (pics and textures) for the model file
   names it loads (ckpt, unet, lora, vae, clip, controlnet, upscale). Mark each image.video file USED-BY <workflow> or UNUSED.

## Deliverable
`docs/lanes/2026-11-13-models-inventory-report.md` with:
- a table per family: path | apparent | exclusive | role (source / variant / bake / pack) | USED-BY or UNUSED | evidence
- a "candidates" table sorted by exclusive GB, each with the risk of deleting it (what breaks, what can be rebuilt and how)
- the image.video section by workflow
- totals: freeable if all UNUSED went, and if only superseded bakes and twin variants went
No em dashes. Commit only the report: `git commit -m "docs(lanes): models inventory report" -- <report path>`.
