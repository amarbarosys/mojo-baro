# Model settings card: Qwen3.8-27B-OBLITERATED (2026-10-18)

Template: `~/iTools/llm/model-library/model-settings-card.md`. Every number cites a receipt in `exchange/receipts/open3/item1/`.
Same structural id as OrcaSAQ-2-27B (`qwen35-eec459db896e8bc6`); this card lists only what differs. Shape, flags, windows and caveats: `docs/models/orcasaq-2-27b.md`.

## Identity

- Model: mradermacher quant of OBLITERATUS/Qwen3.8-27B-OBLITERATED (abliterated, qwen35 hybrid: 48 SSM + 16 attention layers, dense FFN, MTP head)
- Model file and quant: `~/Models/qwen3.8-27b-obliterated-q4_K_M/Qwen3.8-27B-OBLITERATED.Q4_K_M.gguf`, Q4_K_M, 16.03 GB
- Model SHA256: a554c3727f11eb110542b5c4e2828a80f2085987d9a22e9da39b8f44b7970e83
- Architecture: `qwen35.block_count` 65 (64 layers + 1 NextN layer), `qwen35.nextn_predict_layers` 1, tensors `blk.64.nextn.{eh_proj,enorm,hnorm,shared_head_norm}` present (read with gguf-py from the file, not assumed)
- Runtime and version: mojo-baro at 36e440b, self-describing bake `Qwen3.8-27B-OBLITERATED.Q4_K_M-BARO-36e440b.gguf` (16.8 GB, reflink of the source file)
- Host and GPU: 7900 XTX gfx1100, 24 GB, 290 W cap, ROCm 7.2.4
- Date: 2026-10-18

## Load and memory

- Pack: `--q4`, 17.20 GB on disk, 14.66 GB on the device with `BARO_EMBD_HOST=1` (engine echo `device bytes: 14661914624`, `ref-run.log`)
- Context: f32 KV, 131,072 B/token (engine echo). `BARO_TMAX=16384` loads and serves at pool 2% (serve) and 10% (verify, closure default); `pack-fit` at TMAX 16384: FITS, headroom 5.56 GB.
- Peak VRAM: NOT MEASURED on this model (same shape and pack size as OrcaSAQ, 17.02 GB there)

## Best settings (what the bake carries)

Read back from the bake with `gguf-kv` (`bake-kv-readback.txt`), identical to the OrcaSAQ bake:

```
build flags  -D BARO_SEQ_CAP=1 -D BARO_SLOTS=4 -D BARO_DOT_UNROLL=2     (+ shape flags from tools/model-id.py)
run env      BARO_MEGA=1 BARO_SPEC=1 BARO_EMBD_HOST=1
pack         --q4 (tools/engine-pack.mojo)
serve        BARO_TMAX=16384 tools/baro serve Qwen3.8-27B-OBLITERATED.Q4_K_M-BARO-36e440b.gguf
```

`BARO_SPEC=1` is valid because the NextN head exists. The serve receipt shows it working: `drafted 34, accepted 28` on a 45-token reply (`response.json`).

## Prompt and generation

- Chat template: embedded in the GGUF; the OrcaSAQ template file `serve/templates/qwen35.jinja` was not tried on this model
- Thinking: the reply closes its reasoning with `</think>` and then answers (one request, `response.json`); tool calls UNVERIFIED
- Sampling: T>0 runs without speculation, as on OrcaSAQ

## Evidence

- Reference ids: 64 ids from the repo-built engine at 36e440b (`ref-ids.txt`, engine sha256 prefix in `engine-sha.txt`, run log `ref-run.log`)
- Verify: `tools/gguf-verify.sh FILE OUT` under `gpu-wait` with `BARO_TMAX=16384`: **PASS: 64 tokens match**, TMAX 16384 echoed, no NOT-RESIDENT (`gguf-verify-job.log`, `gguf-verify-run.log`)
- Serve: `tools/baro serve` with `BARO_TMAX=16384`, health `tmax 16384, spec_k 2`, one chat request with `max_tokens` 16000: `finish_reason stop`, 45 completion tokens, 73.3 tok/s_gen (one request, a receipt not a claim) (`response.json`, `health.json`, `serve-readback.txt`)
- Quality result: teacher-forced agreement with llama.cpp on the same GGUF, 20 prompts x 64 tokens, engine `engine-kv8-fix2`, window 16384: **91.41% median, 79.69% minimum**; control OrcaSAQ-2-27B in the same job 93.75 / 81.25; frozen bars 90.75 / 76.25 (`gate/`, `control/`, `bench/import3-protocol.md`)
- Known failures or caveats: the gate engine was `engine-kv8-fix2` (the OrcaSAQ serve engine); the bake's engine is rebuilt at 36e440b and the 20-prompt agreement was not repeated on it. The first gate run hit one NOT-RESIDENT in 60 runs (p13); the gate now reads the fail word (6c05cc6).
- Status: `PARTIAL` (bake verified and served; agreement vs llama.cpp measured once against the frozen bars; 20-prompt throughput, peak VRAM, tool calls UNVERIFIED)

## Resource pages

- Source model: https://huggingface.co/OBLITERATUS/Qwen3.8-27B-OBLITERATED, quant by mradermacher
- Local evidence: `bench/import3-protocol.md`, `~/Brain/mojo/mojo-baro/2026-10-18-import3-three-models.md`
