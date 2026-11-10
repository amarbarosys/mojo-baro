# lane-turbobake report

Brief `.work/briefs/turbobake.md` (primary checkout). Self-describing bake of the Qwen3.8-27B TurboFCFusion NEO-CODER MAX MTP Q4_K_M gguf.

Verdict: **PASS.** `tools/baro verify` on the baked file passed (64 tokens match, TMAX 16384), and `tools/baro run` answered one real prompt with text.

## The file

- Source: `~/Models/qwen3.8-27b-turbo-fcf-mtp-q4_K_M/Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf`, sha256 `bc7a6cf2bcc78d1190aaf04d1ab1c5cb845b6ff23aa0e7d24fe0d2ea6d3a7c7c`, not touched (mtime 2026-10-30).
- Bake: same directory, `...-Q4_K_M-BARO-c71dbdc.gguf`, 18,500,060,704 bytes, sha256 `57a7bf8249da85a3cea8bea985b0647090ca3c710ea3a148c6c7212c8bef5660`. btrfs reflink of the source (shared 17.22 GiB, `btrfs filesystem du`).
- Chat template unchanged: `tokenizer.chat_template` kv hash `661a75222e17b829` in source and bake.
- Settings, taken from the OBLITERATED bake: arch qwythos (qwen35 trunk, id `qwen35-eec459db896e8bc6`), pack `--q4`, build flags `-D BARO_SEQ_CAP=1 -D BARO_SLOTS=4 -D BARO_DOT_UNROLL=2`, run env `BARO_MEGA=1 BARO_SPEC=1 BARO_EMBD_HOST=1`. The gguf has the NextN head (`drafted 228 accepted 179`).
- Reference ids: 64 ids, p01-water (`bench/mtp-prompts/p01-water.tokens`), from the repo-built engine at 81a50f7 (sha256 prefix `85aae0d4bbdf9340`), TMAX 16384, 49.6 tok/s_gen. serve/ and kernels/ did not change between 81a50f7 and c71dbdc, so the ids hold for the baked commit.

## Checks (the way a user meets it)

| check | result |
|---|---|
| pack-fit, `--tmax 16384 --kv-bytes 131072 --pool-pct 10`, pack 17.20 GB | FITS, headroom 1.52 GB (device bytes 14.66 GB with `BARO_EMBD_HOST=1`) |
| `env BARO_TMAX=16384 tools/baro verify BAKE` under `gpu-wait run` | **PASS: 64 tokens match**, rebuilt engine 66.0 tok/s_gen (one prompt, a receipt not a bar) |
| `env BARO_TMAX=16384 tools/baro run BAKE` under `gpu-wait run`, chat request "In three sentences, explain why the sky is blue.", `max_tokens` 16000 | finish `stop`, 293 completion tokens, 4.72 s wall, text answer, engine **66.3 tok/s_gen** (62.3 total), health `tmax 16384, spec_k 2` |

tok/s numbers are one prompt each: receipts, not claims. No 20-prompt throughput was run, so `baro.hw.*` is empty in the file.
Receipts: `.work/turbobake/` (verify-job.log, serve-job.log, response.json, health.json, serve.log, ref-run.log, ref-ids.txt, bake-sha.txt).

## Arm labels of .work/turbo27 (resolved)

`agree.log` prints the control block first, then the Turbo block. `arm.txt` in each folder names the gguf, and the last row of `results.txt` matches (p20 57/64 in `ctl`, 59/64 in `new`):

- `ctl/` = OrcaSAQ-2-27B control (pack `.work/bake27/verify3/pack`): **median 93.75, min 79.69**.
- `new/` = this Turbo gguf (pack `bc7a6cf2...`, the same pack the reference ids used): **median 96.88, min 85.94**.

The brief had these reversed. Turbo is above the control. Both arms ran engine `engine-12` (sha `d3b33abd97151afb`), not the bake's engine; the agreement was not repeated on the c71dbdc rebuild.

## Two failed attempts, one tool bug

1. First bake at 81a50f7: `tools/baro verify` failed in the closure (`unable to locate module 'shard_index'`). Cause: `kernels/gemv_kq.mojo` imports `q4x_kernels`, so `q4x_kernels.mojo` is now in the file list of every qwen35 bake, and `tools/gguf-closure.sh` took the qwen4exp branch. Fixed in `c71dbdc` (gate the branch on `window.mojo` being absent, as the moe branch does); rebaked at c71dbdc, which passed. Every qwen35 bake made since the kq lane landed would have failed the same way.
2. `tools/baro run` serves with no tokenizer by design (the tokenizer step is in `serve` mode), so the first request returned 503 `no tokenizer.json loaded`. Not a bake fault. I built `tools/gguf-tokenizer-json.mojo` and wrote `tokenizer.json` into the run pack, then the request returned text. Open: `run` mode could do this step itself.

GPU time: about 6 minutes in three `gpu-wait` jobs (reference ids, verify, run), estimated from the job wall times, not metered. Limit was 15.

## run-tests

run-tests: `RUN_ALL=1 ./run-tests.sh` under `gpu-wait run --timeout 5400` (timeout taken from the moeprefill lane's full run, which finished under it; this tree has no pass-cache markers, so every test ran), rc=0, log .work/turbobake/run-tests.log (last line rc=0, before it "PASS run-tests: all tests, pack ~/Projects/mojo/mojo-baro/.work/engine-pack-q4", 137 PASS lines, "FAIL: 0 (none)", 0 SKIP). Job held the card 381 s (gpu-wait wall), 479 s from launch including the queue wait behind the turbokernel lane. State engine `.work/engine` built from this tree at c71dbdc (9B shape, default flags); kq pack and dense pack from the primary checkout. The bake itself is not part of run-tests: its checks are the table above.

Not done: 20-prompt throughput, peak VRAM, tool calls, agreement on the baked commit's engine, a `baro.hw.*` receipt.
