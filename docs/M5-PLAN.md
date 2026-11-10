# Milestone 5 — make the self-contained engine fast

Handoff plan, written 2026-09-01 at the close of the M1–M4 session.
Read `docs/BASELINE.md`, `docs/ENGINE-ROADMAP.md`, `docs/qwen35-ssm-notes.md`
(incl. §7b/7c) and `~/Brain/mojo-baro/2026-09-01-decode-shape-sweep.md`
before touching anything. Board: `~/Brain/mojo-baro/whiteboard.md`.

## State you inherit (all verified 2026-09-01)

> **SUPERSEDED 2026-09-05 — read this box before trusting anything below.**
> The engine is q8-only and does **68.8 tok/s_gen** no-spec (`c3752e7`),
> **69.69** rebuilt from inside the gguf (`tools/gguf-closure.sh`, 64/64).
> MTP is landed and defaults to k=2: **100.7 median over the 20-prompt set**,
> 145.6 on the single race prompt. Work items 1, 4 and 5 below are DONE;
> item 2 is FALSIFIED. Only items 3 and 6 are still live.
> Current truth: `docs/BASELINE.md`, `README.md`, and the ranked plan in
> `~/Brain/mojo-baro/2026-09-05-engine-next-milestone.md`.

- `serve/engine.mojo` decodes Qwythos-9B **token-identical to llama.cpp**
  (16/16 greedy, prompt "The capital of France is") at **25.5 tok/s**.
- Regression gate (run after EVERY change, non-negotiable):
  `./.work/engine` output must equal `.work/engine-pack/ref-tokens.txt`
  (`prompt-tokens.txt` beside it). Engine pack = `.work/engine-pack/`
  (built FROM the BARO gguf; `-orig` sibling from the original file).
- Self-describing model: `~/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/
  Qwythos-9B-Claude-Mythos-5-1M-MTP-BF16-BARO-3824e20.gguf` embeds all kernel +
  engine sources (11 `baro.kernel.src.*` KVs). Closure verified 2026-09-11 (`3824e20`):
  `tools/gguf-closure.sh` builds shim + engine from the file's own sources,
  PASS 64/64 at 136.61 tok/s_gen. **Re-embed after any kernel/engine change**
  (`tools/gguf-embed.py`) or the model ships stale kernels — that bug already
  happened once, and the pre-`9b8a399` file sat 78 commits stale.
  **`serve/registry.mojo` MUST be in the file list**: `engine.mojo` does
  `from registry import *`, so omitting it makes the closure fail to compile.
  **Since 2026-09-08 the list is `$(tools/embed-files.py)`** — `serve/window.mojo`
  (the per-window body, what the loop may edit) + `serve/registry.mojo` + the
  kernel modules they import — and `serve/engine.mojo` (pack load, stopwatch,
  prints) is NOT embedded: `gguf-closure.sh`/`loop-gate.sh` take it from git at
  `baro.kernel.commit`, so a loop candidate cannot reach the clock
  (`exchange/scorer-integrity-report.md`, P-A). Legacy ggufs (engine.mojo with
  `main()` embedded) still build the old way.
- Key traps already paid for: gdn v-head h pairs with k-head **h % 16**
  (ggml_repeat tiles); q/k L2-norm not RMS; state decays BEFORE the delta
  error term; yarn ext_factor=1 validated; head_dim 256, contiguous q|gate.

## GPU sharing discipline (llama-server owns 22 GB VRAM)

- Server: PID via `pgrep -af '^~/llama.cpp'`, local port per
  `~/Brain/OS/local-service-ports.md` ("Local LLM (llama.cpp)" entry).
- The full restart command line is preserved byte-exact in
  `~/Brain/mojo-baro/llama-server-cmdline.txt` — restore from there only.
- Pattern: kill by exact PID (never pkill with a pattern containing
  'llama-server' — it self-matches your own shell; that bit twice), run
  engine, restart with setsid nohup, curl /health until ok. Always restore
  before ending a stint.

## Work items, in order

1. **[DONE 2026-09-01, verdict `6b99693`]** **Measure the bar**: GPU tok/s (its /metrics + a timed
   /completion, greedy, with and without MTP if feasible). Preregister the
   comparison protocol first (bench/coldcache-protocol.md style, frozen by
   commit) — every perf claim in this repo is preregistered or it is noise.
2. **[FALSIFIED 2026-09-04 — DO NOT ATTEMPT]** The round closed at stage 0
   (`bench/launch-fusion-protocol.md`): 646 launches/token, measured launch
   floor 2.57 us, so the whole ceiling is **6.9%** of a 24 ms token, not the
   ~2x claimed here. S0 fired before a fusion kernel was written; the ~20% gap
   to the HBM roof is NOT the launch floor. Original text follows, for record
   only. ~~**Launch-count reduction** (biggest lever, target ~2x)~~: the engine issues
   ~20 launches/layer at M=1. Fuse: reduce+cast pairs, gates_k into the
   alpha/beta reduce, amar_rmsnorm+cast, split/norm/rope chains. Consider one
   fused "layer prologue" and "layer epilogue" kernel. Keep each fusion
   behind the token-identity gate.
3. **[PARTLY IMPLEMENTED — audited 2026-09-05]** **Prefill batching**: the engine already batches prompt projections in chunks of up to 8, and `amar_attn_decode` uses `T=t_len+row` for causal attention within each chunk. Remaining work is a measured prefill throughput/TTFT improvement, including tiled attention for longer prompts. Registry `TMAX=128` sizes the current KV/token allocation; attention's `MAX_T=1024` is only score-buffer capacity. See [engine diagnostic audit](ENGINE-DIAGNOSTIC-AUDIT.md#5-prefill-already-uses-chunks-and-causal-attention).
4. **[DONE 2026-09-04 `c3752e7` — 41.7 -> 68.8 tok/s_gen, +65%, 64/64]** **q8b weight path** end-to-end (kernels exist, parity 9.2e-4): halves the
   bandwidth ceiling (~50 -> ~100 tok/s roof). Quantize the pack, add a
   `--q8` engine mode. Token identity may legitimately drift under quant —
   define and preregister an acceptance gate first (e.g. greedy match on
   >=90% tokens + a perplexity spot-check), never hand-wave it.
5. **[DONE 2026-09-04 `058dd28`/`41d0361`; k=2 is the default]** **MTP** (blk.32 nextn tensors, `nextn_predict_layers=1`): the model's
   built-in draft head; ~2x on top. Needs a llama.cpp source read for the
   eh_proj wiring (the sonnet research-agent pattern in the Brain notes
   worked well; file:line receipts mandatory).
6. **[STILL LIVE]** **Cold-cache GEMM efficiency** (54% vs vendor 85% of HBM peak):
   multi-column/thread + wider loads on the B-layout skinny;
   `bench/bench_coldcache.mojo` is the instrument.

## Repo rules that bind you (CLAUDE.md)

Kernel files carry zero comments/docstrings. Perf numbers only from
preregistered protocols. Commit per concern with why-bodies, no model
attribution lines. Findings -> `~/Brain/mojo-baro/` dated notes; board
updated same turn work completes. Results also -> `~/AMDHQ/runs/`.
