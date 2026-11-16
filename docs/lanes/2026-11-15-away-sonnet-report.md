# Report: away-run Sonnet lane (2026-11-15)

Branch `lane-away-sonnet`, worktree `../mojo-baro-lanes/away-sonnet`, base main 3ba6c8b. "DONE" means a check that
exercises the row the way a client meets it passed, and the same line names it. Everything else says UNVERIFIED and what
is missing. Arm read-back and receipts: `exchange/receipts/away-sonnet/`.

## Job 1: FSM rows

| Row | Status | Evidence |
|---|---|---|
| C15 shim `tune_count` | DONE: `shim-tune-count-check` ran on the card (gfx1100, under `gpu-wait`): counts 1, 1, 2, 3, PASS | `c15-shim-tune-count.txt` |
| A-M13 text between tool calls | DONE: the base parser returned `before` for `before / call / middle / call / after`; the fix returns all three pieces in the non-stream and the streaming parser; `cargo test` 145 + 24 pass | c8320d0 |
| B-M1 live fail word -> 502 | DONE: real engine + real `baro-serve`, `BARO_TEST_FAILWORD=1` gives HTTP 502 `engine_invalid_tokens` (fail word 1); the same engine without the knob gives 200; wire probe: done line `fail:1` (cand), `fail:0` (control, base) | `http502/`, `wire-live/`, a4e0f2c |
| B-M13 `captured` | DONE live: cand done line `captured>0` on a `capture` request; control (engine started without `BARO_CAPTURE`) and pre-fix base eb35717 show no field | `wire-live/verdict.txt` |
| C6 `grejected` on a served grammar | DONE live for the wire word: a `schema` request ends `grammar:done, gdraws 8, gaccepted 8, grejected 0`; control and base lack the field. UNVERIFIED: a non-zero `grejected` (no live knob makes the matcher reject a draw) | `wire-live/verdict.txt` |
| B-M7 KV pool exhaustion | DONE for mid-decode exhaustion: `BARO_SEQS=3 BARO_KV_POOL=1`, two n=16000 requests: cand error line `kv pool exhausted at pos 8191` then `finish:"error"`, engine alive (next request ok); pre-fix base: same error line then `finish:"cancelled"`. The brief says 429, but 429 is the admission refusal `kv pool full`, a different path, NOT EXERCISED; mid-decode exhaustion is a 502 | `wire-live/kvfull-*` |
| B-M4 | NOT TESTABLE as a symptom: `git show 4f05e45` only routes four hand-written writes of the same field values through `Checkpoint.install` / `set_valid`; no request can observe a difference. The CPU test stays the check | |
| B-M2 probe | NOT TESTABLE on a served request: the leak window is the lines that `continue` before the reset (control lines that decode nothing, the a3b/a3c batch modes); every served request resets in its own parse block | |
| B-M10 | NOT EXERCISED: the hang guard (`expert tier: HANG GUARD released the spinning kernels`) fires only after a doorbell or CPU-worker wait passes 5 s; a run needs the qwen35moe pack in tier mode plus a knob that withholds a doorbell, i.e. deliberately stalling a persistent kernel on the GPU that drives the desktop. Not done; the CPU test stays | |
| spark half of B-M14 | NOT EXERCISED: `serve/spark.mojo` now carries `semantic_req_loaded` (a request-loaded semantic input is dropped for the next request, boot-path input stays); a live check needs a pinned manifest, a values file and an adapter for the receiving model (digests, runtime and tokenizer hashes), and none exists on disk for any spark profile. The 0.5B spark engine built for job 2 would host it | |

How the live fail word works: the engine zeroes the megakernel counters per request; the test knob writes `fail=1`
right after, so the real barrier code sees a non-zero fail word and the run ends with invalid tokens. No kernel change.

## Job 2: Qwen2.5-0.5B token 0

DONE (the premise is stale; evidence below). Commit 7e08caf, receipts `exchange/receipts/away-sonnet/qwen05b/`.

- The engine does not emit token 0 at head. p01-water, 16 tokens: `279 9315 315 279 3720 518 9396 2188 374 220 16 15 15 12348 61347 13`
  (coherent text). The fix that ended the token-0 run is the q8 GEMV tail fix `2d1ebbe`, as BASELINE already said.
- Forced agreement vs llama.cpp, 20 prompts, `bench/dense-run.sh`, TMAX 16384, llama.cpp CPU f32 KV: **min 92.2 / mean 97.9**, 20 of 20
  prompts at or above the 90 bar (BASELINE row said 68.8 / 85.0 UNVERIFIED; the class bar was 82.8 / 91.4). Arm read-back from the log:
  `matrices: q8 97 kquant 24  act: float32`. The old row used bf16 activations; the f32-activation spark lane landed after it.
  Not bisected: which commit moved the number is UNVERIFIED.
- Per-layer comparison (`tools/layer-compare-oracle.py`): our residual after each of the 24 layers, last token of the 13-token prompt p01-water,
  against llama-eval-callback `l_out` of the fp16 GGUF, with llama.cpp's own q4_K_M as the noise control. Our error stays inside the control's
  band at every layer (for example layer 10: ours 0.211, control 0.171; layer 16: ours 0.299, control 0.398; layer 23: 0.078 vs 0.074).
  First divergent layer: none. Limits: six of 896 columns per layer, one prompt.
- No fix needed. BASELINE.md updated; a 0.5B bake is no longer blocked by the gate (not done: not asked).

## Job 3: `tools/q4x-gtt-trace.py` to Mojo

BLOCK. Boss answer (2026-11-15): neither escape; gates stay on, no iTools move. Gate text, verbatim, from the Write of `tools/q4x-gtt-trace.mojo`:

    BLOCKED (work_gates): q4x-gtt-trace.mojo is code in .../away-sonnet; do these first:
      - mojo-port: q4x-gtt-trace.mojo ports .../tools/q4x-gtt-trace.py. Port through the harness: mojo-port verify q4x-gtt-trace
        (cases in tests/cases.txt, Python as oracle) or mojo-port auto q4x-gtt-trace.
    Escape for one repo: touch .../.claude/gates-off and say why in the commit.

Why it cannot pass as the tree stands: every `mojo-port` verb takes an iTool name and works in `~/iTools/<cat>/<name>/` (tool.toml, `tests/cases.txt`);
`tools/q4x-gtt-trace.py` is a repo tool with no iTools folder, so `verify` has nothing to build. The two ways through are the escape (all work gates
off in the worktree) or an iTools folder; both were declined.

Done and committed (d3f4e90): the Python oracle has a deterministic test mode (`GTT_TRACE_ROOT` path prefix, `GTT_TRACE_FAKE_CLOCK`,
`GTT_TRACE_TICKS` = engine to the end then N samples, `GTT_TRACE_ENGINE_PID`; pid and fd lists sorted; live path unchanged, smoke-run on a fake
engine and the no-card FAIL line), plus `tools/tests/q4x-gtt-trace/mkfix.sh` (fake /sys + /proc tree, fake engine). The Mojo port was written
(single thread, `posix.spawn_pair` + `PollSet`, `jsondump.dumps_str`) but not saved and never compiled: the write was refused. To finish once the
harness path exists: save the port, run both on the fixture with identical env, `cmp` the JSONL and the exit codes, add the gate script.

## Job 4: step 0c byte test

DONE: `kernels/test_warp_sum_order.mojo` ran on the card (gfx1100, under `gpu-wait`) and printed PASS: `warp.sum` equals the xor 1, 2, 4, 8, 16
32-lane tree bit for bit, lane 0 and all 32 lanes, 4096 of 4096 waves. Differing bits: none. Controls (the test can fail): the xor 16..1 tree
differs on 2562 waves, the sequential fold on 2988, the halving tree on 2562 (highest differing bit 14). In `run-tests.sh`. Prediction was frozen in
the docstring before the run. Commit 36b5d8e, receipt `exchange/receipts/away-sonnet/warp-sum-order.txt`. The older
`kernels/test_cpu_dot.mojo` (one q4 row dot, CPU emulation vs GPU on real pack rows) was already in the tree and is not in `run-tests.sh`.

## Mistakes

- Sent `herd tell ... DONE job1-C15` before the report file existed. Rule: write the report line, then tell.
- A control arm of the capture probe kept `BARO_CAPTURE` set, so the control showed the feature (`captured` on every
  request): a control must drop the env that enables the feature, not only the request field.
- The grammar probe ran on a pack with no `tokenizer.json` and returned an error line, read as a missing field.
  Read the arm's log before reading the check name.
