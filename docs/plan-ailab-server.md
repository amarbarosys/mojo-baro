# Plan: mojo-baro server for street-sim AI lab episode 5 "Clone society" (2026-11-06)

Boss: street-sim lane `a lane pane`. Driver: Opus `a lane pane`. Team: two Sonnets (teamson).
Source of the requirement: `~/Projects/street-sim/docs/ailab.md`, "mojo-baro server" row and "Protocol".

## What exists (read 2026-11-06, do not rebuild)

The server mode landed today on `main` (`fbc65fc`..`61aa3a0`, lane baro-serve `a lane pane`, design
`docs/design/multi-session-serve.md`, report `briefs/2026-11-06-multi-session-serve-report.md`):

| requirement | state |
|---|---|
| `POST /v1/chat/completions` with `tools`, `tool_calls`, `tool_choice` | done (`fbc65fc`) |
| N sessions, one weight copy | done: `baro-serve --parallel N --kv-pool P`, engine `BARO_SEQS=N` (`-D BARO_SEQ_CAP>=N -D BARO_SLOTS=2`) |
| shared system prefix | done: prefix checkpoints, pages shared read-only, copy-on-write tail page |
| batched decode | done: groups of 8 rows, `amar_gemv_kq_rows` byte-equal to m = 1 |
| bounded memory, no mid-decode failure | done: admission waits for KV pages, 429 only past `--max-inflight` (64) |
| identity across N | done: 0 mismatches at equal restore point (`bench/ailab/parity.sh`) |
| throughput | N = 32 fresh states: 112.5 tok/s vs llama-server `--parallel 32` 42.5 (p95 9.8 s vs 41.7 s) |

## Gaps this plan closes

1. **The gates ran a stand-in prompt.** `bench/ailab/states/` uses `rules.txt` and a hand-made tool set with a
   required `reason` field. The real runner (street-sim's `ailab_run.py`, `ask_openai`) sends: system = `MANUAL`
   (`ailab_manual.manual()`: `data/ailab/guide.md` + key table, 7,190 bytes, identical for every clone), user =
   persona + `Now: {state json}` + handbook and diary tail, tools = `game_mcp.TOOLS` (minus `screenshot`,
   `backstory`) + `learn`, no `required`, `max_tokens` 200, thinking off, and a tool loop of up to 4 rounds
   (`state`/`actions` results go back as `role: tool` messages).
2. **Parity is against our own engine, not llama-server.** The brief asks for llama-server on the same gguf,
   greedy. Prior data: llama.cpp greedy answers were 53 to 60 tokens, ours 34. A prompt-render or tokenization
   difference is the first suspect, and it can be checked on the CPU.
3. **No start/stop under `gpu-wait`.** Nothing brings the server up as a queued job, warms the shared prefix,
   reads its arm back, and proves the GPU is released at stop (engine child gone, VRAM back to idle).
4. **N = 50 never ran.** Engine built at `SEQ_CAP` 32; VRAM at 32 slots was 17.1 GB.
5. **No step-level target.** Gates measured one request per body; a game step is up to 4 rounds per body, and
   the step ends when the slowest body acts.

## Phases

GPU marks: **CPU** = runs now. **GPU** = script and CPU dry run now, the timed run waits for the boss to lift
"NO GPU work" (guardrail of this dispatch) and then goes through `gpu-wait run --timeout S` only.

| # | item | files | check (the way a user meets it) | LOC | GPU |
|---|---|---|---|---|---|
| P0 | Runner request capture: call street-sim's own `ask_openai` with a fake `urlopen` and a fake bridge (canned `state`/`actions` replies), record every request body it posts, round 1 and later rounds. 20 parity states + 104 throughput states. Street-sim files are imported read-only, never edited. | `bench/ailab/capture.py` (oracle), `bench/ailab/states-run/`, `states-run-tp/` | the recorded body equals the bytes `ask_openai` passed to `urlopen` (assert in the capture); `md5` of the set recorded; a second capture is byte-identical | S | CPU |
| P1 | Prompt parity vs llama-server: render + tokenize every P0 request on both sides, compare token ids. llama-server side: `/apply-template` + `/tokenize` on a CPU server (`-ngl 0`). Ours: a render-only path (no engine) through `serve/src/text.rs` + the pack tokenizer. Fix any difference. | `bench/ailab/render-parity.sh`, render-only entry in `serve/` (bin or flag) | 20/20 round-1 and every later-round request: ids equal; on a difference, the first differing token and both strings printed | S (+M if the template needs a fix) | CPU |
| P2 | Runner wire compat: P0 bodies through our request parser and response builder. `model: "local"`, `tools` without `required`, assistant `tool_calls` + `role: tool` history, `enable_thinking: false`, response fields `ask_openai` reads (`choices[0].message.tool_calls[].function.name/arguments` as a JSON string, `id`). | `serve/src/` cargo tests | `cargo test` passes on all P0 bodies; a 429 can not reach a runner with N <= `--max-inflight` (the runner's HTTPError path would retry WITHOUT tools) | S | CPU |
| P3 | Lifecycle: `up` queues baro-serve as one `gpu-wait run --timeout S --vram V --label ailab-serve` job (knobs as `env K=V` inside the queued command), waits for `/health`, warms the `MANUAL` prefix with one request, reads the arm back from `/props` (engine path, pack sha, `BARO_SEQS`, TMAX, kv-pool, spec off) into `.work/ailab/serve.env`. `down`: SIGTERM, wait for the job to exit, then assert no engine child and VRAM within 100 MB of the pre-`up` idle (`gpu-wait gpu` parsed, never printed). | `bench/ailab/serve.sh up|down|status`; `serve/src/engine.rs` (PDEATHSIG on the engine child if the check fails) | CPU: with a stub engine, SIGTERM and SIGKILL of baro-serve leave no child after 5 s. GPU: up -> 20 asks -> down, 3 cycles, VRAM back each time | S | CPU + GPU |
| P4 | Answer parity vs llama-server: same gguf (`~/Models/ornith-1.5-9b-q4_K_M/Ornith-1.5-9B-Q4_K_M.gguf`), greedy, thinking off, `max_tokens` 16000, TMAX 20480, P0 requests. Per state: (a) same tool call (name + parsed args); (b) teacher-forced agreement of llama-server's completion ids through our engine (`BARO_FORCE`, single-session engine if forcing is refused in multi-session mode; the two are identical at equal restore point). Control arm: llama-server `--parallel 1` vs `--parallel 20` on the same set. Plus our own restore-point parity at N = 20 on the new set. | `bench/ailab/parity-llama.sh` | bar below, frozen by this commit | M | GPU |
| P5 | Step throughput: `bench/ailab/step.sh N` drives N threads of the runner's own `ask_openai` (P0 fakes for the bridge, real HTTP to the server), fresh state per body per step, `max_tokens` raised to 16000 in the body (repo floor; answers are 34 to 60 tokens, the cap does not bind). Step wall = slowest body. Arms: ours and llama-server `--parallel N`, same job, interleaved, N = 20, 32, 50, 5 steps each, median. | `bench/ailab/step.sh` | table below; each arm's knobs read back in its `arm.txt` | M | GPU |
| P6 | Scale and soak: build `-D BARO_SEQ_CAP=64`, `pack-fit` with measured used VRAM, f32 KV first; if 50 sessions do not fit, int8 KV (`BARO_KVQ=int8`) after confirming multi-session mode supports it (else that is a work item, M). Soak: N = 50, 200 steps through `step.sh`. | engine build flags, `bench/ailab/step.sh --soak` | 0 errors, 0 x 429, 0 x 5xx; VRAM drift <= 50 MB over the second half; p95 ask < 60 s (runner timeout 120 s) | S | GPU |
| P7 | Speed lever: batch the prefills of one burst (58% of engine time at N = 32, fresh states). Engine and kernel change: runs as a fable lane (standing rule for kernel work), wavewait capture, same-job A/B, byte or agreement gate per CLAUDE.md. NOT in the Sonnet dispatch. | `serve/engine.mojo`, `serve/rows.mojo` | same-job A/B on `step.sh 20` | L | GPU |

Total for the Sonnet team (P0 to P6): about M + 4 x S + 2 x M, under 600 LOC.

## Parity bar (P4), frozen before any run

- Prompt ids (P1): 20/20 equal, no exceptions. A difference is a defect, fixed before P4 runs.
- Tool call: ours vs llama-server same tool name on >= the control's count minus 1 (control = llama-server
  `--parallel 1` vs `--parallel 20`).
- Greedy prefix agreement (matching prefix of the completion ids over llama `--parallel 1`'s length), ours N = 20
  vs llama `--parallel 1`: mean >= control mean - 1 pp, and min >= control min - 3 pp. Control = llama
  `--parallel 20` vs `--parallel 1`, the SAME metric.
- Teacher-forced agreement of llama's ids through our engine is reported, not a bar term: llama-server cannot be
  teacher-forced, so no control on that metric exists.
- Added 2026-11-07 before the FULL run, after QUICK=2 (7 units: control prefix 25.37%, ours 22.49%, FAIL;
  ours forced 89.63%): a second REQUIRED term, so the bar only gets stricter. Forced control = llama
  `--parallel 20` scored at each position of llama `--parallel 1`'s ids (prompt + prefix, one token, argmax),
  the same per-position metric as ours. Pass needs ours forced mean >= control forced mean - 1 pp and min >=
  control forced min - 3 pp, AND the prefix term above. The prefix term stays as frozen. (Amended 2026-11-06 before any P4 run: the first draft of
  the gate compared our teacher-forced agreement with the control's greedy prefix agreement, two metrics; the
  prefix one is lower after any early flip, so that bar was easier for us than for the control.)
- Reference config (frozen 2026-11-07 before the FULL run, driver; bar unchanged): llama-server runs `-ctk f32 -ctv f32` with `-fa off`
  (under flash attention an f32 K/V cache is converted to f16 inside the kernel, ggml-cuda `fattn.cu` need_f16_K, so f32 KV needs
  `-fa off`), the `--parallel 20` control with one KV per slot (`--no-unified`; with `--kv-unified` the one-token control missed the
  prompt cache: CPU A/B 11 of 40 positions in 420 s against 40 in 175 s), bodies at temperature 0, ids/text/calls from ONE
  unconstrained `/completion` generation per unit. The server's own log is read back (`K (f32)`, flash_attn) into arm.txt.
  QUICK=2 with llama-server's defaults (f16 KV, `--kv-unified`, 7 units, EXPLORE, UNVERIFIED) is kept for the report: prefix ours
  14.22% vs control 40.85%, forced ours 88.57/80.37 vs control 93.74/89.29, tool name 2/7 vs 3/7. Why f32: ours is f32 KV and
  llama's own f16-KV config fails its own f32 reference at 5/7 lengths (repo CLAUDE.md, `bench/PROTOCOL-RULES.md` P14).
- Own identity at N = 20: 0 mismatches at equal restore point.
  Amended 2026-11-07 after P4c/P4d (c31d4d6): equal cached count is not equal checkpoint state; the state depends on
  which request created the checkpoint (arrival order). P4 FULL's own-identity term (14/59) compared different
  creators and is VOID; its llama terms stand. The identity gate is P4d's design: one request in flight, the same
  creator request pinned in both arms (single vs multi-session: 0/19 mismatches, `.work/ailab/p4d-c0/`).

## Throughput target (P5), frozen before any run

Prediction from the measured per-round burst wall (N = 16: 4.8 s, N = 32: 9.8 s) and 2 rounds per ask (104/104
greedy answers called `state` first in the old gate):

| N | llama-server step wall (predicted) | ours today (predicted) | target |
|---|---|---|---|
| 20 | 40 s | 12 s | **< 8 s** (needs P7) |
| 50 | 100 s | 30 s | < 20 s |

A miss means optimize further (P7 first), never drop. The first P5 run also records llama-server's real number;
the target row is not moved after it.

## Team split and rules

- Two Sonnet peers negotiate P0 to P6 between them (`briefs/2026-11-06-ailab-server-team.md`), agreed split in
  `briefs/2026-11-06-ailab-server-split.md`. P0 blocks P1, P2, P4, P5. P3 is disjoint from P0.
- GPU only through `gpu-wait run --timeout S`; until the boss lifts the guardrail, GPU phases stop at the CPU dry
  run (`gate-dryrun`) to the first GPU step.
- Commits by explicit pathspec only; no remote; no em dashes; every mistake becomes a line in `docs/rules.md`.
- The driver reruns every phase check before it is ticked.

## Runner notes for a lane pane (street-sim side, no edit here)

- Keep `--max-inflight` >= bodies: on any HTTP error `ask_openai` re-asks WITHOUT tools.
- With `tools` and `tool_choice` auto, baro-serve (and llama `/completion`) can emit a tool name that is not in
  the list (P4 s14-r3: `beg`); llama's chat path blocks it with a lazy grammar. The runner then calls the bridge
  with an unknown tool. Candidate server item: constrain the call name to the request's tool list (S).
- `ask_openai` sends no `temperature`: llama-server then samples at its default 0.8, baro-serve decodes greedy.
  The same cast gives different behaviour per backend. For clones, set `temperature` (and a per-body `seed`, so a
  run can be replayed) in the body; baro-serve samples per row in multi-session mode.
- 102 of 104 llama.cpp answers called `state` although the state is in the prompt: one round per ask is spent
  reading what the prompt already holds.
- Wall time of the run: the runner sleeps `step_seconds` of wall time per step. If a game day is 17,280 steps
  of 5 s (the `ailab.md` ask budget arithmetic), the world alone takes 24 h of wall time per game day, before
  any thinking time; 3 days at N = 20 and 8 s per step add up to about 110 h.
