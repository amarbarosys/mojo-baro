# Mojo front, round 1: the core path (2026-11-09)

Rust in mojo-baro is one crate, the HTTP front `serve/` (17 files, 9,622 LOC, 12 crates, 40
routes). Our Mojo libraries already cover its crates: `bento/` (HTTP/1.1 + SSE on `aio/`),
`json/`, `posix/`, `serve/tokenizer.mojo`, `minja/`. This round builds the core path as a new
binary beside Rust. Rust stays default and is not edited.

Decisions (the maintainer, 2026-11-09): core path first; libraries vendored like `uregex/`; library gaps
fixed upstream in the library, then re-vendored.

Read first: `serve/front/README.md` (the frozen contract). Audit input, in the MAIN checkout:
`exchange/fsm-audit/report-A.md` (Rust wire states), `report-B.md` (Mojo wire states).

## Skills are mandatory: load them before the first edit (the maintainer, 2026-11-09)

Load each with the Skill tool when it is listed for you; else Read the SKILL.md at the path.
The first section of your report lists every skill you loaded, by path. A report without that
list is not READY.

All lanes:
- `mojo-baro`: `~/Brain/Skills/own/mojo-baro/SKILL.md` + `rules.md`
- `mojo-dev`: `~/Brain/Skills/own/mojo-dev/SKILL.md` + `rules.md` #1 to #19, and through it
  `~/Projects/mojo/.claude/skills/mojo-syntax/SKILL.md`, `~/Brain/Skills/own/mojo-lean/SKILL.md`,
  `~/Brain/Skills/own/mojo-nightly-lane-builder/SKILL.md`
- `lanes`: `~/Brain/Skills/own/lanes/SKILL.md`, `rules.md` #1 to #16
- `memory-optimization`: `~/Brain/Skills/own/memory-optimization/SKILL.md` (name the hot path; bytes before and after in the commit body)
- `ops`: `~/Brain/Skills/own/ops/SKILL.md` (memory caps, background jobs, never `pgrep -f` your own pattern)
- `research`: `~/Brain/Skills/own/research/SKILL.md` (receipt before cause)
- `bench-gates`: `~/Brain/Skills/own/bench-gates/SKILL.md` (a gate that cannot fail is not a gate; every gate here has a mutant)
- `article-craft`: `~/Brain/Skills/own/article-craft/SKILL.md` (Read it before you write the report; a hook refuses prose files otherwise)

Per lane:
- front-text: `~/Brain/Skills/own/model-bringup/SKILL.md` (tokenizer defaults), `~/Brain/Skills/own/mojo-port/SKILL.md` (byte parity against an oracle), `~/Brain/Skills/own/mojo-tokenizers/SKILL.md`, `~/Projects/mojo/mojo-json/skills/mojo-json/SKILL.md`
- front-engine: `~/Projects/mojo/mojo-posix/skills/mojo-posix/SKILL.md`, `~/Projects/mojo/mojo-subprocess/skills/mojo-subprocess/SKILL.md`, `~/Brain/Skills/own/mojo-aio/SKILL.md`, `~/Projects/mojo/mojo-json/skills/mojo-json/SKILL.md`
- front-http: `~/Projects/mojo/mojo-bento/skills/mojo-bento/SKILL.md`, `~/Brain/Skills/own/mojo-aio/SKILL.md`, `~/Projects/mojo/mojo-json/skills/mojo-json/SKILL.md`

Code navigation: codebase-memory graph first (`search_graph`, `trace_path`, `get_code_snippet`,
project `home-user-Projects-mojo-mojo-baro`); grep and Read for literals and the lines you edit.

## Standing rules (all lanes)

- New files only, inside your Owns list. No edit to `serve/*.mojo`, `serve/src/*.rs`, `kernels/`,
  `bench/`, `tools/` or the vendored packages. A defect in a vendored library is a finding in
  your report (fixed upstream, then re-vendored), not a local patch.
- No GPU job, no `gpu-wait`, no subagents.
- Mojo 1.2.0: `./.venv/bin/mojo build FILE -I . -o .work/OUT`. Type every build and test command
  under `systemd-run --user --scope -p MemoryMax=10G -p MemorySwapMax=0`; never put that in a committed script.
- Tests are committed Mojo files (`serve/front/test_*.mojo`), short; each prints `PASS n/m` and exits non-zero on a failure.
- Commit each step with a pathspec: `git commit -m "..." -- <paths>`. Conventional subject, why
  in the body, numbers in the body. Never `git add -A`. `git diff --stat main..HEAD | grep -E '\.venv|\.work'` stays empty.
- A lane that builds Rust first runs `rm -f serve/target/release/baro-serve` (it is a link to main's binary).
- Three repairs on one failure, then `.work/<item>-failure.md` and BLOCKED.
- Gate = your test + `JOBS=2 bench/preflight.sh`, output to `.work/<item>-gate.txt` (exit code
  from `${PIPESTATUS[0]}`); small receipts copied to `exchange/receipts/front/` with `git add -f`.
- End READY (committed sha, gate green), PARTIAL or BLOCKED. Report
  `exchange/lane-<item>-report.md` in the MAIN checkout: VERDICT block, skills list, DELIBERATE
  DIFFERENCES, findings for upstream. Then send the reply line from your brief.

## Item front-text

Owns: `serve/front/text.mojo`, `serve/front/test_text.mojo`, `serve/src/text_oracle_test.rs`
(test-only; registered by one `#[cfg(test)] mod` line in `serve/src/main.rs`, the only edit allowed there).
Reads: `serve/src/text.rs`, `serve/tokenizer.mojo`, `minja/`, `serve/src/wire_test.rs`, `docs/TOKENIZER.md`.
Reuse: `serve/tokenizer.mojo` (the engine's own, not `mojo-tokenizers`), `minja/` (canonical here, gate `tools/minja-gate.sh`), `json.parse`.

Steps:
1. Oracle. `cargo test text_oracle` writes `exchange/receipts/front/text-oracle.jsonl`: for every
   P0 body of `wire_test.rs` and for the 20 gate prompts, the rendered prompt, the ids (add_special both ways) and the decode of the ids.
2. `Text` and `Detok` per the contract.
3. `test_text.mojo`: 100% equal to the oracle (prompt bytes, ids, decode). `Detok` pushed id by id
   equals `decode` on every row, including a row that splits a 4-byte codepoint. A mismatch is a
   finding with the first differing byte, never a tolerance. Mutant: flip `add_special`; the test must fail.

Pitfalls, at the step they hit:
- step 1: the fixture names the tokenizer file it used (path + sha256); the Mojo test refuses another file.
- step 2, template: Rust overrides `tojson` (`py_tojson`, `text.rs:40`: Python separators, no HTML escape) and enables `pycompat` and `loop_controls`. Render one tool-call body through `minja/` FIRST. A gap in `minja/` is a QUESTION: it is canonical here and the boss decides who edits it.
- step 2, time: `minja/render.mojo:46` takes `now` from `perf_counter_ns`, so `strftime_now` prints a 1970 date unless `MINJA_NOW` is set (report-C). Pin `MINJA_NOW` in the test and report it.
- step 2, files: `String.splitlines()` splits at tabs; use `.split("\n")`.
- step 3, floats: Mojo folds float literals at compile time; fixture values go through `json.parse`.

## Item front-engine

Owns: `serve/front/engine_client.mojo`, `serve/front/test_engine_client.mojo`, `serve/front/stub_engine.mojo`.
Reads: `serve/src/engine.rs`, `serve/src/protocol.rs`, `serve/serve_proto.mojo`, `serve/PROTOCOL.md`, report-A ("Wire states seen from Rust", M1 M2 M3 M5), report-B ("Wire states seen from Mojo").
Reuse: `posix` (spawn with env, signalfd, poll, non-blocking read and write), `json`.

Steps:
1. `Request.line()`: bytes equal to Rust `Request::line` for the vectors of the `protocol.rs` test `request_line_is_the_engine_shape` (copy the expected strings into the test).
2. `stub_engine.mojo` speaks the line protocol on stdin/stdout. Modes by argv: normal, log lines before ready, ready with a missing field, never ready, dies idle, dies mid-stream, drops the done line, slow.
3. `EngineClient` per the contract: states as Int tags, one `_set`.
4. Test every transition. Idle death is seen within one `poll`. Mid-stream death gives EV_DEAD with the exit code or signal in `dead_reason`. Submit when DEAD raises the named error. Cancel works. Shutdown returns in under 1 s on a healthy stub. 1000 spawn/shutdown cycles leave no fd (count of `/proc/self/fd` equal before and after) and no zombie. Mutant: the stub drops DONE; the client must give a named timeout, not hang.

Pitfalls:
- step 1: a log line is any line that is not JSON with a known key (`protocol.rs:297-319`); keep ONE classifier.
- step 3: declare no new libc externs (`external_call["open"]` and `["read"]` must match signatures already in the build): use `posix`. Importing a `_private` name from `posix` is a finding, not a workaround.
- step 3: a raw pointer does not keep a `List` alive (ASAP destruction): keep owners in the struct.
- step 3: a write to a dead child must not kill the front: check the write's return code and confirm SIGPIPE is off on the path you use.
- step 4: a struct created in a loop whose body loops over a call on it hangs the 1.2.0 front end. Put the body in a function; `timeout 20` on bisect builds.

## Item front-http

Owns: `serve/front/http.mojo`, `serve/front/api.mojo`, `serve/front/test_http.mojo`, `serve/front/fakes.mojo`.
Reads: `serve/src/main.rs` (only `health`, `models`, `props`, `tokenize`, `detokenize`, `completions`, `chat_completions`, `cancel`, `sse_stream`, `collect`, `ApiError`), `bento/server.mojo`, `bento/http.mojo`, `bento/sync.mojo`, report-A M1 M2 M3 M16.
Reuse: `bento.server.Server` and `Handler`, `bento.http.sse_frame` and `Response`, `json`.

Steps:
1. `api.mojo`: request parsing and response building for the routes (OpenAI shapes, `ApiError` shapes), pure functions over `Json`.
2. `fakes.mojo`: `FakeText`, `FakeEngine` with scripted events (normal, error before a token, error after a token, no terminal, slow).
3. `Front[T, E]` with the dispatcher; SSE; client disconnect sends cancel.
4. Oracle: capture the Rust front's responses for the same requests to `serve/front/testdata/http-oracle.jsonl`. If the Rust binary cannot run without a real engine, build the oracle from Rust unit tests that call the response builders, and say which rows are covered that way.
5. `test_http.mojo` over a real socket (`Server` on port 0). Per route and for `stream: true`: status, the headers a client reads, PARSED body equal to the oracle with `id` and `created` masked; SSE event sequence with `data: [DONE]` last. Errors: bad JSON, unknown model, engine DEAD, engine stream with no terminal, disconnect mid-stream (FakeEngine saw cancel). Mutant: omit `[DONE]`; the test must fail.

Pitfalls:
- step 3: a bento streaming handler holds one worker while it waits; size `serve(workers)` and state the number. bento has no idle-connection timeout: a finding for upstream.
- step 3: handlers run on many threads on one shared value; only the dispatcher touches the engine.
- steps 4 and 5: key order differs (Rust uses `preserve_order`): compare parsed values; for SSE chunks compare field sets.
- step 5: no fixed ports (8080 and 8090 are in use on this box): port 0, then `server.port()`.

## Out of scope: round 2

Checkpoints, fork, latent, ipc, turn, audio, vision, embeddings, ollama, the responses adapter,
neurons, web assets, metrics, `bin/router.rs`, multipart, TLS, and `serve/front/main.mojo` (the boss integrates).

## Verification: what must pass before the Mojo front is more than UNVERIFIED

Done by the boss after the merges, the way a user meets it:

1. `bench/preflight.sh` and `tools/ci-checks.sh` on main.
2. `baro-front` on a real pack through `gpu-wait run` (`env K=V` inside the command, `BARO_TMAX=16384`, `max_tokens` 16000); arm parameters read back from the running system.
3. `curl` to `/v1/chat/completions`, stream and non-stream. The same request to the Rust front, greedy: ids equal.
4. `tools/baro-serve-gate.sh` against the Mojo front.
5. Engine child killed mid-stream: a named error reaches the client and the front stays up.
6. Binary size, RSS and request overhead of both fronts on the 20 prompts, in the commit.

Any lane test passing against a stub proves the module, not the front. Step 3 is the falsifier.
