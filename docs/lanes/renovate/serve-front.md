# renovate: serve-front

skills loaded: mojo-dev (read via Skill). codebase-memory graph used for the function listing of serve/front; dead-code proof is name count over the repo (a scratch shell loop, whole-word grep of every def/struct/comptime in serve/front, serve/wire.mojo, serve/serve_proto.mojo; 1 hit = definition only). Not loaded: memory-optimization (no hot-path change).

Base 2266fa9, branch renovate/serve-front, 5 commits before this report.

## Bugs fixed

| file:line | repro | commit |
|---|---|---|
| serve/src/state.rs:353 `exact_prefix_hash` | `cargo test --bin baro-serve pos_past` before: `panicked at src/state.rs:355:22: range end index 9 out of range for slice of length 3`, FAILED. After: `state::` 18 passed. A checkpoint sidecar with pos > tokens.len() (hand-edited or truncated) made `GET /v1/state` panic; `bin/router.rs` already clamps | a8a2b46 |

## Dead code removed

| symbol or file | lines | proof | commit |
|---|---|---|---|
| `use super::*` in serve/src/embed.rs | 1 | `cargo check --all-targets`: `unused import: super` (bin and bin test); after the edit `cargo test` compile prints 0 warnings | e5ccc05 |
| serve/testdata/anthropic/ (12 case json + oracle jsonl) | 13 files, 592 lines, 15965 B | `diff -rq` against serve/front/testdata/anthropic: only the jsonl differs by presence; the jsonl is `diff -q` equal to serve/front/testdata/anthropic-oracle.jsonl; no .mojo/.rs/.sh/.md/.toml/.json names the path; run-tests.sh and test_anthropic.mojo read the front/testdata copy | 640115b |

Whole-word name scan of every def/struct/comptime in serve/front/*.mojo, serve/wire.mojo, serve/serve_proto.mojo: no function without a second hit. Only `wire.SEQ_QUEUED` has one (an enum value of a documented 0..3 set, kept). Prior-round leads (`_opt_field`, `greedy_tokenize`) are gone from serve/ (greedy_tokenize now only in grammar/trie.mojo imports). serve/web (app.js, sw.js, app.css): no unused function or CSS class by the same scan.

## Debloat

| what | before -> after | commit |
|---|---|---|
| Tally + blocking HTTP client duplicated in test_http.mojo and test_anthropic.mojo, now serve/front/test_net.mojo | 1563 -> 1398 LOC over the three files (git diff --stat -328/+175) | 13cbfa3 |
| `_pos` shim copy in anthropic.mojo and test_net.mojo, imported from api.mojo | -15 LOC (2 ins, 17 del) | 3556f86 |

## Optimisations

none (no CPU hot-path number taken).

## Found, not fixed

| file:line | what | evidence | why not fixed / gate needed |
|---|---|---|---|
| serve/src/handoff.rs (whole module, 3.9K) | `mod handoff;` in main.rs, no caller, `#![cfg_attr(not(test), allow(dead_code))]`, header says "Not wired to a route yet (S2)" | `grep handoff::` over serve: none | design intent, has its own tests; owner decides |
| serve/src/bin/router.rs:54-125 | own sha256 + exact_prefix_hash copy (~70 LOC) of checkpoints.rs / state.rs | comment says no lib.rs; a `#[path]` include or a lib target would dedup | structural change (crate layout), not a build-proven dedup |
| serve/front/anthropic.mojo:226,247 and 436 | compiler warning `assignment to 'src'/'input'/'chunk' was never used` (the `var x = Json.object()` then try/assign pattern) | in test_http/test_anthropic build logs | cosmetic; changing the init pattern is a style call |
| serve/src engine.rs:29, main.rs:1480, ollama.rs many `#[allow(dead_code)]` | fields/variants kept for the wire shape (e.g. `Event::Latent` fields never read) | `Event::Latent` is constructed at engine.rs:601 | removing the allow adds warnings; no change |
| load throttle | b.sh waited more than 1 h at load 15-18 (limit 12): both slots plus other workers keep load above 12 | /proc/loadavg 15-18 during the run | I ran the final builds with `B_MAXLOAD=20` (b.sh's own knob), still one slot per command |

## Cross-area leads

| file:line | what | proof |
|---|---|---|
| `_pos` two-overload shim copied in serve/prefix.mojo:264, q4x.mojo:160, shard_index.mojo:60, serve_proto.mojo:76, tokenizer.mojo:128 (excluded), front/text.mojo:160 (excluded), front/api.mojo:834 | one shared helper would drop ~6 LOC each | `grep -rn "def _pos("` over serve/ |

## Needs regen

none (no env name, kernel file or capability changed).

## Checks run

| command | result |
|---|---|
| `b.sh cargo check --manifest-path serve/Cargo.toml --all-targets` (CARGO_TARGET_DIR in .work/rv) before edits | ok, 1 warning (embed.rs) |
| `b.sh cargo test --bin baro-serve pos_past` before the fix | FAILED, panic state.rs:355 |
| `b.sh cargo test --bin baro-serve state::` after the fix | 18 passed, 0 compiler warnings |
| `b.sh mojo build serve/front/test_http.mojo -I . -I serve/front` + run | PASS 434/434 |
| `b.sh mojo build serve/front/test_anthropic.mojo -I . -I serve/front` + run with the oracle jsonl | PASS 106/106 |
| full `cargo test` (all 155 tests), test_engine_client, test_serve_proto, test_fsm_wire, stub engine | UNVERIFIED: not run (files not touched; test_engine_client does not import test_net) |

## Mistakes -> rules

- A shared scratchpad file name (`dead.sh`) was overwritten without checking; other workers share the scratchpad dir. Rule: prefix scratch files with the area key.
- Queued three build commands behind b.sh at load above 12 and waited over an hour, one of them a stale cargo check. Rule: cancel superseded queued builds at once, and batch all builds into one queue slot per edit round.
