# Implicit state machines in mojo-baro (audit 2026-11-09)

**A dead engine in a pool of two is still the preferred target, and `/health` says `ok`.** That is
the worst of 49 implicit state machines found on `main` at 02b2ebc: state held in loose flags,
sentinels and call order, where no type names the phase and no wire field shows it. Five more
arrive with the unmerged lane `neuron2b` at a1a608a.

Method: three readers, one per area, graph first then source. I checked the top three findings
of each area against the cited lines (9 of 9 match the source). **Nothing was executed**: every
"reachable" below is a reading of the code, not a reproduction. Full records with carriers and
transitions are local, in `exchange/fsm-audit/report-{A,B,C}.md`.

## Status 2026-11-10: 46 rows have code on main; 24 symptoms were reproduced first (16 front and tools, 1 engine on CPU, 7 on the card)

Merged in b921f4d. The tables below are the audit as written; this section is what happened to it.

| Area | Rows | State |
|---|---|---|
| Rust front | A-M1 to A-M16 | code + `cargo test` (107 + 20 -> 135 + 23). 8 of 16 failed on the base commit first. A-M13 is partial (text between tool calls is still dropped); A-M14 is the `state_source` field only, the single resolver is not built |
| Engine | B-M1 to B-M7, B-M9, B-M10, B-M12 to B-M15, C6 | code + `serve/test_fsm_wire.mojo` (12 groups). `bench/fsm-wire-gate.sh` PASS, 25 checks: the pre-fix engine eb35717 showed the symptom and the merged engine did not, for B-M1 (window, done fields), B-M5 (resume, fork), B-M9, B-M12, B-M15 |
| Engine | B-M8, B-M11 | NO CHANGE: no reachable failure found |
| Tools, latentos | C1 to C5, C7 to C14, C16, C17, `strftime_now` | code + a test each (`tools/tests/`, `latentos/tests/`, `ncm/`, `miniagi/latent/`, `minja/tests/`) |
| Shim | C15 | code; its check compiles and has NOT run on the card |
| Not touched | C18, P5 | NO CHANGE by plan. P1 to P4 landed with `neuron2b` |

NOT EXERCISED by any run, so UNVERIFIED past a CPU test with injected values: a live non-zero `fail`
word (B-M1; no knob forces the barrier timeout), KV pool exhaustion (B-M7), `captured` (B-M13),
`grejected` on a served grammar (C6), B-M4, B-M10, the spark half of B-M14, and B-M2 (its gate probe
is an identity arm that holds on the pre-fix engine too, so it cannot fail). C8's boot actors compile
and were not run. The wire words are listed in `serve/PROTOCOL.md`, section State words.

## The decision: one tag, one writer, one word on the wire

The same defect repeats in all three areas. The state exists, but a client can only infer it.
Rust infers engine death from a pipe error. The engine reports an invalid-token run as
`finish:"length"`. `baro multi-serve` scrapes a human log line for a path. So one rule decides
every row below:

1. **A phase with 3+ carriers gets one Int or enum tag and one transition function** that logs
   `OLD -> NEW (why)`. Developers read one place.
2. **Every terminal state gets a word on the wire**: the ready line, the done line, `/health`, a
   `receipt:` line. Users see a named error, never a hang or a plausible wrong answer. Agents read
   a field, never prose.
3. **In timed code the fix is the wire word only.** No tag, no branch, no host sync inside the
   decode loop; the value is copied out where the done line is already built.

Fix codes in the tables: **1** tag + transition function, **2** guard with a named error at the
entry, **3** expose on the wire with no internal change, **4** leave as is.
Effort is LOC: XS < 20, S < 60, M < 150.

## Do first: nine fixes, about 190 LOC, none in a timed loop

| # | id | what a user or agent meets today | fix | LOC |
|---|---|---|---|---|
| 1 | B-M1 | a run whose grid barrier timed out (tokens invalid) ends `finish:"length"`; the only signal is a log line. A prompt extended past the window by `state_prefix` kills the engine with no `id` | 3: `"fail":N` on the done line; error line instead of `exit(2)` | XS |
| 2 | A-M1 | pool of 2, one dead: every other request is 503 while `/health` is `ok`; an engine that dies idle is unseen | 1, first step XS: `submit` skips dead engines | XS, then M |
| 3 | B-M5 | `latent_resume` with an unknown id kills the engine, no error line | 2: named error line, as the ipc ops do | XS |
| 4 | C3 | `baro multi-serve` fails with "prepared engine missing" on every run: its regex wants a path ending `/engine`, the cache writes `engine-k<commit>` | 3: `engine_path=` and `pack_path=` on the `receipt:` line | XS |
| 5 | A-M2 | a closed browser tab keeps the GPU decoding, up to the window, and every later request waits behind it | 2: cancel on drop | S |
| 6 | A-M3 | a stream that ends with no terminal event is returned as a successful partial answer | 2: named 502, error chunk before `[DONE]` | XS |
| 7 | B-M7 | KV pool exhaustion prints an error and then `finish:"cancelled"` for the same id | 1: `finish_reason` tag with ERROR | S |
| 8 | C1 | two gguf files with one basename share a work dir: `baro run` serves the first model for the second | 2: `model.sha` stamp, refuse on mismatch | S |
| 9 | C5 | `baro chat` stores a cut reply as a full turn; a `max_tokens` cut looks like a stop | 3: read and print `finish_reason` | XS |

Rows 2, 5 and 6 are already in the contract of the Mojo front (`serve/front/README.md`), so the
Rust fixes are only worth their LOC if the Rust front stays past round 2 of that work.

## Rust front `serve/src/` (16 machines)

`state.rs` and `turn.rs` hold no enum. `Event`, `EngineMsg`, `ApiError` are explicit already.

| id | machine | failure | fix | LOC |
|---|---|---|---|---|
| A-M1 | engine child lifecycle + pool liveness (`alive`, `queued`, `current`, `child`) | above; also stop waits 30 s (stdin never closed), `--audio-only` reports `engine_dead` | 1 | M |
| A-M2 | cancel and client disconnect | above; a queued request cannot be cancelled and the reply does not say why | 2 | S |
| A-M3 | "tokens, then exactly one terminal" contract | above; 400 vs 502 is inferred from "any token yet" | 2 | XS |
| A-M4 | `Gen` request kind, built by hand in 10 places | a new route that forgets the length guard kills the engine | 2: one check with `base_pos` | S |
| A-M5 | wire `Request` kind (15 Option fields) | two ops in one line dispatch silently | 2 | XS |
| A-M6 | checkpoint registry: memory entry, `.baro`, `.json` | sweep deletes a file between lookup and load: 400 instead of 410; orphans pile up | 2: lease + named 410 | M |
| A-M7 | scratch state files, 14 manual `remove_file` | early return leaves `lat1-*.baro` of hundreds of MB, uncounted by the disk cap | 2: RAII guard | S |
| A-M8 | IPC stage, cross-node fork phases | a failed release leaks the stage and drops the relay result | 2: guard releases on drop | S |
| A-M9 | whisper sidecar (child, `last_used`, reaper) | second concurrent call gets 502; a call longer than 300 s is killed mid-request | 1 | M |
| A-M10 | router health: 3 bools + hysteresis | 503 for 5 s after adding an engine; router `/health` always `ok` | 1 | S |
| A-M11 | router proxy phases; `SendState` decides nothing | load counted until headers only, so a streaming engine looks idle; workload list grows forever | 1 | S |
| A-M12 | router prefix single-flight | an upstream that never answers holds every waiter, no deadline | 2: wait timeout | XS |
| A-M13 | streaming tool-call parser, 5 flags | streaming returns only the first of several tool calls; a length cut leaves invalid JSON arguments | 1 | S |
| A-M14 | state-source ladder, written 4 times | an agent must infer which checkpoint source ran | 1 + `state_source` field | M |
| A-M15 | checkpoint tier guessed from `load_s < 0.002` | a guess is reported as fact | 3: engine sends the tier | XS |
| A-M16 | server startup | port closed for minutes during load and warm: "connection refused", not `starting` | 3: bind first | M |

## Mojo engine `serve/*.mojo` (15 machines)

Mojo here has no sum types with payloads: a tag is an Int with named constants. `engine.mojo`
and `window.mojo` are on the timed path and under the perf and identity gates.

| id | machine | failure | fix | LOC | timed |
|---|---|---|---|---|---|
| B-M1 | request finish + fail word | above; any `raise` in main is EOF for the client | 3 | XS | no |
| B-M2 | per-request knobs on shared buffers (steer, gain, implants) | control lines run with the previous request's knobs (candidate, not traced to the end) | 1: one reset function | S | no |
| B-M3 | path ladder: mega, mega_win, spec, prefill | the chosen path is not stated; `mega_win` does not exclude images | 1 + `"path"` on the done line | M | no |
| B-M4 | checkpoint `valid` / `pending`, written from 4 files | `latent_resume` leaves a valid item with a stale hash | 1: two functions in `prefix.mojo` | S | no |
| B-M5 | latent handle table | above | 2 | XS | no |
| B-M6 | grammar matcher phase | draws != accepted is a log line only | 3 | XS | no |
| B-M7 | PairSeq lifecycle (3 Bools) | above | 1 | S | host loop, no new branch |
| B-M8 | wstore slot residency | none found | 4 | 0 | yes |
| B-M9 | ExpertTier mode flags | a slow tier mode has no name on the wire | 3: `"tier"` on the ready line | XS | no |
| B-M10 | bell protocol | a host sync with doorbells pending hangs; guard is a docstring | 3: count and name hang-guard releases | XS | log off the loop |
| B-M11 | q4x `pending` / `pend_ffn` | none found | 4 | 0 | yes |
| B-M12 | idle and stale cancel lines | three engines, three behaviours | 2: one `is_cancel_line` | XS | no |
| B-M13 | capture session | a raise leaves ragged capture files | 3: `"captured":N` | XS | no |
| B-M14 | spark semantic input | persists into the next request when boot paths are set | 4: owner decides the intent | 0 | no |
| B-M15 | steer lens accumulators | a missing base raises inside the decode loop and kills the engine | 2 | XS | no |

The uncommitted edit in `serve/expert_tier.mojo` opens a second fd with `open64` that is never
closed: a candidate defect in the working tree, not on main.

### Pending merge: `neuron2b` at a1a608a (not on main)

| id | machine | failure | fix | LOC |
|---|---|---|---|---|
| P1 | splice (`spl_layer`, `spl_d`, 1024 rows) | a request longer than 1024 positions reads past the buffer; a shorter file after a longer one leaves old rows live | 2: refuse at admission, record the row count | S |
| P2 | implant sweep knobs | fields are ignored silently when no implants are loaded | joins B-M2 + error line | XS |
| P3 | lens foil and absolute NLL | `slens_foil` without a target prints a zero that reads as measured | joins B-M15 | XS |
| P4 | capture through batched prefill | a chunk over 1024 rows raises mid-request with files half written | 2: refuse at startup | XS |
| P5 | expert trace with routing weights | old and new layouts in one appended file, told apart by `gates=1` | 4: the reader checks the header | 0 |

The lane owner has P1 to P4 as pre-merge fixes (2026-11-09).

## latentos, grammar, templates, tools (18 machines)

| id | machine | failure | fix | LOC |
|---|---|---|---|---|
| C1 | `baro run` work dir has no model identity | above | 2 | S |
| C2 | `baro serve` cache markers | stale `tune.txt`; tokenizer rebuild fails on every serve after one failure | 3 (with C3) | S |
| C3 | multi-serve reads paths from a log | above | 3 | XS |
| C4 | `baro verify --append` | writes `mode=verified` when the harness check was skipped | 3: `harness_checked` | XS |
| C5 | `baro chat` reply completion | above; no read timeout | 3 | XS |
| C6 | grammar `Matcher` accept protocol | `window.mojo` counts attempts, drops the result: a rejected token is emitted and counted as accepted | 3: `rejected` counter | XS |
| C7 | agent boot ladder L0 to L7 | `READY=1` is sent after the manifest write failed | 3: `STATUS=` per stage | XS |
| C8 | S0/S1 boot actors | exit 0 with a failed mount and a "complete" last line | 3: one RESULT line, non-zero exit | XS |
| C9 | `LatentStore` handles | `put` revives an expired entry; `get` hands out an fd with no refcount | 2 | XS |
| C10 | `ReferenceStateStore` records | `release` False means two things | 2 | XS |
| C11 | `TransferSession` / `RelayStore` | "identity" error when the cause is "no manifest yet" | 2 | XS |
| C12 | memfd mint, seal, send | an unsealed fd can go out as immutable; 5 failures look alike | 2: read the seals | S |
| C13 | crypto init order | no production caller of `init_crypto` | 2: init inside the shim | XS |
| C14 | route decision flags | `wrong=True` on every route when no oracle is set | 2 | XS |
| C15 | hipBLASLt shim tune cache | a shape change re-tunes inside a timed arm, with no flag | 3: `tune_count` | XS |
| C16 | miniagi curriculum stage ladder | `rule` without `every` holds stage 0 for the whole run | 2 | XS |
| C17 | `ncm` mode by argc | a forgotten directory argument turns a batch run into a self-check | 2 | XS |
| C18 | minja dry-run depth | none today | 4 | 0 |

Also found: `minja/render.mojo:46` takes `now` from uptime, so `strftime_now` prints a 1970 date
unless `MINJA_NOW` is set.

## What this does not show

- No finding was reproduced by a run. B-M2 and B-M3 are marked candidates by their reader. The
  first step of any fix is the reproduction: symptom seen, then gone.
- Not read: `step_window` beyond the grammar lines, `serve_pair` and `serve_seqs` middles,
  most of `spark.mojo`, `ollama.rs`, the audio handlers, `miniagi/train/`. A machine there is not in this count.
- C15 and B-M7 touch code inside a timed arm. Their fixes add a counter in a branch that exists;
  that claim needs the same-job A/B and identity check before it lands.
