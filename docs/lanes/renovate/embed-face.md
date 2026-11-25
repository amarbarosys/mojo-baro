# renovate: embed-face

skills loaded: mojo-dev, codebase-memory, memory-optimization (the graph covers this area poorly: `embed/` and `face/` are an excluded subtree, so every dead-code claim below rests on `git grep`, not on the graph).

Area: `embed/`, `face/`, `bento/`, `assets/`. `imkit/` is vendored v0.9.0 (report only). Three parts of the area are vendored and were not edited: `bento/` (identical to `mojo-libs/libs/bento/src/bento/` after the 6-line header, checked with `diff`), `embed/bge.mojo`, `embed/wordpiece.mojo` and `embed/oracle/bge_*.py` (`tools/ci-checks.sh` step "embed/" diffs them against newera `tools/`).

## Bugs fixed
| file:line | repro | commit |
|---|---|---|
| `face/presets.mojo:96` `is_bare` | A user preset value whose text `Float64()` accepts but TOML does not (`.5`, `+.5`, `infinity`, `NaN`, `Inf`) was written as a bare literal. `save_user_preset` then produced a `presets.toml` that `load_presets` refuses; `PresetsUi()` calls `load_presets` at face start, so the face could not start until the file was removed by hand. Repro (CPU, no GPU), `.work/rv/bare_repro` with key K: `'.5' bare=True FAIL parse: Unexpected token in value position at line 5, column 18` before. The new `presets_test` case ("non-TOML number text round trips") built against the old `presets.mojo` ends with `Unhandled exception caught during execution: Unexpected token in value position at line 11, column 18`; with the fix `PASS presets_test`. A float is now written bare only when its first character after an optional sign is a digit; other text is quoted and reads back as typed. Values that round-tripped before keep their form. | d95659a |

## Dead code removed
None. Candidate scan: every `def/fn/struct/comptime` of `embed/serve_embed.mojo` and `face/*.mojo` counted over the whole repo (`git grep -w`, `docs/` and `*.md` excluded, own definition line excluded, test and gate files counted separately). Result: no symbol without a use. Only `restart_keys`, `ApplyPlan.needs_restart` and `startup_staged` have test-only callers (see Found, not fixed).

## Debloat
None. See Found, not fixed for unused PNG icons (35.3 KB), not removed.

## Optimisations
None. `embed/serve_embed.mojo` `read_line` reads one byte per `read(2)`. Measured the real cost bound: 5 batches of 30 texts at 512 tokens = 630150 request bytes = 630k syscalls, about 0.2 s (estimate, UNVERIFIED as a measurement) of a 10.5 to 11.8 s run (`/usr/bin/time`, 3 runs, `.work/rv/runemb.sh`, machine shared with other build jobs). Below 5 percent: left alone.

## Found, not fixed
| file:line | what | evidence | why not fixed / gate needed |
|---|---|---|---|
| `face/app.mojo:134`, `face/presets.mojo:181` | `startup_staged` ("Balanced is the startup default", owner decision 2026-11-12, commit 3b888cb) is never called by the face. After `var pu = PresetsUi()` nothing stages the default. | `git grep -nw startup_staged` shows only the definition and `presets_test.mojo`; `docs/settings.toml:2937` has a `Balanced` preset. | Changes what the face shows at start (every Balanced key marked staged, Apply enabled). Needs a look at the offscreen gates (`face/settings_gate.sh`, OCR) and the owner. Fix: `_ = startup_staged(pu.presets, st.staged)` after the two constructors. |
| `face/settings.mojo:161`, `face/chat.mojo:223` | `Settings.sampling_json` has no caller outside `settings_test.mojo`. The Apply dialog tells the user that `sampling.*` changes "apply to the next request", but `Chat.submit` builds its body from `stream`, `max_tokens` (fixed 16000) and `messages` only, so the Sampling tab changes nothing in the chat. | `git grep -n sampling_json` and `grep sampling face/chat.mojo face/app.mojo` (no hit). | Design call: `sampling.max_tokens` in `docs/settings.toml` would duplicate the `max_tokens` key that carries the 16K floor (`bench/min16k.sh`). Needs the owner to say which wins, then a `chat_test` case on the request body (CPU only). |
| `face/chat.mojo:235` | `submit()` appends an empty assistant `Msg` before the connection attempt. After a refused connection, a failed send, Stop before the first token, or an error event, the empty assistant turn stays in `msgs`, and the next request carries `{"role":"assistant","content":""}` in its history. | Read from code; not run against a front. UNVERIFIED whether the front or the chat template refuses it. | Behaviour of the front on empty assistant content is unknown; needs one request on a live front (small GPU job) or a CPU check of the front's message validation. Candidate fix: skip empty assistant turns when building the body. |
| `face/dash.mojo:232-240` | The spark line divides by `hi`, the maximum of `d.hist`. If every kept run reports `tok_s_gen` 0, `hi` is 0 and the line gets NaN coordinates. | Read from code; no repro (no window). | Cosmetic and rare; needs a frame check. |
| `face/presets_ui.mojo:122-127` and `face/presets_ui.mojo:48-54` | The preset reload (`load_presets("docs/settings.toml")` plus user presets) is written twice: in `PresetsUi.__init__` and in the Save handler. | Read from code. | A helper would save about 8 lines; the GUI cannot be run here, so not changed. |
| `assets/icon/mojo-baro-{16,32,48,64,128,256,512}.png` | Seven PNGs, 35.3 KB in total, are referenced by nothing: `scripts/install-desktop.sh` installs only the SVG and the `.desktop` file. | `git grep -e 'mojo-baro-[0-9]' -e 'icon/' -e hicolor` over the whole repo (receipts and `exchange/` excluded) hits only `scripts/install-desktop.sh`. | They are a ready icon set (packaging, README art may use them outside the repo); deleting is the owner's call. |
| `face/apply.mojo:150,154`, `embed/serve_embed.mojo:2,95,96`, `face/presets.mojo` (old line 109, fixed) | Compiler deprecation warnings (`as_c_string_slice`, `unsafe_ptr`, positional `__getitem__`, docstring style). | `.work/rv/build_emb0.log`, `.work/rv/build1.log`. | Cosmetic; a toolchain bump is the time to change them. |
| `embed/serve_embed.mojo:19-31` | One `read(2)` per input byte (see Optimisations). | 630k syscalls for a 630 KB request. | Below 5 percent of the run; a buffered reader would also change a file that the `embed-gate` mutant script copies. |

## Cross-area leads
| file:line | what | proof |
|---|---|---|
| `bento/server.mojo:13,16,17` | Imports `alloc`, `PollSet`, `POLLIN` and `Parsed` are never used in the file (vendored copy; upstream is `mojo-libs/libs/bento/src/bento/server.mojo`, identical after the header). | A word count of each name in the file is 1 (the import line). Fix belongs upstream in mojo-libs, then a hand sync. |
| `bento/server.mojo:23` | `comptime _MutBytes` has no use. | `git grep -w _MutBytes` over the repo: one hit, the definition. Same upstream rule. |

## Needs regen
None. No env name, kernel file or setting was added or removed.

## Checks run
| command | result |
|---|---|
| `bash .work/renovate/setup.sh embed-face` | PASS |
| `b.sh mojo build -I . -I face face/presets_test.mojo` then run, before the fix (baseline) | build 0, `PASS presets_test` |
| same test with the new case, built against `presets.mojo` from HEAD (copy in `.work/rv/old`) | FAIL: `Unexpected token in value position at line 11, column 18` |
| `b.sh mojo build -I . -I face face/presets_test.mojo` then run, with the fix | build 0, `PASS presets_test: fixture, apply, user round trip, plan text, verdict` |
| `b.sh mojo build -I . -I face face/app.mojo -o .work/rv/app` (imports `presets.mojo` through `presets_ui`) | build 0, binary 1.0 MB (compile proof only; the window was not opened) |
| `b.sh mojo build embed/serve_embed.mojo -I embed` | build 0 (baseline for the read cost; no embed file changed) |
| `.work/rv/runemb.sh` x3 (5 batches x 30 texts x 512 tokens) | 11.82 s, 10.47 s, 11.04 s wall, 282 MB RSS |

## Mistakes -> rules
- Plain Read/Bash calls in a row were denied by `toolscript_gate` (2nd plain read); `mcp__hub__run_js` has only `tools.read/grep/glob/sh` (read-only, no `require`, no bare `sh`). Rule: batch reads in one `run_js` with `tools.read(path)` and `tools.grep(pattern, {path, output_mode, '-n'})`; use Bash for git and builds.
- The worktree sandbox refuses compound shell lines that contain `git` together with pipes, loops or heredocs. Rule: put loops in a script file under the scratchpad and run it with `bash <file>`; keep `git` lines plain.
- Skill `mojo-dev`, step "building": I first wrote the new test and the fix together and only afterwards built the old `presets.mojo` for the failing arm. Rule: run the test against the old source before touching the source (a `git show HEAD:<file>` copy in `.work/rv/old` costs one build).
