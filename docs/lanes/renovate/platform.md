# renovate: platform

Area: `cpu/ phone/ aio/ shim/ systemd/ scripts/ lab/ profiles/ skills/` (4,942 tracked lines in 123 files; `android-baro` is git-ignored and skipped).
Branch `renovate/platform`, base 2266fa9.

skills loaded: mojo-dev, memory-optimization, codebase-memory, toolscript. No GPU, no gpu-wait, no run-tests.sh.

The area is small and the 2026-11-15 dead-code round already went through it. The survey found no dead code and no code bug that a build or a CPU run proves. The two commits below are the real changes. The rest is leads for other areas.

## Bugs fixed

| file:line | repro (command + output before/after) | commit |
|---|---|---|
| `skills/mojo-baro-operate/SKILL.md`, `references/settings.md`: 37 `file:line` citations into README.md, docs/CAPABILITIES.md, serve/src/main.rs, tools/baro | Checked 26 against the tree at 2266fa9, 19 pointed at the wrong text: `sed -n 30,31p README.md` is the contents list, not Quick start; `sed -n 175p README.md` is the Ornith K-quant row, not the 92.97% row (that row is at 174); `sed -n 128p docs/CAPABILITIES.md` is the LAT1 HMAC entry, not `/api/pull` (129); `serve/src/main.rs` `health()` is at 490-510, not 484-529. After: `grep -E '(README.md\|CAPABILITIES.md\|main.rs\|tools/baro):[0-9]'` over both files is empty. Citations are now section titles, entry names and handler names. Two README Troubleshooting rows the skill table lacked (4+ HIP processes corrupt results; temperature-0 near-tie flips) were added. | 66a5c60 |

## Dead code removed

None. Proof of absence, so nobody repeats it:

| symbol or file | lines | proof | commit |
|---|---|---|---|
| (none) | 0 | `cpu/ops.mojo`: every function has a caller in `cpu/engine_cpu.mojo` (`git grep -w`); every import of `engine_cpu.mojo` is used (`grep -c -w` >= 2 each); `aio/` and `systemd/` are vendored copies (headers say so), their unused exports (`modify`, `remove`, `after`, `reloading`, `watchdog`) belong to the upstream API and bento/face use the rest; `lab/wavegroup/kernels`: every def has a caller or is passed as a comptime kernel argument (`embed_rows`, `cast16`, `epi` are only used inside `enqueue_function[...]`); `wg_rope_norm` is used only by `test_rope.mojo`, on purpose, it tests the shared `rot`; `phone/cl/q4_0_ours_*.cl` are picked by the `ours_*` glob in `race.sh` | - |

## Debloat

| what | bytes/LOC before -> after | commit |
|---|---|---|
| (none) | - | - |

## Optimisations

| what | number before -> after, command | commit |
|---|---|---|
| `cpu/engine_cpu.mojo` build warning, a docstring that began with a lower-case letter. Comment-only change, no speed claim | warnings 1 -> 0; binary sha256 c685bf22749e7633 before and after; `mojo build -I . -I kernels -I cpu -I serve cpu/engine_cpu.mojo` | e24a465 |

## Found, not fixed

| file:line | what | evidence | why not fixed / gate needed |
|---|---|---|---|
| `shim/tools/README.md:14-16` | The build example writes the binary to `/tmp/hipblaslt-tune` | the user rule keeps artifacts in `.work/`, not `/tmp` | Documentation example only, and the tool needs a ROCm box to run. Left as is |
| `shim/src/amarbaro_shim.cpp:194-199` | `hipEventCreate(&ev1)` failing after `ev0` succeeded leaks `ev0`; the 64 MB search workspace stays allocated after a failed search (`:176`, `:244-247`) | read only | No repro: needs a failing HIP call. Both are error paths of a tuning search. Needs a HIP fault-injection test, not done |
| `cpu/pack.mojo:31-48` | `Pack.__init__` raises after `mmap` succeeded (tensor past end of pack.bin) and `__deinit__` does not run on a failed init, so the mapping leaks | read only | The process exits right after the `FAIL pack` error. No repro that matters |
| `phone/ph.sh:12`, `phone/race.sh:11-13` | Hard-coded `~/Android/baro`, `~/Projects/mojo/wavewait`, `~/Projects/imports/llama.cpp`, an NDK 29.0.13113456 path | `ls -d` shows all exist on this box (`~/Android/baro` is a symlink to `~/Projects/mojo/baro`) | Phone scripts need the Xiaomi attached; not run. They are box-specific by design |
| `lab/wavegroup/` | Needs its own Mojo 1.3 nightly venv (`tools/setup.sh`) and a GPU; the root `.venv` is Mojo 1.2 | `lab/wavegroup/README.md`, `tools/setup.sh` | Not built here, so no edit to lab kernels. A lab kernel edit needs `tools/gate.sh` under gpu-wait |

## Cross-area leads

| file:line | what | proof |
|---|---|---|
| `tools/profile.py:44` (`for section, kinds in (("env", ("env", "server-env")), ...`) | The Python oracle rejects `MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT` in `[env]` of 4 profiles because the template kind is `external`, and prints "belongs under [env]" for a key that is under `[env]`. The Mojo port `tools/profile.mojo:57` accepts `external` and is what `tools/baro:163` runs. Fix: add `"external"` to the env tuple | `python3 -I tools/profile.py check` -> `FAIL profiles: 6 checked, 4 problem(s)`; the Mojo build (`mojo build tools/profile.mojo -I . -o ...`) then `profile check` -> `PASS profiles: 6 checked, 0 problem(s)`. Same 6 files, two verdicts, so the oracle drifted. One-line fix, area tools/ |
| `docs/operate-capabilities.toml` (source of `skills/mojo-baro-operate/references/capabilities.md`) | The same stale `file:line` citations (README.md:304-340, serve/src/main.rs:484-529, docs/CAPABILITIES.md:128 ...) are in the generated skill reference; `grep -o ... | wc -l` shows 43 of them in capabilities.md. They are wrong after any README edit | `grep -o -E '(README.md\|docs/CAPABILITIES.md\|serve/src/main.rs\|tools/baro):[0-9]' skills/mojo-baro-operate/references/capabilities.md` | Generated from the toml by `tools/operate-audit.mojo`; the fix is to cite section names in the toml and regenerate (area docs) |
| `README.md:326` | The endpoint table says `GET /props` reports "window, speculation depth, KV type". `props()` in `serve/src/main.rs:512` returns only `default_generation_settings.n_ctx`, `n_predict`, `model` and `chat_template_caps`; spec depth and KV type are not there | `sed -n 512,525p serve/src/main.rs` | README is area docs |

## Needs regen

| generated doc | reason |
|---|---|
| `skills/mojo-baro-operate/references/capabilities.md`, `docs/OPERATE-AUDIT.md`, `docs/operate-capabilities.toml` | Only if the cross-area lead on citation style is taken; this branch changes none of the generated inputs |

## Checks run

| command | result |
|---|---|
| `bash .work/renovate/setup.sh platform` | PASS |
| `b.sh ./.venv/bin/mojo build -I . -I kernels -I cpu -I serve cpu/engine_cpu.mojo -o .work/rv/engine_cpu` before and after the docstring edit | rc 0 both; warnings 1 -> 0; sha256 identical (c685bf22749e7633) |
| `b.sh ./.venv/bin/mojo build tools/profile.mojo -I . -o .work/rv/profile`, then `profile check` | build rc 0 (only vendored `toml/parser.mojo` warnings); `PASS profiles: 6 checked, 0 problem(s)` |
| `python3 -I tools/profile.py check` | `FAIL profiles: 6 checked, 4 problem(s)` (oracle drift, see cross-area leads) |
| `grep -E '(README.md\|CAPABILITIES.md\|main.rs\|tools/baro):[0-9]' skills/mojo-baro-operate/SKILL.md skills/mojo-baro-operate/references/settings.md` | no match |
| `git diff --shortstat 2266fa9..HEAD` before this report | 3 files changed, 33 insertions(+), 31 deletions(-) |
| `bench/check-cpu-engine.sh`, `check-cpu-layer.sh`, `check-cpu-parity.sh` (CPU engine gates) | NOT RUN: they need the GPU engine as reference. The edit changed no code (identical binary), so UNVERIFIED is not a risk here |
| `aio/` and `systemd/` against upstream mojo-libs | NOT RUN: `tools/ci-checks.sh` does it when the upstream is present; vendored, report only |
| phone scripts, shim build with HIP, lab build | NOT RUN: need the phone, a HIP toolchain or a GPU |

## Mistakes -> rules

- `mojo-baro-operate` skill (written 2026-11, the "cite the source" step): it cited README, CAPABILITIES and source by line number. CAPABILITIES.md is regenerated after every merge and the README moves with every edit, so the numbers were wrong within days -> cite section titles, entry names and handler names in skills; never a line number into a file that is generated or edited often.
- Tooling, worker run: a worktree-isolated agent's Bash is refused when `git` appears inside a `for` loop, a `$(...)` or after `&&` with other commands ("too complex to verify"). Use one plain `git grep -E 'a|b|c'` per call or the `read_many` / `sed_range` toolscript presets -> no git inside loops in worktree workers.
