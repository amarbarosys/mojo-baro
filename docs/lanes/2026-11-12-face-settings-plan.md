# Plan: face settings tabs, presets, apply (2026-11-12)

The face (`tools/baro gui`, `face/`) gets tabs for every setting, each with a one-line description and a full help
panel, basic presets, user presets, and an Apply flow for settings that need an engine restart. Owner decisions
2026-11-12 (question dialogs): all four groups (Sampling, Engine, Model + pack, Front + system); inline line AND help
panel; presets Fast/Balanced/Quality, Chat/Code/Reasoning, Low VRAM/Long context, user-saved; restart via Apply
button + confirm; Basic view with an Advanced toggle; one `docs/settings.toml` as the single source; lanes now,
CPU-only work first, GPU gates later (the card is lent to other repos).

## Shared contract (all three lanes build against this; F3 owns the file)
`docs/settings.toml`, one table per setting:

```toml
[[setting]]
key = "BARO_TMAX"              # env knob, or "sampling.temperature" for per-request fields
group = "engine"               # sampling | engine | model | front
label = "Context window"
type = "int"                   # int | float | bool | enum | path | string
default = 16384
min = 16384                    # the 16K floor (CLAUDE.md) is the min for context and max_tokens
max = 262144
choices = []                   # enum only
restart = true                 # engine restart needed
level = "basic"                # basic | advanced
status = "stable"              # stable | experimental | bench-only
summary = "Longest prompt + answer the engine keeps, in tokens."
help = """How to use, when to change it, what it costs (VRAM per token, speed), measured effect with a receipt path."""
```

`[[preset]]` tables: `name`, `group` ("speed" | "task" | "fit" | "user"), `summary`, `values = { KEY = value, ... }`.
User presets live in `~/.config/mojo-baro/presets.toml`, same shape.

## Standing rules (every lane)
- Load skills: `.claude/skills/mojo-baro`, `mojo-dev`, `lanes`; list them in the report.
- imkit is vendored read-only; a missing widget (tabs, slider, dropdown, tooltip) is built in `face/` on imkit
  primitives, and named as a QUESTION for the imkit owner in the report.
- Display isolation: render offscreen to PNG, OCR gate; every gate under `systemd-run --user --scope -p MemoryMax=8G`.
- CPU first. GPU only through `gpu-wait run --shared --vram 1`, and only after the owner frees the card; until then
  GPU steps are listed as NOT RUN in the report.
- Help text facts come from the code and docs (`serve/`, README, `docs/`, `bench/` receipts), never from memory; a
  claim with no source says "unmeasured".
- Commit each step with a pathspec; no co-author tag; no em dashes. Report `docs/lanes/2026-11-12-face-ITEM-report.md`
  committed on the branch, with remaining steps a cold session can follow. Cited receipts copied to
  `docs/receipts/face-ITEM/`.

## F3. Settings schema (owns `docs/settings.toml`, `tools/settings-check.*`)
1. Inventory every `getenv("BARO_` in `serve/`, `kernels/`, `tools/baro`, plus the request fields the front accepts
   (`serve/PROTOCOL.md`, OpenAI fields in the front). Fill `docs/settings.toml` (all knobs, level, status, summary, help,
   sources) and the presets (Fast/Balanced/Quality, Chat/Code/Reasoning, Low VRAM/Long context).
2. Drift check in Mojo (`tools/settings-check.mojo`), wired into `tools/ci-checks.sh`: fails if a knob in code is
   missing from the TOML or a TOML key is gone from code; falsifier: a deleted entry must fail.
3. Generate the README "Configuration" table from the TOML (same tool, `--readme`), checked by ci.

## F4. Tabs + help (owns `face/settings*.mojo`, `face/widgets*.mojo`; one mount call in `face/app.mojo`)
Tab bar (Chat, Sampling, Engine, Model, Front, Presets), controls by type (slider/number, toggle, dropdown, path
field), inline summary under each control, help panel on selection (summary, help, default, range, restart, status),
Basic/Advanced toggle, staged changes marked. Read `docs/settings.toml` at start (use a 5-entry sample until F3 lands;
switch to the real file after F3 merges). Sampling values flow into F1's chat request. Gate: offscreen frames per tab,
OCR reads labels and one help text; unit test for TOML load and value staging.

## F5. Presets + Apply + Model tab (owns `face/presets*.mojo`, `face/apply*.mojo`; after F4's mount exists)
Preset list per group, apply a preset to staged values, save / load / delete user presets
(`~/.config/mojo-baro/presets.toml`). Apply button: shows the restart list and a `pack-fit` VRAM check for the staged
engine values, then restarts the engine through the front (or `tools/baro serve`), with the done line's fail word shown.
Model tab: pick GGUF, find or build the pack (`tools/baro`), model card. Gate CPU: preset round trip, apply plan text
(what restarts, fit verdict) OCR-read; GPU (later): one real restart with a changed `BARO_TMAX`, read back from the
engine log (CLAUDE.md P1 read-back).

## Merge order
F3, then F4 (switch to the real TOML), then F5. `lane-merge --report docs/lanes/2026-11-12-face-ITEM-report.md`.

## Owner decisions for the night (2026-11-12, question dialogs)
- Boss merges F3, F4, F5 in order when each passes boss preflight + ci-checks + lane-merge READY. No GitHub push.
- GPU: shared small jobs only (`gpu-wait run --shared --vram 1`); no exclusive or long runs.
- Startup default preset: Balanced (spec/MTP k=2, f32 KV if it fits else int8, context 32K, temperature 0.7, top_p 0.9).
- Reasoning preset: thinking ON, max_tokens >= 16000, chat shows the thinking block folded with a toggle.
- Help text: user level, what it does, when to change, the cost, one measured number with its receipt when one exists.
- Look: the amarbaro imkit theme as the chat, tabs across the top, help panel on the right of each tab.
- A lane stuck after 3 repairs: boss takes over the step.
- lane-rdna4-epi: left untouched.
- 2026-11-12 night: owner asleep, no images or windows opened on the desktop (overrides CLAUDE.md s14 xdg-open for tonight); PNG paths listed in the morning note instead.
