# Brief: code-fire, mojo-baro as a two-layer neuron network (2026-11-15)

Boss pane: a lane pane. Model: Sonnet, effort high; a Haiku sub-assistant may do reads. Plan (approved by the maintainer, follow it exactly,
steps 1 to 5 each its own commit): the section below. Report: `docs/lanes/2026-11-15-code-fire-report.md`, then
`~/iTools/bin/herd tell a lane pane 'DONE code-fire -> docs/lanes/2026-11-15-code-fire-report.md'`.

Rules:
- Load skills first: brainnet, mojo-lean, memory-optimization; read `~/Brain/SINDEX/tooling.md` entries before editing iTools.
- Graph is the standard: navigate with codebase-memory (`search_graph`, `trace_path`, `get_code_snippet`, `query_graph`), batch in
  `mcp__hub__run_js`; grep only for literals. Project `home-user-Projects-mojo-mojo-baro`.
- iTools commits in ~/iTools (no push). mojo-baro: only the report + `.work/code-fire/` outputs; no code change in mojo-baro.
- Freeze the Verification bars in the Brain note `~/Brain/mojo/mojo-baro/2026-11-15-code-fire.md` and commit it BEFORE the first
  fire run. A missed bar is reported, never re-cut.
- GPU only for the embed step, via `gpu-wait run --timeout S` with env inside `bash -c`.
- Every mistake -> mojo-baro `docs/rules.md` line + SINDEX lesson-entry (Edit tool). Board card on `~/Brain/mojo/mojo-baro/whiteboard.md`.
- Never claim done without the check named in the plan's Verification; else UNVERIFIED + what is missing.

## Context
the maintainer (2026-11-15): convert all of mojo-baro to neurons, convert the codebase-memory graph to neurons too, link the two,
put signals on them, and fire the network thousands of times to find bugs, dead code and optimization potential.
Scope picked: both layers + signals (L).

Found while scouting:
- Layer A exists: `~/.local/share/coios-brain/repo-mojo-baro` (repo-net), 6217 text neurons at HEAD 73574e4, merged into the one
  Brain network by `~/iTools/brain/brain-ask/refresh.sh`.
- Its `links.tsv` is EMPTY (0 B). `~/iTools/brain/repo-net/code-links.sh` finds functions with `func NAME(` (GDScript) and
  keeps only cross-file calls, so not one Mojo/Python call edge of mojo-baro lands. This is a bug in itself.
- Graph: project `home-user-Projects-mojo-mojo-baro`, 33,353 nodes, 85,278 edges (fresh, Haiku report
  `docs/lanes/2026-11-15-graph-refresh-report.md`). Edge kinds include CALLS, USAGE, DATA_FLOWS, IMPORTS, MENTIONS,
  FILE_CHANGES_WITH, SIMILAR_TO. Gap: kernels launched by `enqueue_function` have no CALLS edge (amar_mega_window: 0 callers).
- Reusable: newera `brainnet` (`graph STORE EDGES.tsv`, `spread STORE K SHOW PATH...`, `serve` resident at ~0.09 s a query,
  `kcore`, `stats`), `textnet-build.sh` (GPU embed under gpu-wait), `code-net` (atlas symbols as text, no synapses),
  `brain_spread`. brainnet verdict: spread adds value on DENSE graphs (Wikipedia +13 to 22%); the code graph is dense
  (85k edges / 33k nodes), the vault was not.

## Design
```
layer A  text/mojo-baro/<path>/N        repo-net chunks (exists)            embeddings = meaning
           ^  lives-in (symbol line range -> chunk(s) holding it)
layer B  code/mojo-baro/<qualified_name> one neuron per graph symbol        edges = structure
           synapses from graph edges; enqueue_function launches added from source (literal scan)
signals  signals.tsv per symbol: loc, churn, bugs, cost_cycles, tests
both     merged into the one Brain network (notes, SINDEX lessons, rules link in through MENTIONS / code-links notes)
```

## Steps (each its own commit)
1. **Fix layer A links** (`repo-net/code-links.sh`, ~15 LOC): function matcher per language (`def NAME(` / `fn NAME(` for
   Mojo and Python, `func` kept for GDScript), keep same-file calls too. Check: rebuild links for mojo-baro, `links.tsv` > 0,
   and a known pair (`attn_phases` -> `dattn_split_body`) present; street-sim (GDScript) link count unchanged (cmp).
2. **Layer B export** (new `~/iTools/brain/code-graph-net/`, ~60 LOC, Mojo per §9 where it is more than glue; the export is
   `query_graph` JSON -> TSV, shell + jq like code-links):
   - neurons: Function/Method/Class/Module rows -> text "qn | signature | file:start-end | first doc line", embedded with
     `textnet-build.sh` into store `repo-mojo-baro-graph`.
   - synapses: graph edges -> `edges.tsv` (U TAB V), built with `brainnet graph`. Typed weights are not supported by
     `brainnet graph` (weight 1, both ways): keep edge type in a side file for the reports; weights only if step 5 shows a need.
   - launch edges: `enqueue_function[...NAME...]` / `comptime k = NAME[...]` literals -> host fn -> kernel edges (closes the
     graph gap).
   - lives-in: symbol -> layer A chunk whose line range covers its start line (leads.tsv carries path/N; chunk line spans read
     from the chunker).
   - merge: links.tsv in the store so refresh.sh joins it (same mechanism as repo-net).
3. **Signals** (`signals.sh`, ~40 LOC): loc (end-start); churn = commits touching the symbol's lines (`git log -L` is slow:
   use `git log --numstat` per file over 90 days, spread by line overlap of `git blame --line-porcelain` once per file);
   bugs = symbols touched by commits whose subject starts `fix` plus symbols/files named in `docs/rules.md` and
   `~/Brain/SINDEX/mojo-baro.md` backticks; cost = wavewait goldens `bench/ww-golden/` cycles per kernel; tests = inbound edges
   from `test_*` / `kernels/test_*.mojo` symbols.
4. **code-fire** (`~/iTools/brain/code-fire/code-fire.sh`, ~60 LOC around `brainnet serve` + `spread`): seed sets, N spreads
   each (thousands: every entry point, every bug site, every hot kernel), aggregate activation, write 4 reports to
   `<repo>/.work/code-fire/`:
   - `dead.tsv`: symbols with zero activation from entry seeds (engine mains, tools/, bench/, tests) minus entry points.
   - `bug-candidates.tsv`: seeded from bug sites, leave-one-out; rank non-bug symbols by activation x similarity (SIMILAR_TO or
     cosine) x churn / tests.
   - `hot.tsv`: seeded from cost, activation x cost x loc; SIMILAR_TO clusters = duplicate-kernel candidates (`mojo-lean` step 3).
   - `risk.tsv`: seeded from `detect_changes` symbols of a diff (on demand: `code-fire risk [REV]`).
5. **Graph-standard docs**: iTool headers (= skill), `tool.toml`, `index-gen --write`, `skilldex`; Brain note
   `~/Brain/mojo/mojo-baro/2026-11-15-code-fire.md`; board card.

## Verification (frozen before the first run, in the Brain note)
- Determinism: two runs of code-fire on the same store give byte-identical reports (`cmp`).
- Dead code against an oracle: `search_graph(max_degree=0, exclude_entry_points=true)` is the oracle; dead.tsv must contain
  >= 90% of the oracle's Mojo functions outside tests, and every extra it lists is read and classed (truly dead / launched by
  string / false positive). A planted dead function in a scratch copy must appear.
- Bug candidates, held-out: hide each of 5 known past bug sites (from SINDEX/rules: e.g. the MEGA2 attention partials
  overflowing p_ffn2_d, the row-0 side-copy norm kernel, the KQ dead-branch prologue drift, the 192-VGPR spill in
  amar_ssm_delta_rows, the q4 dot unroll-4 turbo row) and seed from the rest: bar = held-out site in the top 50 for >= 3 of 5,
  control = the same run with random seeds of equal size (must do worse). Missed bar -> report, not re-cut.
- Layer A fix: links.tsv non-empty, known pair present, GDScript repo unchanged.
- Cost: one full fire (thousands of spreads) under 5 min on CPU with `brainnet serve`; measured and in the note.

## Effort
L (~190 LOC: 15 + 60 + 40 + 60 + headers). CPU only, except the embed step (GPU under gpu-wait, minutes).

## Open after the plan
- Whether typed synapse weights help (decided by the held-out bug bar, not upfront).
- The nondeterminism hunt (Pg partials dump) stays next on the mojo-baro board; code-fire's first bug-candidates run may point at it.

## Addendum (the maintainer, 2026-11-15): note -> symbol links as bug seeds
In step 2 add: resolve every backticked name in `~/Brain/SINDEX/mojo-baro.md`, `docs/rules.md` and `~/Brain/mojo/mojo-baro/*.md`
against the graph's qualified names (exact name, then unique suffix match; ambiguous -> skip and count) and write note -> layer-B
symbol links. Those symbols are the `bugs` signal's second source and the seeds of bug-candidates.tsv. Report how many names
resolved, how many were ambiguous, how many unknown. ~20 LOC. The held-out bar stays as frozen (a held-out site's own lesson
links are hidden with it).
