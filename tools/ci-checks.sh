#!/usr/bin/env bash
# usage: tools/ci-checks.sh
# Every invariant that can be checked without a GPU. Needs the Mojo compiler
# (repo .venv, or `mojo` on PATH in CI) for the kernel census only.
# Runs in CI and locally. Kernel work still needs ./run-tests.sh on the card.
set -u
cd "$(dirname "$0")/.."
# Raise this when a rule is added to bench/PROTOCOL-RULES.md, so dropping any
# existing rule fails the check. It read 6 while the file had grown to 20.
PROTOCOL_RULES_N=20
fails=0
step() { printf '\n== %s\n' "$1"; }
ok()   { echo "  OK  $1"; }
bad()  { echo "  FAIL $1"; fails=$((fails + 1)); }
skip() { echo "  SKIP $1"; }

MOJO=./.venv/bin/mojo; [ -x "$MOJO" ] || MOJO=mojo
# Hosts without MAX (the CI runner) cannot build anything importing max.*; those checks skip.
mkdir -p .work; printf 'from max.gpu.host import DeviceContext\ndef main():\n    pass\n' > .work/ci-max-probe.mojo
hasmax=1; "$MOJO" build .work/ci-max-probe.mojo -o .work/ci-max-probe >/dev/null 2>&1 || hasmax=0
# cached NAME CMD...: a slow step reruns only when a file outside bench/gate-scope.tsv's `none` rows
# changed (key = those files' sha256 + NAME + Mojo version); CI_CACHE=0 reruns everything.
codetree=$( { "$MOJO" --version 2>&1; git ls-files -co --exclude-standard | sort | while IFS= read -r f; do
  [ -f "$f" ] || continue; skip=0
  while IFS=$'\t' read -r pat lv; do [[ -z $pat || $pat == \#* ]] && continue; [[ $f == $pat ]] && { [ "$lv" = none ] && skip=1; break; }; done < bench/gate-scope.tsv
  [ "$skip" = 1 ] || printf '%s\0' "$f"; done | xargs -0 sha256sum --; } | sha256sum | cut -c1-32)
mkdir -p .work/ci-pass
cached() { local k; k=$(printf '%s %s' "$codetree" "$1" | sha256sum | cut -c1-32); shift
  if [ "${CI_CACHE:-1}" = 1 ] && [ -f ".work/ci-pass/$k" ]; then echo "  (cached pass)"; return 0; fi
  "$@" && : > ".work/ci-pass/$k"; }
accel=(--target-accelerator "${BARO_TARGET_ACCELERATOR:-gfx1100}")
mkdir -p .work
CENSUS=.work/kernel-census
"$MOJO" build "${accel[@]}" tools/kernel-census.mojo -o "$CENSUS" || { echo "kernel census failed to build"; exit 1; }

step "kernel census (no orphaned kernels)"
if "$CENSUS" --check; then ok "every amar_* kernel is reachable"
else bad "orphaned kernel: it is in kernels/ but no engine, bench or test calls it"; fi

step "docs/KERNELS.md is current"
cp docs/KERNELS.md .work/ci-kernels-before.md
"$CENSUS" >/dev/null
if diff -q .work/ci-kernels-before.md docs/KERNELS.md >/dev/null; then ok "generated census matches the committed file"
else bad "docs/KERNELS.md is stale: run ./.work/kernel-census (no --check) to regenerate it, then commit docs/KERNELS.md"; diff -u .work/ci-kernels-before.md docs/KERNELS.md | head -20; fi
mv .work/ci-kernels-before.md docs/KERNELS.md

step "pre-tokenizer regexes match llama.cpp (serve/pretok-table.json)"
if python3 tools/pretok-check.py > .work/ci-pretok.log 2>&1; then ok "$(tail -n1 .work/ci-pretok.log)"
else bad "pre-tokenizer regex differs from llama.cpp, see .work/ci-pretok.log (a deliberate difference goes in serve/pretok-known-diffs.txt with its reason)"; grep -A3 "DIFF" .work/ci-pretok.log | head -12; fi

step "self-describing GGUF transitive closure resolution (tools/embed-files.py)"
if python3 tools/embed-files.py --arch spark >/dev/null 2>&1 && python3 tools/embed-files.py --arch qwythos >/dev/null 2>&1; then ok "all GGUF self-describing closure roots resolve"
else bad "embed-files.py closure resolution failed: missing imported source module"; fi


step "python sources parse"
pyfiles=$(git ls-files '*.py')
if python3 -m py_compile $pyfiles 2>&1; then ok "$(echo "$pyfiles" | wc -l) files"
else bad "python syntax error"; fi

step "shell scripts parse"
shbad=0
for f in $(git ls-files '*.sh'); do
  bash -n "$f" 2>/dev/null || { bad "$f"; shbad=1; }
done
[ "$shbad" = 0 ] && ok "$(git ls-files '*.sh' | wc -l) scripts"

step "issue templates are valid yaml"
if python3 - <<'PY'
import sys, yaml, pathlib
for p in sorted(pathlib.Path(".github").rglob("*.yml")):
    try:
        yaml.safe_load(p.read_text())
    except Exception as e:
        print(f"  {p}: {e}"); sys.exit(1)
PY
then ok "all .github yaml loads"; else bad "invalid issue template yaml"; fi

step "protocol rules P1-P$PROTOCOL_RULES_N present"
missing=""
for n in $(seq 1 "$PROTOCOL_RULES_N"); do
  grep -qE "^## P$n\." bench/PROTOCOL-RULES.md || missing="$missing P$n"
done
if [ -z "$missing" ]; then ok "P1-P$PROTOCOL_RULES_N all present"
else bad "PROTOCOL-RULES.md lost rules:$missing"; fi

step "tools/itools links resolve (iTools the repo calls, linked so they are tracked here)"
if [ -n "${CI:-}" ]; then skip "tools/itools links: ~/iTools absent on CI runners"; else
dang=""; for l in tools/itools/*; do [ "$(basename "$l")" = README.md ] && continue; [ -x "$l" ] || dang="$dang $(basename "$l")"; done
if [ -z "$dang" ]; then ok "$(ls tools/itools | grep -vc README.md) links, every target executable"; else bad "dangling tools/itools links:$dang (repair with itools-doctor)"; fi
fi

step "vendored tools/gguf_reader.mojo matches its upstream"
UP=~/iTools/lib/mojo/gguf-reader.mojo
if [ ! -f "$UP" ]; then ok "upstream not on this machine, nothing to compare"
elif diff -q <(tail -n +6 tools/gguf_reader.mojo) "$UP" >/dev/null; then ok "vendored copy is in sync"
else bad "tools/gguf_reader.mojo has drifted from $UP"; diff -u <(tail -n +6 tools/gguf_reader.mojo) "$UP" | head -20; fi

step "vendored uregex/, toml/ and the front libraries match their upstream"
# serve/tokenizer.mojo imports uregex; it is
# vendored as real files (repo root uregex/, -I .), not a symlink
# or a build flag pointing outside the tree, so a clone builds without the
# sibling trees checked out. latentos/ and minja/ are not vendored: canonical here
# (next step).
# serve/front/ imports aio, json, posix, systemd and bento (the Mojo HTTP front, docs/lanes/2026-11-09-mojo-front.md).
# A third field `subset` marks a package vendored in part (bento: http, server, sync only): upstream files
# that are not vendored are not drift there.
M=$HOME/Projects/mojo
for pkg_up in "uregex:$M/mojo-uregex/src/uregex" "toml:$HOME/Projects/imports/databooth-mojo-toml/src/toml" \
    "aio:$M/mojo-aio/src/aio" "json:$M/mojo-json/src/json" "posix:$M/mojo-posix/src/posix" \
    "systemd:$M/mojo-systemd/src/systemd" "bento:$M/mojo-bento/src/bento:subset"; do
  pkg=${pkg_up%%:*}; up=${pkg_up#*:}; subset=""
  case $up in *:subset) subset=1; up=${up%:subset} ;; esac
  if [ ! -d "$up" ]; then ok "$pkg: upstream not on this machine, nothing to compare"; continue
  fi
  drift=""
  for f in "$pkg"/*.mojo; do
    base=$(basename "$f")
    if [ ! -f "$up/$base" ]; then drift="$drift $base(missing-upstream)"
    elif ! diff -q <(tail -n +7 "$f") "$up/$base" >/dev/null; then drift="$drift $base"; fi
  done
  for f in "$up"/*.mojo; do
    base=$(basename "$f")
    [ -n "$subset" ] || [ -f "$pkg/$base" ] || drift="$drift $base(missing-vendored)"
  done
  if [ -z "$drift" ]; then ok "$pkg: vendored copy is in sync"
  else bad "$pkg: drifted from $up:$drift"; fi
done

step "latentos/ is the one copy (plan 12 M1): AMDHQ src/latentos points here"
# serve/latent.mojo and serve/engine.mojo import latentos, whose package and
# boot actors live here. AMDHQ keeps design, receipts, labiso/ and its
# tests, and reaches the code through a symlink; a directory there is a
# second copy that drifts (all four modules had, 2026-09-24).
amdhq_l="$HOME/Projects/hardware/AMDHQ/src/latentos"
[ -e "$amdhq_l" ] || [ -L "$amdhq_l" ] || amdhq_l="$HOME/AMDHQ/src/latentos"
if [ ! -e "$amdhq_l" ] && [ ! -L "$amdhq_l" ]; then ok "AMDHQ not on this machine, nothing to compare"
elif [ -L "$amdhq_l" ] && [ "$(realpath "$amdhq_l")" = "$(realpath latentos)" ]; then ok "AMDHQ src/latentos -> $(readlink "$amdhq_l")"
elif [ -L "$amdhq_l" ]; then ok "AMDHQ src/latentos -> $(readlink "$amdhq_l") (another checkout of this repo)"
else bad "AMDHQ src/latentos is a directory, a second copy; run AMDHQ tools/latent-os/m1-point-at-mojo-baro.sh after merging this branch"; fi

step "docs/settings-template.toml matches source settings (tools/settings-check.py)"
if s_out=$(python3 tools/settings-check.py 2>&1); then ok "$s_out"
else bad "$s_out"; fi

step "referenced protocol and doc files exist"
if python3 - <<'PY'
import pathlib, re, sys
# Repo-relative paths only: a match preceded by "/" or a word character belongs
# to a URL (github.com/ggml-org/ggml/blob/master/docs/gguf.md) and is not ours.
PAT = re.compile(r"(?<![\w/])((?:bench|docs|tools|kernels|serve|shim)/[A-Za-z0-9_.-]+\.(?:md|py|sh|mojo))")
docs = ["README.md", "CONTRIBUTING.md"] + [str(p) for p in sorted(pathlib.Path("docs").glob("*.md"))] \
     + [str(p) for p in sorted(pathlib.Path("bench").glob("*.md"))]
refs, bad = set(), 0
for d in docs:
    for m in PAT.finditer(pathlib.Path(d).read_text()):
        refs.add((m.group(1), d))
for r, d in sorted(refs):
    if not pathlib.Path(r).exists():
        print(f"  {d} references {r}, which does not exist"); bad = 1
print(f"  {len(refs)} referenced paths")
sys.exit(bad)
PY
then ok "every referenced repo path resolves"; else bad "dangling reference in the docs"; fi

step "hardware receipts are well-formed"
if python3 - <<'PY'
import json, pathlib, sys
bad = 0
for p in sorted(pathlib.Path("results").glob("*.json")):
    d = json.loads(p.read_text())
    for k in ("schema", "commit", "env", "valid", "problems", "sizes"):
        if k not in d:
            print(f"  {p.name}: missing '{k}'"); bad = 1
    if d.get("valid") != (not d.get("problems")):
        print(f"  {p.name}: valid flag disagrees with problems list"); bad = 1
    for s in d.get("sizes", []):
        if abs(s["ratio"] - s["ours_gflops"] / s["hipblaslt_gflops"]) > 1e-6:
            print(f"  {p.name}: ratio at {s['size']} does not match its own arms"); bad = 1
sys.exit(bad)
PY
then ok "$(ls results/*.json 2>/dev/null | wc -l) receipts consistent"; else bad "malformed receipt"; fi

step "bench sources compile (no GPU needed to build)"
# Every bench/*.mojo with a main() must build against the current kernel and
# serve signatures. bench/bench_launch_floor.mojo drifted silently for a week
# because nothing built bench/. grammar/ is vendored at the repo root (-I .), so every bench builds;
# only a '# ci-checks: needs' marker (extra link flags) skips, and it is listed.
benchbad=0; benchn=0; benchskip=""
# The shim links ROCm HIP. Where HIP is absent, benches still compile to objects, which is
# what this check guards (signature drift); only the final link is skipped, and that is printed.
link=(-Xlinker -L.work/shim-build -Xlinker -lamarbaro_shim -Xlinker -rpath -Xlinker "$PWD/.work/shim-build")
emit=()
if ! { cmake -S shim -B .work/shim-build -DCMAKE_BUILD_TYPE=Release && cmake --build .work/shim-build -j; } > .work/ci-shim-build.log 2>&1; then
  if grep -q '"hip"' .work/ci-shim-build.log; then echo "  note: no ROCm HIP on this host, benches compile to objects without linking"; link=(); emit=(--emit object)
  else bad "shim build: .work/ci-shim-build.log"; fi
fi
# The engine targets gfx1100; naming it lets hosts without that GPU build benches too.
accel=(--target-accelerator gfx1100)
for f in bench/*.mojo bench/e11b/e11b_harness.mojo; do
  grep -q '^def main' "$f" || continue
  if grep -q '^# ci-checks: needs' "$f"; then benchskip="$benchskip $(basename "$f")"; continue; fi
  benchn=$((benchn + 1))
  echo "$MOJO build ${accel[*]} ${emit[*]} $f -o .work/ci-bench-bin-$(basename "$f" .mojo) -I . -I kernels -I serve -I bench ${link[*]}"
done > .work/ci-bench-cmds.txt
if [ "$hasmax" = 0 ]; then skip "bench builds: no MAX on this host"; benchbad=1; else
tools/build-many.sh ci-bench < .work/ci-bench-cmds.txt > .work/ci-bench-build.log 2>&1 || { bad "bench builds: $(tail -1 .work/ci-bench-build.log)"; grep '  FAIL' .work/ci-bench-build.log | head -5; benchbad=1; }
fi
[ "$benchbad" = 0 ] && ok "$benchn bench sources build"
[ -n "$benchskip" ] && echo "  skip ('# ci-checks: needs' marker, extra link flags):$benchskip"

step "LatentOS standalone reference lifecycle"
if [ "$hasmax" = 0 ]; then skip "latentos-standalone: no MAX on this host"; elif cached latentos-standalone latentos/tests/standalone.sh --backend reference; then ok "reference lifecycle, router, AEAD chunks and relay fixture"; else bad "latentos standalone reference gate"; fi
step "rectangular semantic binding export parity"
if [ "$hasmax" = 0 ]; then skip "latent-binding: no MAX on this host"; elif cached latent-binding bench/latent-binding-check.sh .work/ci-latent-binding; then ok "CPU exporter/runtime parity and malformed-input negatives"; else bad "latent binding parity gate"; fi
step "semantic binding loader"
if [ "$hasmax" = 0 ]; then skip "serve/test_semantic_latent: no MAX on this host"; elif "$MOJO" build serve/test_semantic_latent.mojo -I . -I serve -o .work/ci-semantic-loader >/dev/null 2>&1 && .work/ci-semantic-loader latentos/tests/fixtures/binding.manifest latentos/tests/fixtures/expected.f32 3 2 >/dev/null; then ok "semantic manifest/loader contract"; else bad "semantic binding loader"; fi

step "prefill tune table parser"
if [ "$hasmax" = 0 ]; then skip "serve/test_latent_ipc: no MAX on this host"; elif "$MOJO" build serve/test_latent_ipc.mojo -I . -I serve -I kernels -o .work/ci-latent-ipc >/dev/null 2>&1 && .work/ci-latent-ipc >/dev/null 2>&1; then ok "T3b IPC header: wrong pack/prefix refused before HIP"; else bad "serve/test_latent_ipc.mojo"; fi
if "$MOJO" build serve/test_tune.mojo -I . -I serve -o .work/ci-tune >/dev/null 2>&1 && .work/ci-tune >/dev/null 2>&1; then ok "buckets, fallthrough, malformed rows raise"; else bad "serve/test_tune.mojo"; fi

step "VRAM pool size default check"
if python3 -c 'import sys, tomllib; d = tomllib.load(open("docs/settings-template.toml", "rb")); val = d["settings"]["MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT"]["default"]; sys.exit(0 if val == "2" or val == "10" else 1)'; then ok "MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT default is capped (<= 10)"; else bad "MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT default in settings-template.toml is uncapped (> 10)"; fi

printf '\n'
if [ "$fails" = 0 ]; then echo "all non-GPU checks passed"; else echo "$fails check(s) failed"; fi
exit $((fails > 0))
