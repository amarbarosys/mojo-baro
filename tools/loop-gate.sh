#!/usr/bin/env bash
# Mechanical gate ladder for loop candidates. Serial, fail-closed, one receipt per candidate.
# usage: [LOOP_PROMPT2=ids-file] tools/loop-gate.sh ITER CHAMPION_TOKPS   (llama-server must be stopped: engine needs the GPU)
set -uo pipefail
cd "$(dirname "$0")/.."
iter=$1; champ=${2:-}; dir=.work/loop/$iter
export MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=10
: > "$dir/SURVIVORS.md"
# The reference is read ONCE, before any candidate binary runs, and every
# identity check compares against this snapshot. The on-disk file lives in
# .work/, which the candidate's process can write; a candidate that rewrites
# it to its own output would otherwise pass identity (audit 2026-09-08, cand-3).
# Identity is bit-identity with the CHAMPION BUILD of the iteration's own sources: refsnap is taken from the
# champion's warm-up run below, in this shell, before any candidate binary exists. The pack's ref-tokens-64.txt
# is only checked for tampering and compared once, informationally: since 2026-10-09 it holds llama.cpp's greedy
# ids (bench/claims-verify.sh:45), which our engine is not required to match (CLAUDE.md, identity gates; 008 run 1
# stopped on it at position 31).
ref=${LOOP_REF:-.work/engine-pack-q4/ref-tokens-64.txt}; packsnap=$(cat "$ref"); refsnap=""
check() { # check RUNLOG -> identity against the champion snapshot, and the fixture must be untouched
  printf '%s\n' "$refsnap" > "$work/ref.txt"
  if ! printf '%s\n' "$packsnap" | cmp -s - "$ref"; then printf '%s\n' "$packsnap" > "$ref"; echo "candidate rewrote $ref (restored)"; return 1; fi
  tools/check-tokens.sh "$work/ref.txt" "$1"
}
# The iteration's own pristine sources are built once per gate run. That binary is
# (a) the wall-clock reference for the plausibility term, (b) the ISA baseline for
# stage 4, (c) the oracle for the optional second workload and, since P-D
# (2026-09-08 evening, bench/loop-protocol.md), (d) the ACCEPTANCE DENOMINATOR:
# champion and candidate are the same binary pair on the same warm GPU in the
# same run. Iteration 006 showed why: the gguf-closure median from a separate
# session sat 2.2% below the in-gate build of the same sources, and a no-op
# comment reached +1.7% against it. The optional second argument is the closure
# run's tok/s_gen, recorded in the receipt as champion_closure, never compared.
# Harness assembly (split layout, transitive serve/ + kernels/ walk): tools/loop-harness.sh, shared with the proposer.
eval "$(tools/loop-harness.sh "$dir")" || { echo "harness assembly failed"; exit 1; }
[ "$split" = 1 ] && echo "split layout: harness serve/engine.mojo@$kcommit, body window.mojo"
./.venv/bin/mojo build "$dir/$entry" -I "$dir/src" -I "$dir/harness" -o "$dir/champion-engine" > "$dir/champion-build.log" 2>&1 || { echo "champion build failed: $(grep -m1 error: "$dir/champion-build.log")"; exit 1; }
# Warm-up run, discarded (iteration 007: 5.5% spread on the unchanged binary right after a model unload).
./"$dir/champion-engine" > "$dir/champion-warm.log" 2>&1 || { echo "champion warm-up run failed"; exit 1; }
refsnap=$(grep -m1 '^GENERATED:' "$dir/champion-warm.log" | sed 's/^GENERATED://' | tr -s ' ' '\n' | sed '/^$/d')
[ "$(echo "$refsnap" | wc -l)" -ge 64 ] || { echo "FAIL champion warm-up: $(echo "$refsnap" | wc -l) generated ids, want 64"; exit 1; }
echo "champion vs pack reference $ref (informational): $(tools/check-tokens.sh "$ref" "$dir/champion-warm.log" 2>&1 | head -1)"
cw=(); ct=()
for k in 1 2 3; do
  s0=$(date +%s%N); ./"$dir/champion-engine" > "$dir/champion-run$k.log" 2>&1 || { echo "champion run $k failed"; exit 1; }; s1=$(date +%s%N)
  cw+=("$(awk -v a="$s0" -v b="$s1" 'BEGIN{printf "%.3f", (b-a)/1e9}')"); ct+=("$(grep -oE 'tok/s_gen: [0-9.]+' "$dir/champion-run$k.log" | awk '{print $2}')")
done
cwall=$(printf '%s\n' "${cw[@]}" | sort -n | sed -n 2p); ctok=$(printf '%s\n' "${ct[@]}" | sort -n | sed -n 2p)
python3 tools/isa-spills.py "$dir/champion-engine" > "$dir/champion-isa.json" || { echo "champion ISA census failed"; exit 1; }
echo "champion binary from $dir/src: wall_s median $cwall (${cw[*]}), tok/s_gen median $ctok (${ct[*]}), closure ${champ:-none}; ISA baseline $(python3 -c "import json;d=json.load(open('$dir/champion-isa.json'));print(sum(1 for v in d['families'].values() if v['spills'] or v['scratch']),'spilling families of',len(d['families']))")"
# Optional second, unpublished workload (audit 2026-09-08, lead 2): LOOP_PROMPT2=<ids file>.
# Every candidate must reproduce the champion binary's 64 tokens on it as well as
# on the published fixture. Behind a flag so the default ladder's cost is unchanged.
if [ -n "${LOOP_PROMPT2:-}" ]; then
  [ -f "$LOOP_PROMPT2" ] || { echo "LOOP_PROMPT2 not found: $LOOP_PROMPT2"; exit 1; }
  BARO_PROMPT=$LOOP_PROMPT2 ./"$dir/champion-engine" > "$dir/champion-run2.log" 2>&1 || { echo "champion run on $LOOP_PROMPT2 failed"; exit 1; }
  ref2snap=$(grep -m1 '^GENERATED:' "$dir/champion-run2.log" | sed 's/^GENERATED://' | tr -s ' ' '\n' | sed '/^$/d')
  echo "second fixture: $LOOP_PROMPT2 ($(wc -l < "$LOOP_PROMPT2") prompt tokens), $(echo "$ref2snap" | wc -l) reference tokens from the iteration's own sources"
fi
for d in "$dir"/cand-*.diff; do
  c=$(basename "$d" .diff); r="$dir/$c.receipt.json"; work="$dir/$c"; rm -rf "$work"; mkdir -p "$work"
  fail() { echo "{\"cand\":\"$c\",\"stage\":\"$1\",\"result\":\"FAIL\",\"why\":\"$2\"}" > "$r"; echo "$c: FAIL $1: $2"; return 1; }
  [ -s "$d" ] || { fail parse "no diff fence"; continue; }
  # stage 0: scope -- only files listed in the gguf, no gate/timing/profiling edits
  touched=$(grep -E '^\+\+\+ ' "$d" | sed -E 's#^\+\+\+ (b/)?##; s/\t.*//' | sort -u)
  bad=""; for f in $touched; do grep -qx "$(basename "$f")" "$dir/FILES" || bad="$bad $f"; done
  [ -z "$bad" ] || { fail scope "files outside gguf list:$bad"; continue; }
  # The banned-token list alone is evadable: iteration 004's cand-1 commented out
  # `if pf4:`, a profiling GUARD, whose own line carries none of these tokens while
  # disabling the block underneath it. Guard names are in the pattern for that reason.
  # Audit 2026-09-08 (exchange/scorer-integrity-report.md): the timing STATE is
  # as reachable as the timing CALLS. `t_prefill_end += 500_000_000`, deleting
  # `prefill_done = True`, or deleting the synchronize before `dt` each carried
  # none of the tokens above and each passed the whole ladder with an inflated
  # tok/s_gen. Every identifier the tok/s_gen arithmetic reads, every host sync,
  # file writes and the fixture paths are banned on +/- lines. None of these
  # names occur in any kernel file; a legitimate kernel edit is unaffected.
  # Split layout: the clock is not in the candidate's files at all, so the
  # stopwatch identifiers come off the list (pos/ring/e are honest window state
  # there); profiling, prints, file writes and fixture paths stay banned.
  if [ "$split" = 1 ]; then
    scope_pat='^[+-].*(BARO_PROFILE|perf_counter_ns|tok/s|check-tokens|ref-tokens|getenv|print\(|open\(|synchronize|engine-pack|GEN_N|\b(prof|pf2|pf3|pf4|pf_[a-z]+|tq|tp|nw|now[0-9]?|t_acc)\b)'
  else
    scope_pat='^[+-].*(BARO_PROFILE|perf_counter_ns|tok/s|check-tokens|ref-tokens|getenv|print\(|open\(|synchronize|engine-pack|GEN_N|generated|toks_h|\b(prof|pf2|pf3|pf4|pf_[a-z]+|t0|tq|tp|dt|nw|now[0-9]?|t_acc|t_load|t_host|t_prefill_end|prefill_done|prefill_s|decode_s)\b)'
  fi
  grep -E "$scope_pat" "$d" >/dev/null && { fail scope "touches timing/print/profile/fixture code"; continue; }
  # stage 1: apply + compile on a copy of the gguf sources
  # Hunk counts are rewritten from the hunk body first (iteration 002 lost 4/4
  # here to headers declaring 7 context lines while supplying 4). Content is
  # untouched, so a hunk describing nonexistent code still fails -- at compile,
  # where it belongs. apply_mode goes in the receipt: a fuzzy apply is recorded,
  # never silent.
  cp -r "$dir/src" "$work/src"
  python3 tools/diff-normalise.py "$d" "$work/norm.diff" > "$work/normalise.log" 2>&1 || { fail apply "diff-normalise failed"; continue; }
  nfix=$(grep -c '^line ' "$work/normalise.log")
  mode=""
  for try in "-p1|" "-p0|" "-p1|-l --fuzz=3" "-p0|-l --fuzz=3"; do
    strip=${try%%|*}; extra=${try#*|}
    if patch $strip $extra -d "$work/src" --dry-run -s < "$work/norm.diff" >/dev/null 2>&1; then
      patch $strip $extra -d "$work/src" -s < "$work/norm.diff" >/dev/null 2>&1
      mode="$strip${extra:+ $extra}"; break
    fi
  done
  [ -n "$mode" ] || { fail apply "patch does not apply (hunks renumbered: $nfix)"; continue; }
  [ "$split" = 1 ] && cp -r "$dir/harness" "$work/harness"
  ./.venv/bin/mojo build "$work/$entry" -I "$work/src" -I "$work/harness" -o "$work/engine" > "$work/build.log" 2>&1 || { fail compile "$(grep -m1 error: "$work/build.log" | cut -c1-160)"; continue; }
  # stage 2: token identity at 64
  ./"$work/engine" > "$work/run0.log" 2>&1 || { fail run "engine exited $?"; continue; }
  check "$work/run0.log" > "$work/gate.log" 2>&1 || { fail identity "$(head -1 "$work/gate.log")"; continue; }
  if [ -n "${LOOP_PROMPT2:-}" ]; then
    BARO_PROMPT=$LOOP_PROMPT2 ./"$work/engine" > "$work/run0b.log" 2>&1 || { fail run "engine exited $? on second fixture"; continue; }
    printf '%s\n' "$ref2snap" > "$work/ref2.txt"
    tools/check-tokens.sh "$work/ref2.txt" "$work/run0b.log" > "$work/gate0b.log" 2>&1 || { fail identity2 "$(head -1 "$work/gate0b.log")"; continue; }
  fi
  # stage 3: preregistered perf -- candidate's own PREDICT is the preregistration; read tok/s_gen back
  pred=$(cat "$dir/$c.predict"); t=(); w=(); g=(); idfail=""
  for k in 1 2 3; do
    s0=$(date +%s%N); ./"$work/engine" > "$work/run$k.log" 2>&1; s1=$(date +%s%N)
    t+=("$(grep -oE 'tok/s_gen: [0-9.]+' "$work/run$k.log" | awk '{print $2}')")
    w+=("$(awk -v a="$s0" -v b="$s1" 'BEGIN{printf "%.3f", (b-a)/1e9}')")
    g+=("$(grep -oE 'gpu_total_s: [0-9.]+' "$work/run$k.log" | awk '{print $2}')")
    check "$work/run$k.log" > "$work/gate$k.log" 2>&1 || idfail="run$k: $(head -1 "$work/gate$k.log")"
  done
  # The identity rule is 64/64 on the engine's output; a candidate that is right
  # once and wrong on the timed runs is not right. wall_s is the gate's own clock
  # around the whole process (load + prefill + decode + receipt tail): a receipt
  # field for cross-checking the self-reported tok/s_gen, not an acceptance term.
  [ -z "$idfail" ] || { fail identity "$idfail"; continue; }
  med=$(printf '%s\n' "${t[@]}" | sort -n | sed -n 2p); lo=$(printf '%s\n' "${t[@]}" | sort -n | head -1); hi=$(printf '%s\n' "${t[@]}" | sort -n | tail -1)
  spread=$(awk -v l="$lo" -v h="$hi" 'BEGIN{printf "%.3f", (h-l)/l}')
  ok=$(awk -v m="$med" -v c="$ctok" -v s="$spread" 'BEGIN{print (m >= c*1.02 && s < 0.05) ? 1 : 0}')
  [ "$ok" = 1 ] || { fail perf "median $med vs in-gate champion $ctok (need +2%), spread $spread"; continue; }
  # Plausibility (rule 2026-09-08): the claimed decode saving must show in the
  # gate's own clock. Champion and candidate wall_s come from the same gate run.
  wmed=$(printf '%s\n' "${w[@]}" | sort -n | sed -n 2p)
  # Both sides of the plausibility term come from THIS gate run: the champion
  # binary's own tok/s_gen median against its own wall. Using the CHAMPION_TOKPS
  # argument here (another session's median) against the in-gate wall mixed two
  # references -- the split-layout dry run with a nonsense argument of 60 rejected
  # a benign +0.15% candidate for "0.567 s of decode saved".
  plaus=$(awk -v cw="$cwall" -v ww="$wmed" -v c="$ctok" -v m="$med" 'BEGIN{print ((cw-ww) >= 0.5*(63/c-63/m)) ? 1 : 0}')
  [ "$plaus" = 1 ] || { fail perf "tok/s_gen $med claims $(awk -v c="$ctok" -v m="$med" 'BEGIN{printf "%.3f", 63/c-63/m}') s of decode saved vs the in-gate champion $ctok, wall clock moved $(awk -v cw="$cwall" -v ww="$wmed" 'BEGIN{printf "%.3f", cw-ww}') s (champion wall $cwall, candidate $wmed)"; continue; }
  # stage 4: ISA -- no kernel family with more scratch or more spills than the
  # champion build of the same sources, none new with any (rule 2026-09-08; the
  # absolute form rejected the champion itself: delta-step variants spill).
  python3 tools/isa-spills.py "$work/engine" --baseline "$dir/champion-isa.json" > "$work/isa.log" 2>&1 || { fail isa "$(sed -n 2p "$work/isa.log" | sed 's/^ *//')"; continue; }
  echo "{\"cand\":\"$c\",\"stage\":\"all\",\"result\":\"PASS\",\"predict_pct\":\"$pred\",\"tokps\":[${t[0]},${t[1]},${t[2]}],\"median\":$med,\"champion\":$ctok,\"champion_closure\":${champ:-null},\"spread\":$spread,\"wall_s\":[${w[0]},${w[1]},${w[2]}],\"champion_wall_s\":[${cw[0]},${cw[1]},${cw[2]}],\"champion_tokps_ingate\":[${ct[0]},${ct[1]},${ct[2]}],\"gpu_total_s\":[${g[0]},${g[1]},${g[2]}],\"fixture2\":\"${LOOP_PROMPT2:-none}\",\"apply_mode\":\"$mode\",\"hunks_renumbered\":$nfix}" > "$r"
  echo "$c: PASS median $med vs in-gate champion $ctok (predict $pred%)"; { echo "## $c  median $med vs in-gate champion $ctok (predict $pred%)"; echo '```diff'; cat "$d"; echo '```'; } >> "$dir/SURVIVORS.md"
done
echo "survivors: $(grep -c '^## ' "$dir/SURVIVORS.md")"
