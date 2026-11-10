#!/usr/bin/env bash
# usage: gpu-wait run --room cpu --cores 8 --mem 16 --timeout 2400 -- env PATH=$PATH HOME=$HOME bench/ailab/build-engines-head.sh
# Builds the multi-session engine (-D BARO_SEQ_CAP=32 -D BARO_SLOTS=2, as docs/design/multi-session-serve.md) and the single-session
# engine from the SAME commit (HEAD) into .work/ailab/engine-{seqs32s2,main}-head and records commit + sha256 in
# .work/ailab/engines-head.txt. P4c: the P4 own-identity reference engine-main was a d18eba5 build; HEAD has +442 lines in
# serve/{engine,prefix,rows}.mojo since. Built from `git archive HEAD` (other lanes' uncommitted files stay out). The binary is removed before each build, so a failed build cannot leave a stale one to run.
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -n "${GPU_WAITING_ROOM_JOB:-}" ] || { echo "FAIL gate: not inside a gpu-wait job (--room cpu)"; exit 1; }
head=$(git rev-parse --short HEAD); root=$PWD; src=$root/.work/ailab/head-src
# Build from a clean export of HEAD, not the working tree: other lanes keep uncommitted edits in serve/ (expert_tier.mojo at the time of writing)
# and those must neither enter the binary nor be touched.
rm -rf "$src"; mkdir -p "$src" .work/ailab/build-head; git archive HEAD | tar -x -C "$src"
M=$root/.venv/bin/mojo; [ -x "$M" ] || { echo "FAIL tool: no $M"; exit 1; }
build() {  # name flags...
  local n=$1; shift; rm -f "$root/.work/ailab/engine-$n-head"
  (cd "$src" && "$M" build serve/engine.mojo -I . -I kernels "$@" -o "$root/.work/ailab/engine-$n-head") > "$root/.work/ailab/build-head/$n.log" 2>&1 \
    || { echo "FAIL build $n: $(grep -m1 'error' "$root/.work/ailab/build-head/$n.log")"; return 1; }
  [ -x "$root/.work/ailab/engine-$n-head" ] || { echo "FAIL build $n: no binary"; return 1; }
}
build seqs32s2 -D BARO_SEQ_CAP=32 -D BARO_SLOTS=2 & p1=$!
build main & p2=$!
r1=0; r2=0; wait "$p1" || r1=$?; wait "$p2" || r2=$?
[ "$r1" = 0 ] && [ "$r2" = 0 ] || { echo "FAIL builds: seqs rc=$r1 main rc=$r2, see .work/ailab/build-head/*.log"; exit 1; }
{ echo "commit=$head built=$(date -Is)"
  for n in seqs32s2 main; do echo "engine-$n-head sha256=$(sha256sum ".work/ailab/engine-$n-head" | cut -c1-16)"; done; } | tee .work/ailab/engines-head.txt
echo "PASS build-engines-head"
