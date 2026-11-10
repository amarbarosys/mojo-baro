#!/usr/bin/env bash
# usage: bench/gate-scope.sh [BASE]   (default: merge-base with main)
# Prints the gate level a change needs (none | cpu | full) from bench/gate-scope.tsv, the highest
# over every file changed since BASE plus the working tree, and the files that set it. Unmatched
# path = full.  GATE_SCOPE_FILES="a b" replaces the git list (selftest). Scope picks WHICH gates run; a gate that runs keeps its full bar, and claims
# (champion, merge) always run the full set.
set -euo pipefail
cd "$(dirname "$0")/.."
base=${1:-$(git merge-base HEAD main)}
if [ -n "${GATE_SCOPE_FILES:-}" ]; then mapfile -t files < <(tr " " "\n" <<< "$GATE_SCOPE_FILES"); else mapfile -t files < <({ git diff --name-only "$base"; git ls-files -o --exclude-standard; } | sort -u); fi
((${#files[@]})) || { echo "none (no changes since $base)"; exit 0; }
rank() { case $1 in none) r=0;; cpu) r=1;; *) r=2;; esac; }
pats=(); lvs=()
while IFS=$'\t' read -r pat l; do
  [[ -z $pat || $pat == \#* ]] && continue
  pats+=("$pat"); lvs+=("$l")
done < bench/gate-scope.tsv
best=none; why=(); rank none; rb=$r
for f in "${files[@]}"; do
  lv=full
  for i in "${!pats[@]}"; do
    # shellcheck disable=SC2053
    if [[ $f == ${pats[i]} ]]; then lv=${lvs[i]}; break; fi
  done
  rank "$lv"
  [ "$r" -lt "$rb" ] || { [ "$lv" = "$best" ] || why=(); best=$lv; rb=$r; why+=("$f"); }
done
echo "$best"
printf '  %s\n' "${why[@]:0:8}"
