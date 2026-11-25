#!/usr/bin/env bash
# usage: tools/fixture-check.sh [MANIFEST]   (default bench/fixtures.tsv)
# Checks every fixture row against the MAIN checkout's .work/ before a run, in seconds: exists, no dangling
# link on the way, sha256 equal. A required row that fails prints FAIL with its rebuild command and exits 1;
# an optional row prints SKIP. Born 2026-11-25: a repointed .work/ailab/pack-kq cost three GPU suite runs.
set -euo pipefail
cd "$(dirname "$0")/.."
man=${1:-bench/fixtures.tsv}
main=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")
bad=0; n=0
while IFS=$'\t' read -r f sha need rebuild; do
  [[ -z $f || $f == \#* ]] && continue
  n=$((n + 1)); p=$main/$f; why=""
  if [ ! -e "$p" ]; then why="missing"; [ -L "$(dirname "$p")" ] && why="dangling link $(dirname "$f") -> $(readlink "$(dirname "$p")")"
  elif [ "$sha" != - ] && [ "$(sha256sum "$p" | cut -c1-64)" != "$sha" ]; then why="sha256 differs (want ${sha:0:12})"; fi
  [ -z "$why" ] && continue
  if [ "$need" = required ]; then echo "FAIL fixture $f: $why; rebuild: $rebuild"; bad=$((bad + 1))
  else echo "SKIP fixture $f: $why; rebuild: $rebuild"; fi
done < "$man"
[ "$bad" = 0 ] || { echo "FAIL fixture-check: $bad of $n required fixture(s)"; exit 1; }
echo "OK fixture-check: $n fixture(s) in $man"
