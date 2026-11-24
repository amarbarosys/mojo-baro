#!/usr/bin/env bash
# commit-msg-counts: a commit body line "FILE A -> B lines" must name the line count FILE has in the commit (the index).
# Twice a count was typed before `wc -l` (b7a4478: 500 for 508; 51dcf95: 572 for 570). Called by .git/hooks/commit-msg.
#   tools/commit-msg-counts.sh MSG_FILE        exit 1 with "FAIL commit-msg: ..." on a wrong count
set -euo pipefail
M=${1:?MSG_FILE}; bad=0
while read -r name after; do
  p=$(git ls-files | awk -v n="$name" '$0 == n || substr($0, length($0) - length(n)) == "/" n' | head -2)
  [ -n "$p" ] && [ "$(echo "$p" | wc -l)" = 1 ] || continue  # no unique file of that name: nothing to check
  have=$(git show ":$p" | wc -l)
  [ "$have" = "$after" ] || { echo "FAIL commit-msg: body says $name has $after lines, the index has $have ($p)"; bad=1; }
done < <(sed -nE -e 's/(^|.*[^A-Za-z0-9_./-])([A-Za-z0-9_./-]+\.[A-Za-z]+)`? [0-9]+ -> ([0-9]+)`? lines.*/\2 \3/p' \
  -e 's/(^|.*[^A-Za-z0-9_./-])loc ([A-Za-z0-9_./-]+\.[A-Za-z]+) [0-9]+ -> ([0-9]+).*/\2 \3/p' "$M")
exit $bad
