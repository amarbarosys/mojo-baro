#!/usr/bin/env bash
# tune-table.sh SWEEP.out... > tune.txt
# Per shape (bench/bench_tile_sweep.mojo output), the fastest median variant at each
# swept m becomes row `N K m_max variant` when it beats the rule arm's median, else the
# rule's own variant; the largest m covers everything above it.
set -euo pipefail
[ $# -gt 0 ] || { echo "FAIL usage: tune-table.sh SWEEP.out..." >&2; exit 1; }
for f in "$@"; do
  grep -q '^TS shape' "$f" || { echo "FAIL $f: no 'TS shape' header" >&2; exit 1; }
  awk '
    BEGIN { split("wmma16x256 wmma32x256 wmma64x128 wmma128x128 lds64x128 lds128x128", nm, " "); for (i in nm) id[nm[i]] = i - 1 }
    /^TS shape/ { for (i = 1; i <= NF; i++) { if ($i == "N") n = $(i+1); if ($i == "K") k = $(i+1) } next }
    $1 ~ /^[0-9]+$/ && $2 ~ /^rule=/ { m = $1; r = $2; sub(/^rule=/, "", r); rt[m] = $5; rid[m] = id[r]; next }
    $1 ~ /^[0-9]+$/ { m = $1; if (!(m in best) || $5 < best[m]) { best[m] = $5; win[m] = id[$2] }; if (m > last) last = m }
    END { c = asorti(best, ms, "@ind_num_asc"); for (i = 1; i <= c; i++) { mm = ms[i]; print n, k, (mm == last ? 1073741824 : mm), (best[mm] < rt[mm] ? win[mm] : rid[mm]) } }
  ' "$f"
done
