#!/usr/bin/env bash
# Device time per iteration for every target of a bench/dattn-run.sh stint, both arms, one method:
# sum of (End - Start) over the arm's own kernels in tg/trace_kernel_trace.csv, divided by warmup + iters
# (20 + 200 on both harnesses). R's kernels: flash_attn*; O's: amar_dattn*. Per-kernel rows carry the
# trace's own VGPR / scratch / LDS / workgroup / grid read-back (P1: the instrument, not the flag).
# usage: bench/dattn-summarize.sh OUT_DIR [ITERS]  -> OUT_DIR/summary.md
set -euo pipefail
OUT=${1:?OUT_DIR}; IT=$(( ${2:-200} + 20 ))
{
echo "| arm | target | device us/iter | kernel | calls | us/iter | VGPR | scratch | LDS | WG | grid (threads) |"
echo "|---|---|---|---|---|---|---|---|---|---|---|"
for arm in R O; do
  for d in "$OUT/$arm"/*/; do
    [ -d "$d" ] || continue
    label=$(basename "$d"); csv="$d/tg/trace_kernel_trace.csv"
    [ -s "$csv" ] || { echo "| $arm | $label | NO TRACE | | | | | | | | |"; continue; }
    pat='flash_attn'; [ "$arm" = O ] && pat='amar_dattn'
    gawk -v arm="$arm" -v L="$label" -v it="$IT" -v pat="$pat" '
      BEGIN { FPAT = "([^,]*)|(\"[^\"]*\")" }
      NR > 1 && $8 ~ pat {
        name = $8; gsub(/"/, "", name); sub(/\(.*/, "", name); sub(/^void /, "", name)
        n[name]++; ns[name] += $11 - $10; vg[name] = $14; sc[name] = $13; ld[name] = $12; wg[name] = $17; gr[name] = $20; tot += $11 - $10
      }
      END {
        if (tot == 0) { printf "| %s | %s | NO MATCH | | | | | | | | |\n", arm, L; exit }
        first = 1
        for (k in n) {
          printf "| %s | %s | %s | %s | %d | %.2f | %s | %s | %s | %s | %s |\n", arm, (first ? L : ""), (first ? sprintf("%.2f", tot / it / 1e3) : ""), k, n[k], ns[k] / it / 1e3, vg[k], sc[k], ld[k], wg[k], gr[k]
          first = 0
        }
      }' "$csv"
  done
done
echo
echo "Receipts (arm lines as printed by each harness before timing):"
echo
for arm in R O; do
  for d in "$OUT/$arm"/*/; do
    [ -f "$d/stdout.txt" ] || continue
    echo "- $arm/$(basename "$d"): $(grep -E '^arm' "$d/stdout.txt" | head -1)"
    echo "  $(grep -E '^wall' "$d/stdout.txt" | head -1)"
  done
done
} > "$OUT/summary.md"
cat "$OUT/summary.md"
