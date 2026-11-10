#!/usr/bin/env bash
# One fact file per amar_ kernel in docs/kernel-facts/, one neuron each through repo-net.
# Sources: docs/KERNELS.md (file, roles, callers), ISA receipts under exchange/receipts or .work (args), docs/BASELINE.md, git log.
# usage: tools/kernel-facts.sh [ISA_RECEIPT...]; wavewait DIR --json saved as .work/kfacts/ww/<kernel>.json adds the cycle share line.
set -euo pipefail
cd "$(dirname "$0")/.."
O=docs/kernel-facts; rm -rf "$O"; mkdir -p "$O"
mkdir -p .work; isa=$(mktemp .work/kernel-facts.XXXXXX); trap 'rm -f "$isa"' EXIT
if [ $# -eq 0 ]; then set -- $(ls -t exchange/receipts/*/isa-*.txt 2>/dev/null); fi
[ $# -gt 0 ] || { echo "FAIL isa: no receipts"; exit 1; }
for r in "$@"; do gawk -v r="$r" '$2 ~ /amar_/ && match($2, /amar_[a-z0-9_]+/) && split($3, a, "=") == 2 {
  k = substr($2, RSTART, RLENGTH); sub(/_[0-9a-f]{16}$/, "", k); sub(/_[0-9]*$/, "", k); print k "\t" a[1] "\t" a[2] "\t" r }' "$r"; done > "$isa"
n=0
while IFS=$'\t' read -r k f roles used; do
  [ -f "kernels/$f" ] || { echo "FAIL source: $f"; exit 1; }
  tests=$(tr ',' '\n' <<<"$used" | grep -c test_ || true)
  { echo "# Kernel $k"
    echo; echo "Source kernels/$f. Registry roles: ${roles:-none}. Default path: $([ -n "$roles" ] && echo in registry || echo not in registry, bench or test only)."
    echo "Callers: $used. Tests naming it: $tests."
    gawk -F'\t' -v k="$k" 'function up(key, v) { v += 0; if (!(key in m) || v > m[key]) m[key] = v }
      $1 == k { up($2, $3); s[$4] = 1 }
      END { c = 0; for (x in s) c++; if (!c) { print "ISA: not in any receipt, unknown."; exit }
        printf "ISA, max over instances in %d receipts: vgpr %d, sgpr %d, vgpr spills %d, sgpr spills %d, LDS bytes %d, scratch bytes %d.\n", c, m["vgpr_count"], m["sgpr_count"], m["vgpr_spill_count"], m["sgpr_spill_count"], m["group_segment_fixed_size"], m["private_segment_fixed_size"]
        print (m["vgpr_spill_count"] + m["sgpr_spill_count"] > 0 ? "Spills: yes." : "Spills: none measured.") }' "$isa"
    w=${KERNEL_FACTS_WW:-.work/kfacts/ww}/$k.json
    [ -f "$w" ] && jq -r '"Wavewait capture: spill \(.classes.spill.share)% of cycles, barrier \(.classes.barrier.share)%, total \(.grand) cycles."' "$w"
    m=$(grep -n -F "$k" docs/BASELINE.md | head -3 | cut -c1-220 || true)
    [ -n "$m" ] && { echo "Measured in docs/BASELINE.md:"; echo "$m"; }
    echo "Recent commits on the file:"; git log -3 --format='%h %s' -- "kernels/$f" | cut -c1-100
  } > "$O/$k.md"
  n=$((n + 1))
done < <(gawk -F'|' 'function t(s) { gsub(/^[ `]+|[ `]+$/, "", s); return s } /^\| `amar_/ { print t($2) "\t" t($3) "\t" t($5) "\t" t($6) }' docs/KERNELS.md)
[ "$n" -gt 0 ] || { echo "FAIL census: 0 kernels"; exit 1; }
echo "PASS kernel-facts: $n kernels, $(grep -l '^Spills: yes' $O/*.md | wc -l) spill, $(grep -l 'not in any receipt' $O/*.md | wc -l) without ISA"
