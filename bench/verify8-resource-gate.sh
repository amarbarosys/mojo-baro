#!/usr/bin/env bash
# CPU preflight before queueing a new verify8 kernel variant.
# Usage: bench/verify8-resource-gate.sh BINARY KERNEL_SUBSTRING MAX_VGPR
set -euo pipefail
cd "$(dirname "$0")/.."
[ "$#" -eq 3 ] || { echo 'FAIL usage: BINARY KERNEL_SUBSTRING MAX_VGPR'; exit 1; }
bin=$(realpath "$1"); kernel=$2; limit=$3
[[ "$limit" =~ ^[0-9]+$ ]] || { echo 'FAIL limit: expected integer'; exit 1; }
dir=".work/verify8-resource-gate/$(sha256sum "$bin" | cut -c1-16)"
mkdir -p "$dir"
tools/isa-receipt.py "$bin" --dump "$dir" --hist 0 > "$dir/receipt.txt"
awk -v kernel="$kernel" -v limit="$limit" '
index($2,kernel) && $3 ~ /^vgpr_count=/ {
  split($3,a,"="); n++; vgpr=a[2]+0
}
index($2,kernel) && $3 ~ /spill_count=/ {
  split($3,a,"="); spills+=a[2]+0
}
END {
  if(n!=1 || vgpr>limit || spills) {
    print "FAIL resources: matches=" n " vgpr=" vgpr " limit=" limit " spills=" spills; exit 1
  }
  print "PASS resources: vgpr=" vgpr " limit=" limit " spills=" spills
}' "$dir/receipt.txt"
