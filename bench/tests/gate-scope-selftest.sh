#!/usr/bin/env bash
# CPU selftest of bench/gate-scope.sh against the frozen map: expected level per file set.
set -euo pipefail
cd "$(dirname "$0")/../.."
bad=0; n=0
while IFS='|' read -r want fs; do
  n=$((n + 1)); got=$(GATE_SCOPE_FILES="$fs" bench/gate-scope.sh); got=${got%%$'\n'*}
  [ "$got" = "$want" ] || { echo "  FAIL '$fs': got $got want $want"; bad=$((bad + 1)); }
done <<'CASES'
none|docs/BASELINE.md README.md
cpu|serve/src/main.rs docs/x.md
full|serve/window.mojo docs/x.md
full|bench/gate-scope.tsv
full|bench/a2-gate.sh
cpu|tools/foo.sh
full|newdir/x
CASES
[ "$bad" = 0 ] || { echo "FAIL $bad/$n gate-scope cases"; exit 1; }
echo "PASS gate-scope selftest $n/$n"
