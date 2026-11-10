#!/usr/bin/env bash
# Differential test: tools/state_diff.mojo vs its oracle tools/state_diff.py on synthetic MoE-shaped state files
# (bench/tests/state-diff-fixture.py: f32 drift, int8 KVG 256 and 32, reversed, self, header mismatch, usage).
# Contract: same exit code, header and FAIL lines, line labels; every number within 2e-3 relative (three printed
# significant digits). Builds .work/state_diff through tools/build-many.sh (closure-cached).
set -euo pipefail
cd "$(dirname "$0")/../.."
t=.work/state-diff-test; rm -rf "$t"; mkdir -p "$t"
echo "./.venv/bin/mojo build tools/state_diff.mojo -o .work/state_diff" | tools/build-many.sh state-diff > "$t/build.log" 2>&1 || { echo "FAIL build: see $t/build.log"; exit 1; }
python3 bench/tests/state-diff-fixture.py "$t/fix"
F=$t/fix; bad=0; n=0
while IFS= read -r c; do
  n=$((n + 1))
  o=$(eval "python3 tools/state_diff.py $c" 2>/dev/null) && oe=0 || oe=$?
  m=$(eval ".work/state_diff $c" 2>/dev/null) && me=0 || me=$?
  python3 - "$o" "$m" "$oe" "$me" "$c" <<'PY' || bad=$((bad + 1))
import re, sys
o, m, oe, me, c = sys.argv[1:]
num = re.compile(r"\d\.\d{3}e[+-]\d\d")
ol, ml = o.splitlines(), m.splitlines()
ok = oe == me and len(ol) == len(ml)
for a, b in zip(ol, ml):
    if num.sub("#", a) != num.sub("#", b): ok = False; break
    for x, y in zip(num.findall(a), num.findall(b)):
        if abs(float(x) - float(y)) > 2e-3 * max(abs(float(x)), abs(float(y)), 1e-300): ok = False
if not ok:
    print(f"  FAIL '{c}': exit {oe} vs {me}")
    for a, b in zip(ol, ml):
        if a != b: print(f"    py   {a}\n    mojo {b}")
    sys.exit(1)
PY
done <<CASES
--moe $F/ref.st $F/drift.st
--moe $F/ref.st $F/i8.st
--moe --kvg 32 $F/ref.st $F/i8g32.st
--moe $F/i8.st $F/ref.st
--moe $F/ref.st $F/ref.st
--moe $F/ref.st $F/short.st
\$NOARGS
CASES
[ "$bad" = 0 ] || { echo "FAIL $bad/$n state_diff cases"; exit 1; }
echo "PASS state_diff $n/$n against tools/state_diff.py"
