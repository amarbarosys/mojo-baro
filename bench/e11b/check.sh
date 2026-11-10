#!/usr/bin/env bash
# Verifies the E11b item bank: Mojo builder byte-identical to the oracle,
# the independent Mojo verifier, and the Mojo scorer numerically identical
# to the oracle's score() on the fixture and on a generated synthetic file.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
MOJO=./.venv/bin/mojo
WORK=.work/e11b-check
rm -rf "$WORK"
mkdir -p "$WORK/fixture-in" "$WORK/synth-in" "$WORK/mojo-fixture" "$WORK/mojo-synth"

if grep -n "std.python\|Python" bench/e11b/*.mojo; then
    echo "FAIL guard: python usage found in bench/e11b/*.mojo"
    exit 1
fi

"$MOJO" build bench/e11b/build_items.mojo -o "$WORK/build_items" \
    || { echo "FAIL build: build_items.mojo"; exit 1; }
"$MOJO" build bench/e11b/test_items.mojo -I . -o "$WORK/test_items" \
    || { echo "FAIL build: test_items.mojo"; exit 1; }
"$MOJO" build bench/e11b/score.mojo -I . -o "$WORK/score" \
    || { echo "FAIL build: score.mojo"; exit 1; }

"$WORK/build_items" || { echo "FAIL run: build_items"; exit 1; }
cp bench/e11b/items.jsonl "$WORK/items.mojo.jsonl"

python3 tools/e11b_items_oracle.py build || { echo "FAIL run: oracle build"; exit 1; }
cp bench/e11b/items.jsonl "$WORK/items.oracle.jsonl"

cmp "$WORK/items.mojo.jsonl" "$WORK/items.oracle.jsonl" \
    || { echo "FAIL cmp: items.jsonl differs between mojo builder and oracle"; exit 1; }

"$WORK/test_items" || { echo "FAIL run: test_items"; exit 1; }
python3 tools/e11b_items_oracle.py test || { echo "FAIL run: oracle test"; exit 1; }

cp bench/e11b/fixtures/records.jsonl "$WORK/fixture-in/records.jsonl"

python3 - "$WORK/synth-in/records.jsonl" <<'PYEOF'
import json, sys

# Every picked item gets a record on EVERY arm (including "0"), so the
# scorer's paired delta_vs_0/lower95 and follows-vector code paths are
# actually exercised, not just the empty-dif branch.
out_path = sys.argv[1]
items = [json.loads(ln) for ln in open("bench/e11b/items.jsonl")]
base = [r for r in items if not r["is_twin"]]
by_game = {}
for r in base:
    by_game.setdefault(r["game"], []).append(r)
picked = by_game.get("countries", [])[:14] + by_game.get("tips", [])[:14] + by_game.get("relations", [])[:8]
arms = ["0", "T", "S", "G", "C", "X", "M", "I"]

recs = []
k = 0
for it in picked:
    for arm in arms:
        m = k % 4
        if m == 0:
            out = it["gold"]
        elif m == 1:
            out = it["answers"][k % len(it["answers"])][0]
        elif m == 2 and it["twin_gold"]:
            out = it["twin_gold"]
        else:
            other = it["answers"][(k + 1) % len(it["answers"])][0]
            out = it["gold"] + " or " + other
        recs.append(
            {"item": it["id"], "arm": arm, "L": None, "vec_src": None, "out": out, "n_tokens": 1}
        )
        k += 1

with open(out_path, "w") as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", len(recs), "synthetic records to", out_path)
PYEOF

"$WORK/score" "$WORK/fixture-in/records.jsonl" bench/e11b/items.jsonl "$WORK/mojo-fixture" \
    || { echo "FAIL run: mojo score (fixture)"; exit 1; }
python3 -c "
import sys
sys.path.insert(0, 'tools')
import e11b_items_oracle as o
o.score('$WORK/fixture-in/records.jsonl')
" || { echo "FAIL run: oracle score (fixture)"; exit 1; }

"$WORK/score" "$WORK/synth-in/records.jsonl" bench/e11b/items.jsonl "$WORK/mojo-synth" \
    || { echo "FAIL run: mojo score (synthetic)"; exit 1; }
python3 -c "
import sys
sys.path.insert(0, 'tools')
import e11b_items_oracle as o
o.score('$WORK/synth-in/records.jsonl')
" || { echo "FAIL run: oracle score (synthetic)"; exit 1; }

compare_tables() {
    python3 -c "
import json, sys
a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))
if len(a) != len(b):
    print('length differs:', len(a), len(b))
    sys.exit(1)
for ra, rb in zip(a, b):
    if set(ra) != set(rb):
        print('key set differs:', ra, rb)
        sys.exit(1)
    for k in ra:
        va, vb = ra[k], rb[k]
        if isinstance(va, float) or isinstance(vb, float):
            if abs(float(va) - float(vb)) > 1e-6:
                print('field', k, 'differs:', va, vb)
                sys.exit(1)
        elif va != vb:
            print('field', k, 'differs:', va, vb)
            sys.exit(1)
print('OK', len(a), 'rows numerically match')
" "$1" "$2"
}

compare_tables "$WORK/mojo-fixture/table.json" "$WORK/fixture-in/table.json" \
    || { echo "FAIL compare: table.json differs on fixture"; exit 1; }
compare_tables "$WORK/mojo-synth/table.json" "$WORK/synth-in/table.json" \
    || { echo "FAIL compare: table.json differs on synthetic"; exit 1; }

echo "PASS e11b items check"
