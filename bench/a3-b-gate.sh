#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

out=${1:-.work/a3-b}
pack=${BARO_PACK:-.work/engine-pack-q4}
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
  env -u LC_ALL bench/preflight.sh --check
  exec gpu-wait run --priority 70 --timeout 1800 --vram 24 -- env A3_B_JOB=1 GATE_DRYRUN="${GATE_DRYRUN:-0}" LC_ALL= "$0" "$@"
fi
mkdir -p "$out"
rm -f "$out"/ref.out "$out"/ref.err "$out"/refs.json "$out"/identity.out "$out"/reverse.out
: > "$out/receipt.md"
fail() { echo "FAIL $1: $2" | tee -a "$out/receipt.md" >&2; exit 1; }
[ -x .work/engine ] || fail setup "missing .work/engine"
[ -f "$pack/index.txt" ] || fail setup "missing pack index"
engine_sha=$(sha256sum .work/engine | cut -d' ' -f1)
pack_sha=$(sha256sum "$pack/index.txt" | cut -d' ' -f1)
cat > "$out/arm.txt" <<EOF
commit=$(git rev-parse HEAD)
dirty=$(git status --short)
engine=.work/engine sha256=$engine_sha
pack=$pack index_sha256=$pack_sha
prompts=.work/a2-prompts/L8192/p*.tokens pairs=20
n=64
temperature=0
BARO_MEGA=0 BARO_SPEC=0 BARO_CKPT=0
EOF
if [ "${GATE_DRYRUN:-0}" = 1 ]; then
  echo "FAIL GPU step: dry-run reached engine startup" | tee -a "$out/receipt.md" >&2
  exit 1
fi

python3 - "$out" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
files = sorted((out.parent / "a2-prompts" / "L8192").glob("p*.tokens"))[:20]
if len(files) != 20:
    raise SystemExit(f"need 20 prompt files, found {len(files)}")
refs = []
pairs = []
for i, path in enumerate(files):
    toks = [int(x) for x in path.read_text().split()]
    refs.append({"id": 1000 + i, "prompt": toks, "n": 64, "spec": False, "temperature": 0})
    other = [int(x) for x in files[(i + 1) % len(files)].read_text().split()]
    pairs.append({"id": 10000 + i * 2, "prompt": toks, "n": 64, "spec": False, "temperature": 0, "a3b": True,
                  "peer": {"id": 10001 + i * 2, "prompt": other, "n": 64, "spec": False, "temperature": 0, "a3b": True}})
(out / "ref-requests.jsonl").write_text("".join(json.dumps(x) + "\n" for x in refs))
(out / "pair-requests.jsonl").write_text("".join(json.dumps({k: v for k, v in x.items() if k != "peer"}) + "\n" + json.dumps(x["peer"]) + "\n" for x in pairs))
PY

refkey=$(~/iTools/bin/refcache key "$engine_sha" "$pack_sha" "A3b n64 T0 spec0" "@$out/ref-requests.jsonl")
export REFCACHE_ROOT=.work/refcache/a3b
if refjson=$(~/iTools/bin/refcache get "$refkey" refs.json 2> "$out/refcache.txt"); then
  cp "$refjson" "$out/refs.json"
else
  BARO_PACK="$pack" BARO_SERVE=1 BARO_MEGA=0 BARO_SPEC=0 BARO_CKPT=0 BARO_TMAX=32768 .work/engine < "$out/ref-requests.jsonl" > "$out/ref.out" 2> "$out/ref.err" || fail reference "engine reference arm failed"
  python3 - "$out/ref.out" "$out/refs.json" <<'PY'
import json, sys
tokens = {}
for line in open(sys.argv[1]):
    try: d = json.loads(line)
    except Exception: continue
    if "tok" in d and "id" in d:
        tokens.setdefault(str(d["id"]), []).append(d["tok"])
if len(tokens) != 20:
    raise SystemExit(f"reference returned {len(tokens)} token streams, expected 20")
json.dump(tokens, open(sys.argv[2], "w"), sort_keys=True)
PY
  ~/iTools/bin/refcache put "$refkey" refs.json "$out/refs.json" >> "$out/refcache.txt"
fi
cat "$out/refcache.txt" >> "$out/receipt.md"
echo "- ready source: reference and resident arms use .work/engine" >> "$out/receipt.md"
echo "- refcache key: $refkey" >> "$out/receipt.md"

for mode in identity reverse; do
  BARO_PACK="$pack" BARO_SERVE=1 BARO_MEGA=0 BARO_SPEC=0 BARO_CKPT=0 BARO_TMAX=32768 BARO_KVTAB="$mode" \
    .work/engine < "$out/pair-requests.jsonl" > "$out/$mode.out" 2> "$out/$mode.err" || fail "$mode" "resident pair arm failed"
  grep -q 'A3B slots:' "$out/$mode.out" || fail "$mode" "no slot allocation receipt"
  grep -q 'A3B boundary: resident ids' "$out/$mode.out" || fail "$mode" "no resident boundary receipt"
  count=$(grep -c "mapping $mode" "$out/$mode.out" || true)
  [ "$count" -eq 20 ] || fail "$mode" "expected 20 mapping receipts, got $count"
  echo "- $mode slot receipts: $count" >> "$out/receipt.md"
  python3 - "$out" "$mode" <<'PY' >> "$out/receipt.md"
import json, pathlib, sys
out = pathlib.Path(sys.argv[1]); mode = sys.argv[2]
expected = json.loads((out / "refs.json").read_text())
got = {}
done = {}
for line in (out / f"{mode}.out").read_text().splitlines():
    try: d = json.loads(line)
    except Exception: continue
    if "tok" in d and "id" in d: got.setdefault(str(d["id"]), []).append(d["tok"])
    if d.get("done") and "id" in d: done[str(d["id"])] = d
passed = 0
for i in range(20):
    for j in (0, 1):
        rid = str(10000 + i * 2 + j)
        refid = str(1000 + (i if j == 0 else (i + 1) % 20))
        assert got.get(rid) == expected.get(refid), (mode, rid, len(got.get(rid, [])), len(expected.get(refid, [])))
        assert done.get(rid, {}).get("n") == len(got[rid]), (mode, rid)
    passed += 1
print(f"- {mode} identity: A {passed}/20, B {passed}/20; terminal responses: {len(done)}/40")
PY
done
echo "- clean shutdown: both resident engine arms reached EOF with exit 0" >> "$out/receipt.md"
echo "PASS A3(b) resident-state gate: $out/receipt.md"
