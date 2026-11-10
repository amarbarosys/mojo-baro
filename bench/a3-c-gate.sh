#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

out=${1:-.work/a3-c-serial}
pack=${BARO_PACK:-.work/engine-pack-q4}
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
  env -u LC_ALL bench/preflight.sh --check
  exec gpu-wait run --priority 70 --timeout 1800 --vram 24 -- env A3_C_JOB=1 GATE_DRYRUN="${GATE_DRYRUN:-0}" LC_ALL= "$0" "$@"
fi

mkdir -p "$out"
rm -f "$out"/refs.json "$out"/ref.out "$out"/ref.err "$out"/serial.out "$out"/serial.err
: > "$out/receipt.md"
fail() { echo "FAIL $1: $2" | tee -a "$out/receipt.md" >&2; exit 1; }

engine_sha=DRYRUN
pack_sha=DRYRUN
if [ "${GATE_DRYRUN:-0}" != 1 ]; then
  [ -x .work/engine ] || fail setup "missing .work/engine"
  [ -f "$pack/index.txt" ] || fail setup "missing pack index"
  engine_sha=$(sha256sum .work/engine | cut -d' ' -f1)
  pack_sha=$(sha256sum "$pack/index.txt" | cut -d' ' -f1)
fi
cat > "$out/arm.txt" <<EOF
commit=$(git rev-parse HEAD)
dirty=$(git status --short)
engine=.work/engine sha256=$engine_sha
pack=$pack index_sha256=$pack_sha
mode=a3c2-serial-prekernel rows=2
prompts=bench/mtp-prompts/p*.tokens count=20
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
files = sorted(pathlib.Path("bench/mtp-prompts").glob("p*.tokens"))
if len(files) != 20:
    raise SystemExit(f"need 20 exact prompt files, found {len(files)}")
refs = []
rows = []
for i, path in enumerate(files):
    prompt = [int(x) for x in path.read_text().split()]
    refs.append({"id": 1000 + i, "prompt": prompt, "n": 64, "spec": False, "temperature": 0})
    rows.append({"id": 2000 + i, "prompt": prompt, "n": 64, "spec": False, "temperature": 0})
(out / "ref-requests.jsonl").write_text("".join(json.dumps(x) + "\n" for x in refs))
with (out / "serial-requests.jsonl").open("w") as f:
    for i in range(0, 20, 2):
        first = dict(rows[i], a3c2=True)
        f.write(json.dumps(first) + "\n")
        f.write(json.dumps(rows[i + 1]) + "\n")
PY

refkey=$(~/iTools/bin/refcache key "$engine_sha" "$pack_sha" "A3c serial prekernel n64 T0" "@$out/ref-requests.jsonl")
export REFCACHE_ROOT=.work/refcache/a3c
if refjson=$(~/iTools/bin/refcache get "$refkey" refs.json 2> "$out/refcache.txt"); then
  cp "$refjson" "$out/refs.json"
else
  BARO_PACK="$pack" BARO_SERVE=1 BARO_MEGA=0 BARO_SPEC=0 BARO_CKPT=0 BARO_TMAX=32768 \
    .work/engine < "$out/ref-requests.jsonl" > "$out/ref.out" 2> "$out/ref.err" || fail reference "engine reference arm failed"
  python3 - "$out/ref.out" "$out/refs.json" <<'PY'
import json, sys
got = {}
for line in open(sys.argv[1]):
    try: d = json.loads(line)
    except Exception: continue
    if "tok" in d and "id" in d:
        got.setdefault(str(d["id"]), []).append(d["tok"])
if len(got) != 20:
    raise SystemExit(f"reference returned {len(got)} token streams, expected 20")
json.dump(got, open(sys.argv[2], "w"), sort_keys=True)
PY
  ~/iTools/bin/refcache put "$refkey" refs.json "$out/refs.json" >> "$out/refcache.txt"
fi
cat "$out/refcache.txt" >> "$out/receipt.md"
echo "- refcache key: $refkey" >> "$out/receipt.md"

BARO_PACK="$pack" BARO_SERVE=1 BARO_MEGA=0 BARO_SPEC=0 BARO_CKPT=0 BARO_TMAX=32768 \
  .work/engine < "$out/serial-requests.jsonl" > "$out/serial.out" 2> "$out/serial.err" || fail candidate "serial pre-kernel arm failed"
grep -q 'BARO_MEGA: False' "$out/serial.out" || fail candidate "effective BARO_MEGA was not read back"
grep -q 'A3C batch: mode serial rows 2' "$out/serial.out" || fail candidate "no two-row batch receipt"
steps=$(grep -c 'A3C step: mode serial ids' "$out/serial.out" || true)
[ "$steps" -eq 10 ] || fail candidate "expected 10 id-per-step receipts, got $steps"
done_count=$(grep -c '"done":true' "$out/serial.out" || true)
[ "$done_count" -eq 20 ] || fail candidate "expected 20 terminal rows, got $done_count"

python3 - "$out" <<'PY' >> "$out/receipt.md"
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
expected = json.loads((out / "refs.json").read_text())
got = {}
done = set()
for line in (out / "serial.out").read_text().splitlines():
    try: d = json.loads(line)
    except Exception: continue
    if "tok" in d and "id" in d:
        got.setdefault(str(d["id"]), []).append(d["tok"])
    if d.get("done") and "id" in d:
        done.add(str(d["id"]))
passed = 0
for i in range(20):
    rid = str(2000 + i)
    assert got.get(rid) == expected.get(str(1000 + i)), (rid, len(got.get(rid, [])))
    assert rid in done, rid
    passed += 1
print(f"- staged two-request identity: {passed}/20 rows")
print(f"- id-per-launch receipts: 10/10, distinct slots 0,1; terminal rows: {len(done)}/20")
print("- verdict: PASS host descriptor/serial harness only; no one-launch throughput claim")
PY
echo "- clean EOF: engine exit 0" >> "$out/receipt.md"
echo "PASS A3(c) staged serial gate: $out/receipt.md"
