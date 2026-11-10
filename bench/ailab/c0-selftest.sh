#!/usr/bin/env bash
# usage: gpu-wait run --room cpu --cores 2 --mem 4 --timeout 600 -- env PATH=$PATH HOME=$HOME bash bench/ailab/c0-selftest.sh OUT
# CPU selftest of own-identity-c0.sh + wire-drive.py against a token-emitting stand-in engine, through the fake queue (clean env). Three runs:
#   equal   stand-in identical in both arms          -> "S vs Q: 0 mismatches", conditional arms NOT run
#   flip    stand-in changes token 3 when BARO_SEQS  -> "S vs Q: N mismatches", Q0 and M0 DO run (negative control: the verdict can say MISMATCH)
#   dead    stand-in exits after ready               -> FAIL drive-S, nonzero exit (an engine that dies is never a green job)
# Also checks the wire: creator first with n=1 and ckpt [3569], then the scored units in order with ckpt [].
set -euo pipefail
cd "$(dirname "$0")/../.."
out=$(realpath -m "${1:?usage: c0-selftest.sh OUT}"); rm -rf "$out"; mkdir -p "$out"
cat > "$out/stub.py" <<'PY'
#!/usr/bin/env python3
import json, os, sys
log = open(os.environ["FAKE_OUT"], "a")
print("BARO_MEGA:", "True" if os.environ.get("BARO_MEGA", "1") == "1" else "False")
print(json.dumps({"ready": True, "tmax": int(os.environ["BARO_TMAX"]), "mrows": 8, "kmax": 8, "spec_k": 0, "pack": os.environ["BARO_PACK"]}), flush=True)
if os.environ.get("STUB_DEAD") == "1":
    sys.exit(0)
print("BARO_SEQS:", "7" if os.environ.get("STUB_BADSEQS") == "1" and not os.environ.get("BARO_SEQS") else os.environ.get("BARO_SEQS", "1"), flush=True)  # HEAD single prints BARO_SEQS: 1
for line in sys.stdin:
    log.write(os.environ.get("BARO_SEQS", "-") + " " + line)
    log.flush()
    r = json.loads(line)
    toks = [(sum(r["prompt"]) + i) % 1000 for i in range(8)]
    if os.environ.get("STUB_FLIP") == "1" and os.environ.get("BARO_SEQS") and r["ckpt"] == []:
        toks[3] += 1
    for t in toks:
        print(json.dumps({"id": r["id"], "tok": t}), flush=True)
    print(json.dumps({"id": r["id"], "done": True, "n": 8, "cached": 0 if r["ckpt"] else int(os.environ["STUB_CACHED"])}), flush=True)
PY
chmod +x "$out/stub.py"
run() { local n=$1; shift; DRY=1 QUEUE=stub STUB_ENGINE="$out/stub.py" "$@" bash bench/ailab/own-identity-c0.sh "$out/$n" > "$out/$n.log" 2>&1; }
run equal env STUB_FLIP=0 || { tail -5 "$out/equal.log"; echo "FAIL selftest equal: see $out/equal.log"; exit 1; }
grep -q '^S vs Q: 0 mismatches of 19 comparable' "$out/equal/verdict.txt" || { echo "FAIL selftest equal: no 'S vs Q: 0 mismatches of 19 comparable'"; cat "$out/equal/verdict.txt"; exit 1; }
[ ! -d "$out/equal/Q0" ] || { echo "FAIL selftest equal: Q0 ran although S and Q agree"; exit 1; }
run flip env STUB_FLIP=1 || { tail -5 "$out/flip.log"; echo "FAIL selftest flip: see $out/flip.log"; exit 1; }
grep -q '^S vs Q: 19 mismatches of 19 comparable' "$out/flip/verdict.txt" || { echo "FAIL selftest flip: the flipped token was not seen"; cat "$out/flip/verdict.txt"; exit 1; }
[ -d "$out/flip/Q0" ] && [ -d "$out/flip/M0" ] || { echo "FAIL selftest flip: conditional arms Q0 and M0 did not run"; exit 1; }
grep -q 'BARO_SEQS_ROWS=0' "$out/flip/arm.txt" || { echo "FAIL selftest flip: Q0 knob not in arm.txt"; exit 1; }
req=$out/equal/stub-engine-requests.jsonl
python3 - "$req" <<'PY' || { echo "FAIL selftest wire: see above"; exit 1; }
import json, sys
rows = [l.split(" ", 1) for l in open(sys.argv[1]).read().splitlines()]
S = [json.loads(b) for a, b in rows if a == "-"]
Q = [json.loads(b) for a, b in rows if a != "-"]
for name, R in (("S", S), ("Q", Q)):
    assert len(R) == 20, (name, len(R))
    assert R[0]["n"] == 1 and R[0]["ckpt"] == [3569], (name, R[0]["n"], R[0]["ckpt"])
    assert all(r["ckpt"] == [] and r["n"] == 400 and r["spec"] is False for r in R[1:]), name
assert [r["prompt"] for r in S] == [r["prompt"] for r in Q], "arms did not get the same requests in the same order"
print("wire ok: creator n=1 ckpt [3569] first, 19 units ckpt [], same order on both arms")
PY
if DRY=1 QUEUE=stub STUB_ENGINE="$out/stub.py" env STUB_DEAD=1 bash bench/ailab/own-identity-c0.sh "$out/dead" > "$out/dead.log" 2>&1; then
  echo "FAIL selftest dead: a dead engine gave exit 0"; exit 1
fi
grep -q '^FAIL drive-S' "$out/dead.log" || { echo "FAIL selftest dead: no 'FAIL drive-S' line"; tail -3 "$out/dead.log"; exit 1; }
if DRY=1 QUEUE=stub STUB_ENGINE="$out/stub.py" env STUB_BADSEQS=1 bash bench/ailab/own-identity-c0.sh "$out/badseqs" > "$out/badseqs.log" 2>&1; then
  echo "FAIL selftest badseqs: a single arm echoing BARO_SEQS 7 passed the readback"; exit 1
fi
grep -q '^FAIL readback-S' "$out/badseqs.log" || { echo "FAIL selftest badseqs: no 'FAIL readback-S' line"; tail -3 "$out/badseqs.log"; exit 1; }
grep -q 'BARO_MOE_DOT=1 BARO_TIER_ZC=1 BARO_TIER_PINNED=1' "$out/equal/arm.txt" || { echo "FAIL selftest: arm.txt lacks the explicit MOE_DOT/TIER knobs"; exit 1; }
[ "$(grep -c '^engine environ' "$out/equal/arm.txt")" = 2 ] || { echo "FAIL selftest: arm.txt must carry the engine environ of both arms"; exit 1; }
echo "PASS selftest: equal arms stay equal, a flipped token is seen and triggers Q0+M0, the wire is creator-first, a dead engine fails, a wrong single-arm BARO_SEQS echo fails readback-S, both arms print their environ"
