#!/usr/bin/env bash
# usage: bench/a2-gate.sh ENGINE_REF ENGINE_CAND OUTDIR [ENV]
#   QUICK=N    first N prompts, 32k set only (iteration gate, P18); default: 20 prompts x 3 sets (merge gate)
#   MIN_PCT=P  forced-agreement bar per prompt (default 100; an int8 KV arm sets its own P14 bar)
#   SETS="L.." subset of 8192 16384 32768 (scoped run, N prompts each, default 20): verdict SCOPED, never a merge gate
#   CAND_ENV="K=V .." env for the candidate arm only (a runtime knob on a copy of the reference binary);
#              the reference records stay cached. Recorded in arm.txt and in the candidate record key.
#   FAILFAST=1 (default) kills the candidate at the first prompt under MIN_PCT: a FAIL in minutes, a PASS
#              still runs every prompt, so the verdict is the full gate's. FAILFAST=0 for a full FAIL table
#              (resume with FAILFAST=0 continues past the recorded failure).
# A2 identity gate (docs/NEXT-PLAN.md A2) on the shared-document sets of bench/a2-prompts.sh:
# teacher-forced agreement of ENGINE_CAND against ENGINE_REF's greedy ids at 8k, 16k and 32k, both
# engines resident (BARO_SERVE=1, serve/PROTOCOL.md), per request "force":[ids].
# RESUMABLE (2026-10-11, bench/a2-arm.py): each set's doc state is saved once per engine to a file
# and every prompt loads it, so prompts are independent (done line "cached" > 0 on every prompt).
# Each prompt's result is a record under .work/a2-rec/<arm>-<engine sha>-<pack sha>-<env sha>/ the
# moment it finishes; kill the job any time and rerun the same command: finished prompts are read
# back, the rest run (one pack load + one state load per arm, no re-prefill). Reference records are
# the P17 reference cache; candidate records are reused only by the same candidate binary. Receipts
# (P1): ready line, BARO_* echoes per arm in receipts.txt, per prompt cached / tok_s and "forced
# agreement a / n". decode medians mix sessions after a resume: never a speed claim. Exit 1 on any
# void or any prompt under MIN_PCT.
set -euo pipefail
cd "$(dirname "$0")/.."
ref=$1; cand=$2; out=$3; envx=${4:-}
quick=${QUICK:-0}; minpct=${MIN_PCT:-100}; setsx=${SETS:-}; nx=${N:-20}; failfast=${FAILFAST:-1}; candenv=${CAND_ENV:-}
pack=${BARO_PACK:-.work/engine-pack-q4}
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ] && command -v gpu-wait >/dev/null; then
  # the queue does not carry this shell's environment: QUICK=3 ran the full gate once (2026-09-16)
  exec gpu-wait run --timeout 1800 -- env QUICK="$quick" MIN_PCT="$minpct" SETS="$setsx" N="$nx" FAILFAST="$failfast" CAND_ENV="$candenv" BARO_PACK="$pack" "$0" "$ref" "$cand" "$out" ${envx:+"$envx"}
fi
mkdir -p "$out"
bench/preflight.sh --check
ha=$(sha256sum "$ref" | cut -c1-16); hb=$(sha256sum "$cand" | cut -c1-16)
[ "$ha" != "$hb" ] || { echo "REFUSED: ref and cand are the same binary ($ha)"; exit 2; }
[ -d .work/a2-prompts/L32768 ] || bench/a2-prompts.sh
# pack identity: sha256 of pack.bin, cached beside it against size+mtime (6.8 GB, ~20 s once)
pk="$pack/pack.bin"; st=$(stat -c '%s-%Y' "$pk")
if [ ! -f "$pack/pack.sha256" ] || [ "$(cut -d' ' -f2 "$pack/pack.sha256")" != "$st" ]; then
  echo "$(sha256sum "$pk" | cut -c1-16) $st" > "$pack/pack.sha256"
fi
packsha=$(cut -d' ' -f1 "$pack/pack.sha256")
{
  echo "ref=$ref cand=$cand shaRef=$ha shaCand=$hb pack=$pack packsha=$packsha env='$envx' cand_env='$candenv' quick=$quick sets='$setsx' n=$nx failfast=$failfast min_pct=$minpct"
  echo "power_cap_uW=$(cat /sys/class/drm/card1/device/hwmon/hwmon*/power1_cap) vddgfx=$(grep -A1 OD_VDDGFX_OFFSET /sys/class/drm/card1/device/pp_od_clk_voltage | tail -1)"
  echo "commit=$(git rev-parse --short HEAD) dirty='$(git status --porcelain | tr '\n' ';')'"
} | tee "$out/arm.txt"

# request set: id = set_index * 100 + prompt_index (the doc-state request takes set_index * 100)
if [ "$quick" -gt 0 ]; then sets="32768"; n=$quick; elif [ -n "$setsx" ]; then sets=$setsx; n=$nx; else sets="8192 16384 32768"; n=20; fi
python3 - "$out/requests.jsonl" "$n" $sets <<'PY'
import json, pathlib, sys
dst, n, sets = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
with open(dst, "w") as f:
    for si, L in enumerate(sets, 1):
        d = pathlib.Path(f".work/a2-prompts/L{L}"); doclen = int((d / "doclen").read_text())
        for i, tf in enumerate(sorted(d.glob("p*.tokens"))[:n], 1):
            ids = [int(x) for x in tf.read_text().split()]
            f.write(json.dumps({"id": si * 100 + i, "set": f"L{L}", "name": tf.stem, "doclen": doclen,
                "req": {"id": si * 100 + i, "prompt": ids, "n": 64, "spec": False, "temperature": 0}}) + "\n")
PY
runenv="BARO_SERVE=1 BARO_SPEC=0 BARO_TMAX=32768"
envsha=$(printf '%s %s' "$runenv" "$envx" | sha256sum | cut -c1-12)
envsha_c=$envsha; [ -z "$candenv" ] || envsha_c=$(printf '%s %s | %s' "$runenv" "$envx" "$candenv" | sha256sum | cut -c1-12)
recref=.work/a2-rec/ref-$ha-$packsha-$envsha; reccand=.work/a2-rec/cand-$hb-$packsha-$envsha_c
echo "records ref=$recref cand=$reccand" | tee -a "$out/arm.txt"
python3 bench/a2-arm.py ref "$out/requests.jsonl" "$recref" "$recref/state" "$minpct" 0 -- env $runenv $envx "$ref"
rm -f "$out/failfast"
python3 bench/a2-arm.py cand "$out/requests.jsonl" "$reccand" "$reccand/state" "$minpct" "$failfast" "$recref" -- env $runenv $envx $candenv "$cand" \
  || { echo "candidate arm stopped: $?"; : > "$out/armstop"; }
echo "== receipts"; sort -u "$reccand/receipts.txt" | grep -E "^(BARO_MEGA|BARO_SPEC|BARO_TMAX|BARO_KVQ):" | head -4 || true

python3 - "$out" "$minpct" "$setsx" "$recref" "$reccand" <<'PY'
import hashlib, json, statistics as st, sys, pathlib
out = pathlib.Path(sys.argv[1]); minpct = float(sys.argv[2]); ref, cand = pathlib.Path(sys.argv[4]), pathlib.Path(sys.argv[5])
sha = lambda r: hashlib.sha256(json.dumps(r["req"], sort_keys=True).encode()).hexdigest()[:16]
reqs = [json.loads(l) for l in open(out / "requests.jsonl")]
bad = False; rows = ["set prompt agree checked pct cached tok_s"]; per = {}
missing = [r for r in reqs if not (cand / f"{r['id']}-{sha(r)}.json").exists()]
for r in reqs:
    p = cand / f"{r['id']}-{sha(r)}.json"
    if not p.exists(): continue
    d = json.loads(p.read_text()); a, c = d["agree"], d["checked"]; pct = 100.0 * a / c if c else 0.0
    if c == 0: print(f"VOID: {r['set']} {r['name']}: nothing checked"); bad = True
    elif pct < minpct: print(f"FAIL {r['set']} {r['name']}: forced agreement {a}/{c} = {pct:.1f}% < {minpct}%"); bad = True
    if d["cached"] <= 0: print(f"VOID: {r['set']} {r['name']}: doc state not restored (cached 0)"); bad = True
    rows.append(f"{r['set']} {r['name']} {a} {c} {pct:.1f} {d['cached']} {d['tok_s']:.2f}")
    per.setdefault(r["set"], []).append((pct, d["tok_s"]))
(out / "results.txt").write_text("\n".join(rows) + "\n"); print("\n".join(rows))
for s, v in sorted(per.items(), key=lambda x: int(x[0][1:])):
    print(f"{s}: {len(v)} prompts, min agreement {min(x[0] for x in v):.1f}%, mean {st.mean(x[0] for x in v):.1f}%, decode median {st.median(x[1] for x in v):.2f} tok/s")
    if s == "L32768": print(f"decode after 32k: {st.median(x[1] for x in v):.2f} tok/s (mixed sessions after a resume, not a claim)")
if missing and not bad: print(f"INCOMPLETE: {len(missing)} of {len(reqs)} prompts not recorded; rerun the same command to resume"); sys.exit(3)
if missing: print(f"stopped with {len(missing)} prompts unrecorded")
if sys.argv[3] and not bad: print(f"SCOPED PASS sets {sys.argv[3]}: iteration evidence, not a merge gate")
elif not bad: print(f"PASS a2-gate: {len(reqs)} prompts, every one >= {minpct}%")
sys.exit(1 if bad else 0)
PY
