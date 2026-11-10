#!/usr/bin/env bash
# usage: bench/kvc-agree.sh INT8_ENGINE PACK ORIG_BAROST03 RECON_BAROST03 P_TOKENS OUT
# C20 stage A (AMDHQ 06 "## C20:"): teacher-forced agreement of BAROKV4's reconstruction
# against the original BAROST03 on the int8-pool engine, 32k prefix, 20 suffixes x 64 ids,
# one engine process per arm (bench/c19-agree.sh's shape):
#   ref  original file, greedy                    -> reference ids
#   ctl  original file, forced to ref             -> must be 64/64 on all 20 (same-arm control)
#   kv4  reconstruction, forced to ref            -> agreement per suffix; bar mean >= 98, min >= 90
# One gpu-wait job. GATE_DRYRUN=1: stub engine echoing forced ids, every post-GPU step runs.
set -euo pipefail
cd "$(dirname "$0")/.."
eng=$1; pack=$2; orig=$3; recon=$4; ptok=$5; out=$6
fail() { echo "FAIL $1: $2"; exit 1; }
mkdir -p "$out"
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ] && [ "${GATE_DRYRUN:-0}" != 1 ]; then
    {
        echo "engine=$eng sha=$(sha256sum "$eng" | cut -c1-16) commit=$(git rev-parse --short HEAD) dirty='$(git status --porcelain | tr '\n' ' ' | head -c 80)'"
        echo "pack=$pack orig=$orig $(sha256sum "$orig" | cut -c1-16) recon=$recon $(sha256sum "$recon" | cut -c1-16) P=$(sha256sum "$ptok" | cut -c1-16)"
        echo "suffixes=$(ls bench/mtp-prompts/p*.tokens | head -20 | xargs sha256sum | sha256sum | cut -c1-16) n=64 tmax=32768 spec=0"
    } > "$out/arm.txt"
    cat "$out/arm.txt"
    exec gpu-wait run --priority 20 --vram 22 --timeout 900 -- "$0" "$@"
fi
if [ "${GATE_DRYRUN:-0}" = 1 ]; then
    stub="$out/stub-engine.py"
    cat > "$stub" <<'PY'
import json, struct, sys
print('{"ready":true}')
for line in sys.stdin:
    r = json.loads(line)
    pos = struct.unpack("<q", open(r["state_load"], "rb").read(16)[8:16])[0]
    print(f"state loaded: {r['state_load']}  pos {pos}")
    ids = r.get("force") or [1000 + i for i in range(r["n"])]
    for t in ids:
        print(json.dumps({"id": r["id"], "tok": t}))
    if r.get("force"):
        print(f"forced agreement: {len(ids)} / {len(ids)}")
PY
    eng="python3 $stub"
fi
echo "${GPU_WAITING_ROOM_JOB:-dryrun}" > "$out/gpu-wait-job.txt"
trap 'pkill -TERM -P $$ 2>/dev/null || :' EXIT
run_arm() {
    local arm=$1 st=$2 ref=${3:-}
    local d="$out/$arm"; mkdir -p "$d"
    python3 - "$ptok" "$st" "$d/requests.jsonl" "$ref" <<'PY'
import glob, json, sys
P = [int(x) for x in open(sys.argv[1]).read().split()]
st, reqf, ref = sys.argv[2], sys.argv[3], sys.argv[4]
refids = json.load(open(ref)) if ref else None
sfx = sorted(glob.glob("bench/mtp-prompts/p*.tokens"))[:20]
with open(reqf, "w") as f:
    for i, s in enumerate(sfx):
        r = {"id": i + 1, "prompt": P + [int(x) for x in open(s).read().split()], "n": 64, "spec": False, "state_load": st}
        if refids is not None:
            r["force"] = refids[str(i + 1)]
        f.write(json.dumps(r) + "\n")
PY
    local t0=$SECONDS
    env BARO_SERVE=1 BARO_SPEC=0 BARO_PACK="$pack" BARO_TMAX=32768 $eng < "$d/requests.jsonl" > "$d/out.txt" 2> "$d/err.txt" || fail "engine $arm" "$d/err.txt"
    echo "$((SECONDS - t0))" > "$d/wall_s.txt"
    grep -q '"ready":true' "$d/out.txt" || fail "engine $arm" "no ready line in $d/out.txt"
    python3 - "$d" "$st" "$ref" <<'PY'
import json, re, sys
d, st, ref = sys.argv[1], sys.argv[2], sys.argv[3]
lines = open(f"{d}/out.txt").read().splitlines()
loads = [l for l in lines if l.startswith("state loaded:") or l.startswith("state hot:")]
if len(loads) != 20 or any(st not in l for l in loads):
    sys.exit(f"FAIL parse: {d}: {len(loads)} state loaded/hot lines naming {st}, want 20")
pos = [int(re.search(r"pos (\d+)", l).group(1)) for l in loads]
if any(p != 31999 for p in pos):
    sys.exit(f"FAIL void: {d}: state loaded at pos {sorted(set(pos))}, want 31999")
toks = {}
for l in lines:
    if l.startswith("{") and '"tok"' in l:
        j = json.loads(l); toks.setdefault(str(j["id"]), []).append(j["tok"])
if len(toks) != 20 or any(not 1 <= len(v) <= 64 for v in toks.values()):
    sys.exit(f"FAIL parse: {d}: {len(toks)} requests with ids, lengths {sorted(set(len(v) for v in toks.values()))}")
json.dump(toks, open(f"{d}/ids.json", "w"))
agree = [tuple(int(x) for x in re.findall(r"\d+", l)) for l in lines if l.startswith("forced agreement:")]
if ref:
    if len(agree) != 20:
        sys.exit(f"FAIL parse: {d}: {len(agree)} forced agreement lines, want 20")
    with open(f"{d}/agree.tsv", "w") as f:
        for i, (a, c) in enumerate(agree):
            f.write(f"{i + 1}\t{a}\t{c}\n")
print(f"{d}: 20 requests, load pos {sorted(set(pos))}, agreement lines {len(agree)}")
PY
}
run_arm ref "$orig"
run_arm ctl "$orig" "$out/ref/ids.json"
run_arm kv4 "$recon" "$out/ref/ids.json"
python3 - "$out" <<'PY'
import statistics as st, sys
out = sys.argv[1]
def pct(arm):
    rows = [l.split("\t") for l in open(f"{out}/{arm}/agree.tsv").read().splitlines()]
    return [100.0 * int(a) / int(c) for _, a, c in rows]
ctl = pct("ctl"); kv4 = pct("kv4")
lines = [f"ctl: min {min(ctl):.2f}% mean {st.mean(ctl):.2f}% (must be 100 on all 20)", f"kv4: min {min(kv4):.2f}% mean {st.mean(kv4):.2f}% per-suffix {[round(x, 1) for x in kv4]}"]
void = min(ctl) < 100
verdict = "VOID (control not 20/20)" if void else ("PASS" if st.mean(kv4) >= 98 and min(kv4) >= 90 else ("KILL" if st.mean(kv4) < 96 or min(kv4) < 85 else "FAIL (between bar and kill)"))
lines.append(f"verdict {verdict}: bar mean >= 98 and min >= 90; kill mean < 96 or min < 85")
open(f"{out}/summary.md", "w").write("# kvc-agree (C20 stage A)\n\n" + "\n".join("- " + l for l in lines) + "\n")
print("\n".join(lines))
sys.exit(0 if verdict == "PASS" else 1)
PY
