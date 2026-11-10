#!/usr/bin/env bash
# usage: bench/c19-agree.sh ENGINE PACK OUT
# C19 stage 2 (AMDHQ 06 "## C19:"): teacher-forced agreement of state files
# rebuilt from CacheGen's codec against the f32 state they came from.
# Per length (8k, 32k), each arm in its own engine process (no shared chain):
#   f32-ref  greedy, 20 requests P + suffix, 64 ids each: the reference ids
#   f32-ctl  same file, forced to f32-ref's ids (must be 20/20 at 64/64)
#   int8 / cg-L2 / cg-L1  forced likewise
# Inputs under C19_DIR (default .work/c19): states/{f32,int8}-lat_state_pos*.baro,
# s1-8k/ and s1-32k/ from tools/c19_cachegen.py --write-recon, and
# P-8000.tokens / P-32000.tokens. One gpu-wait job for both lengths.
# GATE_DRYRUN=1 runs the whole script on the CPU against a stub engine that
# echoes the forced ids (every post-GPU step runs; the verdict is not scored).
set -euo pipefail
cd "$(dirname "$0")/.."
eng=$1; pack=$2; out=$3
dir=${C19_DIR:-.work/c19}
fail() { echo "FAIL $1: $2"; exit 1; }
mkdir -p "$out"

state_for() {
    local n=$1 arm=$2 p=$((n - 1)) sub=s1-8k
    [ "$n" = 32000 ] && sub=s1-32k
    case $arm in
        f32-ref|f32-ctl) ls "$dir"/states/f32-lat_state_pos${p}_*.baro ;;
        int8) ls "$dir"/states/int8-lat_state_pos${p}_*.baro ;;
        cg-L2|cg-L1) echo "$dir/$sub/$arm.baro" ;;
    esac
}

if [ -z "${GPU_WAITING_ROOM_JOB:-}" ] && [ "${GATE_DRYRUN:-0}" != 1 ]; then
    [ -x "$eng" ] || fail setup "engine $eng missing"
    [ -f "$pack/pack.bin" ] || fail setup "pack $pack missing"
    for n in 8000 32000; do
        [ -f "$dir/P-$n.tokens" ] || fail setup "$dir/P-$n.tokens missing"
        for arm in f32-ref int8 cg-L2 cg-L1; do
            f=$(state_for $n $arm); [ -f "$f" ] || fail setup "state $arm $n missing: $f"
        done
    done
    {
        echo "engine=$eng sha=$(sha256sum "$eng" | cut -c1-16) commit=$(git rev-parse --short HEAD) dirty='$(git status --porcelain | tr '\n' ' ')'"
        echo "pack=$pack pack_bin_sha256=$(sha256sum "$pack/pack.bin" | cut -c1-64)"
        for n in 8000 32000; do
            for arm in f32-ref int8 cg-L2 cg-L1; do
                f=$(state_for $n $arm); echo "state n=$n arm=$arm $(sha256sum "$f" | cut -c1-64) $(stat -c %s "$f") $f"
            done
            echo "P n=$n $(sha256sum "$dir/P-$n.tokens" | cut -c1-64)"
        done
        echo "suffixes=$(ls bench/mtp-prompts/p*.tokens | head -20 | xargs sha256sum | sha256sum | cut -c1-16) n=64 tmax=32768 spec=0"
    } > "$out/arm.txt"
    cat "$out/arm.txt"
    exec gpu-wait run --priority 20 --vram 22 --timeout 900 -- env C19_DIR="$dir" "$0" "$eng" "$pack" "$out"
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
    local n=$1 arm=$2 st=$3 ref=${4:-}
    local d="$out/n$n-$arm"
    mkdir -p "$d"
    python3 - "$dir/P-$n.tokens" "$st" "$d/requests.jsonl" "$ref" <<'PY'
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
    env BARO_SERVE=1 BARO_SPEC=0 BARO_PACK="$pack" BARO_TMAX=32768 $eng < "$d/requests.jsonl" > "$d/out.txt" 2> "$d/err.txt" \
        || fail "engine n=$n $arm" "$d/err.txt"
    echo "$((SECONDS - t0))" > "$d/wall_s.txt"
    grep -q '"ready":true' "$d/out.txt" || fail "engine n=$n $arm" "no ready line in $d/out.txt"
    python3 - "$d" "$n" "$st" "$ref" <<'PY'
import json, re, sys
d, n, st, ref = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
lines = open(f"{d}/out.txt").read().splitlines()
loads = [l for l in lines if l.startswith("state loaded:") or l.startswith("state hot:")]
if len(loads) != 20 or any(st not in l for l in loads):
    sys.exit(f"FAIL parse: {d}: {len(loads)} state loaded/hot lines naming {st}, want 20")
pos = [int(re.search(r"pos (\d+)", l).group(1)) for l in loads]
if any(p != n - 1 for p in pos):
    sys.exit(f"FAIL void: {d}: state loaded at pos {sorted(set(pos))}, want {n - 1}")
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
            f.write(f"{i + 1}\t{a}\t{c}\t{pos[i]}\n")
print(f"{d}: 20 requests, load pos {sorted(set(pos))}, agreement lines {len(agree)}")
PY
}

for n in 8000 32000; do
    run_arm $n f32-ref "$(state_for $n f32-ref)"
    for arm in f32-ctl int8 cg-L2 cg-L1; do
        run_arm $n $arm "$(state_for $n $arm)" "$out/n$n-f32-ref/ids.json"
    done
done

python3 - "$out" <<'PY' | tee "$out/summary.md"
import sys, pathlib
out = pathlib.Path(sys.argv[1])
print("| n | arm | mean agreement | min | 64/64 count | verdict |")
print("|---|---|---|---|---|---|")
void = False
for n in (8000, 32000):
    for arm in ("f32-ctl", "int8", "cg-L2", "cg-L1"):
        rows = [l.split("\t") for l in (out / f"n{n}-{arm}" / "agree.tsv").read_text().splitlines()]
        pct = [100.0 * int(a) / int(c) for _, a, c, _ in rows]
        mean, mn, full = sum(pct) / len(pct), min(pct), sum(p == 100.0 for p in pct)
        if arm == "f32-ctl":
            v = "control OK" if full == 20 else "VOID (control below 20/20)"
            void |= full != 20
        else:
            v = "QUALITY PASS" if mean >= 98.0 and mn >= 90.0 else "QUALITY FAIL"
        print(f"| {n} | {arm} | {mean:.2f}% | {mn:.2f}% | {full}/20 | {v} |")
print()
print("VOID" if void else "valid")
PY
if [ "${GATE_DRYRUN:-0}" = 1 ]; then echo "PASS gate-dryrun: c19-agree post-steps ran on stub output"; exit 0; fi
grep -qx valid "$out/summary.md" || fail void "$out/summary.md"
echo "DONE c19-agree: $out/summary.md"
