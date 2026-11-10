#!/usr/bin/env bash
# MOEPF gate 1b (bench/moe-prefill-protocol.md, round 1b): above PF_EXACT_T = 256 prompt tokens
# the chunk path uses WMMA attention, which reorders sums, so identity there is TEACHER-FORCED
# AGREEMENT against the replay path's own greedy tokens (CLAUDE.md: never greedy equality past
# ~256 ids). Same binary, three processes: replay greedy (the reference ids), replay forced (the
# same-arm control, must read NGEN/NGEN or the instrument is broken), batched forced.
#   bench/moe-prefill-agree.sh ENGINE PACK OUT [resident|tier]
# env: LENS="0512 1024 8192" (bench/prefill-prompts), NGEN=64, MIN_MEAN=99 (percent), EXPLORE=1
#      skips the preflight check and caps the verdict at UNVERIFIED exit 3.
# PASS = control NGEN/NGEN on every prompt AND batched mean agreement >= MIN_MEAN. The per-prompt
# minimum is reported, not gated (P5b: a min-over-prompts bar fails the unpatched base too).
set -euo pipefail
cd "$(dirname "$0")/.."
eng=$1; pack=$2; out=$3; mode=${4:-tier}
lens=${LENS:-0512 1024 8192}; ngen=${NGEN:-64}; minmean=${MIN_MEAN:-99}; explore=${EXPLORE:-0}
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ] && command -v gpu-wait >/dev/null; then
  exec gpu-wait run --timeout 3600 -- env LENS="$lens" NGEN="$ngen" MIN_MEAN="$minmean" EXPLORE="$explore" "$0" "$@"
fi
mkdir -p "$out"
if [ "$explore" = 1 ]; then echo "EXPLORE=1: preflight not checked, this run can end UNVERIFIED (exit 3) or FAIL, never PASS"
else bench/preflight.sh --check || { echo "FAIL preflight: tree changed since the last passing bench/preflight.sh"; exit 1; }; fi
case "$mode" in
  resident) modeenv="" ;;
  tier) modeenv="BARO_TIER=64 BARO_TIER_PINNED=1 BARO_TIER_ZC=1" ;;
  *) echo "FAIL args: mode '$mode' is not resident|tier"; exit 1 ;;
esac
longest=0
for n in $lens; do
  [ -f "bench/prefill-prompts/p$n.tokens" ] || { echo "FAIL args: bench/prefill-prompts/p$n.tokens missing"; exit 1; }
  c=$(wc -w < "bench/prefill-prompts/p$n.tokens"); if [ "$c" -gt "$longest" ]; then longest=$c; fi
done
tmax=$((longest + ngen + 1024))
{ echo "gate=moe-prefill-agree mode=$mode lens='$lens' ngen=$ngen min_mean=$minmean explore=$explore tmax=$tmax"
  echo "eng=$eng sha=$(sha256sum "$eng" | cut -c1-16)"
  echo "pack=$pack index_sha=$(sha256sum "$pack/index.txt" | cut -c1-16)"
  echo "commit=$(git rev-parse --short HEAD) dirty='$(git status --short | tr '\n' ';')'"; } | tee "$out/arm.txt"

run_arm() { # name BARO_PREFILL reqfile
  # shellcheck disable=SC2086
  env BARO_SERVE=1 BARO_MEGA=0 BARO_SPEC=0 BARO_CKPT=0 BARO_TMAX="$tmax" BARO_PREFILL="$2" BARO_PACK="$pack" $modeenv "$eng" \
    < "$3" > "$out/$1.log" 2>&1 || { echo "FAIL $1 arm: engine exited non-zero, see $out/$1.log"; exit 1; }
  if grep -qiE 'NOT-RESIDENT|^Error|error:|Unhandled exception' "$out/$1.log"; then
    echo "FAIL $1 arm: fail word in $out/$1.log"; exit 1
  fi
  grep -m1 '^BARO_PREFILL:' "$out/$1.log" | sed "s/^/$1 readback: /" | tee -a "$out/arm.txt"
  grep -q "^{\"ready\":true,\"tmax\":$tmax," "$out/$1.log" || { echo "FAIL $1 arm VOID: ready line does not carry tmax $tmax"; exit 1; }
}

python3 - "$out" "$ngen" greedy $lens <<'EOF'
import sys
out, ngen, _, *lens = sys.argv[1:]
with open(f"{out}/req-greedy.jsonl", "w") as f:
    for i, n in enumerate(lens, 1):
        ids = open(f"bench/prefill-prompts/p{n}.tokens").read().split()
        f.write('{"id":%d,"prompt":[%s],"n":%s,"spec":false}\n' % (i, ",".join(ids), ngen))
EOF
run_arm greedy 0 "$out/req-greedy.jsonl"
python3 - "$out" "$ngen" $lens <<'EOF'
import json, sys
out, ngen, *lens = sys.argv[1:]
toks = {}
for line in open(f"{out}/greedy.log", errors="replace"):
    if line.startswith('{"id":') and '"tok":' in line:
        d = json.loads(line); toks.setdefault(d["id"], []).append(d["tok"])
with open(f"{out}/req-force.jsonl", "w") as f:
    for i, n in enumerate(lens, 1):
        ref = toks.get(i, [])
        if len(ref) != int(ngen):
            print(f"FAIL greedy arm: p{n} produced {len(ref)} tokens, expected {ngen}"); sys.exit(1)
        ids = open(f"bench/prefill-prompts/p{n}.tokens").read().split()
        f.write('{"id":%d,"prompt":[%s],"n":%s,"spec":false,"force":[%s]}\n' % (i, ",".join(ids), ngen, ",".join(map(str, ref))))
EOF
run_arm control 0 "$out/req-force.jsonl"
run_arm batched 1 "$out/req-force.jsonl"
grep -q '^BARO_PREFILL: False' "$out/control.log" || { echo "FAIL control arm VOID: BARO_PREFILL echo is not False"; exit 1; }
grep -q '^BARO_PREFILL: True' "$out/batched.log" || { echo "FAIL batched arm VOID: BARO_PREFILL echo is not True"; exit 1; }

python3 - "$out" "$ngen" "$minmean" "$explore" $lens <<'EOF'
import re, sys
out, ngen, minmean, explore, *lens = sys.argv[1:]
ngen = int(ngen)
def scores(arm):
    s, rows = [], []
    for line in open(f"{out}/{arm}.log", errors="replace"):
        m = re.search(r"forced agreement: (\d+) / (\d+)", line)
        if m:
            s.append((int(m.group(1)), int(m.group(2))))
        m = re.search(r"prefill rows: (\d+)", line)
        if m:
            rows.append(int(m.group(1)))
    return s, rows
c, crows = scores("control")
b, brows = scores("batched")
if len(c) != len(lens) or len(b) != len(lens):
    print(f"FAIL VOID: forced-agreement lines control {len(c)} batched {len(b)}, expected {len(lens)}"); sys.exit(1)
if any(crows) or not all(brows):
    print(f"FAIL VOID: prefill rows echo control {crows} batched {brows}"); sys.exit(1)
with open(f"{out}/summary.tsv", "w") as f:
    f.write("len\tcontrol\tbatched\tof\tprefill_rows\n")
    for n, (ck, cn), (bk, bn), r in zip(lens, c, b, brows):
        f.write(f"{n}\t{ck}\t{bk}\t{bn}\t{r}\n")
        print(f"p{n}: control {ck}/{cn}  batched {bk}/{bn}  prefill rows {r}")
bad = [n for n, (ck, cn) in zip(lens, c) if ck != cn or cn != ngen]
if bad:
    print(f"FAIL {len(bad)}/{len(lens)} control arm below {ngen}/{ngen} on {' '.join('p' + x for x in bad)}: the instrument is broken, not the candidate"); sys.exit(1)
pct = [100.0 * bk / bn for bk, bn in b]
mean, low = sum(pct) / len(pct), min(pct)
verdict = f"mean {mean:.2f}% (bar {minmean}%), min {low:.2f}% (reported, not gated), see {out}/summary.tsv"
if mean < float(minmean):
    print(f"FAIL agreement {verdict}"); sys.exit(1)
if explore == "1":
    print(f"UNVERIFIED (EXPLORE=1, no preflight): agreement {verdict}"); sys.exit(3)
print(f"PASS agreement {verdict}")
EOF
