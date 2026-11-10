#!/usr/bin/env bash
# MOEPF gate 1 (bench/moe-prefill-protocol.md): greedy tokens after batched MoE prefill equal
# the replay path's, same binary, one process per arm.
#   bench/moe-prefill-identity.sh ENGINE PACK OUT [resident|tier]
# env: SET=mtp (20 bench/mtp-prompts, the brief's gate) | long (prefill-prompts 128/512/1024,
#      crosses chunk boundaries with PFC) | all;  QUICK=N first N prompts;  PFC=chunk rows (default
#      engine CP);  XENV='K=V ...' extra engine env, both arms (diagnostic toggles);  EXPLORE=1 skips the preflight check and caps the verdict at UNVERIFIED, exit 3;  NGEN=64;  TMAXV=BARO_TMAX (default 2048, read back from the ready line).
# PASS = every prompt's NGEN tokens equal AND the batched arm's own echo shows prefill rows > 0 on
# every prompt long enough to prefill. Prompts under PF_MIN + 1 tokens are listed as NOT EXERCISED.
# Receipts: OUT/arm.txt, OUT/{replay,batched}.log, OUT/summary.tsv.
set -euo pipefail
cd "$(dirname "$0")/.."
eng=$1; pack=$2; out=$3; mode=${4:-resident}
set_=${SET:-mtp}; quick=${QUICK:-0}; pfc=${PFC:-0}; ngen=${NGEN:-64}; tmax=${TMAXV:-2048}; explore=${EXPLORE:-0}; xenv=${XENV:-}
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ] && command -v gpu-wait >/dev/null; then
  exec gpu-wait run --timeout 3600 -- env SET="$set_" QUICK="$quick" PFC="$pfc" NGEN="$ngen" TMAXV="$tmax" EXPLORE="$explore" XENV="$xenv" "$0" "$@"
fi
mkdir -p "$out"
if [ "$explore" = 1 ]; then echo "EXPLORE=1: preflight not checked, this run can end UNVERIFIED (exit 3) or FAIL, never PASS"
else bench/preflight.sh --check || { echo "FAIL preflight: tree changed since the last passing bench/preflight.sh"; exit 1; }; fi
case "$mode" in
  resident) modeenv="" ;;
  tier) modeenv="BARO_TIER=64 BARO_TIER_PINNED=1 BARO_TIER_ZC=1" ;;
  *) echo "FAIL args: mode '$mode' is not resident|tier"; exit 1 ;;
esac
pfcenv=""
if [ "$pfc" != 0 ]; then pfcenv="BARO_PREFILL_C=$pfc"; fi

files=()
if [ "$set_" = mtp ] || [ "$set_" = all ]; then files+=(bench/mtp-prompts/p*.tokens); fi
if [ "$set_" = long ] || [ "$set_" = all ]; then files+=(bench/prefill-prompts/p0128.tokens bench/prefill-prompts/p0512.tokens bench/prefill-prompts/p1024.tokens); fi
[ "${#files[@]}" -gt 0 ] || { echo "FAIL args: SET '$set_' selects no prompts"; exit 1; }
if [ "$quick" != 0 ]; then files=("${files[@]:0:$quick}"); fi

{ echo "gate=moe-prefill-identity mode=$mode set=$set_ explore=$explore quick=$quick pfc=$pfc ngen=$ngen tmax=$tmax prompts=${#files[@]}"
  echo "eng=$eng sha=$(sha256sum "$eng" | cut -c1-16)"
  echo "pack=$pack index_sha=$(sha256sum "$pack/index.txt" | cut -c1-16) pack_bytes=$(stat -c %s "$pack/pack.bin")"
  echo "modeenv='$modeenv' pfcenv='$pfcenv' xenv='$xenv'"
  echo "pcie=$(cat /sys/bus/pci/devices/0000:00:01.1/current_link_speed)"
  echo "commit=$(git rev-parse --short HEAD) dirty='$(git status --short | tr '\n' ';')'"; } | tee "$out/arm.txt"

: > "$out/req.jsonl"; : > "$out/names.tsv"
id=0
for tf in "${files[@]}"; do
  id=$((id + 1))
  ids=$(tr -s ' \n' ',,' < "$tf" | sed 's/^,//; s/,$//')
  echo "{\"id\":$id,\"prompt\":[$ids],\"n\":$ngen,\"spec\":false}" >> "$out/req.jsonl"
  printf '%s\t%s\t%s\n' "$id" "$(basename "$tf" .tokens)" "$(wc -w < "$tf")" >> "$out/names.tsv"
done

run_arm() { # name BARO_PREFILL
  # shellcheck disable=SC2086
  env BARO_SERVE=1 BARO_MEGA=0 BARO_SPEC=0 BARO_TMAX="$tmax" BARO_PREFILL="$2" BARO_PACK="$pack" $modeenv $pfcenv $xenv "$eng" \
    < "$out/req.jsonl" > "$out/$1.log" 2>&1 || { echo "FAIL $1 arm: engine exited non-zero, see $out/$1.log"; exit 1; }
  if grep -qiE 'NOT-RESIDENT|^Error|error:|Unhandled exception' "$out/$1.log"; then
    echo "FAIL $1 arm: fail word in $out/$1.log"; exit 1
  fi
  grep -m1 '^BARO_PREFILL:' "$out/$1.log" | sed "s/^/$1 readback: /" | tee -a "$out/arm.txt"
  grep -q "^{\"ready\":true,\"tmax\":$tmax," "$out/$1.log" || { echo "FAIL $1 arm VOID: ready line does not carry tmax $tmax, see $out/$1.log"; exit 1; }
}
run_arm replay 0
run_arm batched 1
grep -q '^BARO_PREFILL: False' "$out/replay.log" || { echo "FAIL replay arm VOID: engine did not echo BARO_PREFILL: False"; exit 1; }
grep -q '^BARO_PREFILL: True' "$out/batched.log" || { echo "FAIL batched arm VOID: engine did not echo BARO_PREFILL: True"; exit 1; }

python3 - "$out" "$ngen" "$explore" <<'EOF'
import json, re, sys
out, ngen, explore = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "1"
names = {int(a): (b, int(c)) for a, b, c in (l.rstrip("\n").split("\t") for l in open(f"{out}/names.tsv"))}
def load(arm):
    toks, rows = {}, []
    for line in open(f"{out}/{arm}.log", errors="replace"):
        m = re.search(r"prefill rows: (\d+)", line)
        if m:
            rows.append(int(m.group(1)))
        if line.startswith('{"id":') and '"tok":' in line:
            d = json.loads(line)
            toks.setdefault(d["id"], []).append(d["tok"])
    return toks, rows
rt, rrows = load("replay")
bt, brows = load("batched")
if any(rrows):
    print(f"FAIL replay arm VOID: prefill rows echo {rrows} is not all zero"); sys.exit(1)
if len(brows) != len(names):
    print(f"FAIL batched arm VOID: {len(brows)} 'prefill rows' echoes for {len(names)} requests"); sys.exit(1)
bad, idle = [], []
with open(f"{out}/summary.tsv", "w") as f:
    f.write("id\tname\tprompt_tokens\tprefill_rows\tequal\tfirst_divergence\n")
    for i, (name, n) in sorted(names.items()):
        a, b = rt.get(i, []), bt.get(i, [])
        rows = brows[i - 1]
        if len(a) != ngen or len(b) != ngen:
            print(f"FAIL {name}: token counts replay {len(a)} batched {len(b)}, expected {ngen}"); sys.exit(1)
        if n - 1 >= 16 and rows == 0:
            print(f"FAIL batched arm VOID: {name} has {n} tokens but the engine echoed 0 prefill rows"); sys.exit(1)
        div = next((k for k in range(ngen) if a[k] != b[k]), -1)
        if rows == 0:
            idle.append(name)
        if div >= 0:
            bad.append(f"{name}@{div}")
        f.write(f"{i}\t{name}\t{n}\t{rows}\t{int(div < 0)}\t{div}\n")
ex = len(names) - len(idle)
print(f"exercised {ex}/{len(names)} (NOT EXERCISED, under PF_MIN+1 tokens: {' '.join(idle) or 'none'})")
if bad:
    print(f"FAIL {len(bad)}/{len(names)} {' '.join(bad)}  (name@first divergent token index), see {out}/summary.tsv"); sys.exit(1)
if explore:
    print(f"UNVERIFIED (EXPLORE=1, no preflight): {len(names)}/{len(names)} equal over {ngen} tokens, {ex} exercised prefill"); sys.exit(3)
print(f"PASS identity {len(names)}/{len(names)} equal over {ngen} tokens, {ex} exercised prefill, see {out}/summary.tsv")
EOF
