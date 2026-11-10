#!/usr/bin/env bash
# Lane moeprefill (bench/moeprefill-protocol.md P4, P5): same-job A/B of two MoE engine binaries on batched prefill, tier mode.
#   bench/moe-prefill-ab.sh ENGINE_A ENGINE_B TIER_PACK OUT
# 20 prompts of 768 tokens cut from bench/prefill-prompts/p16384.tokens, one BARO_SERVE=1 process per arm run, order A B B A,
# each process warms up on p0128 first; every request's 16 greedy ids are compared across all four runs (identity on every
# timed run), prefill_s comes from the engine's done line. Then one rocprofv3 kernel trace per arm on p1024 (one-shot) for the
# per-kernel receipt. Read-back: ready tmax, prefill_rows and cached per request, the tier line, no fail word. Equal shas refuse.
# env: NP=20 prompts, LEN=768 tokens, VRAM=14, KERNEL=moe_down_q6k_grouped (receipt filter), SKIP_AB=1 reuses OUT's arm logs and
# runs only the trace receipt (after a trace-step failure; the verdict is then recomputed from the existing logs).
set -euo pipefail
cd "$(dirname "$0")/.."
a=$1 b=$2 pack=$3 out=$4
np=${NP:-20}; len=${LEN:-768}; kern=${KERNEL:-moe_down_q6k_grouped}
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
  exec gpu-wait run --vram "${VRAM:-14}" --timeout 1200 --label "moe-prefill-ab" -- env NP="$np" LEN="$len" KERNEL="$kern" SKIP_AB="${SKIP_AB:-0}" "$0" "$@"
fi
rp=${ROCPROF:-/opt/rocm/bin/rocprofv3}
[ -x "$rp" ] || { echo "FAIL setup: no rocprofv3 at $rp"; exit 1; }
[ -f "$pack/experts.bin" ] || { echo "FAIL setup: $pack is not a tier-split pack"; exit 1; }
sa=$(sha256sum "$a" | cut -c1-16); sb=$(sha256sum "$b" | cut -c1-16)
[ "$sa" != "$sb" ] || { echo "FAIL setup: both arms hash $sa"; exit 1; }
mkdir -p "$out/prompts"
tr -s ' \n' '\n' < bench/prefill-prompts/p16384.tokens | awk -v d="$out/prompts" -v L="$len" -v NP="$np" \
  'NF && NR <= NP * L { f = sprintf("%s/q%02d.tokens", d, int((NR - 1) / L) + 1); printf "%s%s", (((NR - 1) % L) ? " " : ""), $1 > f; if (NR % L == 0) printf "\n" > f }'
[ "$(ls "$out"/prompts | wc -l)" = "$np" ] || { echo "FAIL setup: wanted $np prompt files"; exit 1; }
tenv=(BARO_TIER=64 BARO_TIER_PINNED=1 BARO_SPEC=0 BARO_MEGA=0 BARO_CKPT=0 BARO_TMAX=16384)
echo "A=$a sha=$sa B=$b sha=$sb pack=$pack np=$np len=$len env=${tenv[*]} load=$(cut -d' ' -f1-3 /proc/loadavg) commit=$(git rev-parse --short HEAD)" | tee "$out/arm.txt"
req() { # id file n
  printf '{"id":%d,"prompt":[%s],"n":%d,"spec":false}\n' "$1" "$(tr -s ' \n' ',' < "$2" | sed 's/,$//')" "$3"
}
run_arm() { # arm engine tag
  local arm=$1 eng=$2 tag=$3 log="$out/$3.log"
  { req 0 bench/prefill-prompts/p0128.tokens 16; for i in $(seq 1 "$np"); do req "$i" "$out/prompts/q$(printf %02d "$i").tokens" 16; done; } \
    | env "${tenv[@]}" BARO_SERVE=1 BARO_PACK="$pack" "$eng" > "$log" 2>&1 || { echo "FAIL $tag: engine exit"; tail -3 "$log"; exit 1; }
  grep -q '"ready":true,"tmax":16384,' "$log" || { echo "FAIL $tag: ready tmax read-back"; exit 1; }
  grep -q 'expert tier: cap 64 ' "$log" || { echo "FAIL $tag: tier line"; exit 1; }
  grep -q 'BARO_PREFILL: True' "$log" || { echo "FAIL $tag: prefill off"; exit 1; }
  ! grep -qiE 'NOT-RESIDENT|fail word: [1-9]|Unhandled exception' "$log" || { echo "FAIL $tag: fail word"; exit 1; }
  echo "$tag sha=$(sha256sum "$eng" | cut -c1-16) binary ran: $(grep -c '"done":true' "$log") done lines" >> "$out/arm.txt"
}
if [ "${SKIP_AB:-0}" = 1 ]; then echo "SKIP_AB=1: arm runs reused from $out, verdict below is from those logs" | tee -a "$out/arm.txt"
else run_arm A "$a" A1; run_arm B "$b" B1; run_arm B "$b" B2; run_arm A "$a" A2; fi
python3 -I - "$out" "$np" "$len" <<'PY'
import json, sys, statistics as st
out, np_, L = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
runs = {}
for tag in ("A1", "B1", "B2", "A2"):
    toks, done = {}, {}
    for line in open(f"{out}/{tag}.log"):
        if not line.startswith("{"): continue
        d = json.loads(line)
        if "tok" in d: toks.setdefault(d["id"], []).append(d["tok"])
        elif d.get("done"): done[d["id"]] = d
    for i in range(1, np_ + 1):
        if i not in done: sys.exit(f"FAIL {tag}: request {i} has no done line")
        if done[i]["prefill_rows"] != L - 1 or done[i]["cached"] != 0:
            sys.exit(f"FAIL {tag}: request {i} prefill_rows {done[i]['prefill_rows']} cached {done[i]['cached']}")
        if len(toks[i]) != 16: sys.exit(f"FAIL {tag}: request {i} emitted {len(toks[i])} tokens")
    runs[tag] = (toks, done)
same = sum(all(runs[t][0][i] == runs["A1"][0][i] for t in ("B1", "B2", "A2")) for i in range(1, np_ + 1))
pa = {i: st.mean([runs[t][1][i]["prefill_s"] for t in ("A1", "A2")]) for i in range(1, np_ + 1)}
pb = {i: st.mean([runs[t][1][i]["prefill_s"] for t in ("B1", "B2")]) for i in range(1, np_ + 1)}
ma, mb = st.median(pa.values()), st.median(pb.values())
paired = st.median([pb[i] / pa[i] for i in pa])
with open(f"{out}/summary.tsv", "w") as f:
    f.write("prompt\tA_s\tB_s\tB_over_A\tids_equal\n")
    for i in range(1, np_ + 1):
        f.write(f"{i}\t{pa[i]:.4f}\t{pb[i]:.4f}\t{pb[i] / pa[i]:.4f}\t{int(all(runs[t][0][i] == runs['A1'][0][i] for t in ('B1', 'B2', 'A2')))}\n")
print(f"prefill_s median A {ma:.4f} B {mb:.4f} ratio B/A {mb / ma:.4f} paired median {paired:.4f} n={np_} rows={L - 1}")
print(f"identity {same}/{np_} prompts with all four runs' 16 ids equal")
if same != np_: sys.exit("FAIL identity")
PY
for arm in A B; do
  eng=$a; [ $arm = B ] && eng=$b
  d="$out/tr-$arm"; rm -rf "$d"; mkdir -p "$d"
  "$rp" --kernel-trace --output-format csv -d "$d" -o tr -- env "${tenv[@]}" BARO_PACK="$pack" BARO_PROMPT=bench/prefill-prompts/p1024.tokens "$eng" > "$d/run.log" 2>&1 \
    || { echo "FAIL trace $arm: engine exit"; exit 1; }
  grep -q '^GENERATED' "$d/run.log" || { echo "FAIL trace $arm: no GENERATED"; exit 1; }
  csv=$(find "$d" -name '*kernel_trace.csv' -print -quit)
  awk -F, -v k="$kern" -v arm="$arm" 'NR == 1 { for (i = 1; i <= NF; i++) { gsub(/"/, "", $i); if ($i == "Start_Timestamp") st = i; if ($i == "End_Timestamp") en = i }; next }
    index($0, k) { n++; s += $en - $st } END { if (!st || !en || !n) { print "FAIL trace " arm ": no timestamp columns or no " k " rows"; exit 1 }
    printf "%s %s launches %d total_us %.0f\n", arm, k, n, s / 1e3 }' "$csv" | tee -a "$out/arm.txt"
done
cmp -s <(grep '^GENERATED' "$out/tr-A/run.log") <(grep '^GENERATED' "$out/tr-B/run.log") && echo "trace runs: p1024 GENERATED equal" | tee -a "$out/arm.txt" || { echo "FAIL trace: p1024 GENERATED differ"; exit 1; }
echo "PASS moe-prefill-ab: see $out/summary.tsv and $out/arm.txt"
