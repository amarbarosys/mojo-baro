#!/usr/bin/env bash
# usage: bench/moe-token-profile.sh ENGINE PACK OUTDIR [ENV...]
# Per-token GPU time per kernel name on the MoE decode path, from rocprofv3 kernel traces of
# one request at two generated lengths (16 and 48); the difference cancels load and prefill.
# Prints us/token per kernel (top 25), total kernel us/token, launches/token, and wall us/token
# from the engine's own tok/s_gen, so the gap (launch + idle) is visible. Run under gpu-wait.
# PROMPT=FILE (default bench/mtp-prompts/p01-water.tokens, 13 ids) sets the context the two lengths decode after.
set -euo pipefail
cd "$(dirname "$0")/.."
eng=$1; pack=$2; out=$3; shift 3
mkdir -p "$out"
command -v rocprofv3 >/dev/null || { echo "FAIL setup: no rocprofv3 on PATH"; exit 1; }
prompt=${PROMPT:-bench/mtp-prompts/p01-water.tokens}   # PROMPT=ids file: decode cost at another context length
[ -s "$prompt" ] || { echo "FAIL setup: prompt $prompt missing"; exit 1; }
p=$(tr -s ' \n' ',' < "$prompt" | sed 's/,$//')
for n in 16 48; do
  d=$out/gen$n; rm -rf "$d"; mkdir -p "$d"
  echo "{\"id\":1,\"prompt\":[$p],\"n\":$n,\"spec\":false}" > "$d/req.jsonl"
  env BARO_SERVE=1 BARO_SPEC=0 BARO_PACK="$pack" "$@" rocprofv3 --kernel-trace -f csv -d "$d" -o trace -- "$eng" < "$d/req.jsonl" > "$d/stdout.txt" 2> "$d/stderr.txt" || { echo "FAIL run n=$n: see $d/stderr.txt"; exit 1; }
  grep -q '"done"' "$d/stdout.txt" || { echo "FAIL run n=$n did not finish: see $d/stderr.txt"; exit 1; }
done
python3 - "$out" <<'PY'
import csv, glob, re, sys, collections
out = sys.argv[1]
def load(n):
    f = glob.glob(f"{out}/gen{n}/**/*kernel_trace.csv", recursive=True)
    if not f: sys.exit(f"FAIL no trace csv for n={n}")
    t = collections.Counter(); c = collections.Counter()
    for r in csv.DictReader(open(f[0])):
        k = re.sub(r'_[0-9A-Za-z]{6,}.*$', '', r['Kernel_Name'])[:60]
        t[k] += int(r['End_Timestamp']) - int(r['Start_Timestamp']); c[k] += 1
    return t, c
(t1, c1), (t2, c2) = load(16), load(48)
# doorbell_* kernels spin until the host rings: their time is host stall, not GPU work, and it differs between
# the two runs (1140 vs 154 ms, S4 2026-10-12), so they are left out of the per-token totals
d = {k: (t2[k] - t1[k]) / 32 / 1e3 for k in set(t1) | set(t2) if not k.startswith("doorbell")}
L = sum(c2.values()) - sum(c1.values())
tot = sum(d.values())
wall = []
for n in (16, 48):
    m = re.findall(r'tok/s_gen: ([0-9.]+)', open(f"{out}/gen{n}/stdout.txt").read() + open(f"{out}/gen{n}/stderr.txt").read())
    wall.append(1e6 / float(m[-1]) if m else float('nan'))
print(f"kernel time {tot:.0f} us/token, launches {L/32:.0f}/token, wall (engine tok/s_gen, n=48) {wall[1]:.0f} us/token, gap {wall[1]-tot:.0f} us")
for k, v in sorted(d.items(), key=lambda x: -x[1])[:25]:
    print(f"{v:8.1f} us {100*v/tot:5.1f}%  x{(c2[k]-c1[k])/32:5.0f}  {k}")
PY
