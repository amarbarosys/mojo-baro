#!/usr/bin/env bash
# usage: bench/tier-gpu-busy.sh ENGINE OUTDIR "ENV"
# GPU busy time per decode token (kernels + copies, from rocprofv3) against
# decode wall time per token. Two runs at 16 and 64 tokens; the difference
# cancels load and prefill. busy << wall means the host round trips cost the
# time; busy ~= wall means the GPU work itself does.
set -euo pipefail
eng=$1; out=$2; envs=$3
mkdir -p "$out"
p=$(tr -s ' \n' ',' < bench/mtp-prompts/p01-water.tokens | sed 's/,$//')
run() {
  local n=$1 d=$out/n$1; rm -rf "$d"; mkdir -p "$d"
  echo "{\"id\":1,\"prompt\":[$p],\"n\":$n,\"spec\":false}" > "$d/request.jsonl"
  env BARO_SERVE=1 BARO_SPEC=0 $envs rocprofv3 --kernel-trace --memory-copy-trace -f csv -d "$d" -o trace -- "$eng" \
    < "$d/request.jsonl" > "$d/stdout.txt" 2> "$d/stderr.txt" || { echo "FAIL run n=$n: $d/stderr.txt"; exit 1; }
  grep -q '"done"' "$d/stdout.txt" || { echo "FAIL run n=$n did not finish: $d/stderr.txt"; exit 1; }
}
run 16; run 64
python3 - "$out" <<'PY'
import csv, glob, json, sys
out = sys.argv[1]
def busy(n):
    tot = {}
    for f in glob.glob(f"{out}/n{n}/**/*_trace.csv", recursive=True):
        kind = "copy" if "memory_copy" in f else "kernel"
        with open(f) as fh:
            for r in csv.DictReader(fh):
                tot[kind] = tot.get(kind, 0) + int(r["End_Timestamp"]) - int(r["Start_Timestamp"])
    return tot
def wall(n):
    for line in open(f"{out}/n{n}/stdout.txt"):
        if '"done"' in line:
            return json.loads(line)["decode_s"]
a, b = busy(16), busy(64)
dt = 48
res = {k: (b.get(k, 0) - a.get(k, 0)) / dt / 1e6 for k in ("kernel", "copy")}
res["wall_ms"] = (wall(64) - wall(16)) / dt * 1e3
print(json.dumps({k: round(v, 2) for k, v in res.items()}))
PY
