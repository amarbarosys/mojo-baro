#!/usr/bin/env bash
# Does spark's batched prefill slow the decode that follows? One spark binary, env arms interleaved per prompt, at
# BARO_GEN 64 and 256 (a start transient shrinks at 256, a kernel cause does not). Per run: prefill_s, tok/s_gen and a
# hash of the generated ids (equal across arms). Ratios against the first arm.
# usage: [ARMS="name:K=V,K=V name2:..."] tools/prefill_decode_ab.sh SPARK PACK OUTDIR
set -euo pipefail
cd "$(dirname "$0")/.."
spark=$1 pack=$2 out=$3
arms=${ARMS:-"rows0:BARO_PREFILL_ROWS=0 rows1:BARO_PREFILL_ROWS=1"}
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
  exec gpu-wait run --timeout 1800 -- env ARMS="$arms" "$0" "$@"
fi
mkdir -p "$out"
echo "spark=$spark sha=$(sha256sum "$spark" | cut -c1-16) pack=$pack arms=$arms load=$(cut -d' ' -f1-3 /proc/loadavg)" | tee "$out/arm.txt"
: > "$out/runs.txt"
for f in tests/prompts/*.ids; do
  p=$(basename "$f" .ids)
  tr ',' ' ' < "$f" > "$out/$p.tokens"
  for g in 64 256; do for a in $arms; do
    n=${a%%:*} kv=${a#*:}
    log=$(env ${kv//,/ } BARO_GEN=$g BARO_PROMPT="$out/$p.tokens" BARO_PACK="$pack" BARO_SPEC=0 BARO_MEGA=0 "$spark" 2>&1) \
      || { echo "FAIL run $p $n g$g: engine exit"; echo "$log" | tail -3; exit 1; }
    echo "$log" > "$out/$p.g$g.$n.log"
    pf=$(echo "$log" | grep -oP 'prefill_s: \K[0-9.e-]+' || true)
    ts=$(echo "$log" | grep -oP 'tok/s_gen: \K[0-9.]+' || true)
    gen=$(echo "$log" | grep -oP '^generated: \K.*' || true)
    [ -n "$ts" ] && [ -n "$pf" ] && [ -n "$gen" ] || { echo "FAIL run $p $n g$g: no tok/s_gen, prefill_s or generated ids"; exit 1; }
    ids=$(echo "$gen" | md5sum | cut -c1-8)
    echo "$p g$g $n tok_s $ts prefill_s $pf ids $ids" | tee -a "$out/runs.txt"
  done; done
done
python3 - "$out/runs.txt" $arms <<'PY'
import sys, statistics as st
rows = [l.split() for l in open(sys.argv[1])]
names = [a.split(":")[0] for a in sys.argv[2:]]
for g in ("g64", "g256"):
    by = {n: {r[0]: r for r in rows if r[1] == g and r[2] == n} for n in names}
    ref = by[names[0]]
    for n in names:
        b = by[n]
        ratio = [float(b[k][4]) / float(ref[k][4]) for k in ref]
        same = sum(ref[k][8] == b[k][8] for k in ref)
        print(f"{g} {n:8s} median {st.median(float(r[4]) for r in b.values()):6.1f} tok/s, /{names[0]} median {st.median(ratio):.4f} "
              f"(min {min(ratio):.4f}, max {max(ratio):.4f}), ids same {same}/{len(ref)}, prefill_s median {st.median(float(r[6]) for r in b.values()):.4f}")
PY
