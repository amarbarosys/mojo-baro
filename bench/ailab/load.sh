#!/usr/bin/env bash
# usage: bench/ailab/load.sh URL N ROUNDS OUTDIR
# Lockstep load in the street-sim shape: ROUNDS rounds; each round sends N chat requests at once (body b of round r
# gets state ((r*N + b) % 20) + 1 from bench/ailab/states/) and waits for all N before the next round.
# Writes OUTDIR/r<round>-b<body>.json (response), OUTDIR/lat.tsv (round body state http_code seconds completion_tokens),
# OUTDIR/summary.txt (N, requests, wall s, aggregate completion tok/s, p50/p95 latency s, errors).
# Env: STATE_DIR (bench/ailab/states), STATES ("1 2 ... 20") = the state order (request i gets STATES[i % len]); SAMPLE ("T TOP_K SEED") turns
# every request into a sampled one (default: the states' own temperature 0).
# Exit 0 = every request HTTP 200 with a choice; 1 = any failure (named on stdout).
set -euo pipefail
[ $# -eq 4 ] || { echo "FAIL usage: load.sh URL N ROUNDS OUTDIR"; exit 1; }
url=$1; n=$2; rounds=$3; out=$4
here=$(cd "$(dirname "$0")" && pwd)
read -r -a order <<<"${STATES:-$(seq -s ' ' 1 20)}"
sample=${SAMPLE:-}
mkdir -p "$out"; : > "$out/lat.tsv"
one() {  # round body
  local r=$1 b=$2 s f t0 t1 code
  s=${order[$(( (r * n + b) % ${#order[@]} ))]}; f=$(printf '%s/s%02d.json' "${STATE_DIR:-$here/states}" "$s")
  if [ -n "$sample" ]; then
    read -r st sk ss <<<"$sample"
    jq --argjson t "$st" --argjson k "$sk" --argjson sd "$ss" '.temperature = $t | .top_k = $k | .seed = $sd' "$f" > "$out/body-r$r-b$b.json"
    f=$out/body-r$r-b$b.json
  fi
  t0=$(date +%s.%N)
  code=$(curl -s -o "$out/r$r-b$b.json" -w '%{http_code}' -H 'content-type: application/json' --data-binary @"$f" \
    --max-time 900 "$url/v1/chat/completions" || echo 000)
  t1=$(date +%s.%N)
  printf '%d\t%d\t%d\t%s\t%s\t%s\n' "$r" "$b" "$s" "$code" "$(awk -v a="$t0" -v b="$t1" 'BEGIN {printf "%.3f", b - a}')" \
    "$(jq -r '.usage.completion_tokens // 0' "$out/r$r-b$b.json" 2>/dev/null || echo 0)" >> "$out/lat.tsv"
}
w0=$(date +%s.%N)
for r in $(seq 0 $((rounds - 1))); do
  for b in $(seq 0 $((n - 1))); do one "$r" "$b" & done
  wait
done
w1=$(date +%s.%N)
wall=$(awk -v a="$w0" -v b="$w1" 'BEGIN {printf "%.3f", b - a}')
bad=$(awk -F'\t' '$4 != 200' "$out/lat.tsv" | wc -l)
tok=$(awk -F'\t' '{s += $6} END {print s + 0}' "$out/lat.tsv")
read -r p50 p95 < <(cut -f5 "$out/lat.tsv" | sort -g | awk '{a[NR] = $1} END {i50 = int(NR * 0.50 + 0.5); i95 = int(NR * 0.95 + 0.5);
  if (i50 < 1) i50 = 1; if (i95 < 1) i95 = 1; print a[i50], a[i95]}')
{ echo "url=$url n=$n rounds=$rounds requests=$((n * rounds)) wall_s=$wall"
  echo "completion_tokens=$tok agg_tok_s=$(awk -v t="$tok" -v w="$wall" 'BEGIN {printf "%.2f", t / w}') p50_s=$p50 p95_s=$p95 errors=$bad"; } | tee "$out/summary.txt"
[ "$bad" -eq 0 ] || { echo "FAIL load: $bad requests not HTTP 200 (see $out/lat.tsv)"; exit 1; }
