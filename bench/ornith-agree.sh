#!/usr/bin/env bash
# usage: bench/ornith-agree.sh ENGINE CONTROL_PACK CANDIDATE_PACK OUTDIR
# Teacher-forced agreement of two Ornith packs against llama.cpp's greedy ids (bench/llama-stint-protocol.md run 5).
# One gpu-wait job: llama-server greedy 64 ids per prompt, then each pack forced on them. Exit 0 PASS, 1 FAIL/fault, 3 VOID.
# PREFIX=FILE (run 9): the ids of FILE are prepended to every prompt, so each one crosses a batched prefill chunk; the bar
# becomes candidate median >= control median - 1.0 AND candidate min >= control min - 1.0 (run 9 amendment: the
# same-arm control reads 78.12 min with the prefix, P5b shape), and the logs must read back prefill rows > 0 on the
# candidate and replay rows = N-1 on the control. CTL_ENV / CAND_ENV: extra K=V knobs for one arm (e.g. CTL_ENV=BARO_PREFILL=0).
set -euo pipefail
cd "$(dirname "$0")/.."
eng=$1 ctl=$2 cand=$3 out=$4
G=${ORNITH_GGUF:-$HOME/Models/ornith-1.5-9b-q4_K_M/Ornith-1.5-9B-Q4_K_M.gguf}
srv=${LLAMA_SERVER:-$HOME/Projects/imports/llama.cpp/build/bin/llama-server}
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
  exec gpu-wait run --vram 22 --timeout 1800 -- env ORNITH_GGUF="$G" LLAMA_SERVER="$srv" PREFIX="${PREFIX:-}" CTL_ENV="${CTL_ENV:-}" CAND_ENV="${CAND_ENV:-}" "$0" "$@"
fi
mkdir -p "$out"
prompts=(bench/mtp-prompts/p*.tokens)
if [ -n "${PREFIX:-}" ]; then
  [ -f "$PREFIX" ] || { echo "FAIL arm: no PREFIX file $PREFIX"; exit 1; }
  mkdir -p "$out/prompts"; prompts=()
  for tf in bench/mtp-prompts/p*.tokens; do { { cat "$PREFIX"; echo; cat "$tf"; } | tr -s ' \n' '  ' | sed 's/^ //; s/ $//'; echo; } > "$out/prompts/$(basename "$tf")"; prompts+=("$out/prompts/$(basename "$tf")"); done
fi
for p in "$ctl" "$cand"; do
  grep -q "\"source\": \"$G\"" "$p/identity.json" || { echo "FAIL arm: pack $p was not built from $G"; exit 1; }
done
{ echo "engine=$eng sha=$(sha256sum "$eng" | cut -c1-16) control=$ctl candidate=$cand gguf=$G"
  echo "server=$srv sha=$(sha256sum "$srv" | cut -c1-16) load=$(cut -d' ' -f1-3 /proc/loadavg)"; } | tee "$out/arm.txt"
port=$((20000 + RANDOM % 20000))
"$srv" -m "$G" -c 16384 -ngl 99 -fa on -ctk f16 -ctv f16 -np 1 --no-webui --host 127.0.0.1 --port $port > "$out/server.log" 2>&1 &
pid=$!
trap 'kill $pid 2>/dev/null || true' EXIT
for _ in $(seq 240); do curl -sf localhost:$port/health > /dev/null && break; sleep 0.5; done
curl -sf localhost:$port/health > /dev/null || { echo "FAIL llama: server not healthy, $out/server.log"; exit 1; }
for tf in "${prompts[@]}"; do
  p=$(basename "$tf" .tokens)
  ids=$(tr -s ' \n' ',,' < "$tf" | sed 's/,$//')
  curl -sf localhost:$port/completion -d "{\"prompt\": [$ids], \"n_predict\": 64, \"temperature\": 0, \"top_k\": 1, \"cache_prompt\": false, \"ignore_eos\": true, \"return_tokens\": true}" \
    | jq -r '.tokens | map(tostring) | join(" ")' > "$out/$p.llama.ids" || { echo "FAIL llama $p: completion"; exit 1; }
  [ "$(wc -w < "$out/$p.llama.ids")" = 64 ] || { echo "FAIL llama $p: not 64 ids"; exit 1; }
done
kill $pid; wait $pid 2>/dev/null || true
for arm in control candidate; do
  pack=$ctl; [ $arm = candidate ] && pack=$cand
  : > "$out/$arm.txt"
  extra=$CTL_ENV; [ $arm = candidate ] && extra=$CAND_ENV
  for tf in "${prompts[@]}"; do
    p=$(basename "$tf" .tokens)
    env BARO_PACK="$pack" BARO_SPEC=0 BARO_TMAX=16384 BARO_PROMPT="$tf" BARO_FORCE="$out/$p.llama.ids" $extra "$eng" > "$out/$p.$arm.log" 2>&1 \
      || { echo "FAIL $arm $p: engine exit"; exit 1; }
    grep -q 'TMAX: 16384 ' "$out/$p.$arm.log" || { echo "FAIL $arm $p: TMAX read-back is not 16384"; exit 1; }
    if [ -n "${PREFIX:-}" ]; then
      n=$(wc -w < "$tf"); rows=$(grep -oP ' prefill rows: \K[0-9]+' "$out/$p.$arm.log" | head -1); replay=$(grep -oP ' replay rows: \K[0-9]+' "$out/$p.$arm.log" | head -1)
      [ "$replay" = "$((n - 1))" ] || { echo "FAIL $arm $p: replay rows $replay, want $((n - 1))"; exit 1; }
      [ $arm = candidate ] && { [ "${rows:-0}" -gt 0 ] || { echo "FAIL candidate $p: prefill rows ${rows:-none}, want > 0"; exit 1; }; }
      [ $arm = control ] && { [ "${rows:-0}" = 0 ] || { echo "FAIL control $p: prefill rows $rows, want 0 (replay arm)"; exit 1; }; }
    fi
    a=$(grep -oP '^forced agreement: \K[0-9]+ / [0-9]+' "$out/$p.$arm.log") || { echo "FAIL $arm $p: no forced agreement line"; exit 1; }
    echo "$p $a" >> "$out/$arm.txt"
  done
done
awk '
  { pct = 100 * $2 / $4; v[FILENAME][++n[FILENAME]] = pct }
  END {
    for (f in v) { m = n[f]; asort(v[f]); med = (m % 2) ? v[f][(m + 1) / 2] : (v[f][m / 2] + v[f][m / 2 + 1]) / 2
      split(f, s, "/"); name = s[length(s)]; sub(/\.txt$/, "", name); printf "%s median %.2f min %.2f n %d\n", name, med, v[f][1], m }
  }' "$out/control.txt" "$out/candidate.txt" | sort | tee "$out/summary.txt"
read -r _ _ cmed _ cmin _ < <(grep '^control' "$out/summary.txt")
read -r _ _ kmed _ kmin _ < <(grep '^candidate' "$out/summary.txt")
if [ -n "${PREFIX:-}" ]; then
  awk -v c="$cmed" -v m="$kmed" -v n="$kmin" -v cn="$cmin" 'BEGIN { exit !(m >= c - 1.0 && n >= cn - 1.0) }' && { echo "PASS: candidate median $kmed min $kmin (control median $cmed min $cmin, bar median and min >= control - 1.0)"; exit 0; }
  echo "FAIL: candidate median $kmed min $kmin (control median $cmed min $cmin, bar median and min >= control - 1.0)"; exit 1
fi
awk -v c="$cmed" 'BEGIN { exit !(c >= 95.0) }' || { echo "VOID: control median $cmed < 95.0"; exit 3; }
awk -v m="$kmed" -v n="$kmin" 'BEGIN { exit !(m >= 95.0 && n >= 85.0) }' && { echo "PASS: candidate median $kmed min $kmin"; exit 0; }
echo "FAIL: candidate median $kmed min $kmin (bar median >= 95.0, min >= 85.0)"; exit 1
