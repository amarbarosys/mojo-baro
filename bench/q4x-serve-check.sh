#!/usr/bin/env bash
# q4x serve round check (bench/q4x-serve-protocol.md), one gpu-wait job:
#   I1 OLD vs NEW argv block sums (warm off), S1 OLD argv per-request wall on 5 prompts, R1/R2/R3 NEW resident arms
#   (Q4X_STDIN, 20 prompts + p01 again; default / Q4X_WARM=0 / Q4X_ECACHE_GB=17), I2 reset identity, C1 8192-token
#   prefill chunked (default) vs one forward (Q4X_CHUNK=8192), hidden bytes compared. Every arm's slots, warm and cap
#   lines are read back into OUT/receipt.txt.
# MODE=http (lane q4x-serve16k): NEW behind baro-serve at Q4X_TMAX >= 16384 and max_tokens >= 16000, ids of /v1/completions
#   (the engine's own ids line) vs the Q4X_STDIN run of the same prompts, streamed text vs /detokenize; env TOK=tokenizer.json
#   BSRV=baro-serve NP=prompts MAX_TOKENS=16000 Q4X_TMAX=16384.
# MODE=margin (protocol amendment 1): I1, then R1 vs R3 resident in order R1 R3 R3 R1, gtt_used read before/after.
# usage: bench/q4x-serve-check.sh OUT_DIR      env OLD=bin NEW=bin MODE=margin
set -euo pipefail
OUT=${1:?usage: q4x-serve-check.sh OUT_DIR}
REPO=$(cd "$(dirname "$0")/.." && pwd)
. "$REPO/bench/min16k.sh"
GGUF=${Q4X_GGUF:-$HOME/Models/qwen3.8-flash-next-gsq/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf}
OLD=${OLD:-$REPO/.work/flash-slim/q4x}; NEW=${NEW:-$REPO/.work/q4x-serve/q4x}
LONG=${LONG:-$REPO/.work/flash-slim/prompts/trace.ids}
NG=64
if [ -z "${Q4X_SERVE_INNER:-}" ]; then
  mkdir -p "$OUT"
  if [ "${MODE:-}" = http ]; then OLD=$NEW; LONG=$GGUF; min16k "${Q4X_TMAX:-16384}" "${MAX_TOKENS:-16000}"; for f in "${TOK:?TOK}" "${BSRV:?BSRV}"; do [ -e "$f" ] || { echo "FAIL preflight: $f missing"; exit 1; }; done; fi
  for f in "$OLD" "$NEW" "$LONG" "$GGUF.warm"; do [ -e "$f" ] || { echo "FAIL preflight: $f missing"; exit 1; }; done
  rc=0
  gpu-wait run --vram 20 --timeout "${GPU_TIMEOUT:-1800}" -- env Q4X_SERVE_INNER=1 Q4X_GGUF="$GGUF" OLD="$OLD" NEW="$NEW" LONG="$LONG" MODE="${MODE:-}" TOK="${TOK:-}" BSRV="${BSRV:-}" NP="${NP:-2}" MAX_TOKENS="${MAX_TOKENS:-16000}" Q4X_TMAX="${Q4X_TMAX:-16384}" HOME="$HOME" \
    bash "$REPO/bench/q4x-serve-check.sh" "$OUT" || rc=$?
  gpu-wait stats --days 1 | tail -3 >> "$OUT/receipt.txt"
  [ $rc = 0 ] || { echo "FAIL job rc $rc, see $OUT/receipt.txt"; exit $rc; }
  exit 0
fi
cd "$REPO"
export MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=2
TMAX=${Q4X_TMAX:-16384}; MT=${MAX_TOKENS:-16000}
unset Q4X_WARM Q4X_CHUNK Q4X_ECACHE_GB Q4X_PREFILL Q4X_STDIN Q4X_TMAX
R=$OUT/receipt.txt
fail() { echo "FAIL $1: $2 (receipt $R)" | tee -a "$R"; exit 1; }
vram() { for d in /sys/class/drm/card*/device; do [ -f $d/mem_info_vram_total ] && awk -v t=$(cat $d/mem_info_vram_total) -v u=$(cat $d/mem_info_vram_used) 'BEGIN { if (t > 20e9) printf "%.2f", (t - u) / 2^30 }'; done; :; }
gtt() { for d in /sys/class/drm/card*/device; do [ -f $d/mem_info_vram_total ] && awk -v t=$(cat $d/mem_info_vram_total) -v g=$(cat $d/mem_info_gtt_used) 'BEGIN { if (t > 20e9) printf "%.2f", g / 2^30 }'; done; :; }
res() { fincore --bytes --noheadings --output RES "$1" | awk '{s+=$1} END {printf "%.1f", s/2**30}'; }
ids() { jq -r 'join(" ")' "$1"; }
{
  echo "date $(date -Is) commit $(git rev-parse --short HEAD) dirty=$(git status --porcelain -- serve kernels | wc -l)"
  echo "engOLD=$OLD sha256 $(sha256sum "$OLD" | cut -c1-16)"
  echo "engNEW=$NEW sha256 $(sha256sum "$NEW" | cut -c1-16)"
  echo "gguf $GGUF warm $(readlink -f "$GGUF.warm") sha256 $(sha256sum "$GGUF.warm" | cut -c1-16)"
  echo "vram_free_gb_start $(vram) ram_avail_gb $(awk '/MemAvailable/ {printf "%.1f", $2/2^20}' /proc/meminfo)"
  for f in $(dirname "$GGUF")/*.gguf; do echo "pagecache_resident_gb $(basename $f) $(res $f)"; done
} >> "$R"
P=$(ls bench/q4x-prompts/*.prompt | head -20); P1=$(echo "$P" | head -1)

if [ "${MODE:-}" = http ]; then
  min16k "$TMAX" "$MT"
  PORT=8099
  echo "http arm: tmax=$TMAX max_tokens=$MT np=${NP:-2} bsrv=$BSRV sha256 $(sha256sum "$BSRV" | cut -c1-16) tok sha256 $(sha256sum "$TOK" | cut -c1-16)" >> "$R"
  "$BSRV" --engine "$NEW" --pack "$GGUF" --tokenizer "$TOK" --port $PORT --engine-env Q4X_PREFILL=1 --engine-env Q4X_TMAX="$TMAX" \
    --engine-env MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=2 > "$OUT/server.log" 2>&1 &
  SPID=$!
  stopsrv() { kill "$SPID" 2>/dev/null || true; for _ in $(seq 60); do kill -0 "$SPID" 2>/dev/null || return 0; sleep 1; done; kill -9 "$SPID" 2>/dev/null || true; }
  trap stopsrv EXIT
  for _ in $(seq 900); do
    curl -sf localhost:$PORT/health > "$OUT/health.json" 2>/dev/null && break
    kill -0 "$SPID" 2>/dev/null || fail http "baro-serve exited before /health, see $OUT/server.log"; sleep 1
  done
  [ -s "$OUT/health.json" ] || fail http "no /health in 900 s"
  [ "$(jq -r .limits.tmax "$OUT/health.json")" = "$TMAX" ] || fail http "read-back tmax $(jq -r .limits.tmax "$OUT/health.json") != $TMAX"
  [ "$(jq -r .status "$OUT/health.json")" = ok ] || fail http "engine not ok"
  grep -q 'Q4X_TMAX' "$OUT/server.log" || echo "note: engine env override line not in server log" >> "$R"
  echo "http read-back: $(jq -c '{status,limits,pack}' "$OUT/health.json") vram_free_gb $(vram) gtt_gb $(gtt)" | tee -a "$R"
  HP=$(echo "$P" | head -n "${NP:-2}"); : > "$OUT/http-n.txt"
  for pf in $HP; do
    b=$(basename "$pf" .prompt); nb=$(grep -c '^engine: ids ' "$OUT/server.log" || true)
    jq -c --argjson n "$MT" '{prompt: ., max_tokens: $n, temperature: 0, stream: true}' "$pf" > "$OUT/http-$b.req"
    curl -sfN -H "Content-Type: application/json" localhost:$PORT/v1/completions -d @"$OUT/http-$b.req" > "$OUT/http-$b.sse" || fail http "$b: curl failed, see $OUT/server.log"
    grep -q '^data: \[DONE\]' "$OUT/http-$b.sse" || fail http "$b: stream has no [DONE]"
    grep '^data: {' "$OUT/http-$b.sse" | sed 's/^data: //' > "$OUT/http-$b.ev"
    jq -j '.choices[0].text // empty' "$OUT/http-$b.ev" > "$OUT/http-$b.text"
    jq -r '.choices[0].tokens[]? ' "$OUT/http-$b.ev" > "$OUT/http-$b.sse.ids"
    fin=$(jq -r '.choices[0].finish_reason // empty' "$OUT/http-$b.ev" | tail -1)
    for _ in $(seq 60); do [ "$(grep -c '^engine: ids ' "$OUT/server.log" || true)" -gt "$nb" ] && break; sleep 1; done
    grep '^engine: ids ' "$OUT/server.log" | tail -1 | cut -d' ' -f3- > "$OUT/http-$b.ids" || true
    n=$(wc -w < "$OUT/http-$b.ids"); [ "$n" -gt 0 ] || fail http "$b: no ids line in server log"
    if [ "$fin" = stop ]; then tr -s ' ' '\n' < "$OUT/http-$b.ids" | grep . | head -n -1 > "$OUT/http-$b.idsc"; else tr -s ' ' '\n' < "$OUT/http-$b.ids" | grep . > "$OUT/http-$b.idsc"; fi
    cmp -s "$OUT/http-$b.idsc" "$OUT/http-$b.sse.ids" || fail http "$b: streamed token ids differ from the engine ids line"
    tr '\n' ' ' < "$OUT/http-$b.idsc" | jq -c -R '{tokens: (split(" ") | map(select(. != "") | tonumber))}' | curl -sf -H "Content-Type: application/json" localhost:$PORT/detokenize -d @- | jq -j .content > "$OUT/http-$b.detok"
    echo "$b $n" >> "$OUT/http-n.txt"
    echo "http $b ids $n finish $fin text_bytes $(wc -c < "$OUT/http-$b.text") detok_bytes $(wc -c < "$OUT/http-$b.detok") | $(grep '^engine: gen ms' "$OUT/server.log" | tail -1)" | tee -a "$R"
  done
  nb=$(grep -c '^engine: ids ' "$OUT/server.log" || true); cid=cmpl-$(( $(echo "$HP" | wc -l) + 1 ))
  b1=$(basename "$(echo "$HP" | head -1)" .prompt)
  curl -sfN -H "Content-Type: application/json" localhost:$PORT/v1/completions -d @"$OUT/http-$b1.req" > "$OUT/http-cancel.sse" & CPID=$!
  sleep 3
  cr=$(curl -sf -H "Content-Type: application/json" localhost:$PORT/v1/cancel -d "{\"id\":\"$cid\"}" || echo none)
  wait $CPID || true
  for _ in $(seq 90); do [ "$(grep -c '^engine: ids ' "$OUT/server.log" || true)" -gt "$nb" ] && break; sleep 1; done
  [ "$(grep -c '^engine: ids ' "$OUT/server.log" || true)" -gt "$nb" ] || fail http "cancel: no ids line 90 s after POST /v1/cancel"
  cn=$(grep '^engine: ids ' "$OUT/server.log" | tail -1 | cut -d' ' -f3- | wc -w)
  cf=$(grep '^data: {' "$OUT/http-cancel.sse" | sed 's/^data: //' | jq -r '.choices[0].finish_reason // empty' | tail -1)
  [ "$cn" -lt "$(awk -v b=$b1 '$1 == b {print $2}' "$OUT/http-n.txt")" ] || fail http "cancel: engine ran to $cn ids, cancel response $cr"
  echo "http cancel: POST /v1/cancel $cid after 3 s -> $cr, engine stopped at $cn ids, stream finish_reason $cf" | tee -a "$R"
  echo "http gtt_gb_after $(gtt) vram_free_gb_after $(vram)" | tee -a "$R"
  stopsrv; trap - EXIT
  { for pf in $HP; do b=$(basename "$pf" .prompt); echo "$b $(awk -v b=$b '$1 == b {print $2}' "$OUT/http-n.txt") $(ids "$pf")"; done; } > "$OUT/http-requests.txt"
  env Q4X_STDIN=1 Q4X_PREFILL=1 Q4X_TMAX="$TMAX" "$NEW" "$GGUF" < "$OUT/http-requests.txt" > "$OUT/http-stdin.log" 2>&1 || fail http "Q4X_STDIN reference run failed"
  for pf in $HP; do
    b=$(basename "$pf" .prompt)
    grep "^out $b " "$OUT/http-stdin.log" | cut -d' ' -f3- | sed 's/^ids //' > "$OUT/http-$b.stdin.ids"
    cmp -s <(tr -s ' ' '\n' < "$OUT/http-$b.ids" | grep .) <(tr -s ' ' '\n' < "$OUT/http-$b.stdin.ids" | grep .) || fail http "$b: HTTP ids differ from the Q4X_STDIN ids"
    [ -s "$OUT/http-$b.text" ] || fail http "$b: streamed text empty"
    cmp -s "$OUT/http-$b.text" "$OUT/http-$b.detok" || fail http "$b: streamed text differs from /detokenize of the ids"
    echo "http $b PASS: $(wc -w < "$OUT/http-$b.ids") ids equal Q4X_STDIN, streamed text equals detokenized ids" | tee -a "$R"
  done
  echo "job PASS" | tee -a "$R"; exit 0
fi

# I1: block sums, warm off on both (OLD has no default warm list and reads Q4X_WARM=0 as a path)
env Q4X_DUMP=1 "$OLD" "$GGUF" 4 $(ids "$P1") > "$OUT/i1-old.log" 2>&1 || fail I1 "OLD run failed"
env Q4X_WARM=0 Q4X_DUMP=1 "$NEW" "$GGUF" 4 $(ids "$P1") > "$OUT/i1-new.log" 2>&1 || fail I1 "NEW run failed"
grep '^D ' "$OUT/i1-old.log" > "$OUT/i1-old.d"; grep '^D ' "$OUT/i1-new.log" > "$OUT/i1-new.d"
nd=$(wc -l < "$OUT/i1-old.d"); [ "$nd" -gt 1000 ] || fail I1 "only $nd block sums"
cmp -s "$OUT/i1-old.d" "$OUT/i1-new.d" || fail I1 "block sums differ ($(diff "$OUT/i1-old.d" "$OUT/i1-new.d" | grep -c '^<') of $nd)"
grep -q '^q4x: warmed' "$OUT/i1-new.log" && fail I1 "Q4X_WARM=0 still warmed"
echo "I1 PASS: $nd block sums identical OLD vs NEW" | tee -a "$R"

{ for pf in $P; do echo "$(basename "$pf" .prompt) $NG $(ids "$pf")"; done; echo "again-$(basename "$P1" .prompt) $NG $(ids "$P1")"; } > "$OUT/requests.txt"
if [ "${MODE:-}" = margin ]; then
  for arm in R1a R3a R3b R1b; do
    log=$OUT/$arm.log; g0=$(gtt)
    case $arm in R1*) envs="";; R3*) envs="Q4X_ECACHE_GB=17";; esac
    env Q4X_STDIN=1 $envs /usr/bin/time -f 'wall %e s rss %M KB' "$NEW" "$GGUF" < "$OUT/requests.txt" > "$log" 2>&1 || fail "$arm" "resident run failed"
    [ "$(grep -c '^out ' "$log")" = 21 ] || fail "$arm" "not 21 answers"
    echo "$arm $envs | $(grep -m1 '^q4x: expert cache' "$log" | sed 's/^q4x: //') | gtt_gb before $g0 after $(gtt) | $(grep '^wall ' "$log") | vram_free_gb_after $(vram)" | tee -a "$R"
    awk -v arm=$arm '/^gen ms\/token/ { g = $3 } /^out / { t = $2 } /^expert misses/ { printf "req %s %s gen_ms %.3f misses_per_token %.2f\n", arm, t, g, $3 / ($5 / 480) }' "$log" >> "$R"
  done
  python3 - "$R" <<'EOF2' | tee -a "$R"
import re, statistics, sys
rows = {}
for l in open(sys.argv[1]):
    m = re.match(r'req (R[13])[ab] (\S+) gen_ms ([\d.]+) misses_per_token ([\d.]+)', l)
    if m and not m[2].startswith('p01') and not m[2].startswith('again'): rows.setdefault(m[1], []).append((float(m[3]), float(m[4])))
for a, r in sorted(rows.items()):
    print(f"MARGIN {a} steady (req 2..20, both runs, n={len(r)}) median gen_ms {statistics.median(x[0] for x in r):.3f} misses/token {statistics.median(x[1] for x in r):.2f}")
g1 = statistics.median(x[0] for x in rows['R1']); g3 = statistics.median(x[0] for x in rows['R3'])
print(f"MARGIN RATIO R1/R3 gen_ms {g1 / g3:.4f} misses {statistics.median(x[1] for x in rows['R1']) / statistics.median(x[1] for x in rows['R3']):.4f}")
EOF2
  echo "job PASS" | tee -a "$R"; exit 0
fi

# I2 reference: fresh argv NEW p01
"$NEW" "$GGUF" $NG $(ids "$P1") > "$OUT/i2-argv.log" 2>&1 || fail I2 "argv run failed"
grep -m1 '^ids ' "$OUT/i2-argv.log" > "$OUT/i2-argv.ids" || fail I2 "argv run printed no ids"

# S1: OLD argv, one process per prompt
for pf in $(echo "$P" | head -5); do
  b=$(basename "$pf" .prompt)
  /usr/bin/time -f 'wall %e s rss %M KB' "$OLD" "$GGUF" $NG $(ids "$pf") > "$OUT/s1-$b.log" 2>&1 || fail S1 "$b failed"
  echo "S1 $b gen_ms $(grep -o 'gen ms/token [0-9.]*' "$OUT/s1-$b.log" | awk '{print $3}') $(grep -o 'expert misses [0-9]* of [0-9]*' "$OUT/s1-$b.log") | $(grep '^wall ' "$OUT/s1-$b.log")" | tee -a "$R"
done

# R1..R3: resident arms
resident() { # name env...
  local name=$1; shift
  local log=$OUT/$name.log
  env Q4X_STDIN=1 "$@" /usr/bin/time -f 'wall %e s rss %M KB' "$NEW" "$GGUF" < "$OUT/requests.txt" > "$log" 2>&1 || fail "$name" "resident run failed: $(grep -m1 -E 'Error|error' "$log" | cut -c1-160)"
  local n; n=$(grep -c '^out ' "$log"); [ "$n" = 21 ] || fail "$name" "$n of 21 requests answered"
  echo "$name $* | $(grep -m1 '^q4x: expert cache' "$log" | sed 's/^q4x: //') | warm: $(grep -m1 '^q4x: warmed' "$log" | cut -c5-80 || echo none) | $(grep '^wall ' "$log") | vram_free_gb_after $(vram)" | tee -a "$R"
  awk -v arm=$name '/^gen ms\/token/ { g = $3 } /^out / { t = $2 } /^expert misses/ { printf "req %s %s gen_ms %.3f misses_per_token %.2f\n", arm, t, g, $3 / ($5 / 480) }' "$log" | tee -a "$R"
}
resident R1
resident R2 Q4X_WARM=0
grep -q '^q4x: warmed' "$OUT/R2.log" && fail R2 "Q4X_WARM=0 still warmed"
resident R3 Q4X_ECACHE_GB=17
grep -q '^q4x: warmed' "$OUT/R1.log" || fail R1 "default warm list not loaded"
for arm in R1 R2 R3; do
  a=$(grep "^out $(basename "$P1" .prompt) " "$OUT/$arm.log" | cut -d' ' -f3-); z=$(grep "^out again-" "$OUT/$arm.log" | cut -d' ' -f3-)
  [ "$a" = "$z" ] || fail I2 "$arm: p01 after 19 requests differs from the first p01"
  [ "$a" = "$(cat "$OUT/i2-argv.ids")" ] || fail I2 "$arm: resident p01 differs from the argv run"
done
echo "I2 PASS: p01 ids equal argv, first and last request, in R1 R2 R3" | tee -a "$R"
python3 - "$R" <<'EOF' | tee -a "$R"
import re, statistics, sys
rows = {}; s1 = []
for l in open(sys.argv[1]):
    m = re.match(r'req (\S+) (\S+) gen_ms ([\d.]+) misses_per_token ([\d.]+)', l)
    if m: rows.setdefault(m[1], []).append((m[2], float(m[3]), float(m[4])))
    m = re.match(r'S1 \S+ gen_ms ([\d.]+)', l)
    if m: s1.append(float(m[1]))
old = statistics.median(s1)
print(f"S1 OLD argv median gen ms/token {old:.3f} (n={len(s1)})")
for a, r in sorted(rows.items()):
    st = r[1:20]
    g = statistics.median(x[1] for x in st); mi = statistics.median(x[2] for x in st)
    print(f"{a} req1 gen_ms {r[0][1]:.3f} misses/token {r[0][2]:.2f} | steady (req 2..20 median) gen_ms {g:.3f} misses/token {mi:.2f} | steady/req1 misses {mi / r[0][2]:.3f} | steady gen vs OLD {g / old:.4f}")
EOF

# C1: 8192-token prefill, chunked vs one forward
L=$(tr '\n' ' ' < "$LONG"); nl=$(wc -w <<< "$L"); [ "$nl" = 8192 ] || fail C1 "long prompt has $nl ids"
for arm in chunk one; do
  rm -f "$OUT/c1-$arm.hid" "$OUT/c1-$arm.hid.n"
  if [ $arm = one ]; then ck=8192; else ck=2048; fi
  env Q4X_PREFILL=1 Q4X_TMAX=8320 Q4X_CHUNK=$ck Q4X_HIDDEN="$OUT/c1-$arm.hid" Q4X_HIDDEN_F32=1 /usr/bin/time -f 'wall %e s rss %M KB' \
    "$NEW" "$GGUF" 16 $L > "$OUT/c1-$arm.log" 2>&1 || fail C1 "$arm failed: $(grep -m1 -E 'Error|error' "$OUT/c1-$arm.log" | cut -c1-160)"
  echo "C1 $arm Q4X_CHUNK=$ck | $(grep -m1 '^q4x: expert cache' "$OUT/c1-$arm.log" | sed 's/^q4x: //') | $(grep '^prompt ' "$OUT/c1-$arm.log") | $(grep -o 'expert misses [0-9]* of [0-9]*' "$OUT/c1-$arm.log") | $(grep '^wall ' "$OUT/c1-$arm.log")" | tee -a "$R"
done
[ "$(cat "$OUT/c1-chunk.hid.n")" = "$(cat "$OUT/c1-one.hid.n")" ] || fail C1 "hidden row counts differ"
head -c $((8192 * 2560 * 4)) "$OUT/c1-chunk.hid" > "$OUT/c1-chunk.p"; head -c $((8192 * 2560 * 4)) "$OUT/c1-one.hid" > "$OUT/c1-one.p"
cmp -s "$OUT/c1-chunk.p" "$OUT/c1-one.p" || fail C1 "prompt hidden bytes differ chunked vs one forward"
[ "$(grep -m1 '^ids ' "$OUT/c1-chunk.log")" = "$(grep -m1 '^ids ' "$OUT/c1-one.log")" ] || fail C1 "generated ids differ"
rm -f "$OUT"/c1-*.hid "$OUT"/c1-*.p
echo "C1 PASS: 8192 prompt hidden rows byte-identical, ids equal" | tee -a "$R"
echo "job PASS" | tee -a "$R"
