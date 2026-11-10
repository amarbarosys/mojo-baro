#!/usr/bin/env bash
# usage: bench/ailab/step.sh OUTDIR              P5: step wall, ours vs llama-server --parallel N, N in NS, STEPS steps each
#        bench/ailab/step.sh --soak OUTDIR       P6: ours only, N = SOAK_N (50), SOAK_STEPS (200) steps, error / VRAM / p95 verdict
# One queued job (`gpu-wait run --timeout`, knobs as `env K=V` inside the queued command); both servers are started and
# stopped INSIDE it (serve.sh QUEUE=none, llama.sh), never as separate queue jobs. Per N the two arms run back to back, the
# arm that goes first alternates with N (ours, llama, ours, ...), same job and same card state. Each arm: server up, arm read
# back (ours: serve.sh arm.txt from the running engine; llama: llama.sh /props + cmdline), STEPS lockstep steps of N bodies
# through street-sim's own ask_openai (bench/ailab/step.py, fresh state per body per step, max_tokens floored to 16000),
# server down (ours: engine gone, VRAM back to idle). Table and verdict at the end.
# PASS (P5) = no error and no non-200 post (429, 4xx, 5xx) in any arm; the step-wall target (N=20 < 8 s, N=50 < 20 s, plan
# frozen before any run) is printed as MET / MISSED and never changes the exit code (a miss means optimize, not drop).
# PASS (P6) = 0 errors, 0 x 429, 0 x 4xx, 0 x 5xx, the ENGINE process's VRAM (fdinfo, not the card: the desktop moves that) drift <= 50 MB over the second half AND slope x 50,000
# steps <= 1 GiB (a 3-day run); the global card number is printed as a second line,
# ask p95 < 60 s.
# Env: NS ("20 32 50") STEPS (5) ENGINE (.work/ailab/engine-seqs32s2; N > 32 uses ENGINE64 .work/ailab/engine-seqs64) PACK
#   TMAX (24576) KV_POOL (default ceil(N*2700/TMAX), at least 3) VRAM_GB (22) TIMEOUT (job limit, default 3600 / soak 14400)
#   ENGINE_EXTRA ("K=V K=V" more engine env on the ours arm, e.g. BARO_SEQS_TIME=1: an instrumented arm, named in arm.txt)
#   MIN_HEADROOM (6000, serve.sh: prompt tokens TMAX must leave beside max_tokens 16000) STATES_DIR (step.py --states, negative controls)
#   POOLS ("3 6": one ours arm per KV pool size, labelled ours-p3, ours-p6, in the same job) PROMPT_EST (4800: worst-case prompt tokens per cold
#   session for the pool formula) MIN_CACHED (3500: every first-round request of the burst, and of the second warm ask, must restore at least
#   this many cached tokens, else FAIL cached; 0 disables)
#   FOREIGN_CHECK=1 / FOREIGN_NAMES (selftest: the guard also in a stub run, on a differently named process)
#   (the job runs `--timed`; a foreign llama-server / baro-serve / engine process before or during an arm FAILs it, own pids excluded)
#   ARMS ("ours llama") SOAK_N SOAK_STEPS STEP_STUB=1 (CPU dry run: stub engine + stub llama-server, no queue, no GPU)
# Exit 0 = PASS; 1 = FAIL (named); the dry run prints the same verdict lines.
set -euo pipefail
cd "$(dirname "$0")/../.."
soak=0; if [ "${1:-}" = --soak ]; then soak=1; shift; fi
[ $# -eq 1 ] || { echo "FAIL usage: step.sh [--soak] OUTDIR"; exit 1; }
out=$1   # relative to the repo root (serve.sh DIR contract)
case $out in /*) echo "FAIL args: OUTDIR must be relative to the repo root"; exit 1 ;; esac
stub=${STEP_STUB:-}
ns=${NS:-20 32 50}; steps=${STEPS:-5}; tmax=${TMAX:-24576}; arms=${ARMS:-ours llama}
engine=${ENGINE:-.work/ailab/engine-seqs32s2}; engine64=${ENGINE64:-.work/ailab/engine-seqs64}; pack=${PACK:-.work/ailab/pack-kq}
vramgb=${VRAM_GB:-22}
if [ -n "${POOLS:-}" ]; then  # one ours arm per pool size
  na=""; for a in $arms; do if [ "$a" = ours ]; then for pl in $POOLS; do na="$na ours-p$pl"; done; else na="$na $a"; fi; done; arms=${na# }
fi
if [ "$soak" = 1 ]; then ns=${SOAK_N:-50}; steps=${SOAK_STEPS:-200}; arms=ours; tl=${TIMEOUT:-14400}; else tl=${TIMEOUT:-3600}; fi
GW=${GW:-$HOME/.local/bin/gpu-wait}
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ] && [ -z "$stub" ]; then
  mkdir -p "$out"
  exec "$GW" run --timed --timeout "$tl" --vram "$vramgb" --label ailab-step -- env NS="$ns" STEPS="$steps" TMAX="$tmax" ARMS="$arms" \
    ENGINE="$engine" ENGINE64="$engine64" PACK="$pack" KV_POOL="${KV_POOL:-}" MAX_INFLIGHT="${MAX_INFLIGHT:-}" SOAK_N="${SOAK_N:-}" SOAK_STEPS="${SOAK_STEPS:-}" \
    ENGINE_EXTRA="${ENGINE_EXTRA:-}" POOLS="${POOLS:-}" PROMPT_EST="${PROMPT_EST:-}" MIN_CACHED="${MIN_CACHED:-}" MIN_HEADROOM="${MIN_HEADROOM:-}" STATES_DIR="${STATES_DIR:-}" "$PWD/$0" $([ "$soak" = 1 ] && echo --soak) "$out"
fi
mkdir -p "$out"
if [ -n "$stub" ]; then
  export STUB_VRAM=${STUB_VRAM:-1200000000} FAKE_OUT=$PWD/$out/stub-req.jsonl LLAMA=$PWD/bench/stub-llama-server.py
  export STUB_CACHED=${STUB_CACHED:-3569}
  engine=bench/stub-engine.py; engine64=bench/stub-engine.py; : > "$FAKE_OUT"; ngl=0
else ngl=99; fi
vram() { if [ -n "${STUB_VRAM:-}" ]; then echo "$STUB_VRAM"; else cat /sys/class/drm/card*/device/mem_info_vram_used | sort -n | tail -1; fi; }
fail() { echo "FAIL $1: $2"; exit 1; }
for f in "$pack/identity.json" serve/target/release/baro-serve bench/ailab/step.py bench/ailab/serve.sh bench/ailab/llama.sh; do
  [ -e "$f" ] || fail arm "missing $f"
done
mi=${MAX_INFLIGHT:-64}
for n in $ns; do
  # the runner re-asks WITHOUT tools on any HTTPError (a 429 included): N must stay inside the in-flight bound
  [ "$n" -le "$mi" ] || fail arm "N=$n > --max-inflight $mi: a 429 would reach the runner"
  e=$engine; if [ "$n" -gt 32 ]; then e=$engine64; fi
  [ -e "$e" ] || fail arm "no engine for N=$n: $e (build -D BARO_SEQ_CAP=64, plan P6)"
done
{
  echo "date=$(date -Is) commit=$(git rev-parse --short HEAD) dirty='$(git status --short -- serve bench | tr '\n' ' ')'"
  echo "mode=$([ "$soak" = 1 ] && echo soak || echo step) stub=${stub:-0} ns='$ns' steps=$steps tmax=$tmax arms='$arms' ngl=$ngl job=${GPU_WAITING_ROOM_JOB:-none}"
  echo "scripts: step.sh $(sha256sum bench/ailab/step.sh | cut -c1-12) step.py $(sha256sum bench/ailab/step.py | cut -c1-12) serve.sh $(sha256sum bench/ailab/serve.sh | cut -c1-12)"
  echo "server sha=$(sha256sum serve/target/release/baro-serve | cut -c1-16) built=$(date -r serve/target/release/baro-serve -Is) src_newer_than_binary='$(find serve/src serve/Cargo.toml serve/Cargo.lock -type f -newer serve/target/release/baro-serve | tr '\n' ' ')' pack_identity=$(sha256sum "$pack/identity.json" | cut -c1-16)"
  echo "power_cap_uw=$(cat /sys/class/drm/card*/device/hwmon/hwmon*/power1_cap 2> /dev/null | head -1 || echo n/a) vram_idle_mib=$(( $(vram) / 1048576 ))"
} | tee "$out/arm.txt"
cur=""   # "ours:DIR" or "llama:" of the arm that is up
cleanup() {
  case $cur in
    ours:*) bench/ailab/serve.sh down DIR="${cur#ours:}" > /dev/null 2>&1 || :
      pkill -TERM -P "$(sed -n 's/^PID=//p' "${cur#ours:}/serve.env" 2> /dev/null)" 2> /dev/null || : ;;
    llama:*) bench/ailab/llama.sh down > /dev/null 2>&1 || : ;;
  esac
}
trap cleanup EXIT
# KV pool sizing from the P3 arm (cycle1/arm.txt: N=20, pool 3 x 20480, warm global 16,657 MiB over idle 1,265): engine MiB =
# 6,896 (pack.bin) + 2,749 (fixed: MAX pool, scratch, prefill workspace) + 95.4 x N (SSM ring, 100 MB per slot) + P x TMAX x 65,536 B
# (f32 KV, 8 attention layers). pack-fit models one sequence at 40,960 B/token, which is not what the engine measured; this is.
# P = min(ceil(N x (PROMPT_EST + 512) / TMAX) [worst case: nothing shared, 20 cold 4.1K-token prompts needed 20 x 32 = 640 pages > 576 and 4 x 429],
# what fits in card - idle - 512 MiB), at least 2.
# Foreign load: this job's step wall reads the host clock, so another lane's llama-server / baro-serve / engine (a CPU llama at 790% CPU overlapped the
# first minute of the first P5 FULL) voids the arm. `--timed` makes the cpu room wait; this catches what is started outside the queue: no foreign
# server process at arm start, and a sampler every 2 s during the arm; any sighting FAILs the arm (own pids excluded).
foreign() {  # own pids...  -> prints foreign server pids (none when the guard is off: stub runs are not timing claims)
  local p nm own=" $* "
  if [ -n "$stub" ] && [ "${FOREIGN_CHECK:-}" != 1 ]; then return 0; fi
  for p in $(for nm in ${FOREIGN_NAMES:-llama-server}; do pgrep -x "$nm"; done; if [ -z "${FOREIGN_NAMES:-}" ]; then pgrep 'engine-seqs'; fi); do
    case $own in *" $p "*) ;; *) echo "$p" ;; esac
  done
}
watch_foreign() {  # logfile own_pid_file ; loops until killed
  local own
  while :; do
    own=$(cat "$2" 2> /dev/null || :)
    # shellcheck disable=SC2046
    if [ -n "$(foreign $own)" ]; then echo "$(date +%T) foreign: $(foreign $own | tr '\n' ' ')" >> "$1"; fi
    sleep 2
  done
}
own_pids() {  # pids of this arm's server processes, one line
  case $1 in
    ours) sed -n 's/^PID=//p;s/^ENGINE_PID=//p' "$2/serve.env" | tr '\n' ' ' ;;
    llama) sed -n 's/^PID=//p' .work/ailab/llama.env | tr '\n' ' ' ;;
  esac
}
pool_for() {
  local want cap total avail
  want=$(( ($1 * (${PROMPT_EST:-4800} + 512) + tmax - 1) / tmax ))
  if [ -n "${STUB_VRAM:-}" ]; then total=25165824000; else total=$(cat /sys/class/drm/card*/device/mem_info_vram_total | sort -n | tail -1); fi
  avail=$(( total / 1048576 - $(vram) / 1048576 - 512 - 6896 - 2749 - 954 * $1 / 10 ))
  cap=$(( avail * 1048576 / (tmax * 65536) ))
  if [ "$cap" -lt 2 ]; then echo "FAIL pool: N=$1 TMAX=$tmax leaves room for $cap KV pool units on this card (f32 KV); int8 KV needs its own engine build" >&2; exit 1; fi
  local p=$want; if [ "$cap" -lt "$p" ]; then p=$cap; fi
  p=${KV_POOL:-$p}
  echo "pool N=$1 TMAX=$tmax: want $want, card fits $cap, using $p; predicted warm global $(( $(vram) / 1048576 + 6896 + 2749 + 954 * $1 / 10 + p * tmax * 65536 / 1048576 )) MiB of $(( total / 1048576 ))" >&2
  echo "$p"
}
run_arm() {  # label n   (label = ours | ours-pP | llama)
  local label=$1 n=$2 arm pool="" e d=$out/$1-n$2 port minc=${MIN_CACHED:-3500} xc=()
  arm=${label%%-p*}; case $label in *-p*) pool=${label##*-p} ;; esac
  mkdir -p "$d"
  e=$engine; if [ "$n" -gt 32 ]; then e=$engine64; fi
  fp=$(foreign | tr '\n' ' ')
  [ -z "$fp" ] || fail "$label-n$n" "foreign server process(es) running before the arm: $fp (a timed arm needs a quiet box; their CPU load voids the wall times)"
  echo "arm $label n=$n start: load1=$(cut -d' ' -f1 /proc/loadavg)" >> "$out/arm.txt"
  if [ "$arm" = ours ]; then
    port=$(( 18600 + n ))
    cur="ours:$d"
    [ -n "$pool" ] || pool=$(pool_for "$n")
    bench/ailab/serve.sh up QUEUE=none DIR="$d" ENGINE="$e" PACK="$pack" N="$n" MAX_INFLIGHT="$mi" KV_POOL="$pool" MIN_CACHED="$minc" TMAX="$tmax" PORT="$port" MIN_HEADROOM="${MIN_HEADROOM:-6000}" \
      HEALTH_WAIT=900 EXTRA="${FAKE_OUT:+FAKE_OUT=$FAKE_OUT} ${ENGINE_EXTRA:-}" > "$d/up.log" 2>&1 || { tail -3 "$d/up.log"; fail "$label-n$n" "serve.sh up, see $d/up.log"; }
    cp "$d/arm.txt" "$d/arm-server.txt"
    url=$(sed -n 's/^URL=//p' "$d/serve.env"); epid=$(sed -n 's/^ENGINE_PID=//p' "$d/serve.env")
    # serve.sh up warmed through the runner (two asks, ask 2 asserted cached >= MIN_CACHED); the burst below asserts it again from step 0
    [ "$minc" = 0 ] || xc=(--expect-cached "$minc" --expect-from-step 0)
  else
    cur="llama:"
    ctx=$(( n * 8192 ))
    bench/ailab/llama.sh up --ngl "$ngl" --parallel "$n" --ctx "$ctx" --port $(( 18700 + n )) > "$d/up.log" 2>&1 \
      || { tail -3 "$d/up.log"; fail "$label-n$n" "llama.sh up, see $d/up.log"; }
    cp .work/ailab/llama.arm "$d/arm-server.txt"
    nctx=$(sed -n 's/.*n_ctx=\([0-9]*\).*/\1/p' "$d/arm-server.txt")
    # unified KV shares no prefix across slots: every live session holds its whole prompt + the answer (about 100 tokens). llama.cpp clamps -c to the
    # model's n_ctx_train (262,144), so N=50 gets 5,242 per slot: the margin is 256 tokens, and a 500 'Context size has been exceeded' still FAILs the arm.
    need=$(( n * (${PROMPT_EST:-4800} + 256) ))
    [ "${nctx:-0}" -ge "$need" ] || fail "$label-n$n" "llama n_ctx ${nctx:-?} < N x (prompt + 256) = $need: 'Context size has been exceeded' 500s"
    url=$(sed -n 's/^URL=//p' .work/ailab/llama.env); epid=$(sed -n 's/^PID=//p' .work/ailab/llama.env)
  fi
  echo "arm $label n=$n pool=${pool:-none}: $(head -c 300 "$d/arm-server.txt" | tr '\n' '|')" >> "$out/arm.txt"
  : > "$d/foreign.log"; own_pids "$arm" "$d" > "$d/own.pids"
  watch_foreign "$d/foreign.log" "$d/own.pids" &
  wpid=$!
  python3 -I bench/ailab/step.py "$url" "$n" "$steps" "$d/run" --pid "$epid" ${xc[@]+"${xc[@]}"} ${STATES_DIR:+--states "$STATES_DIR"} > "$d/step.log" 2>&1 || { tail -5 "$d/step.log"; fail "$label-n$n" "step.py, see $d/step.log"; }
  kill "$wpid" 2> /dev/null || :; wait "$wpid" 2> /dev/null || :
  echo "arm $label n=$n end: load1=$(cut -d' ' -f1 /proc/loadavg) foreign_samples=$(wc -l < "$d/foreign.log")" >> "$out/arm.txt"
  [ ! -s "$d/foreign.log" ] || { cat "$d/foreign.log" | head -3; fail "$label-n$n" "foreign server process seen during the timed arm, see $d/foreign.log: rerun on a quiet box"; }
  grep -E '^N=' "$d/step.log" | tail -1 | tee "$d/summary.txt"
  if [ "$arm" = ours ]; then
    bench/ailab/serve.sh down DIR="$d" > "$d/down.log" 2>&1 || { cat "$d/down.log"; fail "$label-n$n" "serve.sh down"; }
    cat "$d/down.log"
    grep 'seqs time:' "$d/serve.log" | tail -1 > "$d/seqs-time.txt" || :
    [ ! -s "$d/seqs-time.txt" ] || echo "engine timers (cumulative, BARO_SEQS_TIME): $(cut -c1-300 "$d/seqs-time.txt")"
    [ ! -s "$d/run/rounds.tsv" ] || column -t -s "$(printf '\t')" "$d/run/rounds.tsv"
  else bench/ailab/llama.sh down; fi
  cur=""
}
i=0
for n in $ns; do
  order=($arms); if [ $(( i % 2 )) = 1 ] && [ "${#order[@]}" = 2 ]; then order=("${order[1]}" "${order[0]}"); fi
  for arm in "${order[@]}"; do run_arm "$arm" "$n"; done
  i=$((i + 1))
done
# verdict
fails=0; unresolved=0
field() { sed -n "s/.*\b$2=\([^ ]*\).*/\1/p" "$1"; }
printf '%-6s %-6s %12s %10s %10s %7s %6s %6s %6s %5s\n' N arm step_median ask_p50 ask_p95 errors 429 4xx 5xx exc | tee "$out/table.txt"
for n in $ns; do for arm in $arms; do
  s=$out/$arm-n$n/summary.txt
  [ -s "$s" ] || { echo "FAIL table: no summary for $arm n=$n"; fails=$((fails + 1)); continue; }
  printf '%-6s %-6s %12s %10s %10s %7s %6s %6s %6s %5s\n' "$n" "$arm" "$(field "$s" step_median)" "$(field "$s" ask_p50)" "$(field "$s" ask_p95)" \
    "$(field "$s" errors)" "$(field "$s" http429)" "$(field "$s" http4xx)" "$(field "$s" http5xx)" "$(field "$s" exc)" | tee -a "$out/table.txt"
  # every non-200 counts: the runner re-asks WITHOUT tools after any HTTPError, so a 400 is a silently degraded ask, not an error
  bad=$(( $(field "$s" errors) + $(field "$s" http429) + $(field "$s" http4xx) + $(field "$s" http5xx) + $(field "$s" exc) ))
  if [ "$bad" -ne 0 ]; then echo "FAIL $arm n=$n: $bad error/429/4xx/5xx/exc, see $out/$arm-n$n/run/asks.tsv"; fails=$((fails + 1)); fi
done; done
if [ "$soak" = 1 ]; then
  d=$out/ours-n$ns/run; half=$(( steps / 2 ))
  # engine column = this process's own VRAM (fdinfo of the engine child), global column = the whole card incl. the desktop
  if [ -z "$stub" ] && [ "$(awk -F'\t' 'NR == 1 {print $2}' "$d/vram.tsv")" -le 0 ]; then
    echo "FAIL soak: engine VRAM unreadable from /proc/<engine pid>/fdinfo (drm-resident-vram), see $d/vram.tsv"; fails=$((fails + 1))
  fi
  p95=$(field "$out/ours-n$ns/summary.txt" ask_p95)
  slope() {  # column: "slope se resid_sd" in MiB per step, least squares over the second half only (the allocator pool fills in the first)
    awk -F'\t' -v h="$half" -v c="$1" '$1 >= h {n++; x = $1; y = $c / 1048576; sx += x; sy += y; sxy += x * y; sxx += x * x; syy += y * y}
      END {d = n * sxx - sx * sx; if (n < 4 || d == 0) {print "nan nan nan"; exit}
        b = (n * sxy - sx * sy) / d; sxxc = sxx - sx * sx / n; ssr = (syy - sy * sy / n) - b * b * sxxc; if (ssr < 0) ssr = 0
        printf "%.5f %.5f %.4f", b, sqrt(ssr / (n - 2) / sxxc), sqrt(ssr / (n - 2))}' "$d/vram.tsv"
  }
  v1=$(awk -F'\t' -v h="$half" '$1 == h - 1 {print $2}' "$d/vram.tsv"); v2=$(awk -F'\t' 'END {print $2}' "$d/vram.tsv")
  g1=$(awk -F'\t' -v h="$half" '$1 == h - 1 {print $3}' "$d/vram.tsv"); g2=$(awk -F'\t' 'END {print $3}' "$d/vram.tsv")
  drift=$(( (v2 - v1) / 1048576 )); read -r slope se sd <<<"$(slope 2)"; read -r gslope _ _ <<<"$(slope 3)"
  echo "soak engine: VRAM $(( v1 / 1048576 )) -> $(( v2 / 1048576 )) MiB over the second half (drift $drift MiB), slope $slope +- $se MiB/step (noise sd $sd MiB; x 50000 steps = $(awk -v s="$slope" 'BEGIN {printf "%.0f", s * 50000}') MiB), ask p95 $p95 s"
  echo "soak global (card, desktop included, informational): $(( g1 / 1048576 )) -> $(( g2 / 1048576 )) MiB, drift $(( (g2 - g1) / 1048576 )) MiB, slope $gslope MiB/step"
  [ "$drift" -le 50 ] || { echo "FAIL soak: VRAM drift $drift MiB > 50"; fails=$((fails + 1)); }
  # bar: slope x 50,000 steps <= 1 GiB (0.02048 MiB/step; a fixed 50 MB bar over 100 steps passes 0.4 MiB/step, about 20 GB over a 3-day run).
  # FAIL when the slope is over the bar; PASS only when slope + 2 se is under it; else UNRESOLVED (exit 3): the soak is too short for this noise.
  verdict=$(awk -v s="$slope" -v e="$se" -v sd="$sd" -v n="$(( steps - half ))" 'BEGIN {bar = 1024 / 50000
    if (s == "nan") {print "UNRESOLVED nan"; exit}
    if (s > bar) {print "FAIL"; exit}
    if (s + 2 * e > bar) {need = (12 * sd * sd / ((bar / 2) ^ 2)) ^ (1 / 3); printf "UNRESOLVED needs about %d second-half steps at noise sd %.3f MiB", need + 1, sd; exit}
    print "PASS"}')
  case $verdict in
    PASS) echo "soak slope: PASS ($slope + 2 x $se <= 0.02048 MiB/step)" ;;
    FAIL) echo "FAIL soak: VRAM slope $slope MiB/step x 50000 steps > 1024 MiB"; fails=$((fails + 1)) ;;
    *) echo "UNRESOLVED soak: slope $slope +- $se MiB/step cannot show the 0.02048 bar: ${verdict#UNRESOLVED }"; unresolved=$((unresolved + 1)) ;;
  esac
  awk -v p="$p95" 'BEGIN {exit !(p < 60)}' || { echo "FAIL soak: ask p95 $p95 s >= 60 s"; fails=$((fails + 1)); }
else
  for pair in "20 8" "50 20"; do
    set -- $pair; s=$out/ours-n$1/summary.txt
    if [ -s "$s" ]; then
      m=$(field "$s" step_median)
      if awk -v m="$m" -v t="$2" 'BEGIN {exit !(m < t)}'; then echo "target N=$1: ours $m s < $2 s MET"; else echo "target N=$1: ours $m s >= $2 s MISSED (optimize, never drop)"; fi
    fi
  done
fi
[ "$fails" -eq 0 ] || { echo "FAIL step: $fails problem(s), table $out/table.txt"; exit 1; }
[ "$unresolved" -eq 0 ] || { echo "UNVERIFIED step: $unresolved verdict(s) UNRESOLVED, table $out/table.txt"; exit 3; }
echo "PASS step: $out/table.txt"
