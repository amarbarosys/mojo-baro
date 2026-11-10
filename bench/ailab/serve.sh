#!/usr/bin/env bash
# usage: bench/ailab/serve.sh up [K=V ...] | down | status
# Lifecycle of the ailab server (docs/plan-ailab-server.md P3). `up` queues baro-serve as ONE job (`gpu-wait submit
# --timeout`, knobs as `env K=V` inside the queued command), waits for /health, warms the shared MANUAL prefix with one
# request (WARM_BODY), reads the arm back from the RUNNING processes (the engine child's /proc environ and exe sha, the
# engine's echo lines, /health limits; never from the command line) and refuses a run whose read-back differs from the
# request. Writes DIR/serve.env (URL PORT JOB PID ENGINE_PID ARM LOG IDLE_VRAM) and DIR/arm.txt. `down` sends SIGTERM,
# waits for the job, then asserts the engine child is gone and VRAM is within 100 MB of the pre-`up` idle.
# Knobs (K=V after `up`, or the environment): ENGINE PACK SERVER N (20) KV_POOL (3) TMAX (24576; floor 16384, and at least MIN_HEADROOM + the warm body's max_tokens 16000)
#   MAX_INFLIGHT (64) PORT (18433) VRAM_GB (20) TIMEOUT (14400 s job limit) QUEUE (gpu|cpu|none; none = no queue, CPU
#   dry runs only) WARM_BODY (first P0 body, else bench/ailab/states/s01.json) EXTRA ("K=V K=V" more engine env)
#   DIR (.work/ailab) QUEUE_WAIT (3600 s) HEALTH_WAIT (900 s) PRIORITY (50) STUB_VRAM (set: use this idle VRAM instead of sysfs)
#   MIN_CACHED (3500; 0 = skip): after the stand-in warm, two asks through the runner's own ask_openai (bench/ailab/step.py, states from the far end of
#   the set) leave the shared system+tools checkpoint; ask 2 must restore >= MIN_CACHED tokens or `up` FAILs (a cold first burst of 20 prompts needs
#   20 x 32 pages, overflows the pool and answers 429 'kv pool full'; seen in P5 quick and P4 FULL).
#   MIN_HEADROOM (6000: prompt tokens the window must leave beside max_tokens 16000; the runner's real prompts are 4,041 to 4,550+ tokens,
#   and baro-serve answers 400 exceed_context when prompt + max_tokens > TMAX: 20480 left 4,480 and the runner swallowed the 400s).
set -euo pipefail
cd "$(dirname "$0")/../.."
. bench/min16k.sh
GW=${GW:-$HOME/.local/bin/gpu-wait}

mode=${1:-}; shift || :
case $mode in up | down | status) ;; *) echo "FAIL usage: serve.sh up [K=V ...] | down | status"; exit 2 ;; esac
for kv in "$@"; do
  case $kv in
    ENGINE=* | PACK=* | SERVER=* | N=* | KV_POOL=* | TMAX=* | MAX_INFLIGHT=* | PORT=* | VRAM_GB=* | TIMEOUT=* | QUEUE=* | \
      WARM_BODY=* | EXTRA=* | DIR=* | QUEUE_WAIT=* | HEALTH_WAIT=* | PRIORITY=* | STUB_VRAM=* | MIN_HEADROOM=* | MIN_CACHED=*) export "${kv?}" ;;
    *) echo "FAIL args: unknown knob '$kv'"; exit 2 ;;
  esac
done
dir=${DIR:-.work/ailab}; envf=$dir/serve.env
engine=${ENGINE:-.work/ailab/engine-seqs32s2}; pack=${PACK:-.work/ailab/pack-kq}
server=${SERVER:-serve/target/release/baro-serve}
n=${N:-20}; kvp=${KV_POOL:-3}; tmax=${TMAX:-24576}; mi=${MAX_INFLIGHT:-64}; port=${PORT:-18433}
queue=${QUEUE:-gpu}; vramgb=${VRAM_GB:-20}; tlimit=${TIMEOUT:-14400}; extra=${EXTRA:-}
qwait=${QUEUE_WAIT:-3600}; hwait=${HEALTH_WAIT:-900}
priority=${PRIORITY:-50}

vram() {  # used bytes, the fullest card (sysfs: gpu-wait is not on PATH inside a queued job)
  if [ -n "${STUB_VRAM:-}" ]; then echo "$STUB_VRAM"; return; fi
  cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null | sort -n | tail -1
}
alive() { [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null; }
jobstate() { "$GW" status --json "$1" 2>/dev/null | jq -r '.job.state // "unknown"' 2>/dev/null || echo unknown; }

teardown() {  # pid job engine_pid idle  -> 0 when clean, prints FAIL line and returns 1 otherwise
  local pid=$1 job=$2 epid=$3 idle=$4 i now
  if alive "$pid"; then kill -TERM "$pid" 2>/dev/null || :; fi
  for i in $(seq 120); do alive "$pid" || break; sleep 0.5; done
  if alive "$pid"; then kill -KILL "$pid" 2>/dev/null || :; echo "FAIL down: baro-serve $pid ignored SIGTERM for 60 s, killed"; return 1; fi
  if [ -n "$job" ] && [ "$queue" != none ]; then
    for i in $(seq 120); do
      case $(jobstate "$job") in succeeded | failed | cancelled | timeout | timed_out | killed) break ;; esac
      sleep 1
    done
  fi
  for i in $(seq 10); do alive "$epid" || break; sleep 0.5; done
  if alive "$epid"; then kill -KILL "$epid" 2>/dev/null || :; echo "FAIL down: engine child $epid still alive after baro-serve exit"; return 1; fi
  now=$idle
  for i in $(seq 40); do  # the driver releases the card a moment after the process is gone
    now=$(vram)
    [ $(( now - idle )) -le 104857600 ] && break
    sleep 0.5
  done
  if [ $(( now - idle )) -gt 104857600 ]; then
    echo "FAIL down: VRAM $(( now / 1048576 )) MiB, idle was $(( idle / 1048576 )) MiB (limit +100 MiB)"; return 1
  fi
  echo "down: ok, engine $epid gone, VRAM $(( now / 1048576 )) MiB (idle $(( idle / 1048576 )) MiB)"
}

case $mode in
status)
  [ -f "$envf" ] || { echo "FAIL status: no $envf (not up)"; exit 1; }
  # shellcheck disable=SC1090
  . "$envf"
  alive "$PID" || { echo "FAIL status: baro-serve $PID is dead (job $JOB $(jobstate "$JOB"))"; exit 1; }
  h=$(curl -sf --max-time 5 "http://127.0.0.1:$PORT/health") || { echo "FAIL status: no /health on $PORT"; exit 1; }
  [ "$(jq -r .status <<<"$h")" = ok ] || { echo "FAIL status: health $(jq -c '{status}' <<<"$h")"; exit 1; }
  echo "status: up pid=$PID engine=$ENGINE_PID job=$JOB $(jobstate "$JOB") $(jq -c '{queue, limits}' <<<"$h") vram_mib=$(( $(vram) / 1048576 ))"
  ;;
down)
  [ -f "$envf" ] || { echo "FAIL down: no $envf (not up)"; exit 1; }
  # shellcheck disable=SC1090
  . "$envf"
  teardown "$PID" "$JOB" "$ENGINE_PID" "$IDLE_VRAM" || exit 1
  rm -f "$envf"
  ;;
up)
  if [ -f "$envf" ]; then
    # shellcheck disable=SC1090
    . "$envf"
    if alive "$PID"; then echo "FAIL up: already up (pid $PID); run down first"; exit 1; fi
    echo "up: stale $envf (pid $PID dead), removed"; rm -f "$envf"
  fi
  case $queue in gpu | cpu | none) ;; *) echo "FAIL up: QUEUE=$queue (gpu|cpu|none)"; exit 1 ;; esac
  for f in "$engine" "$pack/identity.json" "$server"; do [ -e "$f" ] || { echo "FAIL up: missing $f"; exit 1; }; done
  warm=${WARM_BODY:-}
  if [ -z "$warm" ]; then
    warm=$(ls bench/ailab/states-run/s01-r1.json bench/ailab/states/s01.json 2>/dev/null | head -1 || :)
  fi
  [ -n "$warm" ] && [ -s "$warm" ] || { echo "FAIL up: no warm body (WARM_BODY)"; exit 1; }
  mkdir -p "$dir"
  jq '.max_tokens = ([.max_tokens // 0, 16000] | max)' "$warm" > "$dir/warm-body.json"  # P0 bodies carry the runner's 200: repo floor
  warm=$dir/warm-body.json; min16k "$tmax" "$(jq -r .max_tokens "$warm")"
  head=$(( tmax - $(jq -r .max_tokens "$warm") ))
  [ "$head" -ge "${MIN_HEADROOM:-6000}" ] || { echo "FAIL up: TMAX $tmax leaves $head prompt tokens beside max_tokens $(jq -r .max_tokens "$warm") (need ${MIN_HEADROOM:-6000}): the server answers 400 exceed_context"; exit 1; }
  idle=$(vram); pidf=$dir/serve.pid; log=$dir/serve.log; arm=$dir/arm.txt
  [ -n "$idle" ] || { echo "FAIL up: cannot read VRAM from sysfs"; exit 1; }
  rm -f "$pidf"; : > "$log"
  eenv="--engine-env BARO_TMAX=$tmax --engine-env BARO_SPEC=0"
  for kv in $extra; do eenv="$eenv --engine-env $kv"; done
  jobsh=$dir/serve-job.sh  # a script file: gpu-wait re-quotes an inline command (`$$` arrived as `$`)
  {
    echo '#!/usr/bin/env bash'
    echo "echo \$\$ > $PWD/$pidf"
    echo "exec env BARO_SPEC=0 $(realpath "$server") --engine $engine --pack $pack --port $port --parallel $n --kv-pool $kvp --max-inflight $mi $eenv > $PWD/$log 2>&1"
  } > "$jobsh"
  chmod +x "$jobsh"; inner="$PWD/$jobsh"
  job=""
  case $queue in
    none) setsid "$inner" > /dev/null 2>&1 < /dev/null & ;;
    *)
      out=$("$GW" submit --json --room "$queue" --priority "$priority" --vram "$vramgb" --timeout "$tlimit" --label ailab-serve --cwd "$PWD" -- "$inner") \
        || { echo "FAIL up: gpu-wait submit: $out"; exit 1; }
      job=$(jq -r '.job_id // empty' <<<"$out")
      [ -n "$job" ] || { echo "FAIL up: no job id in: $(cut -c1-200 <<<"$out")"; exit 1; }
      ;;
  esac
  fail() {  # step reason
    echo "FAIL up/$1: $2 (log $log)"
    if [ -n "${pid:-}" ] && alive "$pid"; then kill -TERM "$pid" 2>/dev/null || :; sleep 2; kill -KILL "$pid" 2>/dev/null || :; fi
    if [ -n "$job" ]; then "$GW" cancel "$job" > /dev/null 2>&1 || :; fi
    exit 1
  }
  pid=""
  for _ in $(seq "$qwait"); do  # the job may wait in the queue behind other lanes
    [ -s "$pidf" ] && { pid=$(cat "$pidf"); break; }
    if [ -n "$job" ]; then
      case $(jobstate "$job") in failed | cancelled | timeout | timed_out | killed) fail queue "job $job $(jobstate "$job") before it started" ;; esac
    fi
    sleep 1
  done
  [ -n "$pid" ] || fail queue "job $job did not start in $qwait s"
  echo "up: job=$job pid=$pid, waiting for /health"
  ok=""
  for _ in $(seq $(( hwait * 2 ))); do
    if curl -sf --max-time 3 "http://127.0.0.1:$port/health" 2> /dev/null | jq -e '.status == "ok"' > /dev/null 2>&1; then ok=1; break; fi
    alive "$pid" || fail health "baro-serve died: $(tail -3 "$log" | tr '\n' '|')"
    sleep 0.5
  done
  [ -n "$ok" ] || fail health "no /health in $hwait s"
  code=$(curl -s -o "$dir/warm.json" -w '%{http_code}' -H 'content-type: application/json' --data-binary @"$warm" \
    --max-time 900 "http://127.0.0.1:$port/v1/chat/completions" || echo 000)
  [ "$code" = 200 ] || fail warm "HTTP $code from $warm: $(head -c 200 "$dir/warm.json" 2> /dev/null)"
  minc=${MIN_CACHED:-3500}; warm_runner="skipped (MIN_CACHED=0)"
  if [ "$minc" != 0 ]; then
    python3 -I bench/ailab/step.py "http://127.0.0.1:$port/v1" 1 2 "$dir/warm-runner" --offset 100 --expect-cached "$minc" --expect-from-step 1 > "$dir/warm-runner.log" 2>&1 \
      || fail warm "two asks through the runner: $(grep -E '^(FAIL|Traceback)' "$dir/warm-runner.log" | head -1 | cut -c1-230) (see $dir/warm-runner.log)"
    warm_runner="ok, ask 2 cached $(awk -F'\t' '$1 == 1 && $4 == 1 {print $11}' "$dir/warm-runner/posts.tsv" | head -1) tokens (>= $minc)"
  fi
  health=$(curl -sf --max-time 5 "http://127.0.0.1:$port/health") || fail readback "no /health after warm"
  epid=$(grep -o 'engine child pid: [0-9]*' "$log" | tail -1 | awk '{print $NF}' || :)
  [ -n "$epid" ] && alive "$epid" || fail readback "no live engine child pid in the log"
  renv() { tr '\0' '\n' < "/proc/$epid/environ" | awk -F= -v k="$1" '$1 == k {print substr($0, length(k) + 2)}'; }
  esha=$(sha256sum "/proc/$epid/exe" | cut -c1-16); eexe=$(readlink "/proc/$epid/exe")
  e_seqs=$(renv BARO_SEQS); e_tmax=$(renv BARO_TMAX); e_spec=$(renv BARO_SPEC); e_pool=$(renv BARO_KV_POOL)
  e_share=$(renv BARO_KV_SHARE); e_kvq=$(renv BARO_KVQ)
  echo_seqs=$(grep -m1 '^engine: BARO_SEQS:' "$log" | awk '{print $NF}' || :)
  h_tmax=$(jq -r .limits.tmax <<<"$health"); h_spec=$(jq -r .limits.spec_k <<<"$health")
  {
    echo "date=$(date -Is) commit=$(git rev-parse --short HEAD) label=ailab-serve job=$job queue=$queue"
    stale=$(find serve/src serve/Cargo.toml serve/Cargo.lock -type f -newer "$server" | tr '\n' ' ')
    echo "server=$server sha=$(sha256sum "$server" | cut -c1-16) built=$(date -r "$server" -Is) head=$(git rev-parse --short HEAD) src_newer_than_binary='${stale:-none}'"
    echo "engine=$eexe sha=$esha engine_pid=$epid"
    echo "pack=$(jq -r .pack <<<"$health") identity_sha=$(sha256sum "$pack/identity.json" | cut -c1-16) index_sha=$(sha256sum "$pack/index.txt" 2> /dev/null | cut -c1-16)"
    echo "want: N=$n kv_pool=$kvp tmax=$tmax max_inflight=$mi spec=0 extra='$extra'"
    echo "engine_environ: BARO_SEQS=$e_seqs BARO_TMAX=$e_tmax BARO_SPEC=$e_spec BARO_KV_POOL=$e_pool BARO_KV_SHARE=$e_share BARO_KVQ=${e_kvq:-f32(unset)}"
    echo "engine_echo: BARO_SEQS=${echo_seqs:-none}"
    echo "health_limits: tmax=$h_tmax spec_k=$h_spec mrows=$(jq -r .limits.mrows <<<"$health") kmax=$(jq -r .limits.kmax <<<"$health")"
    echo "warm_runner: $warm_runner"
    echo "warm: HTTP $code cached=$(jq -r '.timings.cached // "n/a"' "$dir/warm.json") completion_tokens=$(jq -r '.usage.completion_tokens // "n/a"' "$dir/warm.json")"
    echo "vram_idle_mib=$(( idle / 1048576 )) vram_warm_mib=$(( $(vram) / 1048576 ))"
  } | tee "$arm"
  bad=""
  [ "$e_seqs" = "$n" ] || bad="$bad BARO_SEQS environ '$e_seqs' != $n;"
  [ "$echo_seqs" = "$n" ] || bad="$bad BARO_SEQS echo '${echo_seqs:-none}' != $n;"
  [ "$e_tmax" = "$tmax" ] && [ "$h_tmax" = "$tmax" ] || bad="$bad tmax environ '$e_tmax' health '$h_tmax' != $tmax;"
  [ "$e_spec" = 0 ] && [ "$h_spec" = 0 ] || bad="$bad spec environ '$e_spec' health '$h_spec' != 0;"
  [ "$e_pool" = "$kvp" ] || bad="$bad BARO_KV_POOL environ '$e_pool' != $kvp;"
  [ "$e_share" = 1 ] || bad="$bad BARO_KV_SHARE environ '$e_share' != 1;"
  [ -z "$bad" ] || fail readback "arm differs from the request:$bad"
  {
    echo "URL=http://127.0.0.1:$port/v1"; echo "PORT=$port"; echo "JOB=$job"; echo "PID=$pid"; echo "ENGINE_PID=$epid"
    echo "ARM=$PWD/$arm"; echo "LOG=$PWD/$log"; echo "IDLE_VRAM=$idle"
  } > "$envf"
  echo "PASS up: $n sessions, tmax $tmax, engine $esha, serve.env $envf"
  ;;
esac
