#!/usr/bin/env bash
# RegesCore-35B on the Blender skill-eval tasks (16 Blender 5.2 scripts, run and
# checked headless), one engine per call so one engine holds the card per job.
# Brief: .work/briefs/regescore-blender-ab.md; report exchange/regescore-blender-ab.md.
#
#   gpu-wait run --vram 23 --timeout 3600 -- bench/regescore-blender-ab.sh baro  OUT
#   gpu-wait run --vram 12 --timeout 3600 -- bench/regescore-blender-ab.sh baro-tier OUT
#   gpu-wait run --vram 22 --timeout 3600 -- bench/regescore-blender-ab.sh llama OUT
#   gpu-wait run --vram 23 --timeout 5400 -- bench/regescore-blender-ab.sh llama,baro RUN
#
# A comma list runs the engines in order in ONE job, each into RUN/<engine>, then writes
# RUN/TABLE.tsv: the reference arm is an arm, rerun beside ours, never reused from an
# earlier run (run 9 reused the morning's llama numbers). The table flags a gap over 1%
# between the engines' generated-token totals.
#
# Load gate: other lanes' CPU load cost decode 5-10% in run 9. A timed run waits up to
# 60 s for the 1-minute load average to reach LOAD_START_MAX (4), else FAILs before any
# GPU work; vram.tsv samples it (4th column) and the run is INVALID once it crosses
# LOAD_RUN_MAX (8): the sampler stops eval.py at first sight. BARO_EXPERTS runs
# (instrumentation) and the LEASE_SERVE job (serves, times nothing) are exempt; arm.txt
# says which. The llama arm reads its sampling back from llama-server's /props.
#
# Both engines get the same gguf, the same persona template, the same requests
# (eval.py: thinking off, temperature 0.35, top-p 0.95, top-k 20, min-p 0.05 in
# every request) and one request in flight. eval.py writes each request's usage
# and timings into OUT/<arm>/*/summary.json.
#
# baro: resident pack (all experts in VRAM, regescore-fast's BARO_MEGA=0) with
# BARO_PREFILL=1, served by an engine built from this checkout, because the
# gguf's own harness (e340ee1) predates BARO_PREFILL. BARO_TMAX=8192, not 32768:
# pack-fit puts the resident pack at 32768 over MAX's pool by 0.84 GB, and the
# largest request here is 5765 prompt + 1200 generated ids.
# The resident pack leaves ~1 GB beside a desktop that holds ~2 GB of VRAM
# (plasmashell + browser, 2026-09-24): the driver then moves the pack to GTT and
# decode drops from 90 to 8.6 tok/s with nothing in the logs. OUT/vram.tsv samples
# vram/gtt every 5 s; the resident arm FAILs if GTT passes 4 GB. baro-tier keeps
# the experts in pinned host memory (BARO_TIER=64, zero-copy), the arm that fits.
# ENGINE=/path/engine overrides the engine (default .work/regescore-ab/engine,
# built with -D BARO_MODEL=qwen35moe). ARMS="identity persona" (default).
# WARM=1 starts baro-serve with --warm: the persona prefix is prefilled and
# checkpointed before the first request (its time is in the server log's warm: line).
# MEM_PCT=N caps MAX's device memory pool (MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT):
# by default MAX takes 90% of free VRAM, which left the desktop no room next to the
# resident pack. The resident arm FAILs before any GPU work when the card cannot hold
# its 22.34 GB plus FIT_MARGIN_MB (1536) at the moment the job starts.
# MEGA=1 runs the persistent MoE token kernel (BARO_MEGA=1, resident arm only); the job
# FAILs if the engine sets its fail word (NOT-RESIDENT: garbage tokens at 10x speed).
# PF_CHUNK=N sets the batched prefill chunk (BARO_PREFILL_C; scratch scales with it).
# TIER_CAP=N sets the baro-tier expert cache (default 64 per layer); ONLY="t01_x t02_y"
# runs a task subset (a cap probe: the first task is cold, the rest reuse its prefix).
# JOBS=2 (default) keeps a second request queued at the one-sequence server while
# Blender checks the previous script, so the card never idles through the CPU half.
# Both engines serve one request at a time, so the engine's own prefill/decode
# timers are unchanged; the client's request_s then includes queue wait, so model
# time is read from the engine timings (TABLE.tsv prefill_s + decode_s).
# LEASE=DIR/lease.json sends the requests to a resident server from bench/baro-lease.sh
# instead of launching one (queue the run as `gpu-wait run --shared --vram 0.5`): it FAILs
# `lease mismatch: <field>` unless the lease runs the arm this run would launch (engine,
# baro-serve, pack, tokenizer, template, env knobs, warm), reads P1 back from the lease's
# server.log, and its checkpoints persist across runs. LEASE_SERVE=1 is the lease's own job.
set -euo pipefail

here=$(cd "$(dirname "$0")/.." && pwd)
eng=${1:?usage: regescore-blender-ab.sh baro|llama OUT_DIR}
out=$(realpath -m "${2:?usage: regescore-blender-ab.sh baro|llama OUT_DIR}")
mkdir -p "$out"
if [[ $eng == *,* ]]; then
  IFS=, read -ra engs <<<"$eng"
  [[ -z ${LEASE:-} || " ${engs[*]} " != *" llama "* ]] || { echo "FAIL usage: LEASE holds the card, llama cannot run beside it"; exit 2; }
  for e in "${engs[@]}"; do "$0" "$e" "$out/$e" || { echo "FAIL $e arm, A/B $eng stopped -> $out/$e"; exit 1; }; done
  "$here/bench/regescore-blender-ab-table.sh" "$out" > "$out/TABLE.tsv"
  awk -F'\t' '$1 == "tokens" && $7 == "GAP"' "$out/TABLE.tsv" | while IFS=$'\t' read -r _ a ln e n pct _; do
    echo "WARN tokens: $a llama $ln vs $e $n generated, gap ${pct}% > 1%"; done
  echo "PASS $eng -> $out/TABLE.tsv"; exit 0
fi
models=$HOME/Models/RegesCore-1.0-35
gguf=$models/RegesCore-1.0-35B-UD-Q4_K_S-BARO-e340ee1.gguf
persona=${PERSONA:-$models/personas/persona-video.jinja}   # bench/regescore-persona-pin.sh pins runs 1-5's
if [[ -n ${PERSONA_SHA:-} ]] && [[ $(sha256sum "$persona" | cut -c1-${#PERSONA_SHA}) != "$PERSONA_SHA" ]]; then
  echo "FAIL persona: $persona is not sha256 $PERSONA_SHA"; exit 1
fi
evaldir=$HOME/Projects/gpu-media/Blender/skill-eval
read -ra arms <<<"${ARMS:-identity persona}"
export BLENDER=${BLENDER:-$HOME/.local/bin/blender}
[[ -x $BLENDER ]] || { echo "FAIL setup: no Blender at $BLENDER"; exit 1; }

load1() { cut -d' ' -f1 /proc/loadavg; }
gt() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a > b) }'; }
start_max=${LOAD_START_MAX:-4}; run_max=${LOAD_RUN_MAX:-8}; gate=1
if [[ -n ${BARO_EXPERTS:-} ]]; then gate=0; echo "load gate: exempt, instrumentation run (BARO_EXPERTS)" > "$out/arm.txt"
elif [[ -n ${LEASE_SERVE:-} ]]; then gate=0; echo "load gate: exempt, lease server job (the runs on it are gated)" > "$out/arm.txt"
else
  for _ in $(seq 12); do gt "$(load1)" "$start_max" || break; sleep 5; done
  l=$(load1); gt "$l" "$start_max" && { echo "FAIL load: 1-min load average $l > $start_max at launch after 60 s, timed run refused"; exit 1; }
  echo "load gate: 1-min load $l <= $start_max at launch, INVALID above $run_max mid-run" > "$out/arm.txt"
fi
rm -f "$out/LOAD"
card=$(grep -l . /sys/class/drm/card*/device/mem_info_vram_total | xargs -n1 dirname \
  | while read -r d; do echo "$(cat "$d/mem_info_vram_total") $d"; done | sort -n | tail -1 | cut -d' ' -f2)
( while :; do
    l=$(load1); echo "$(date +%s) $(( $(cat "$card/mem_info_vram_used") >> 20 )) $(( $(cat "$card/mem_info_gtt_used") >> 20 )) $l"
    if ((gate)) && [[ ! -f $out/LOAD ]] && gt "$l" "$run_max"; then
      echo "$(date -Is) load average $l > $run_max" > "$out/LOAD"; pkill -f -- "^/usr/bin/python3 $evaldir/eval.py .*--out $out/" || :
    fi
    sleep 5
  done ) > "$out/vram.tsv" &
sampler=$!
# `|| :` in the traps: under set -e a kill of an already-dead pid would replace the exit status
# (a stopped lease served clean, then its job exited 1).
trap 'kill $sampler 2>/dev/null || :' EXIT
max_gtt() { awk 'BEGIN{m=0} $3>m{m=$3} END{print m}' "$out/vram.tsv"; }
load_check() {
  echo "max load $(awk 'BEGIN{m=0} $4>m{m=$4} END{print m}' "$out/vram.tsv")" >> "$out/arm.txt"
  [[ ! -f $out/LOAD ]] || { echo "FAIL load: $(cat "$out/LOAD"), run INVALID (see $out/vram.tsv)"; exit 1; }
}

run_evals() {
  local url=$1 rc=0 arm
  git -C "$evaldir" log -1 --format='skill-eval %h %s' > "$out/eval-commit.txt"
  for arm in "${arms[@]}"; do
    /usr/bin/python3 "$evaldir/eval.py" --url "$url" --arm "$arm" --name "regescore-$eng-$arm" \
      --out "$out/$arm" --temperature 0.35 --top-p 0.95 --top-k 20 --min-p 0.05 ${ONLY:+--only $ONLY} \
      --jobs "${JOBS:-2}" \
      > "$out/eval-$arm.log" 2>&1 || { echo "FAIL eval $arm: see $out/eval-$arm.log"; rc=1; }
    tail -1 "$out/eval-$arm.log"
  done
  return $rc
}

case $eng in
baro|baro-tier)
  engine=${ENGINE:-$here/.work/regescore-ab/engine}
  pack=$here/.work/moe-ngram/pack
  tok=$here/.work/latent-spec/pack-tier/tokenizer.json
  mode=(); want=()
  if [[ $eng == baro-tier ]]; then
    pack=$here/.work/latent-spec/pack-tier
    cap=${TIER_CAP:-64}
    mode=(BARO_TIER=$cap BARO_TIER_PINNED=1 BARO_TIER_ZC=1)
    want=("expert tier: cap $cap " "mode pinned+zc" "min rows ${BARO_PF_MIN_TIER:-96}")
  fi
  bs=$here/serve/target/release/baro-serve
  mega=${MEGA:-0}; mega_word=$([[ $mega == 1 ]] && echo True || echo False)
  mode+=(BARO_MEGA=$mega)
  # BARO_EXPERTS=<path>: router top-8 trace (bench/moe-locality-protocol.md). Instrumentation run,
  # never a timed one; the readback below proves the engine took it.
  [[ -z ${BARO_EXPERTS:-} ]] || { mode+=(BARO_EXPERTS=$BARO_EXPERTS); want+=("BARO_EXPERTS: $BARO_EXPERTS"); echo "expert trace $BARO_EXPERTS (instrumentation run, tok/s not a receipt)" >> "$out/arm.txt"; }
  [[ -z ${PF_CHUNK:-} ]] || { mode+=(BARO_PREFILL_C=$PF_CHUNK); want+=("chunk $PF_CHUNK "); }
  [[ -z ${MEM_PCT:-} ]] || mode+=(MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=$MEM_PCT)
  knobs=(BARO_SPEC=0 BARO_PREFILL=1 BARO_TMAX=8192 "${mode[@]}")
  if [[ -f $pack/identity.json ]]; then
    pack_id="$(jq -r .pack_sha256 "$pack/identity.json") from $(jq -r .source "$pack/identity.json")"
  else
    pack_id="pack.bin $(stat -c '%s %y' "$pack/pack.bin")"
  fi
  # The arm: what this run launches, or must find already running under LEASE.
  ident=("kind=$eng" "engine=$(sha256sum "$engine" | cut -c1-16)" "baro_serve=$(sha256sum "$bs" | cut -c1-16)"
    "pack=$pack_id" "tokenizer=$(sha256sum "$tok" | cut -c1-16)" "template=$(sha256sum "$persona" | cut -c1-16)"
    "env=${knobs[*]}" "warm=${WARM:+1}")
  {
    echo "checkout $(git -C "$here" rev-parse --short HEAD)"
    echo "engine $engine sha256 ${ident[1]#*=}"
    echo "baro-serve sha256 ${ident[2]#*=}"
    echo "pack $pack $pack_id"
    echo "tokenizer $tok sha256 ${ident[4]#*=}"
    echo "template $persona sha256 ${ident[5]#*=}"
    [[ -z ${MEM_PCT:-} ]] || echo "MAX pool cap ${MEM_PCT}%"
  } >> "$out/arm.txt"
  log=$out/server.log; ldir=
  if [[ -n ${LEASE:-} ]]; then   # bench/baro-lease.sh: the server is resident, this run only sends requests
    LEASE=$(realpath "$LEASE"); ldir=$(dirname "$LEASE")
    for kv in "${ident[@]}"; do
      k=${kv%%=*}; have=$(jq -r --arg k "$k" '.identity[$k] // "MISSING"' "$LEASE")
      [[ $have == "${kv#*=}" ]] || { echo "FAIL lease mismatch: $k (lease '$have', this run '${kv#*=}')"; exit 1; }
    done
    port=$(jq -r .port "$LEASE"); log=$(jq -r .server_log "$LEASE")
    kill -0 "$(jq -r .pid "$LEASE")" 2>/dev/null || { echo "FAIL lease: server pid $(jq -r .pid "$LEASE") is gone ($LEASE)"; exit 1; }
    { echo "lease $(jq -r .job "$LEASE") $(jq -r .started_at "$LEASE") $LEASE"
      echo "checkpoints $(jq -r .ckpt_dir "$LEASE") (persist across runs on this lease)"; } >> "$out/arm.txt"
  else
    if [[ $eng == baro ]]; then   # checked when the job starts, i.e. once the queue gave us the card
      used=$(( $(cat "$card/mem_info_vram_used") >> 20 )); total=$(( $(cat "$card/mem_info_vram_total") >> 20 ))
      # Resident need, MiB: pack 20032 + KV 320 + engine buffers ~560 + prefill scratch (2012 at
      # 1024-row chunks, from run 6's OOM log; ~half at 512) + MAX's pool (MEM_PCT of free, reserved
      # up front). pack-fit's 1 GB scratch guess undercounted it.
      need=${RESIDENT_NEED_MB:-22000}
      margin=$(( total - used - need ))
      echo "fit: vram_used ${used} MiB of ${total}, need ${need}, left ${margin} MiB" >> "$out/arm.txt"
      ((margin >= ${FIT_MARGIN_MB:-1536})) || { echo "FAIL fit: ${margin} MiB left after the resident pack, need ${FIT_MARGIN_MB:-1536} (close Vivaldi?)"; exit 1; }
    fi
    # A free port per run: a fixed one answered /health from a dry run's stub
    # server while this job's engine was still loading (2026-09-24).
    port=$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
    env BARO_CKPT_DIR="$out/ckpt" "${knobs[@]}" \
      "$bs" --engine "$engine" --pack "$pack" --tokenizer "$tok" --chat-template-file "$persona" \
      --port "$port" ${WARM:+--warm} > "$log" 2>&1 < /dev/null &
    spid=$!
    # SIGTERM the engine with baro-serve: left to stdin EOF it held the card 30 s after the last
    # request until baro-serve gave up and killed it (2026-09-24, exchange/lane-LEASE-report.md).
    trap 'pkill -TERM -P $spid 2>/dev/null || :; kill $spid $sampler 2>/dev/null || :; wait $spid 2>/dev/null || :' EXIT
    for _ in $(seq 600); do
      grep -qF "listening on http://127.0.0.1:$port" "$log" && curl -sf "localhost:$port/health" > /dev/null && break
      kill -0 $spid 2>/dev/null || { echo "FAIL serve: baro-serve exited, see $log"; exit 1; }
      sleep 1
    done
  fi
  grep -qF "listening on http://127.0.0.1:$port" "$log" && curl -sf "localhost:$port/health" > /dev/null \
    || { echo "FAIL serve: no /health from ${LEASE:-this job}'s server, see $log"; exit 1; }
  # Read back every arm-defining knob from the running engine (P1).
  # The engine turns batched prefill off while it writes the expert trace (serve/engine.mojo, BARO_EXPERTS).
  pf_word=True; [[ -z ${BARO_EXPERTS:-} ]] || pf_word=False
  want+=("engine: BARO_MEGA: $mega_word" "engine: BARO_SPEC: False" "engine: BARO_PREFILL: $pf_word" "tmax: 8192")
  [[ -z ${WARM:-} ]] || want+=("warm: ")
  for w in "${want[@]}"; do
    grep -qF "$w" "$log" || { echo "FAIL readback: '$w' not in $log"; exit 1; }
    echo "readback ok: $(grep -m1 -F "$w" "$log")" >> "$out/arm.txt"
  done
  if [[ $eng == baro && -z ${LEASE:-} ]]; then   # a spilled pack runs 10x slow: stop at once, not after 32 requests
    ( while kill -0 $spid 2>/dev/null; do
        g=$(( $(cat "$card/mem_info_gtt_used") >> 20 ))
        if ((g > 4096)); then echo "$(date -Is) gtt_used ${g} MB" > "$out/SPILL"; kill $spid; break; fi
        sleep 2
      done ) &
  fi
  rc=0
  if [[ -n ${LEASE_SERVE:-} ]]; then   # bench/baro-lease.sh start: publish the lease, serve until stop
    jq -n --argjson port "$port" --argjson pid $spid --argjson pgid "$(ps -o pgid= -p $$)" \
      --arg job "${GPU_WAITING_ROOM_JOB:-none}" --arg at "$(date -Is)" --arg log "$log" --arg ckpt "$out/ckpt" \
      '{job: $job, started_at: $at, port: $port, pid: $pid, pgid: $pgid, server_log: $log, ckpt_dir: $ckpt,
        identity: ($ARGS.positional | map(capture("^(?<k>[^=]*)=(?<v>.*)$") | {(.k): .v}) | add)}' \
      --args "${ident[@]}" > "$out/lease.json.tmp"
    mv "$out/lease.json.tmp" "$out/lease.json"
    trap 'date -Is > "$out/STOPPED"' TERM   # stop signals the whole group; the server exits with it
    while kill -0 $spid 2>/dev/null; do wait $spid || :; done
    [[ -f $out/STOPPED ]] || { echo "FAIL serve: the lease server exited unasked, see $log"; rc=1; }
  else
    run_evals "http://127.0.0.1:$port/v1" || rc=1
  fi
  load_check
  if grep -qE "mega fail word: [1-9]" "$log"; then   # NOT-RESIDENT megakernel: garbage tokens
    echo "FAIL fail word: $(grep -m1 -oE 'mega fail word: [1-9].*' "$log"), run INVALID"; exit 1
  fi
  for s in "$out/SPILL" ${ldir:+"$ldir/SPILL"}; do
    [[ ! -f $s ]] || { echo "FAIL spill: $(cat "$s"), the resident pack left VRAM; run INVALID"; exit 1; }
  done
  g=$(max_gtt); echo "max gtt_used ${g} MB" >> "$out/arm.txt"
  if [[ $eng == baro ]] && ((g > 4096)); then
    echo "FAIL spill: GTT reached ${g} MB, the resident pack left VRAM; timings invalid (see $out/vram.tsv)"; exit 1
  fi
  ((rc == 0)) || exit 1
  ;;
llama)
  llama=${LLAMA:-$HOME/llama.cpp/build/bin/llama-server}
  echo "llama.cpp $(git -C "$HOME/llama.cpp" log -1 --format='%h %cs') server $llama sha256 $(sha256sum "$llama" | cut -c1-16)" >> "$out/arm.txt"
  # Server defaults = eval.py's per-request sampling, so /props can prove them (P1).
  export LLAMA=$llama LLAMA_ARGS="--chat-template-file $persona -ctk q8_0 -ctv q8_0 --temp 0.35 --top-p 0.95 --top-k 20 --min-p 0.05"
  echo "LLAMA_ARGS $LLAMA_ARGS ctx 32768" >> "$out/arm.txt"
  props_check() {
    local p have kv; p=$(curl -sf "${1%/v1}/props") || { echo "FAIL readback: no /props from ${1%/v1}"; return 1; }
    for kv in temperature=0.35 top_p=0.95 top_k=20 min_p=0.05; do
      have=$(jq -r --arg k "${kv%=*}" '.default_generation_settings.params[$k] // "MISSING"' <<<"$p")
      awk -v a="$have" -v b="${kv#*=}" 'BEGIN { exit !(a != "MISSING" && (a - b)^2 < 1e-10) }' \
        || { echo "FAIL readback: llama-server /props ${kv%=*} $have, want ${kv#*=}"; return 1; }
      echo "readback ok: /props ${kv%=*} $have" >> "$out/arm.txt"
    done
  }
  export -f run_evals props_check
  export out eng evaldir
  port=$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
  ARMS_S="${arms[*]}" "$HOME/iTools/bin/with-llama" --ctx 32768 --port "$port" --log "$out/server.log" "$gguf" -- \
    bash -c 'read -ra arms <<<"$ARMS_S"; props_check "$LLAMA_URL" && run_evals "$LLAMA_URL"'
  load_check
  echo "max gtt_used $(max_gtt) MB" >> "$out/arm.txt"
  ;;
*)
  echo "FAIL usage: engine must be baro, baro-tier or llama, got $eng"; exit 2
  ;;
esac
if [[ -n ${LEASE_SERVE:-} ]]; then echo "PASS $eng lease served -> $out"; else echo "PASS $eng ${#arms[@]} arms -> $out"; fi
