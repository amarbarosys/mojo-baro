#!/usr/bin/env bash
# usage: bench/ailab/parity-llama.sh OUT        env: RESUME=1 (reuse the finished llama stages of an earlier run in OUT), QUICK=N (first N of the 20 states, default 20), DRY=1 (stubs, CPU only),
#        EXPLORE=1 (skip bench/preflight.sh --check; the verdict can never be PASS: exit 3 at best)
# P4 answer parity of baro-serve against llama-server on the runner's own requests (docs/plan-ailab-server.md, P4).
# ONE gpu-wait job runs every arm in turn (the job re-execs itself under `gpu-wait run`; DRY=1 skips the queue):
#   1 llama-server --parallel 1, greedy, drives street-sim's ask_openai (fake game): the units = request bytes (max_tokens 16000),
#     llama's ids (/completion return_tokens) and parsed tool call; round 2+ bodies carry llama's own replies (driver ruling)
#   2 llama-server --parallel 20: the same bytes, lockstep bursts of 20 = the CONTROL (llama against llama)
#   3 baro-serve N=20 (bench/ailab/serve.sh up QUEUE=none, TMAX 24576): the same bytes; 4 baro-serve N=1: own-identity reference
#   5 single-session engine, resident, teacher-forced with llama's ids (BARO_FORCE per request) + forced with our own ids (control)
# Verdict by bench/ailab/parity_llama.py compare (bar frozen in its docstring and in docs/plan-ailab-server.md): exit 0 PASS (full
# 20-state set, no DRY), 3 UNVERIFIED (QUICK or DRY, bar met on the subset), 1 FAIL. Receipts and arm.txt land in OUT.
# Dry run: DRY=1 bench/ailab/parity-llama.sh .work/ailab/p4-dry  (stub llama, bench/stub-engine.py, stub-force.sh).
set -euo pipefail
cd "$(dirname "$0")/../.."
out=${1:?usage: parity-llama.sh OUT}; quick=${QUICK:-20}; dry=${DRY:-0}; export DRY=$dry
case $out in /*) ;; *) out=$PWD/$out ;; esac
GW=$HOME/.local/bin/gpu-wait; self=$(realpath "$0")
# Knobs the queued re-run must carry: the queue starts the job in a clean environment, so every knob is passed as `env K=V` from ONE list
# (RESUME was missing once: the rerun ran the whole pipeline, +19 GPU-min). QUEUE=stub re-execs through a fake queue (env -i) for the CPU selftest.
fwd=(); for k in QUICK EXPLORE RESUME DRY LLAMA_KV TMAX PACK ENGINE FENGINE OURS_PORT; do if [ -n "${!k+x}" ]; then fwd+=("$k=${!k}"); fi; done
if [ "${GPU_WAITING_ROOM_JOB:-}" = stub ] && [ "$dry" != 1 ]; then echo "FAIL queue: the fake queue marker without DRY=1 would run a real GPU job bare"; exit 1; fi
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ] && [ "${QUEUE:-}" = stub ] && [ "$dry" = 1 ]; then
  f=("${fwd[@]}"); if [ "${STUB_FORWARD:-1}" = 0 ]; then f=(DRY=1); fi  # STUB_FORWARD=0: the old bug (every knob dropped), the selftest's negative control; DRY=1 always crosses: the harness is not the knob under test
  exec env -i PATH="$PATH" HOME="$HOME" GPU_WAITING_ROOM_JOB=stub "${f[@]}" "$self" "$out"
fi
if [ "$dry" != 1 ] && [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
  if [ "${EXPLORE:-0}" = 1 ]; then echo "EXPLORE=1: bench/preflight.sh --check skipped, the verdict is capped at UNVERIFIED (exit 3)"
  else bench/preflight.sh --check || { echo "FAIL preflight: bench/preflight.sh --check (no GPU job queued; EXPLORE=1 for an exploratory run)"; exit 1; }; fi
  exec "$GW" run --timeout 5400 --vram 21 --label ailab-parity-llama -- env "${fwd[@]}" "$self" "$out"
fi
mkdir -p "$out"
export LLAMA_ENV=$out/llama.env  # this gate's own llama state file: never another job's
pack=${PACK:-.work/ailab/pack-kq}; ours_port=${OURS_PORT:-18435}
kvt=${LLAMA_KV:-f32}  # llama's KV cache type, f32 to match ours (llama default f16 fails its own f32 reference, CLAUDE.md); LLAMA_KV=default keeps llama-server's default
kvf=(); if [ "$kvt" != default ]; then kvf=(--kv "$kvt"); fi
tmax=${TMAX:-24576}  # runner prompts are 4.0-4.6K tokens; with max_tokens 16000 TMAX 20480 answers 400 past 4480 (P4 quick, 2026-11-06)
if [ "$dry" = 1 ]; then
  export LLAMA=bench/ailab/stub-llama.py; ngl=0; eng=bench/stub-engine.py; feng=bench/ailab/stub-force.sh
  seng=$eng; dk=(STUB_VRAM=1000000000); sk=(QUEUE=none "${dk[@]}" "ENGINE=$eng")
  export FAKE_OUT=$out/stub-engine-requests.jsonl BARO_PACK=$pack  STUB_CACHED=3569  # bench/stub-engine.py reads FAKE_OUT and BARO_PACK; STUB_CACHED lets serve.sh up's warm assert (MIN_CACHED 3500) run on the stub
else
  ngl=99; eng=${ENGINE:-.work/ailab/engine-seqs32s2}; feng=${FENGINE:-.work/ailab/engine-main}
  seng=$feng; dk=(); sk=(QUEUE=none "ENGINE=$eng")
fi
[ -e "$pack/pack.bin" ] || [ "$dry" = 1 ] || { echo "FAIL preflight: no pack at $pack"; exit 1; }
vram() { cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null | sort -n | tail -1; }
if [ "${RESUME:-0}" = 1 ] && [ -f "$out/arm.txt" ]; then cp "$out/arm.txt" "$out/arm-prev.txt"; fi
{ echo "gate=parity-llama quick=$quick dry=$dry explore=${EXPLORE:-0} ngl=$ngl commit=$(git rev-parse --short HEAD) dirty='$(git status --short | grep -v '^??' | tr '\n' ' ')'"
  echo "engine_multi=$eng sha=$(sha256sum "$eng" | cut -c1-16) engine_single=$feng sha=$(sha256sum "$feng" | cut -c1-16) pack=$pack"
  echo "vram_used_start_mib=$(( $(vram) / 1048576 )) job=${GPU_WAITING_ROOM_JOB:-none} bodies=floored to max_tokens 16000, temperature 0, tmax=$tmax llama_kv=$kvt"; } | tee "$out/arm.txt"
srvdirs=(); spid=""
cleanup() {
  if [ -n "$spid" ]; then pkill -TERM -P "$spid" || :; kill -TERM "$spid" 2> /dev/null || :; sleep 1; kill -KILL "$spid" 2> /dev/null || :; fi
  bench/ailab/llama.sh down > /dev/null 2>&1 || :
  for d in "${srvdirs[@]}"; do [ -f "$d/serve.env" ] && bench/ailab/serve.sh down "DIR=${d#"$PWD"/}" > /dev/null 2>&1 || :; done
}
trap cleanup EXIT
PL="python3 bench/ailab/parity_llama.py"
step() { local n=$1; shift; "$@" > "$out/$n.log" 2>&1 || { tail -5 "$out/$n.log"; echo "FAIL $n: see $out/$n.log"; exit 1; }; tail -2 "$out/$n.log"; }
llama_arm() {  # name parallel [extra llama.sh flags] -> up, arm line into arm.txt
  local nm=$1 par=$2; shift 2
  step "up-$nm" bench/ailab/llama.sh up --ngl "$ngl" --parallel "$par" --port 18462 "${kvf[@]}" "$@"
  { echo "arm $nm: $(cut -c1-400 "${LLAMA_ENV%.env}.arm")"; } >> "$out/arm.txt"
}
ours_arm() {  # name N
  local d=$out/serve-$1; srvdirs+=("$d"); mkdir -p "$d"  # serve.sh joins $PWD/$DIR: DIR must be repo-relative
  step "up-$1" bench/ailab/serve.sh up "${sk[@]}" "N=$2" "DIR=${d#"$PWD"/}" TMAX=$tmax PORT=$ours_port
  { echo "arm $1: $(cat "$d/arm.txt" | tr '\n' ' ' | cut -c1-500)"; } >> "$out/arm.txt"
}
soft_down() {  # the lifecycle assert (engine gone, VRAM back) is P3's gate: here its result is recorded, not fatal (other lanes move VRAM)
  local r; r=$(bench/ailab/serve.sh down "DIR=${out#"$PWD"/}/serve-$1" "${dk[@]}" 2>&1 | tail -1) || :
  echo "down $1: $r" | tee -a "$out/arm.txt"
}
serial_up() {  # own-identity reference: baro-serve on the single-session engine, no --parallel (serve.sh refuses N=1: no BARO_SEQS)
  FAKE_OUT=${FAKE_OUT:-/dev/null} setsid env BARO_SPEC=0 serve/target/release/baro-serve --engine "$seng" --pack "$pack" --port "$ours_port" \
    --engine-env BARO_TMAX=$tmax --engine-env BARO_SPEC=0 > "$out/serve-n1.log" 2>&1 < /dev/null &
  spid=$!
  for _ in $(seq 600); do
    kill -0 "$spid" 2>/dev/null || { echo "FAIL up-n1: baro-serve died, see $out/serve-n1.log"; exit 1; }
    if curl -sf -m 2 "http://127.0.0.1:$ours_port/health" 2> /dev/null | jq -e '.status == "ok"' > /dev/null 2>&1; then break; fi
    sleep 0.5
  done
  echo "arm n1: serial baro-serve pid=$spid engine=$seng sha=$(sha256sum "$seng" | cut -c1-16) $(curl -s -m 5 "http://127.0.0.1:$ours_port/health" | jq -c '{limits}')" | tee -a "$out/arm.txt"
}
serial_down() {
  pkill -TERM -P "$spid" || :; kill -TERM "$spid" 2>/dev/null || :
  for _ in $(seq 40); do kill -0 "$spid" 2>/dev/null || break; sleep 0.5; done
  kill -KILL "$spid" 2>/dev/null || :; spid=""
}
fp() { (cd "$out/units" && cat $(ls | grep '\.ref\.json$' | sort) | sha256sum | cut -c1-16); }  # fingerprint of the units a stage ran on
resume_ok() {  # RESUME=1: the llama stages (gen, p20 control burst, forced control) of a finished earlier run in this OUT are reused
  [ "${RESUME:-0}" = 1 ] && [ -f "$out/arm-prev.txt" ] || return 1
  local st now; now=$(fp 2> /dev/null || :)  # every reused stage must have run on the units that are on disk now (a regenerated units/ made a corrupt mix once)
  for st in gen l20 fctl; do
    [ "$(cat "$out/$st.units.sha" 2> /dev/null || :)" = "$now" ] && [ -n "$now" ] || { echo "FAIL resume: stage $st has no receipt for the current units (fingerprint '$now', receipt '$(cat "$out/$st.units.sha" 2> /dev/null || echo none)'): rerun without RESUME"; exit 1; }
  done
  local k; k=$(ls "$out/units" 2> /dev/null | grep -c '\.ref\.json$' || :)
  [ "$k" -gt 0 ] && grep -q '^gen:' "$out/gen.log" 2> /dev/null && grep -q '^fctl:' "$out/fctl.log" 2> /dev/null \
    && [ "$(ls "$out/l20" 2> /dev/null | grep -c '\.json$' || :)" = "$k" ] && [ "$(ls "$out/fctl" 2> /dev/null | grep -c '\.json$' || :)" = "$k" ]
}
if resume_ok; then
  echo "RESUME: reusing gen, llama p20 control burst and forced control from the earlier run in $out ($(grep -c '' "$out/arm-prev.txt") arm lines kept in arm-prev.txt; earlier $(head -1 "$out/arm-prev.txt" | grep -o 'commit=[0-9a-f]*'), this run commit=$(git rev-parse --short HEAD): only the runner changed, not gen/control code paths)" | tee -a "$out/arm.txt"
else
  llama_arm l1 1
  step gen $PL gen --llama http://127.0.0.1:18462 --out "$out" --n "$quick"
  fp > "$out/gen.units.sha"
  step down-l1 bench/ailab/llama.sh down
  llama_arm l20 20 --no-unified  # one KV per slot: with --kv-unified the one-token forced control missed the prompt cache (CPU A/B: 11 of 40 positions in 420 s vs 40 in 175 s)
  step burst-l20 $PL burst --kind llama --ids --url http://127.0.0.1:18462 --units "$out/units" --out "$out/l20"
  fp > "$out/l20.units.sha"
  step fctl $PL fctl --url http://127.0.0.1:18462 --units "$out/units" --out "$out/fctl"
  fp > "$out/fctl.units.sha"
  step down-l20 bench/ailab/llama.sh down
fi
ours_arm n20 20
step burst-n20 $PL burst --kind ours --url "http://127.0.0.1:$ours_port" --units "$out/units" --out "$out/ours20"
step negctl $PL negctl --url "http://127.0.0.1:$ours_port" --units "$out/units" --tmax "$tmax"
soft_down n20
serial_up
step burst-n1 $PL burst --kind ours --url "http://127.0.0.1:$ours_port" --units "$out/units" --out "$out/ours1"
serial_down
force_run() {  # name jsonl log -> one resident single-session engine process, one request line per unit
  env BARO_SERVE=1 BARO_PACK="$pack" BARO_SPEC=0 BARO_TMAX=$tmax MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=2 "$feng" < "$1" > "$2" 2>&1
  ! grep -qE 'fail word: [1-9]|NOT-RESIDENT|Unhandled exception' "$2" || return 1
}
step forcejl-ref $PL forcejl --units "$out/units" --src ref --out "$out/forced-ref.jsonl"
step force-ref force_run "$out/forced-ref.jsonl" "$out/forced-ref.log"
step forcejl-own $PL forcejl --units "$out/units" --src "$out/ours20" --out "$out/forced-own.jsonl"
step force-own force_run "$out/forced-own.jsonl" "$out/forced-own.log"
full=""; if [ "$quick" = 20 ] && [ "$dry" != 1 ] && [ "${EXPLORE:-0}" != 1 ]; then full="--full"; fi
set +e
$PL compare --units "$out/units" --ctl "$out/l20" --ours "$out/ours20" --own1 "$out/ours1" --fctl "$out/fctl" --forced "$out/forced-ref.log" --own "$out/forced-own.log" $full 2>&1 | tee "$out/verdict.txt"
rc=${PIPESTATUS[0]}
if [ "$dry" = 1 ] && grep -q '^units ' "$out/verdict.txt"; then
  echo "DRY RUN reached the verdict (stub engine answers nothing, so the bar is not met; never a verdict)"; rc=3
fi
echo "gpu-wait stats: $("$GW" stats --days 1 2> /dev/null | head -3 | tr '\n' ' ' || :)" | tee -a "$out/verdict.txt"
exit "$rc"
