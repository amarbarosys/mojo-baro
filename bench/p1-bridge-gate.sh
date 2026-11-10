#!/usr/bin/env bash
# usage: bench/p1-bridge-gate.sh MODEL [OUT]        MODEL = qwen25 | lily | ornith
#
# P1 gate 4 (bench/p1-bridge-protocol.md, frozen 599d8fd + amendment 1): E15's three models
# continue in llama.cpp from OUR exported state.
#   tokenize (CPU)      our tokenizer from the GGUF (tools/baro-tokenize), before any GPU job
#   ours     (GPU job)  baro-serve: our cold 32 ids (control N), LAT1 state export
#   convert  (CPU)      tools/state-to-llama-slot per state; K/V-swapped copies for the falsifier
#   llama    (GPU job)  llama-server: cold ids (refcache, P17), control L, primary, falsifier
#   score    (CPU)      verdict by the frozen rules; a void is a failure
# Each GPU phase is this script re-run as ONE blocking `gpu-wait run` job (PHASE=...), the idiom of
# bench/llama-handoff.sh. gpu-wait's daemon owns a job's processes, so a server started as
# `gpu-wait run ... &` is not a descendant of the caller and cannot be reaped by it; started inside
# the phase, it is a direct child and stop() reaps it (ledger gpuwaitingroom.md 2026-09-17). The GPU
# is held only during the two GPU phases, and our engine and llama-server never overlap.
# env: QUICK=N   first N prompts (iterate; a claim needs the full 20, P18)
#      FALSIFY=1 also run the K/V-swapped arm (the protocol runs it on one model, P11)
#      PREFLIGHT=1 CPU only, no GPU job and no engine of ours: llama-server --device none makes a
#        stand-in state through the reverse tool or the oracle. Proves convert, llama, falsifier and
#        score end to end before any GPU minute (P15). No control N; never a gate result.
set -euo pipefail
cd "$(dirname "$0")/.."
name=${1:?usage: bench/p1-bridge-gate.sh qwen25|lily|ornith [OUT]}
pre=${PREFLIGHT:-0}; falsify=${FALSIFY:-0}; quick=${QUICK:-20}; phase=${PHASE:-all}
out=${2:-.work/fork/g4-gate/$name$([ "$pre" = 1 ] && echo -preflight || true)}
server=${LLAMA_SERVER:-$HOME/llama.cpp/build/bin/llama-server}
serve=serve/target/release/baro-serve
fwd=.work/fork/state-to-llama-slot
rev=.work/fork/llama-slot-to-state
tok=.work/fork/baro-tokenize
refcache=$HOME/iTools/harness/refcache/refcache.sh
py=./.venv/bin/python
case "$name" in
  qwen25) gguf=$HOME/Models/qwen2.5-7b-instruct-gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf; engine=.work/fork/g4/qwen25-spark/engine; pack=.work/fork/g4/qwen25-spark/pack; attn_only=1 ;;
  lily)   gguf=$HOME/Models/lily-cybersecurity-7b-v0.2-q6_k/lily-cybersecurity-7b-v0.2-q6_k-BARO-04867e2.gguf; engine=.work/fork/g4/lily-spark/engine; pack=.work/fork/g4/lily-spark/pack; attn_only=1 ;;
  ornith) gguf=$HOME/Models/ornith-1.5-9b-q4_K_M/Ornith-1.5-9B-Q4_K_M-BARO-04867e2.gguf; engine=.work/fork/engine; pack=.work/fork/g4/ornith/pack; attn_only=0 ;;
  *) echo "FAIL setup: unknown model $name"; exit 1 ;;
esac
mkdir -p "$out/states" "$out/slots" "$out/ids"
exec > >(tee -a "$out/gate.log") 2>&1
fail() { echo "FAIL $1: $2 (log $out/gate.log)"; exit 1; }
for f in "$gguf" "$server" "$fwd" "$refcache"; do [ -e "$f" ] || fail setup "missing $f"; done
[ "$pre" = 1 ] || for f in "$engine" "$pack/pack.bin" "$serve"; do [ -e "$f" ] || fail setup "missing $f"; done
mapfile -t prompts < <(ls bench/mtp-prompts/p*.txt | head -n "$quick")
[ "${#prompts[@]}" -ge 1 ] || fail setup "no prompts"
names=(); for t in "${prompts[@]}"; do names+=("$(basename "$t" .txt)"); done

# A server is a direct child of THIS shell. Its own child (baro-serve's engine) is signalled
# first because baro-serve outlives TERM while the engine runs; the wait is bounded.
spid=""
stop() {
  [ -n "$spid" ] || return 0
  pkill -TERM -P "$spid" 2>/dev/null || true; kill -TERM "$spid" 2>/dev/null || true
  for _ in $(seq 1 20); do kill -0 "$spid" 2>/dev/null || { spid=""; return 0; }; sleep 0.5; done
  pkill -KILL -P "$spid" 2>/dev/null || true; kill -KILL "$spid" 2>/dev/null || true; spid=""
}
trap stop EXIT
post() { curl -sS --fail-with-body -X POST "$1" -H 'content-type: application/json' --data @"$2"; }
req() { printf '{"prompt":[%s],"n_predict":%s,"temperature":0,"cache_prompt":true,"return_tokens":true}' "$1" "$2"; }
slot() { curl -sS --fail-with-body -X POST "$lurl/slots/0?action=$1" -H 'content-type: application/json' -d "{\"filename\":\"$2\"}"; }

start_llama() {  # TAG; sets $lurl
  local lport=$((20000 + RANDOM % 20000)) dev=(-ngl 99) hide=()
  [ "$pre" = 1 ] && { dev=(--device none -ngl 0); hide=(HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1); }
  env "${hide[@]}" "$server" -m "$gguf" "${dev[@]}" -fa on -np 1 -c 4096 -ctk f16 -ctv f16 -b 2048 -ub 512 \
    --port "$lport" --no-webui --slot-save-path "$out/slots" > "$out/llama-$1.log" 2>&1 &
  spid=$!
  lurl="http://127.0.0.1:$lport"
  for _ in $(seq 1 1200); do
    curl -sf "$lurl/health" >/dev/null 2>&1 && break
    kill -0 "$spid" 2>/dev/null || fail "llama-$1" "server exited: $(tail -2 "$out/llama-$1.log" | tr '\n' ' ')"
    sleep 0.5
  done
  curl -sf "$lurl/health" >/dev/null 2>&1 || fail "llama-$1" "no /health in 600 s"
  curl -sf "$lurl/props" > "$out/llama-props-$1.json" || fail "llama-$1" "/props"
  echo "llama readback ($1): $($py -c "import json;d=json.load(open('$out/llama-props-$1.json'));print('n_ctx',d['default_generation_settings']['n_ctx'],'build',d.get('build_info'),'model',d.get('model_path'))")" | tee -a "$out/arm.txt"
}

phase_standin() {  # PREFLIGHT only: llama.cpp's own state at |P|-1 stands in for ours
  start_llama standin
  for p in "${names[@]}"; do
    $py - "bench/mtp-prompts/$p.txt" "$lurl" > "$out/ids/$p.ids" <<'PY' || fail tokenize "$p"
import json, sys, urllib.request
b = json.dumps({"content": open(sys.argv[1]).read(), "add_special": True}).encode()
r = urllib.request.urlopen(urllib.request.Request(sys.argv[2] + "/tokenize", b, {"Content-Type": "application/json"}), timeout=60)
print(",".join(str(x) for x in json.load(r)["tokens"]))
PY
    ids=$(cat "$out/ids/$p.ids")
    slot erase x > /dev/null || fail standin "$p erase"
    req "${ids%,*}" 0 > "$out/req.json"; post "$lurl/completion" "$out/req.json" > /dev/null || fail standin "$p prefill"
    slot save "standin-$p.slot" > /dev/null || fail standin "$p save"
  done
  stop
  local hd=""; [ "$attn_only" = 1 ] && hd=$($py tools/gguf-geometry.py "$gguf" | awk '{print $6}')
  for p in "${names[@]}"; do
    if [ "$attn_only" = 1 ]; then $py tools/llama-slot-kv-oracle.py "$out/slots/standin-$p.slot" "$out/states/$p.state" --hd "$hd" > "$out/states/$p.log" 2>&1 || fail standin "$p oracle"
    else "$rev" "$out/slots/standin-$p.slot" preflight-pack "$out/states/$p.state" > "$out/states/$p.log" 2>&1 || fail standin "$p reverse tool"; fi
  done
}

phase_ours() {
  env BARO_PACK="$pack" BARO_SPEC=0 BARO_CKPT_DIR="$out/ckpts" "$serve" --engine "$engine" --pack "$pack" --port 0 > "$out/ours.stdout" 2> "$out/ours.stderr" &
  spid=$!
  for _ in $(seq 1 2400); do
    grep -q '^listening on' "$out/ours.stdout" && break
    kill -0 "$spid" 2>/dev/null || fail ours "server exited: $(tail -2 "$out/ours.stderr" | tr '\n' ' ')"
    sleep 0.5
  done
  local ourl; ourl=$(grep -m1 -oE 'http://[0-9.:]+' "$out/ours.stdout") || fail ours "no listening line in 1200 s"
  echo "ours readback: engine_sha=$(sha256sum "$engine" | cut -c1-16) pack=$pack pack_sha=$($py -c "import json;print(json.load(open('$pack/identity.json'))['pack_sha256'][:16])") state=$(curl -sf "$ourl/v1/state" | $py -c "import json,sys;d=json.load(sys.stdin);print('portable',d['portable'],'kv',d['kv'])")" | tee -a "$out/arm.txt"
  for p in "${names[@]}"; do
    [ -s "$out/ids/$p.ids" ] || fail ours "$p has no ids (tokenized on the CPU before this job)"
    ids=$(cat "$out/ids/$p.ids")
    printf '{"prompt":[%s],"max_tokens":32,"temperature":0,"spec":false}' "$ids" > "$out/req.json"
    post "$ourl/v1/completions" "$out/req.json" > "$out/ids/$p.ours.json" || fail ours "$p completion"
    printf '{"tokens":[%s]}' "$ids" > "$out/req.json"
    curl -sS --fail-with-body -X POST "$ourl/v1/state/export" -H 'content-type: application/json' --data @"$out/req.json" -o "$out/states/$p.state" || fail ours "$p export: $(head -c 300 "$out/states/$p.state")"
  done
  stop
  # serve/engine.mojo prints two "state saved:" lines per save (one names the format, one the
  # time) and serve/spark.mojo prints one; the format line is the one both print exactly once.
  local ns; ns=$(grep -c 'state saved:.*format BAROST' "$out/ours.stderr" || true)
  [ "$ns" = "${#names[@]}" ] || fail ours "$ns state-saved lines for ${#names[@]} prompts"
  echo "ours readback: $ns/${#names[@]} state-saved lines in the engine log" | tee -a "$out/arm.txt"
}

phase_llama() {
  start_llama gate
  run_arm() {  # ARM SLOT-or-"" P IDS
    slot erase x > /dev/null || fail "$1" "$3 erase"
    if [ -n "$2" ]; then slot restore "$2" > "$out/ids/$3.$1.restore.json" || fail "$1" "$3 restore refused: $(cat "$out/ids/$3.$1.restore.json")"; fi
    req "$4" 32 > "$out/req.json"; post "$lurl/completion" "$out/req.json" > "$out/ids/$3.$1.json" || fail "$1" "$3 completion"
  }
  local reqs="$out/ref-requests.txt" key; : > "$reqs"; for p in "${names[@]}"; do cat "$out/ids/$p.ids" >> "$reqs"; done
  key=$("$refcache" key "$(sha256sum "$gguf" | cut -d' ' -f1)" "llama=$(git -C "$HOME/llama.cpp" rev-parse --short HEAD)" "dev=$([ "$pre" = 1 ] && echo cpu || echo ngl99) fa=on np=1 c=4096 ctk=f16 ctv=f16 b=2048 ub=512 n32 T0" "@$reqs")
  if "$refcache" get "$key" cold.tar "$out/cold.tar" 2>/dev/null; then tar -xf "$out/cold.tar" -C "$out/ids"; echo "refcache: HIT $key" | tee -a "$out/arm.txt"; else echo "refcache: MISS $key" | tee -a "$out/arm.txt"; fi
  for p in "${names[@]}"; do
    ids=$(cat "$out/ids/$p.ids")
    [ -s "$out/ids/$p.cold.json" ] || run_arm cold "" "$p" "$ids"
    slot erase x > /dev/null || fail ctrlL "$p erase"
    req "${ids%,*}" 0 > "$out/req.json"; post "$lurl/completion" "$out/req.json" > /dev/null || fail ctrlL "$p prefill"
    slot save "ctrlL-$p.slot" > /dev/null || fail ctrlL "$p save"
    run_arm ctrlL "ctrlL-$p.slot" "$p" "$ids"
    run_arm primary "bridged-$p.slot" "$p" "$ids"
    [ "$falsify" != 1 ] || run_arm falsify "swapped-$p.slot" "$p" "$ids"
  done
  stop
  ( cd "$out/ids" && tar -cf ../cold.tar ./*.cold.json ) || fail refcache "tar"
  "$refcache" put "$key" cold.tar "$out/cold.tar" > /dev/null || fail refcache "put"
}

case "$phase" in ours) phase_ours; exit 0 ;; llama) phase_llama; exit 0 ;; all) ;; *) fail setup "unknown PHASE $phase" ;; esac

echo "arm: model=$name gguf=$gguf llama=$(git -C "$HOME/llama.cpp" rev-parse --short HEAD) prompts=${#names[@]} preflight=$pre falsify=$falsify commit=$(git rev-parse --short HEAD) dirty='$(git status --porcelain | tr '\n' ';')'" | tee "$out/arm.txt"
job() { gpu-wait run --priority 50 --vram 23 --timeout 1500 -- env HOME="$HOME" PATH="$PATH" PHASE="$1" QUICK="$quick" FALSIFY="$falsify" "$PWD/bench/p1-bridge-gate.sh" "$name" "$PWD/$out" || fail "job-$1" "gpu-wait job exited $?"; }
if [ "$pre" = 1 ]; then phase_standin
else
  [ -f "$pack/identity.json" ] || $py tools/engine-pack.py --identity "$pack" > "$out/identity.log" 2>&1 || fail identity "engine-pack.py --identity, $out/identity.log"
  # Our tokenizer, on the CPU, from the GGUF (the protocol's wording; bench/dense-run.sh does the
  # same). Not baro-serve's /tokenize: a spark pack carries no tokenizer.json, so that route is 503,
  # and tokenizing is not GPU work anyway.
  [ -x "$tok" ] || ./.venv/bin/mojo build tools/baro-tokenize.mojo -I . -I serve -o "$tok" > "$out/build-tok.log" 2>&1 || fail build "baro-tokenize, $out/build-tok.log"
  badt=()
  for p in "${names[@]}"; do
    "$tok" encode "bench/mtp-prompts/$p.txt" "$gguf" 2> "$out/ids/$p.tok.log" | paste -sd, - > "$out/ids/$p.ids"
    grep -qE '^[0-9]+(,[0-9]+)+$' "$out/ids/$p.ids" || badt+=("$p")
  done
  [ "${#badt[@]}" = 0 ] || fail tokenize "${#badt[@]}/${#names[@]} ${badt[*]}"
  job ours
fi

geom=(); if [ "$attn_only" = 1 ]; then read -r -a geom <<< "$($py tools/gguf-geometry.py "$gguf")"; [ "${#geom[@]}" = 8 ] || fail convert "gguf-geometry: ${geom[*]:-no output}"; fi
badc=()
for p in "${names[@]}"; do
  "$fwd" "$out/states/$p.state" "$out/slots/bridged-$p.slot" --kv f16 "${geom[@]}" > "$out/slots/$p.convert.log" 2>&1 || badc+=("$p")
  [ "$falsify" != 1 ] || $py tools/slot-swap-kv.py "$out/slots/bridged-$p.slot" "$out/slots/swapped-$p.slot" > "$out/slots/$p.swap.log" 2>&1 || badc+=("$p-swap")
done
[ "${#badc[@]}" = 0 ] || fail convert "${#badc[@]}/${#names[@]} ${badc[*]}"

if [ "$pre" = 1 ]; then phase_llama; else job llama; fi

# Read-back from what the running systems WROTE, not from the flags passed (P1). This build's
# llama-server log is silent on KV type and flash attention, but the slot files it writes are not.
rb=$($py tools/slot-readback.py $(for p in "${names[@]}"; do echo "$out/slots/ctrlL-$p.slot"; done)) || fail readback "$rb"
echo "llama readback (slot files it wrote): $rb" | tee -a "$out/arm.txt"
echo "llama readback (offload, from timings): $($py tools/slot-readback.py --decode-rate "$out" "${names[@]}")" | tee -a "$out/arm.txt"
nb=$(grep -l 'BAROST0[12] pos' "$out"/slots/*.convert.log | wc -l)
[ "$nb" = "${#names[@]}" ] || fail readback "only $nb/${#names[@]} states were read as BAROST by the tool"
echo "ours readback (state files it wrote): $nb/${#names[@]} accepted by the tool's magic and length checks ($(grep -ho 'BAROST0[12]' "$out"/slots/*.convert.log | sort -u | tr '\n' ' '))" | tee -a "$out/arm.txt"
$py tools/p1-bridge-score.py "$out" "$name" "$pre" "$falsify" "${names[@]}"
