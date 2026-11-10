#!/usr/bin/env bash
# Decode tok/s of the resident MoE engine across arms, in ONE gpu-wait job:
#   direct  the engine's own stdin protocol (BARO_SERVE=1), no server in between
#   serve   the same engine behind baro-serve, the same prompts as chat requests
#   harness the champion's method (one process per prompt, bench/mtp-prompts, GEN default)
# Each direct/serve arm runs short, short, long, long (the second long restores the first's
# prompt-end checkpoint, so it times decode at ~5.5k context without a prefill in front).
#
#   gpu-wait run --vram 22 --timeout 1500 -- bench/decode-probe.sh OUT_DIR
#
# Inputs: OUT_DIR/{short,long}.req, one engine request line each ({"id": 0, "prompt": [...],
# "n": 256, ...}), taken on the CPU from what baro-serve sends a stub engine
# (bench/stub-engine.py, FAKE_OUT) for the two chat requests below, so direct and serve arms
# decode the same ids. ENGINE, PACK, PERSONA required; G192 / G288 (grid-size builds) optional.
# MOE_MISS_ONLY=1 runs only direct M0, for BARO_TIER probes incompatible with the mega arm.
# Resident pack, MAX pool 2%, TMAX 8192.
set -euo pipefail

here=$(cd "$(dirname "$0")/.." && pwd)
out=$(realpath -m "${1:?usage: decode-probe.sh OUT_DIR}")
eng=${ENGINE:?ENGINE=path to the engine binary}
pack=${PACK:?PACK=resident pack dir}
persona=${PERSONA:?PERSONA=chat template}
for f in short.req long.req; do [[ -s $out/$f ]] || { echo "FAIL setup: no $out/$f"; exit 1; }; done
card=$(grep -l . /sys/class/drm/card*/device/mem_info_vram_total | xargs -n1 dirname \
  | while read -r d; do echo "$(cat "$d/mem_info_vram_total") $d"; done | sort -n | tail -1 | cut -d' ' -f2)
common=(BARO_PACK="$pack" BARO_TMAX=8192 BARO_SPEC=0 BARO_CKPT=16 MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=2)
printf 'arm\treq\tprompt\tcached\tgen\ttok_s\n' > "$out/results.tsv"
fails=()

gtt_ok() {  # $1 arm: a spilled pack makes every number of the arm invalid
  local g=$(( $(cat "$card/mem_info_gtt_used") >> 20 ))
  echo "$1 gtt_used $g MiB" >> "$out/arm.txt"
  ((g <= 4096)) || { echo "FAIL spill $1: GTT $g MiB, arm INVALID"; fails+=("$1"); return 1; }
}

direct() {  # $1 label, $2 engine, $3.. env
  local label=$1 e=$2; shift 2
  local log=$out/$label.log i=0
  { for f in short short long long; do i=$((i + 1)); sed "s/\"id\": 0/\"id\": $i/" "$out/$f.req"; done; } \
    | env "${common[@]}" BARO_SERVE=1 "$@" "$e" > "$log" 2>&1 || { echo "FAIL $label: engine exit, see $log"; fails+=("$label"); return; }
  { grep -E "^(BARO_MEGA|mega barrier gen|mega fail word)" "$log" || true; } | sort | uniq -c | sed "s/^/$label /" >> "$out/arm.txt"
  if grep -qE "mega fail word: [1-9]" "$log"; then   # NOT-RESIDENT: the tokens and the speed are garbage
    echo "FAIL $label: $(grep -m1 -oE 'mega fail word: [1-9].*' "$log"), arm INVALID"; fails+=("$label"); return
  fi
  local n; n=$(grep -cE '"done": ?true' "$log" || true)
  ((n == 4)) || { echo "FAIL $label: $n of 4 requests done, see $log"; fails+=("$label"); return; }
  grep -E '"done": ?true' "$log" | jq -r --arg a "$label" '[$a, .id, "-", .cached, .n, .tok_s] | @tsv' >> "$out/results.tsv"
  gtt_ok "$label" || true
}

serve() {  # $1 label, $2 engine, $3.. env
  local label=$1 e=$2; shift 2
  local port log=$out/$label.log
  port=$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
  env "${common[@]}" "$@" "$here/serve/target/release/baro-serve" --engine "$e" --pack "$pack" \
    --chat-template-file "$persona" --port "$port" > "$log" 2>&1 < /dev/null &
  local spid=$!
  for _ in $(seq 600); do
    grep -qF "listening on http://127.0.0.1:$port" "$log" && curl -sf "localhost:$port/health" > /dev/null && break
    kill -0 $spid 2>/dev/null || break
    sleep 1
  done
  if ! curl -sf "localhost:$port/health" > /dev/null; then
    echo "FAIL $label: no server, see $log"; fails+=("$label"); kill $spid 2>/dev/null; wait $spid 2>/dev/null; return
  fi
  local q="Write a Python function that returns the first 30 Fibonacci numbers, with a docstring and three unit tests."
  local sys="[{\"role\":\"system\",\"content\":\"You are a helpful assistant.\"},{\"role\":\"user\",\"content\":\"$q\"}]"
  local per="[{\"role\":\"user\",\"content\":\"$q\"}]" m i=0
  for m in "$sys" "$sys" "$per" "$per"; do
    i=$((i + 1))
    curl -sf "localhost:$port/v1/chat/completions" -H 'content-type: application/json' -d "{\"messages\": $m,
      \"max_tokens\": 256, \"temperature\": 0, \"chat_template_kwargs\": {\"enable_thinking\": false}}" \
      | jq -r --arg a "$label" --arg i "$i" '[$a, $i, .usage.prompt_tokens, (.usage.baro.cached_tokens // 0),
          .usage.completion_tokens, .timings.tok_s_gen] | @tsv' >> "$out/results.tsv" \
      || { echo "FAIL $label: request $i"; fails+=("$label"); break; }
  done
  { grep -E "engine: (BARO_MEGA|mega barrier gen)" "$log" || true; } | sort | uniq -c | sed "s/^/$label /" >> "$out/arm.txt"
  if grep -qE "mega fail word: [1-9]" "$log"; then echo "FAIL $label: mega fail word set, arm INVALID"; fails+=("$label"); fi
  gtt_ok "$label" || true
  kill $spid 2>/dev/null; wait $spid 2>/dev/null || true
}

harness() {  # $1 label, $2 engine, $3.. env: the champion's one-process-per-prompt method
  local label=$1 e=$2; shift 2
  local tf p t
  for tf in "$here"/bench/mtp-prompts/p0{1,2,3}-*.tokens; do
    p=$(basename "$tf" .tokens)
    env "${common[@]}" BARO_PROMPT="$tf" "$@" "$e" > "$out/$label.$p.log" 2>&1 < /dev/null || { echo "FAIL $label $p"; fails+=("$label"); continue; }
    t=$(grep -oE 'tok/s_gen: [0-9.]+' "$out/$label.$p.log" | cut -d' ' -f2 || true)
    [[ -n $t ]] || { echo "FAIL $label $p: no tok/s_gen line, see $out/$label.$p.log"; fails+=("$label"); continue; }
    printf '%s\t%s\t%s\t-\t-\t%s\n' "$label" "$p" "$(wc -w < "$tf")" "${t:-nan}" >> "$out/results.tsv"
  done
  gtt_ok "$label" || true
}

echo "engine $eng sha256 $(sha256sum "$eng" | cut -c1-16) checkout $(git -C "$here" rev-parse --short HEAD)" > "$out/arm.txt"
if [[ ${MOE_MISS_ONLY:-0} == 1 ]]; then
  [[ -n ${BARO_TIER:-} ]] || { echo "FAIL setup: MOE_MISS_ONLY requires BARO_TIER"; exit 2; }
  direct direct-tier-m0 "$eng" BARO_MEGA=0
  column -t "$out/results.tsv"
  if ((${#fails[@]})); then echo "FAIL ${#fails[@]} arm(s): ${fails[*]}"; exit 1; fi
  echo "PASS decode-probe tier-only -> $out"
  exit 0
fi
direct  direct-m0 "$eng" BARO_MEGA=0
direct  direct-m1 "$eng" BARO_MEGA=1
serve   serve-m0  "$eng" BARO_MEGA=0
serve   serve-m1  "$eng" BARO_MEGA=1
harness harness-m0 "$eng" BARO_MEGA=0
[[ -z ${G192:-} ]] || direct direct-g192-m1 "$G192" BARO_MEGA=1
[[ -z ${G288:-} ]] || direct direct-g288-m1 "$G288" BARO_MEGA=1
column -t "$out/results.tsv"
if ((${#fails[@]})); then echo "FAIL ${#fails[@]} arm(s): ${fails[*]}"; exit 1; fi
echo "PASS decode-probe -> $out"
