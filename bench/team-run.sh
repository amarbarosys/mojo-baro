#!/usr/bin/env bash
# usage: N=6 bench/team-run.sh OUTDIR   (re-execs under gpu-wait; 3 engines on one weight copy: speaker + workers w1, w2; about 8 minutes)
# First executor for the team prompt (serve/prompts/orcasaq-system.md): crew_assign goes to a worker engine, its note comes back as the tool result.
# Task: two independent mod-1000 chains (tools/crew_tasks, L=3), report both values. Solo baseline: same task, one engine, no tools.
# The final answer is only the text after </think>; a reply cut inside the reasoning counts as no answer (team2 scored a false pass from reasoning text).
# Thinking is ON (the template default; the thinking-off run team1 gave crew_assign 0 on items 0 to 3 and was cancelled).
# Preregistered (committed before the run), greedy, one run, N=4:
#  P1 the speaker calls crew_assign at least twice on at least 3 of N items;
#  P2 the team final answer contains both expected values on at least 3 of N items;
#  P3 (reported, not gated) team correct count versus solo correct count;
#  P4 every tool call the speaker makes is handled (crew_assign only; any other call is counted and answered with an error).
set -euo pipefail
cd "$(dirname "$0")/.."; export LC_ALL=C
. bench/min16k.sh; min16k 24576 16000
out=$(realpath -m "${1:?usage: N=6 bench/team-run.sh OUTDIR}"); mkdir -p "$out"
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
  exec gpu-wait run --priority ${PRIO:-90} --vram 23 --timeout 3600 -- env GPU_WAITING_ROOM_JOB=1 N="${N:-4}" PATH="$PATH" "$PWD/bench/team-run.sh" "$out"
fi
N=${N:-4}; port=8320
fail() { echo "FAIL $1: $2"; exit 1; }
eng=$(realpath ${ENG:-.work/crew/engine-kv8-fix2}); pack=$(realpath .work/bake27/verify3/pack); tok=$(realpath .work/crew/tok/tokenizer.json); tpl=$(realpath serve/templates/qwen35.jinja)
shim=$(realpath .work/shim-build/libamarbaro_shim.so); serve=serve/target/release/baro-serve; rec=$out/weights.rec
used=$(( $(cat /sys/class/drm/card1/device/mem_info_vram_used) >> 20 )); [ "$used" -lt 2560 ] || fail preflight "vram_used ${used} MiB"
pids=(); stop_all() { for p in "${pids[@]}"; do kill -- -"$p" 2>/dev/null || kill "$p" 2>/dev/null || true; done; wait 2>/dev/null || true; }; trap stop_all EXIT
for i in 0 1 2; do
  w=(BARO_WEIGHTS_IMPORT="$rec"); [ "$i" = 0 ] && w=(BARO_WEIGHTS_EXPORT="$rec")
  setsid env "${w[@]}" BARO_EMBD_HOST=1 BARO_PACK="$pack" BARO_SPEC=1 BARO_MEGA=1 BARO_TMAX=24576 BARO_SHIM_LIB="$shim" \
    "$serve" --engine "$eng" --pack "$pack" --tokenizer "$tok" --chat-template-file "$tpl" --port $((port + i)) > "$out/serve-$i.out" 2> "$out/serve-$i.err" &
  pids+=($!)
  for _ in $(seq 180); do curl -sf "http://127.0.0.1:$((port + i))/health" >/dev/null 2>&1 && break; sleep 1; done
  curl -sf "http://127.0.0.1:$((port + i))/health" >/dev/null || fail "health $i" "$out/serve-$i.err"
done
grep -q "weights exported over IPC" "$out/serve-0.err" && grep -q "weights imported over IPC" "$out/serve-1.err" && grep -q "weights imported over IPC" "$out/serve-2.err" || fail readback "weights not shared"
echo "N=$N engines=3 tmax=24576 vram_mib=$(( $(cat /sys/class/drm/card1/device/mem_info_vram_used) >> 20 )) system_sha=$(sha256sum serve/prompts/orcasaq-system.md | cut -c1-16) tools_sha=$(sha256sum serve/prompts/orcasaq-tools.json | cut -c1-16) template_sha=$(sha256sum "$tpl" | cut -c1-16) engine_sha=$(sha256sum "$eng" | cut -c1-16)" > "$out/arm.txt"
chat() { # port, messages-json, tools(1|0), max_tokens -> response json on stdout
  jq -nc --rawfile s serve/prompts/orcasaq-system.md --slurpfile t serve/prompts/orcasaq-tools.json --argjson m "$2" --argjson tl "$3" --argjson mt "$4" \
    '{messages:([{role:"system",content:$s}]+$m),max_tokens:$mt,temperature:0,chat_template_kwargs:{enable_thinking:true}} + (if $tl==1 then {tools:$t[0]} else {} end)' \
  | curl -s -m 300 -X POST "http://127.0.0.1:$1/v1/chat/completions" -H 'content-type: application/json' -d @-
}
.work/crew_tasks 3 $((2 * N)) 21 mod1000 > "$out/chains.json"
: > "$out/results.tsv"
for i in $(seq 0 $((N-1))); do
  p1=$(jq -r ".[$((2*i))].prompt" "$out/chains.json"); e1=$(jq -r ".[$((2*i))].expected" "$out/chains.json")
  p2=$(jq -r ".[$((2*i+1))].prompt" "$out/chains.json"); e2=$(jq -r ".[$((2*i+1))].expected" "$out/chains.json")
  task=$(printf 'Role: speaker\nTask: compute two independent values and report both.\nValue 1: %s\nValue 2: %s\nWorkers: w1, w2.' "$p1" "$p2")
  msgs=$(jq -nc --arg c "$task" '[{role:"user",content:$c}]')
  assigns=0; other=0; recov=0; final=""
  for round in 1 2 3 4 5; do
    r=$(chat $port "$msgs" 1 16000); echo "$r" > "$out/i$i-speaker-r$round.json"
    if jq -e '.error' <<<"$r" >/dev/null 2>&1; then fail request "item $i round $round: $(jq -r .error.message <<<"$r")"; fi
    n=$(jq -r '.choices[0].message.tool_calls // [] | length' <<<"$r")
    if [ "$n" = 0 ]; then
      final=$(jq -r '.choices[0].message.content // ""' <<<"$r" | awk '/<\/think>/{f=1;sub(/.*<\/think>/,"")} f')
      if [ -z "$(tr -d ' \n' <<<"$final")" ] && [ "$recov" -lt 2 ] && [ "$assigns" -gt 0 ]; then
        recov=$((recov+1)); msgs=$(jq -c '. + [{role:"user",content:"All worker notes are in. Write the final answer now."}]' <<<"$msgs"); continue
      fi
      break
    fi
    msgs=$(jq -c --argjson a "$(jq -c '.choices[0].message | {role:"assistant",content:((.content // "") | sub("^.*</think>";"")),tool_calls:.tool_calls}' <<<"$r")" '. + [$a]' <<<"$msgs")
    for k in $(seq 0 $((n-1))); do
      call=$(jq -c ".choices[0].message.tool_calls[$k]" <<<"$r"); cid=$(jq -r '.id // "call"' <<<"$call"); name=$(jq -r '.function.name' <<<"$call")
      if [ "$name" = crew_assign ]; then
        assigns=$((assigns+1)); args=$(jq -r '.function.arguments' <<<"$call"); wk=$(jq -r '.worker // "w1"' <<<"$args"); wt=$(jq -r '.task // ""' <<<"$args")
        wp=$port; [ "$wk" = w1 ] && wp=$((port+1)); [ "$wk" = w2 ] && wp=$((port+2))
        wm=$(jq -nc --arg c "$(printf 'Role: worker (%s)\nTask: %s\nShow each step. End with the note.' "$wk" "$wt")" '[{role:"user",content:$c}]')
        wr=$(chat $wp "$wm" 1 16000); echo "$wr" > "$out/i$i-$wk-$round-$k.json"
        note=$(jq -r '.choices[0].message.content // ""' <<<"$wr" | awk '/<\/think>/{f=1;sub(/.*<\/think>/,"")} f'); [ -n "$(tr -d ' \n' <<<"$note")" ] || note="error: the worker did not finish (no final note)"
      else other=$((other+1)); note="error: $name is not available in this test"; fi
      msgs=$(jq -c --arg id "$cid" --arg n "$note" '. + [{role:"tool",tool_call_id:$id,content:$n}]' <<<"$msgs")
    done
  done
  echo "$final" > "$out/i$i-final.txt"
  ok=0; grep -qw -- "$e1" <<<"$final" && grep -qw -- "$e2" <<<"$final" && ok=1
  # solo baseline: one engine, no tools, same task
  sm=$(jq -nc --arg c "$(printf 'Role: speaker\nTask: compute two independent values and report both.\nValue 1: %s\nValue 2: %s' "$p1" "$p2")" '[{role:"user",content:$c}]')
  sr=$(chat $port "$sm" 0 16000); sfinal=$(jq -r '.choices[0].message.content // ""' <<<"$sr" | awk '/<\/think>/{f=1;sub(/.*<\/think>/,"")} f'); echo "$sfinal" > "$out/i$i-solo.txt"
  sok=0; grep -qw -- "$e1" <<<"$sfinal" && grep -qw -- "$e2" <<<"$sfinal" && sok=1
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' $i $assigns $other $ok $sok $recov | tee -a "$out/results.tsv"
done
echo "items with >=2 crew_assign: $(awk -F'\t' '$2>=2{n++}END{print n+0}' "$out/results.tsv") / $N   (P1 needs 3)" | tee -a "$out/arm.txt"
echo "team correct: $(awk -F'\t' '{n+=$4}END{print n+0}' "$out/results.tsv") / $N   (P2 needs 3)   solo correct: $(awk -F'\t' '{n+=$5}END{print n+0}' "$out/results.tsv") / $N" | tee -a "$out/arm.txt"
echo "other tool calls (unhandled): $(awk -F'\t' '{n+=$3}END{print n+0}' "$out/results.tsv")   recoveries (stop inside the reasoning, one retry): $(awk -F'\t' '{n+=$6}END{print n+0}' "$out/results.tsv")   failword lines nonzero: $(cat "$out"/serve-*.err | grep -c 'fail word: [1-9]' || true)" | tee -a "$out/arm.txt"
