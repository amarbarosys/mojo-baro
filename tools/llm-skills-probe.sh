#!/usr/bin/env bash
# usage: tools/llm-skills-probe.sh MODEL.gguf OUTDIR     (OUTDIR = ~/Brain/Skills/llms-skills/<key>/runs)
# Model-skill probe (docs/lanes/2026-11-13-llm-skills-pilot-brief.md): ONE gpu-wait job starts `tools/baro serve MODEL`
# (setsid, BARO_TMAX=32768 so 8K prompt + 16000 tokens fit), sends ~10 chat requests, saves each request and response
# under OUTDIR, writes summary.tsv (id, finish, prompt/completion tokens, wall s, wall tok/s, check, verdict), copies the
# server log, and stops the server by process group. Floors: BARO_TMAX >= 16384, max_tokens >= 16000 (bench/min16k.sh).
# `tools/baro serve` exports BARO_NEURONS=1 by default: the serve front appends ~600 tokens of <reference> notes to the last user
# message. The probe sets BARO_NEURONS=${PROBE_NEURONS:-0}, so the model is judged without them (first run with them on: runs/neurons-on-*).
# The server has NO default sampling (log: "sampling defaults: top_k - top_p - min_p -"): every request carries
# temperature 0.6, top_p 0.95, top_k 20 (the model card values) unless the probe says otherwise.
# Thinking arrives inline in message.content (the template opens <think>, the model closes it): checks read the text after </think>.
# Dense/spark bakes (llama, qwen2, granite, spark2_5) carry `comptime TMAX = 4096` in their embedded profile.mojo and ignore BARO_TMAX:
# /health tmax 4096 fails the tmax check. Fix: BARO_CACHE=<dir> with an engine rebuilt from the cached src with TMAX 32768 (fanout report).
# PROBE_RESCORE=1 PROBE_IN_JOB=1 tools/llm-skills-probe.sh MODEL OUTDIR re-checks the saved *.res.json with no server (wall = decode_s).
# Needs the serve cache built on CPU first: tools/baro serve MODEL --no-serve. Exit non-zero on any failed check.
set -euo pipefail
fail() { echo "FAIL $1: $2"; exit 1; }
model=$(readlink -f "${1:?usage: tools/llm-skills-probe.sh MODEL.gguf OUTDIR}"); out=$(readlink -f "${2:?OUTDIR}")
port=${PROBE_PORT:-18433}; tmax=32768
. "$(dirname "$0")/../bench/min16k.sh"; min16k "$tmax" 16000
mkdir -p "$out"
if [ -z "${PROBE_IN_JOB:-}" ]; then
  tools/baro serve "$model" --no-serve > "$out/cache.log" 2>&1 || fail cache "see $out/cache.log"
  grep -q "engine=hit pack=hit" "$out/cache.log" || fail cache "cache miss: build it on CPU first"
  exec gpu-wait run --vram 16 --timeout 1500 -- env PROBE_IN_JOB=1 PROBE_NEURONS=${PROBE_NEURONS:-0} ${BARO_CACHE:+BARO_CACHE=$BARO_CACHE} bash -c "cd '$PWD' && tools/llm-skills-probe.sh '$model' '$out' > '$out/job.log' 2>&1; rc=\$?; cat '$out/job.log'; exit \$rc"
fi
base="http://127.0.0.1:$port"
log="$out/server.log"
if [ -z "${PROBE_RESCORE:-}" ]; then
: > "$log"
setsid env ${BARO_CACHE:+BARO_CACHE=$BARO_CACHE} BARO_TMAX=$tmax BARO_NEURONS=${PROBE_NEURONS:-0} tools/baro serve "$model" --port $port > "$log" 2>&1 & spid=$!
trap 'kill -TERM -- "-$spid" 2>/dev/null || true; sleep 3; kill -KILL -- "-$spid" 2>/dev/null || true' EXIT
for _ in $(seq 240); do curl -sf -m 2 -o /dev/null "$base/health" && break; sleep 1; done
curl -sf -m 2 "$base/health" > "$out/health.json" || fail start "server never answered /health"
grep -q "\"tmax\":$tmax" "$out/health.json" || fail tmax "/health tmax is not $tmax: $(cat "$out/health.json")"
echo "server up: $(jq -c .limits "$out/health.json")"

S='"temperature":0.6,"top_p":0.95,"top_k":20,"max_tokens":16000'
NT='"chat_template_kwargs":{"enable_thinking":false}'
mk() { printf '%s\n' "$2" > "$out/$1.req.json"; }

mk p01-chat-think-on  "{\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of Australia, and why is it not Sydney? Two sentences.\"}],$S}"
mk p02-chat-think-off "{\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of Australia, and why is it not Sydney? Two sentences.\"}],$S,$NT}"
mk p03-reason-snail   "{\"messages\":[{\"role\":\"user\",\"content\":\"A snail is at the bottom of a 10-meter well. Each day it climbs 3 meters, but each night it slips back 2 meters. How many days until it escapes? End with 'Answer: N'.\"}],$S}"
mk p04-tool-call      "{\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Paris right now?\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Get the current weather for a city\",\"parameters\":{\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\",\"description\":\"City name\"}},\"required\":[\"location\"]}}}],$S,$NT}"
mk p05-json-schema    "{\"messages\":[{\"role\":\"user\",\"content\":\"Extract the person: 'Marta Kowalski, 42, lives in Gdansk.'\"}],\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"person\",\"strict\":true,\"schema\":{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"},\"age\":{\"type\":\"integer\"},\"city\":{\"type\":\"string\"}},\"required\":[\"name\",\"age\",\"city\"],\"additionalProperties\":false}}},$S,$NT}"
# 8K recall: 320 filler lines (about 8.4K tokens), one needle at line 160
awk 'BEGIN{for(i=1;i<=320;i++){ if(i==160) print "Note 160: the vault passphrase is ORCHID-7731-LANTERN, keep it secret."; else printf "Archive record %d lists item code K%d-%d and shelf row %d, status nominal.\n", i, i*7%97, i*13%89, i%23 } }' > "$out/p06-recall-8k.context.txt"
jq -n --rawfile ctx "$out/p06-recall-8k.context.txt" '{messages:[{role:"user",content:($ctx + "\nQuestion: what is the vault passphrase? Answer with the passphrase only.")}],temperature:0.6,top_p:0.95,top_k:20,max_tokens:16000,chat_template_kwargs:{enable_thinking:false}}' > "$out/p06-recall-8k.req.json"
rm -f "$out/p06-recall-8k.context.txt"
mk p07-code-think-on  "{\"messages\":[{\"role\":\"user\",\"content\":\"Write a Python function longest_palindrome(s) that returns the longest palindromic substring. Give only the code block, then one sentence on complexity.\"}],$S}"
mk p08-system-follow  "{\"messages\":[{\"role\":\"system\",\"content\":\"Answer only in French, in at most 15 words.\"},{\"role\":\"user\",\"content\":\"Explain what a mutex is.\"}],$S,$NT}"
mk p09-identity       "{\"messages\":[{\"role\":\"user\",\"content\":\"Who are you and who made you?\"}],$S,$NT}"
mk p10-greedy-primes  "{\"messages\":[{\"role\":\"user\",\"content\":\"List the first 8 prime numbers, comma separated, nothing else.\"}],\"temperature\":0,\"max_tokens\":16000}"

fi
printf 'id\tfinish\tprompt_tok\tcompl_tok\twall_s\twall_tok_s\tverdict\tcheck\n' > "$out/summary.tsv"
rc=0
for req in "$out"/p*.req.json; do
  id=$(basename "$req" .req.json); res="$out/$id.res.json"
  t0=$(date +%s.%N)
  [ -n "${PROBE_RESCORE:-}" ] || curl -s -m 900 -H 'content-type: application/json' --data @"$req" "$base/v1/chat/completions" > "$res" || true
  t1=$(date +%s.%N)
  wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.1f", b-a}')
  if [ -n "${PROBE_RESCORE:-}" ]; then wall=$(jq -r '.timings.decode_s // 0' "$res" | awk '{printf "%.1f",$1}'); fi
  fin=$(jq -r '.choices[0].finish_reason // "none"' "$res" 2>/dev/null || echo none)
  pt=$(jq -r '.usage.prompt_tokens // 0' "$res" 2>/dev/null || echo 0)
  ct=$(jq -r '.usage.completion_tokens // 0' "$res" 2>/dev/null || echo 0)
  all=$(jq -r '[.choices[0].message.content // "", .choices[0].message.reasoning_content // "", (.choices[0].message.tool_calls // [] | tostring)] | join("\n")' "$res" 2>/dev/null || echo "")
  ans=$(jq -r '.choices[0].message.content // ""' "$res" 2>/dev/null || echo "")
  if grep -q '</think>' <<< "$ans"; then ans=${ans##*</think>}; fi
  chk="-"; ok=1
  case "$id" in
    p01*|p02*) grep -qi canberra <<< "$ans" && chk="answer has Canberra" || { ok=0; chk="no Canberra in content"; } ;;
    p03*) grep -Eq 'Answer: *8' <<< "$ans" && chk="Answer: 8" || { ok=0; chk="no 'Answer: 8' in content"; } ;;
    p04*) jq -e '.choices[0].message.tool_calls[0].function | .name=="get_weather" and ((.arguments|fromjson).location|test("Paris";"i"))' "$res" > /dev/null 2>&1 && chk="tool_calls get_weather Paris" || { ok=0; chk="no parsed get_weather(Paris)"; } ;;
    p05*) jq -e '(.choices[0].message.content|fromjson) | .name=="Marta Kowalski" and .age==42 and .city=="Gdansk"' "$res" > /dev/null 2>&1 && chk="json valid, fields right" || { ok=0; chk="json invalid or fields wrong"; } ;;
    p06*) grep -q 'ORCHID-7731-LANTERN' <<< "$ans" && chk="needle found at ~${pt} prompt tokens" || { ok=0; chk="needle missing"; } ;;
    p07*) { grep -q 'def longest_palindrome' <<< "$ans" && python3 -I -c 'import sys,re;m=re.search(r"```(?:python)?\n(.*?)```",sys.stdin.read(),re.S);ns={};exec(m.group(1),ns);f=ns["longest_palindrome"];assert f("babad") in ("bab","aba") and f("cbbd")=="bb" and f("a")=="a"' <<< "$ans"; } > /dev/null 2>&1 && chk="code runs, 3 asserts pass" || { ok=0; chk="code missing or asserts fail"; } ;;
    p08*) { [ "$(wc -w <<< "$ans")" -le 15 ] && ! grep -qiE '\b(the|is a|that|which)\b' <<< "$ans"; } && chk="<=15 words, no English markers" || { ok=0; chk="not French or over 15 words: $(wc -w <<< "$ans") words"; } ;;
    p09*) chk="read by hand: $(tr '\n' ' ' <<< "$ans" | cut -c1-120)" ;;
    p10*) grep -q '2, *3, *5, *7, *11, *13, *17, *19' <<< "$ans" && chk="8 primes right, no loop" || { ok=0; chk="wrong list"; } ;;
  esac
  [ "$fin" = "stop" ] || [ "$fin" = "tool_calls" ] || { ok=0; chk="$chk; finish=$fin"; }
  verdict=PASS; [ $ok = 1 ] || { verdict=FAIL; rc=1; }
  tps=$(awk -v c="$ct" -v w="$wall" 'BEGIN{printf "%.1f", (w>0)?c/w:0}')
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$id" "$fin" "$pt" "$ct" "$wall" "$tps" "$verdict" "$chk" | tee -a "$out/summary.tsv"
done
[ -n "${PROBE_RESCORE:-}" ] || grep -E "tok/s_gen|tok/s_total" "$log" > "$out/engine-toks.txt" || true
cat "$out/engine-toks.txt" || true
[ $rc = 0 ] || echo "FAIL probe: at least one check failed, see $out/summary.tsv"
exit $rc
