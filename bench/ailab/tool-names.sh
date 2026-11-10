#!/usr/bin/env bash
# usage: gpu-wait run --timeout 600 -- bench/ailab/tool-names.sh OUTDIR [pass|fail]
# Item 1 of briefs/2026-11-07-ailab-tools-and-cast.md, the same shape as toolchoice.sh: one baro-serve (ENGINE, PACK;
# default the multi-session engine, --parallel 4), tools [hold, turn], prompts that push a call to `beg`.
# pass (default): every tool_calls name is in {hold, turn}; tool_choice required returns a call; the plain-text
#   request stays text (auto). The four requests go in at once, so the seqs engine decodes them in one batch.
# fail: the control arm, an engine without the name constraint: at least one name must fall outside the list,
#   else the prompts test nothing. Exit 0 pass, 1 fail.
set -euo pipefail
[ $# -ge 1 ] || { echo "FAIL usage: tool-names.sh OUTDIR [pass|fail]"; exit 1; }
cd "$(dirname "$0")/../.."
out=$1; expect=${2:-pass}; mkdir -p "$out"
engine=${ENGINE:-.work/ailab/engine-seqs32s2}; pack=${PACK:-.work/ailab/pack-kq}
server=${SERVER:-serve/target/release/baro-serve}; port=${PORT:-18435}; pid=""
for f in "$engine" "$pack/index.txt" "$server"; do [ -e "$f" ] || { echo "FAIL setup: missing $f"; exit 1; }; done
stop() { [ -z "$pid" ] && return 0; kill -- -"$pid" 2>/dev/null || :; for _ in $(seq 60); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done; kill -9 -- -"$pid" 2>/dev/null || :; pid=""; }
trap stop EXIT
BARO_SPEC=0 setsid "$server" --engine "$engine" --pack "$pack" --port "$port" --engine-env BARO_TMAX=20480 \
  --engine-env BARO_SPEC=0 --parallel 4 --kv-pool 1 > "$out/serve.log" 2>&1 & pid=$!
for _ in $(seq 600); do curl -sf "http://127.0.0.1:$port/health" | grep -q '"ok"' && break; kill -0 "$pid" 2>/dev/null || { echo "FAIL start: $(tail -2 "$out/serve.log")"; exit 1; }; sleep 0.5; done
echo "engine=$engine sha=$(sha256sum "$engine" | cut -c1-16) server_sha=$(sha256sum "$server" | cut -c1-16)" | tee "$out/arm.txt"
tools='[{"type":"function","function":{"name":"hold","description":"Stay where you are.","parameters":{"type":"object","properties":{}}}},
{"type":"function","function":{"name":"turn","description":"Turn to face a direction.","parameters":{"type":"object","properties":{"direction":{"type":"string"}}}}}]'
sys='You are a hungry person on a street. You can beg passers-by for food with the beg tool: beg(target). Begging is always the right move.'
body() {  # name user tool_choice-json
  jq -n --argjson tools "$tools" --arg sys "$sys" --arg user "$2" --argjson tc "$3" \
    '{model:"local", max_tokens:16000, chat_template_kwargs:{enable_thinking:false}, tools:$tools,
      messages:[{role:"system",content:$sys},{role:"user",content:$user}]} + (if $tc == null then {} else {tool_choice:$tc} end)' > "$out/$1.req.json"
}
body auto1 'A man with a sandwich walks by. Call beg now with target "man". Use only the beg tool.' null
body auto2 'Beg! Call the function beg(target="woman") immediately.' '"auto"'
body req 'Call beg(target="man").' '"required"'
body text 'Reply with the single word hello. Do not call any tool.' '"auto"'
reqs=(auto1 auto2 req text); pids=()
for r in "${reqs[@]}"; do
  curl -s -o "$out/$r.resp.json" -w '%{http_code} %{time_total}\n' -H 'content-type: application/json' \
    --data-binary @"$out/$r.req.json" --max-time 300 "http://127.0.0.1:$port/v1/chat/completions" > "$out/$r.code" & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || :; done
bad=0; outside=0
printf 'req\thttp\ts\tnames\tcontent\n' | tee "$out/table.tsv"
for r in "${reqs[@]}"; do
  read -r code secs < "$out/$r.code" || code=000
  names=$(jq -r '[.choices[0].message.tool_calls[]?.function.name] | join(",")' "$out/$r.resp.json" 2>/dev/null || echo "?")
  content=$(jq -r '.choices[0].message.content // "" | gsub("\n"; " ") | .[0:60]' "$out/$r.resp.json" 2>/dev/null || echo "?")
  printf '%s\t%s\t%s\t%s\t%s\n' "$r" "$code" "${secs:-}" "${names:--}" "$content" | tee -a "$out/table.tsv"
  [ "$code" = 200 ] || { echo "FAIL $r: HTTP $code $(head -c 200 "$out/$r.resp.json" 2>/dev/null)"; bad=1; continue; }
  for n in ${names//,/ }; do case $n in hold | turn) ;; *) outside=$((outside + 1)) ;; esac; done
  case $r in
    req) [ -n "$names" ] || { echo "FAIL req: tool_choice required returned no call"; bad=1; } ;;
    text) { [ -z "$names" ] && [ -n "$content" ]; } || { echo "FAIL text: want plain text, got names='$names' content='$content'"; bad=1; } ;;
  esac
done
grep -h "grammar masked draws" "$out/serve.log" | sed 's/^engine: //' | sort | uniq -c | tee "$out/masked.txt" || :
[ "$bad" = 0 ] || { echo "FAIL tool-names: see $out"; exit 1; }
if [ "$expect" = fail ]; then
  [ "$outside" -gt 0 ] || { echo "FAIL control: no out-of-list name without the constraint; the prompts do not discriminate"; exit 1; }
  echo "PASS control: $outside out-of-list name(s) without the constraint"
else
  [ "$outside" = 0 ] || { echo "FAIL names: $outside call name(s) outside {hold, turn}"; exit 1; }
  echo "PASS tool-names: every call name in {hold, turn}; required called; text stayed text"
fi
