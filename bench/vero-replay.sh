#!/usr/bin/env bash
# Replays newera's vero author loop (tools/vero_author.sh) from a recorded run: every
# assistant turn of every RUN/<n>.chat.json becomes one request whose history is the
# recorded one (teacher-forced), so both engines see byte-identical prompts. Greedy,
# thinking off, the loop's own tools and max_tokens; PAR conversations in flight, each
# conversation's turns in order, as the live loop sends them.
#
#   bench/vero-replay.sh RUN URL OUT [PAR]  -> OUT/req.tsv (one row per request), OUT/resp/
#
# req.tsv: conv turn wall_s prompt cached prefill_s completion decode_s finish out_sha.
# cached / prefill_s / decode_s come from the engine (baro: usage.baro + timings;
# llama-server: timings.cache_n, prompt_ms, predicted_ms). out_sha hashes the generated token
# ids when the engine returns them (baro), else content + tool_calls (ids dropped): equal sha =
# equal reply, the identity gate's input.
set -euo pipefail
export LC_ALL=C
run=${1:?usage: vero-replay.sh RUN URL OUT [PAR]} url=${2:?} out=$(realpath -m "${3:?}") par=${4:-2}
mkdir -p "$out/resp"
curl -sf --max-time 5 "$url/health" >/dev/null || { echo "FAIL server: nothing on $url"; exit 1; }
tools=$(jq -c . "$(dirname "$0")/vero-tools.json")   # the loop's two tools, also baro-serve --warm-tools

conv() {  # $1 = path to <n>.chat.json
	local f=$1 n i k=0 body t resp
	n=$(basename "$f" .chat.json)
	for i in $(jq -r 'to_entries[] | select(.value.role == "assistant") | .key' "$f"); do
		k=$((k + 1))
		body=$(jq --argjson i "$i" --argjson tl "$tools" '{messages: .[:$i], tools: $tl, parallel_tool_calls: false,
			max_tokens: 3000, temperature: 0, chat_template_kwargs: {enable_thinking: false}}' "$f")
		t=$EPOCHREALTIME
		resp=$(curl -sf --max-time 900 "$url/v1/chat/completions" -H 'Content-Type: application/json' -d "$body") \
			|| { echo "FAIL request: conv $n turn $k" >&2; return 1; }
		t=$(awk -v a="$t" -v b="$EPOCHREALTIME" 'BEGIN { printf "%.3f", b - a }')
		echo "$resp" > "$out/resp/$n.$k.json"
		jq -r --arg n "$n" --arg k "$k" --arg t "$t" '[$n, $k, $t, .usage.prompt_tokens,
			(.usage.baro.cached_tokens // .timings.cache_n // ""),
			(.timings.prefill_s // ((.timings.prompt_ms // 0) / 1000)),
			.usage.completion_tokens,
			(.timings.decode_s // ((.timings.predicted_ms // 0) / 1000)),
			.choices[0].finish_reason,
			((.choices[0].tokens // (.choices[0].message | {content, tc: [(.tool_calls // [])[] | .function]})) | tojson | @base64)] | @tsv' <<<"$resp" \
			| awk -F'\t' -v OFS='\t' '{ cmd = "printf %s " $10 " | base64 -d | sha256sum"; cmd | getline s; close(cmd); $10 = substr(s, 1, 16); print }'
	done
}
export -f conv; export out url tools
printf 'conv\tturn\twall_s\tprompt\tcached\tprefill_s\tcompletion\tdecode_s\tfinish\tout_sha\n' > "$out/req.tsv"
t0=$EPOCHREALTIME
ls "$run"/*.chat.json | sort -V | xargs -P "$par" -I{} bash -c 'conv "$1"' _ {} >> "$out/req.tsv"
awk -v a="$t0" -v b="$EPOCHREALTIME" 'BEGIN { printf "wall_s %.1f\n", b - a }' > "$out/wall.txt"
awk -F'\t' 'NR > 1 { w += $3; p += $6; d += $7; ds += $8; r++ } END {
	printf "requests %d  sum wall %.1f s  prefill %.1f s  decode %.1f s  generated %d\n", r, w, p, ds, d }' "$out/req.tsv" | tee -a "$out/wall.txt"
