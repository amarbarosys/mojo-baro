#!/usr/bin/env bash
# usage: gpu-wait run --timeout 900 -- bench/ailab/cast-check.sh OUTDIR LIST.tsv
# Item 2 of briefs/2026-11-07-ailab-tools-and-cast.md: per cast model, one baro-serve, one runner tool request.
# LIST.tsv rows: name<TAB>engine<TAB>pack<TAB>parallel (1 = single-session engine). Each row: start, wait /health,
# read VRAM (sysfs, fullest card, minus the idle reading taken before the start), post BODY (default the runner's
# states-run/s01-r1.json with max_tokens 16000), record HTTP code, seconds, tool_calls names, stop, check VRAM is back.
# Writes OUTDIR/table.tsv. Exit 1 only on a harness failure; a model that fails to load or answer is a row, not an exit.
set -euo pipefail
[ $# -eq 2 ] || { echo "FAIL usage: cast-check.sh OUTDIR LIST.tsv"; exit 1; }
cd "$(dirname "$0")/../.."
out=$1; list=$2; mkdir -p "$out"
server=${SERVER:-serve/target/release/baro-serve}; port=${PORT:-18436}; body=${BODY:-bench/ailab/states-run/s01-r1.json}
[ -x "$server" ] && [ -s "$body" ] || { echo "FAIL setup: missing $server or $body"; exit 1; }
jq '.max_tokens = 16000' "$body" > "$out/body.json"
vram() { cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null | sort -n | tail -1; }
pid=""
stop() { [ -z "$pid" ] && return 0; kill -- -"$pid" 2>/dev/null || :; for _ in $(seq 60); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done; kill -9 -- -"$pid" 2>/dev/null || :; pid=""; }
trap stop EXIT
rows=$list; [ "${ROWS:-1}" = 1 ] || rows=/dev/null  # ROWS=0 / TOGETHER=0 run one phase per job (< 5 min each)
printf 'name\tloads\thealth_s\tvram_mib(load/peak)\thttp\ts_request\tnames\tcompletion_tokens\tnote\n' > "$out/table.tsv"
while IFS=$'\t' read -r name engine pack par; do
  [ -n "$name" ] && [ "${name:0:1}" != "#" ] || continue
  log=$out/$name.serve.log; idle=$(vram); t0=$(date +%s.%N)
  extra=(); [ "$par" -gt 1 ] && extra=(--parallel "$par" --kv-pool 1)
  BARO_SPEC=0 setsid "$server" --engine "$engine" --pack "$pack" --port "$port" --engine-env BARO_TMAX=20480 \
    --engine-env BARO_SPEC=0 "${extra[@]}" > "$log" 2>&1 & pid=$!
  ok=""; for _ in $(seq 480); do
    curl -sf "http://127.0.0.1:$port/health" 2>/dev/null | grep -q '"ok"' && { ok=1; break; }
    kill -0 "$pid" 2>/dev/null || break; sleep 0.5
  done
  hs=$(awk -v a="$t0" -v b="$(date +%s.%N)" 'BEGIN { printf "%.1f", b - a }')
  if [ -z "$ok" ]; then
    printf '%s\tn\t%s\t-\t-\t-\t-\t-\t%s\n' "$name" "$hs" "$(grep -m1 -iE 'error|panic|abort' "$log" | tr '\t' ' ' | cut -c1-160)" | tee -a "$out/table.tsv"
    stop; continue
  fi
  vm=$(( ($(vram) - idle) / 1048576 ))
  curl -s -o "$out/$name.resp.json" -w '%{http_code} %{time_total}\n' -H 'content-type: application/json' \
    --data-binary @"$out/body.json" --max-time 300 "http://127.0.0.1:$port/v1/chat/completions" > "$out/$name.code" & cp=$!
  peak=$(vram); while kill -0 "$cp" 2>/dev/null; do v=$(vram); [ "$v" -gt "$peak" ] && peak=$v; sleep 0.2; done
  wait "$cp" || :; read -r code secs < "$out/$name.code" || { code=000; secs=-; }
  vm="$vm/$(( (peak - idle) / 1048576 ))"
  names=$(jq -r '[.choices[0].message.tool_calls[]?.function.name] | join(",")' "$out/$name.resp.json" 2>/dev/null || echo "")
  ct=$(jq -r '.usage.completion_tokens // "-"' "$out/$name.resp.json" 2>/dev/null || echo "-")
  note=$(jq -r '.choices[0].message.content // .error.message // "" | gsub("[\n\t]"; " ") | .[0:80]' "$out/$name.resp.json" 2>/dev/null || head -c 80 "$out/$name.resp.json")
  printf '%s\ty\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$hs" "$vm" "$code" "$secs" "${names:--}" "$ct" "$note" | tee -a "$out/table.tsv"
  stop
done < "$rows"
# Together: every row's server up at once (ports PORT+1..), total VRAM, one request each, all in flight together.
if [ "${TOGETHER:-1}" = 1 ]; then
  idle=$(vram); pids=(); names_=(); i=0
  while IFS=$'\t' read -r name engine pack par; do
    [ -n "$name" ] && [ "${name:0:1}" != "#" ] || continue
    i=$((i + 1)); extra=(); [ "$par" -gt 1 ] && extra=(--parallel "$par" --kv-pool 1)
    BARO_SPEC=0 setsid "$server" --engine "$engine" --pack "$pack" --port $((port + i)) --engine-env BARO_TMAX=20480 \
      --engine-env BARO_SPEC=0 "${extra[@]}" > "$out/together-$name.serve.log" 2>&1 & pids+=($!); names_+=("$name")
  done < "$list"
  up=0; for _ in $(seq 600); do
    up=0; for j in $(seq ${#pids[@]}); do curl -sf "http://127.0.0.1:$((port + j))/health" 2>/dev/null | grep -q '"ok"' && up=$((up + 1)); done
    [ "$up" = ${#pids[@]} ] && break
    dead=0; for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null || dead=$((dead + 1)); done
    [ $((up + dead)) = ${#pids[@]} ] && break; sleep 0.5
  done
  for n in "${names_[@]}"; do grep -m1 -oE "hipErrorOutOfMemory|engine exited before its ready line" "$out/together-$n.serve.log" | head -1 | sed "s/^/together $n start: /"; done | tee "$out/together-start.txt"
  load=$(( ($(vram) - idle) / 1048576 ))
  cps=(); for j in $(seq ${#pids[@]}); do
    curl -s -o "$out/together-${names_[j-1]}.resp.json" -w '%{http_code} %{time_total}\n' -H 'content-type: application/json' \
      --data-binary @"$out/body.json" --max-time 300 "http://127.0.0.1:$((port + j))/v1/chat/completions" > "$out/together-${names_[j-1]}.code" & cps+=($!)
  done
  peak=$(vram); while kill -0 "${cps[@]}" 2>/dev/null; do v=$(vram); [ "$v" -gt "$peak" ] && peak=$v; sleep 0.2; done
  for c in "${cps[@]}"; do wait "$c" || :; done
  { echo "together: $up/${#pids[@]} healthy, vram over idle load ${load} MiB peak $(( (peak - idle) / 1048576 )) MiB, idle $(( idle / 1048576 )) MiB, card $(( $(cat /sys/class/drm/card*/device/mem_info_vram_total | sort -n | tail -1) / 1048576 )) MiB"
    for n in "${names_[@]}"; do echo "together $n: $(cat "$out/together-$n.code" 2>/dev/null) names=$(jq -r '[.choices[0].message.tool_calls[]?.function.name] | join(",")' "$out/together-$n.resp.json" 2>/dev/null)"; done
    grep -h "GTT\|gtt_used" "$out"/together-*.serve.log | tail -n 4
  } | tee "$out/together.txt"
  for p in "${pids[@]}"; do kill -- -"$p" 2>/dev/null || :; done; sleep 3; for p in "${pids[@]}"; do kill -9 -- -"$p" 2>/dev/null || :; done
fi
echo "PASS cast-check: $out/table.tsv"
