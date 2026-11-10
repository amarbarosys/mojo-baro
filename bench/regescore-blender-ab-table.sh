#!/usr/bin/env bash
# Per-task table from regescore-blender-ab.sh output dirs (OUT/{baro,llama}/<arm>/*/summary.json):
# one row per task and engine, then per engine/arm totals and medians. baro's
# prefill_s here = engine prefill_s + restore_s (the engine starts its prefill
# clock after the checkpoint restore).
#   bench/regescore-blender-ab-table.sh RUN_DIR > RUN_DIR/TABLE.tsv
# Last block: generated-token totals per arm, each engine against llama, GAP when they
# differ by more than 1% (run 9's +2.7% tokens made model time incomparable).
# TASK_RE='^t(0[1-9]|1[0-6])_' keeps the 16 tasks run 1 had (t17-t19 were added 2026-09-24 03:28).
set -euo pipefail
run=${1:?usage: regescore-blender-ab-table.sh RUN_DIR}
shopt -s nullglob
files=("$run"/*/*/*/summary.json)
((${#files[@]})) || { echo "FAIL table: no summary.json under $run"; exit 1; }
printf 'engine\tarm\ttask\tpass1\twall_s\trequest_s\tprompt\treused\tprefilled\tprefill_s\tgen\tdecode_s\ttok_s\n'
rows=$(for f in "${files[@]}"; do
  rel=${f#"$run"/}; eng=${rel%%/*}; rest=${rel#*/}; arm=${rest%%/*}
  jq -r --arg e "$eng" --arg a "$arm" --arg re "${TASK_RE:-.}" '.tasks[] | select(.id | test($re)) | . as $t | ($t.requests[0] // {}) as $r
    | ($r.timings // {}) as $m | ($r.usage // {}) as $u
    | if ($e | startswith("baro")) then
        [($m.cached // 0), ($m.prefill_rows // null), (($m.prefill_s // 0) + ($m.restore_s // 0)), $u.completion_tokens, $m.decode_s, $m.tok_s_gen]
      else
        [($m.cache_n // 0), $m.prompt_n, (($m.prompt_ms // 0) / 1000), $m.predicted_n, (($m.predicted_ms // 0) / 1000), $m.predicted_per_second]
      end
    | [$e, $a, $t.id, $t.pass1, $t.seconds, $r.request_s, $u.prompt_tokens] + . | @tsv' "$f"
done)
printf '%s\n' "$rows"
echo
printf 'engine\tarm\tpass1\ttasks\twall_s_sum\trequest_s_sum\tprefill_s_sum\tdecode_s_sum\tfirst_prefill_s\tmedian_prefill_s_2on\tmedian_tok_s\tmedian_reused_2on\tgen_sum\n'
printf '%s\n' "$rows" | sort -t$'\t' -k1,1 -k2,2 -k3,3 | awk -F'\t' '
  function med(s,   n, a, i, j, t) { n = split(s, a, " "); for (i = 1; i <= n; i++) for (j = i + 1; j <= n; j++) if (a[j] + 0 < a[i] + 0) { t = a[i]; a[i] = a[j]; a[j] = t }
    return n ? (n % 2 ? a[(n + 1) / 2] : (a[n / 2] + a[n / 2 + 1]) / 2) : "" }
  { k = $1 "\t" $2; if (!(k in n)) { order[++m] = k; first[k] = $10 } else { pf[k] = pf[k] " " $10; ru[k] = ru[k] " " $8 }
    n[k]++; p[k] += $4; w[k] += $5; rq[k] += $6; ps[k] += $10; ds[k] += $12; ts[k] = ts[k] " " $13; g[k] += $11; ga[$2] = 1; ge[$1] = 1 }
  END { for (i = 1; i <= m; i++) { k = order[i]
    printf "%s\t%d\t%d\t%.1f\t%.1f\t%.2f\t%.2f\t%.2f\t%.3f\t%.1f\t%s\t%d\n", k, p[k], n[k], w[k], rq[k], ps[k], ds[k], first[k], med(pf[k]), med(ts[k]), med(ru[k]), g[k] }
    if (!("llama" in ge)) exit
    printf "\ntokens\tarm\tllama_gen\tengine\tgen\tgap_pct\tflag\n"
    for (a in ga) for (e in ge) if (e != "llama" && ((e "\t" a) in n) && (("llama\t" a) in n)) {
      l = g["llama\t" a]; b = g[e "\t" a]; mx = l > b ? l : b; gap = mx ? 100 * (b - l) / mx : 0
      printf "tokens\t%s\t%d\t%s\t%d\t%.2f\t%s\n", a, l, e, b, gap, (gap > 1 || gap < -1) ? "GAP" : "ok" } }'
