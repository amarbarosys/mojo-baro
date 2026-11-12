#!/usr/bin/env bash
# GPU idle by bucket for one rocprofv3 trace of q4x decode (bench/q4x-rows-trace.sh dir/A): the gap between consecutive kernel
# dispatches, bucketed by the kernel that precedes it; gaps after the router are split by how many SDMA copies fall inside.
# usage: bench/q4x-gaps.sh TRACE_DIR GEN_MS_PER_TOKEN N_TOKENS      (window = the last GEN_MS * N_TOKENS ms of the trace)
set -euo pipefail
D=${1:?usage: q4x-gaps.sh TRACE_DIR GEN_MS N_TOKENS}; G=${2:?}; N=${3:?}
[ -s "$D/trace_kernel_trace.csv" ] && [ -s "$D/trace_memory_copy_trace.csv" ] || { echo "FAIL q4x-gaps: no trace in $D"; exit 1; }
gawk -v g="$G" -v n="$N" 'BEGIN { FPAT = "([^,]*)|(\"[^\"]*\")" }
FNR == 1 { next }
FILENAME ~ /memory_copy/ { cs[++nc] = $7 + 0; ce[nc] = $8 + 0; next }
{ k++; s[k] = $10 + 0; e[k] = $11 + 0; nm[k] = $8; if (e[k] > mx) mx = e[k] }
END { t0 = mx - g * n * 1e6; PROCINFO["sorted_in"] = "@val_num_asc"; m = 0; for (i in s) if (s[i] >= t0) ord[++m] = i
  delete PROCINFO["sorted_in"]; pe = 0
  for (j = 1; j <= m; j++) { i = ord[j]
    if (pe > 0) { gap = s[i] - pe; if (gap < 0) gap = 0; b = pnm; gsub(/"/, "", b); sub(/^.*amar_q4x_/, "", b); sub(/_.*/, "", b)
      if (b == "router" && gap > 50000) { b = "router>50us"; c = 0; for (x = 1; x <= nc; x++) if (cs[x] >= pe - 2000 && cs[x] <= s[i]) c++; hc[c]++; hg[c] += gap }
      else if (b == "router") b = "router<=50us"
      bg[b] += gap; bn[b]++; tot += gap }
    if (e[i] > pe) pe = e[i]; pnm = nm[i] }
  printf "window %.0f ms (%d tokens), %d kernels, GPU idle %.0f ms = %.2f ms/token (%.0f%% of wall)\n", g * n, n, m, tot / 1e6, tot / 1e6 / n, 100 * tot / 1e6 / (g * n)
  PROCINFO["sorted_in"] = "@val_num_desc"; r = 0
  for (b in bg) if (++r <= 8 || b ~ /router/) printf "after %-13s gaps %6d  total %8.1f ms  %6.2f ms/token  mean %6.1f us\n", b, bn[b], bg[b] / 1e6, bg[b] / 1e6 / n, bg[b] / bn[b] / 1e3
  PROCINFO["sorted_in"] = "@ind_num_asc"; for (c in hc) printf "router>50us gaps with %2d SDMA copies inside: %5d, mean gap %7.1f us\n", c, hc[c], hg[c] / hc[c] / 1e3 }' "$D/trace_memory_copy_trace.csv" "$D/trace_kernel_trace.csv"
