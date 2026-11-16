#!/usr/bin/env bash
# usage: bench/trace-pair.sh ENGINE PACK OUT N_SHORT N_LONG [K=V ...] : two rocprofv3 kernel traces of one resident
# request that differ only in generated length, for tools/q4x-trace-gaps.py (per-token busy / gap split). The
# engine may abort at exit under rocprofv3 (signal 6 after output generation); the CSV is checked, not the exit code.
set -euo pipefail
cd "$(dirname "$0")/.."
e=$(realpath "$1"); pack=$(realpath "$2"); out=$(realpath -m "$3"); ns=$4; nl=$5; shift 5; mkdir -p "$out"
for n in "$ns" "$nl"; do
  printf '{"id":1,"prompt":[%s],"n":%d,"spec":true}\n' "$(tr -s ' \n' ',,' < bench/mtp-prompts/p01-water.tokens | sed 's/,$//')" "$n" > "$out/req-$n.jsonl"
  rm -rf "$out/t$n"
  env BARO_PACK="$pack" BARO_SERVE=1 BARO_TMAX=16384 "$@" /opt/rocm/bin/rocprofv3 --kernel-trace --output-format csv -d "$out/t$n" -- "$e" < "$out/req-$n.jsonl" > "$out/t$n.log" 2>&1 || true
  ls "$out/t$n"/*/*kernel_trace.csv > /dev/null 2>&1 || { echo "FAIL trace $n: no kernel_trace.csv, see $out/t$n.log"; exit 1; }
  grep -q '"done": *true' "$out/t$n.log" || { echo "FAIL trace $n: request not done, see $out/t$n.log"; exit 1; }
  mv "$out/t$n"/*/*kernel_trace.csv "$out/t$n/"
done
python3 tools/q4x-trace-gaps.py "$out/t$ns" "$ns" "$out/t$nl" "$nl" | tee "$out/gaps.txt"
