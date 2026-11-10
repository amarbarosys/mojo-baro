#!/usr/bin/env bash
# Kernel time of batched prefill rows at depth: the standalone engine prefills two prompts that
# share a prefix (SHORT, LONG token files) under rocprofv3 --kernel-trace; the per-kernel
# difference LONG - SHORT is what the rows SHORT..LONG cost at that context depth. MoE prefill
# has no BARO_PROFILE split (moe_prefill_forward prints chunk totals only), hence the trace.
#   gpu-wait run --vram 23 --timeout 900 -- env KNOBS="BARO_TMAX=26624 ..." bench/prefill-trace.sh ENGINE PACK SHORT LONG OUT
# OUT/<short|long>.log (engine output: knob lines, GENERATED ids), OUT/<short|long>.tsv (kernel,
# calls, us), OUT/diff.tsv (kernel, us LONG - SHORT, share), sorted by time. Compare two engines'
# GENERATED lines for parity (rocprofv3 mangles Mojo names: grouped by the prefix before _p<digit>).
set -euo pipefail
export LC_ALL=C
eng=$(realpath "${1:?usage: prefill-trace.sh ENGINE PACK SHORT LONG OUT}") pack=$(realpath "${2:?}")
short=$(realpath "${3:?}") long=$(realpath "${4:?}") out=$(realpath -m "${5:?}")
read -ra knobs <<<"${KNOBS:-}"
rp=${ROCPROF:-/opt/rocm/bin/rocprofv3}   # absolute: a gpu-wait job does not carry the shell's PATH
[[ -x $rp ]] || { echo "FAIL setup: no rocprofv3 at $rp"; exit 1; }
mkdir -p "$out"
echo "engine $eng sha256 $(sha256sum "$eng" | cut -c1-16) pack $pack knobs ${knobs[*]}" > "$out/arm.txt"
for a in short long; do
	tf=$short; [[ $a == long ]] && tf=$long
	rm -rf "$out/tr"; mkdir -p "$out/tr"
	"$rp" --kernel-trace --output-format csv -d "$out/tr" -o tr -- \
		env "${knobs[@]}" BARO_PACK="$pack" BARO_PROMPT="$tf" "$eng" > "$out/$a.log" 2>&1 \
		|| { echo "FAIL $a: engine exited non-zero, see $out/$a.log"; exit 1; }
	grep -q '^GENERATED' "$out/$a.log" || { echo "FAIL $a: no GENERATED line in $out/$a.log"; exit 1; }
	! grep -qE "fail word: [1-9]" "$out/$a.log" || { echo "FAIL $a: mega fail word set"; exit 1; }
	csv=$(find "$out/tr" -name '*kernel_trace.csv' | head -1)
	[[ -n $csv ]] || { echo "FAIL $a: no kernel trace"; exit 1; }
	python3 -I - "$csv" > "$out/$a.tsv" <<'EOF'
import csv, re, sys
t = {}
for r in csv.DictReader(open(sys.argv[1])):
    n = re.sub(r'_p\d.*|\(.*', '', r['Kernel_Name'])[:60]
    c, us = t.get(n, (0, 0.0))
    t[n] = (c + 1, us + (int(r['End_Timestamp']) - int(r['Start_Timestamp'])) / 1e3)
for n, (c, us) in sorted(t.items(), key=lambda x: -x[1][1]):
    print(f"{n}\t{c}\t{us:.0f}")
EOF
	echo "$a: $(tr ' ' '\n' < "$tf" | grep -c .) tokens, $(awk -F'\t' '{s += $3} END {printf "%.3f s kernel time", s / 1e6}' "$out/$a.tsv"), $(grep -m1 -oE 'prefill_s: [0-9.]+' "$out/$a.log")" >> "$out/arm.txt"
done
rm -rf "$out/tr"
awk -F'\t' 'NR == FNR { s[$1] = $3; next } { d[$1] = $3 - s[$1]; tot += $3 - s[$1] } END {
	for (k in d) printf "%s\t%.0f\t%.3f\n", k, d[k], d[k] / tot }' "$out/short.tsv" "$out/long.tsv" | sort -t$'\t' -k2,2nr > "$out/diff.tsv"
cat "$out/arm.txt"; head -12 "$out/diff.tsv"
