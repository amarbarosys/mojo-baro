#!/usr/bin/env bash
# Teacher-forced agreement of a candidate arm against a reference arm on the REAL requests of a
# vero replay (CLAUDE.md: identity past ~256 ids is agreement, never greedy equality).
# Input: the engine wire of one reference replay, captured by bench/engine-tee.sh: WIRE_IN = the
# request lines baro-serve wrote, WIRE_OUT = the engine's stdout (its {"id":..,"tok":..} lines are
# the reference ids). Every request is replayed with n = len(ref), "force" = ref, no stop list, in
# one serve-mode process per arm: the control arm (reference knobs) must agree 100% on every
# request or the instrument is broken; the candidate passes at mean agreement >= MIN_MEAN percent.
#   gpu-wait run --vram 23 --timeout 3600 -- env KNOBS_C="..." KNOBS_X="..." \
#       bench/vero-force-agree.sh ENGINE PACK WIRE_IN WIRE_OUT OUT
# env: MIN_MEAN (default 99), LIMIT (first N requests only, default all).
# OUT/forced.jsonl, OUT/<control|candidate>.log, OUT/summary.tsv (req, n, control, candidate).
set -euo pipefail
export LC_ALL=C
eng=$(realpath "${1:?usage: vero-force-agree.sh ENGINE PACK WIRE_IN WIRE_OUT OUT}") pack=$(realpath "${2:?}")
win=$(realpath "${3:?}") wout=$(realpath "${4:?}") out=$(realpath -m "${5:?}")
read -ra kc <<<"${KNOBS_C:?KNOBS_C (reference arm knobs)}"
read -ra kx <<<"${KNOBS_X:?KNOBS_X (candidate arm knobs)}"
minmean=${MIN_MEAN:-99} limit=${LIMIT:-0}
# MAX's pool uncapped (unset or 0 in a gpu-wait job) segfaults the engine (bench/engine-ab.sh).
p=${MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT:-0}; [[ $p != 0 ]] || p=2
export MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=$p
mkdir -p "$out"
{ echo "gate=vero-force-agree min_mean=$minmean limit=$limit"
  echo "engine $eng sha256 $(sha256sum "$eng" | cut -c1-16) pack $pack"
  echo "wire_in $win sha256 $(sha256sum "$win" | cut -c1-16) wire_out $wout sha256 $(sha256sum "$wout" | cut -c1-16)"
  echo "control ${kc[*]}"; echo "candidate ${kx[*]}"; } > "$out/arm.txt"
grep -E '^\{"id":[0-9]+,"tok":' "$wout" | jq -cs 'group_by(.id) | map({key: (.[0].id | tostring), value: map(.tok)}) | from_entries' > "$out/ref.json"
jq -c --slurpfile ref "$out/ref.json" 'select(.prompt != null) | (.id | tostring) as $k | select($ref[0][$k] != null)
	| .n = ($ref[0][$k] | length) | .force = $ref[0][$k] | del(.stop)' "$win" > "$out/forced.jsonl"
((limit > 0)) && { head -n "$limit" "$out/forced.jsonl" > "$out/f.tmp"; mv "$out/f.tmp" "$out/forced.jsonl"; }
nreq=$(wc -l < "$out/forced.jsonl")
((nreq > 0)) || { echo "FAIL input: no request of $win has reference tokens in $wout"; exit 1; }
echo "requests $nreq, forced ids $(jq -s 'map(.n) | add' "$out/forced.jsonl")" >> "$out/arm.txt"
for a in control candidate; do
	if [[ $a == control ]]; then k=("${kc[@]}"); else k=("${kx[@]}"); fi
	env BARO_SERVE=1 BARO_PACK="$pack" "${k[@]}" "$eng" < "$out/forced.jsonl" > "$out/$a.log" 2>&1 \
		|| { echo "FAIL $a: engine exited non-zero, see $out/$a.log"; exit 1; }
	! grep -qE 'fail word: [1-9]|NOT-RESIDENT|Unhandled exception' "$out/$a.log" || { echo "FAIL $a: fail word in $out/$a.log"; exit 1; }
	for kv in "${k[@]}"; do
		[[ $kv == BARO_PF_GEMM=* ]] || continue
		grep -q "^BARO_PF_GEMM: ${kv#*=}\$" "$out/$a.log" || { echo "FAIL readback: $kv not echoed in $out/$a.log"; exit 1; }
	done
	grep -oE 'forced agreement: [0-9]+ / [0-9]+' "$out/$a.log" | awk '{print $3 "\t" $5}' > "$out/$a.tsv"
	[[ $(wc -l < "$out/$a.tsv") == "$nreq" ]] || { echo "FAIL $a VOID: $(wc -l < "$out/$a.tsv") agreement lines for $nreq requests"; exit 1; }
done
paste <(jq -r '[.id, .n] | @tsv' "$out/forced.jsonl") "$out/control.tsv" "$out/candidate.tsv" \
	| awk -F'\t' 'BEGIN { print "req\tn\tcontrol\tof\tcandidate\tof" } { print }' > "$out/summary.tsv"
awk -F'\t' -v bar="$minmean" 'NR > 1 { r++; if ($3 != $4 || $4 != $2) cb++; p = 100 * $5 / $6; s += p; if (r == 1 || p < lo) lo = p; ca += $5; cn += $6 }
	END { if (cb) { printf "FAIL control below 100%% on %d of %d requests: the instrument is broken, not the candidate\n", cb, r; exit 1 }
		m = s / r; v = sprintf("mean %.2f%% over %d requests (bar %s%%), min %.2f%%, pooled %d / %d ids", m, r, bar, lo, ca, cn)
		if (m < bar) { print "FAIL agreement " v; exit 1 }; print "PASS agreement " v }' "$out/summary.tsv" | tee -a "$out/arm.txt"
