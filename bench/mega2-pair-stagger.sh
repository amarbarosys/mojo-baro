#!/usr/bin/env bash
# MEGA2 refill falsifier (exchange/mega2-report.md, job 2a attempt 2): the SEQ_CAP=2 engine with
# BARO_SEQS=2 gets request 1 (short prompt, N ids) and request 2 (long prompt, N+200 ids) at once,
# then request 3 (the short prompt again) only after request 1's done line: slot 0 is refilled from
# request 1's checkpoints while slot 1 is still decoding, the path that faulted on 2026-10-17.
# PASS: three done lines, the engine exits 0, fail word 0 on all three, request 3's ids == request
# 1's (same prompt, greedy, both served in pair steps), read-backs present.
#   gpu-wait run --vram 23 --timeout 900 -- env N=64 bench/mega2-pair-stagger.sh ENGINE2 OUT
# SWAP=1: the long prompt is request 1 (slot 0, N+200 ids) and the short one request 2 (slot 1, N),
# so request 3 refills SLOT 1 with a 3-row tail while slot 0 decodes (the job-2a hang path).
# T (1200): seconds to wait for done lines before the engine is killed and the run FAILs.
# TAIL=k: request 3's prompt is the short prompt plus the first k ids request 1 (or 2 under SWAP)
# generated, so it restores at that request's end checkpoint and replays k rows on the per-token
# launch path AFTER pair steps ran (the job-2a hang path); its ids must equal the short request's
# ids from index k on.
set -euo pipefail
export LC_ALL=C
eng=$(realpath "${1:?usage: mega2-pair-stagger.sh ENGINE2 OUT}") out=$(realpath -m "${2:?}")
pack=${PACK:-$HOME/Projects/mojo/mojo-baro/.work/moe-ngram/pack}
short=${SHORT:-$HOME/Projects/mojo/mojo-baro/.work/l5x/p13951.tokens}
long=${LONG:-$HOME/Projects/mojo/mojo-baro/.work/l5x/p16146.tokens}
n=${N:-64}
mkdir -p "$out"
export MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=${MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT:-2}
echo "engine $eng sha256 $(sha256sum "$eng" | cut -c1-16) n $n" > "$out/arm.txt"
req() { printf '{"id":%d,"prompt":[%s],"n":%d}\n' "$1" "$(tr -s ' \n' ',' < "$2" | sed 's/^,//; s/,$//')" "$3"; }
log=$out/P.log fifo=$out/P.in
rm -f "$fifo"; mkfifo "$fifo"
env BARO_SERVE=1 BARO_PACK="$pack" BARO_TMAX=26624 BARO_MEGA=1 BARO_PREFILL=1 BARO_SPEC=0 BARO_CKPT=32 BARO_KV_SHARE=2 BARO_PF_GEMM=exact BARO_SEQS=2 ${KNOBS:-} "$eng" < "$fifo" > "$log" 2>&1 &
pid=$!
exec 4>"$fifo"
if [[ ${SWAP:-0} == 1 ]]; then req 1 "$long" $((n + 200)) >&4; req 2 "$short" "$n" >&4; else req 1 "$short" "$n" >&4; req 2 "$long" $((n + 200)) >&4; fi
sid=1; [[ ${SWAP:-0} == 1 ]] && sid=2   # the short-prompt request whose ids request 3 must repeat
echo "swap ${SWAP:-0}: short-prompt request is $sid" >> "$out/arm.txt"
wait_done() {  # wait_done COUNT
	local t=0 nd
	while nd=$({ grep -c '"done":true' "$log" 2>/dev/null || :; }); (( ${nd:-0} < $1 )); do
		kill -0 $pid 2>/dev/null || { echo "FAIL engine exited before $1 done lines, see $log"; exec 4>&-; return 1; }
		sleep 1; t=$((t + 1)); (( t < ${T:-1200} )) || { echo "FAIL ${T:-1200} s without $1 done lines (hang?), engine killed, gpu busy $(cat /sys/class/drm/card*/device/gpu_busy_percent 2>/dev/null | sort -n | tail -1)%"; kill -9 $pid; exec 4>&-; return 1; }
	done
}
wait_done 1
tail_k=${TAIL:-0}
if (( tail_k > 0 )); then
	ids_sid() { { grep "\"id\":$sid,\"tok\":" "$log" || :; } | sed 's/.*"tok":\([0-9-]*\).*/\1/'; }
	tail_ids=$(ids_sid | head -n "$tail_k" | tr '\n' ' ')
	printf '{"id":3,"prompt":[%s,%s],"n":%d}\n' "$(tr -s ' \n' ',' < "$short" | sed 's/^,//; s/,$//')" "$(echo "$tail_ids" | tr -s ' ' ',' | sed 's/,$//')" $((n - tail_k)) >&4
	echo "request 3 = short prompt + $tail_k generated ids ($tail_ids)" >> "$out/arm.txt"
else
	req 3 "$short" "$n" >&4
fi
echo "request 3 sent after done line 1 at $(date +%T)" >> "$out/arm.txt"
wait_done 3
exec 4>&-
wait $pid && echo "engine exit 0" >> "$out/arm.txt" || { echo "FAIL engine exit $? (fault?), see $log"; exit 1; }
for w in "BARO_SEQS: 2" '"tmax":26624' "BARO_PF_GEMM: exact" "BARO_KV_SHARE: pages [1-9]"; do
	grep -q "$w" "$log" || { echo "FAIL read-back '$w' missing in $log"; exit 1; }
done
(( $(grep -c "mega fail word: 0" "$log") == 3 )) || { echo "FAIL fail word not 0 on three requests: $(grep -c 'mega fail word: [1-9]' "$log" || :) non-zero"; exit 1; }
ids() { { grep "\"id\":$1,\"tok\":" "$log" || :; } | sed 's/.*"tok":\([0-9-]*\).*/\1/' | tr '\n' ' '; }
want=$(ids $sid | tr -s ' ' '\n' | tail -n +$((tail_k + 1)) | tr '\n' ' ')
[[ "$want" == "$(ids 3)" ]] || { echo "FAIL request 3 ids != request $sid ids from index $tail_k"; echo "$sid: $want"; echo "3: $(ids 3)"; exit 1; }
grep -o '"id":[0-9]*,"done":true.*"pair_steps":[0-9]*,"slot":[0-9]' "$log" | sed 's/"n":\([0-9]*\).*"pair_steps"/n \1 pair_steps/' >> "$out/arm.txt"
cat "$out/arm.txt"
echo "PASS mega2-pair-stagger: refill of slot $(( ${SWAP:-0} )) while the other slot decodes, tail $tail_k, $(echo "$(ids 3)" | wc -w) ids equal"
