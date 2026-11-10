#!/usr/bin/env bash
# MEGA2 pair gate (exchange/mega2-report.md): the two job-4 prompts served in one engine process,
# arm S (SEQ_CAP=1 engine, one request at a time) and arm P (SEQ_CAP=2 engine, BARO_SEQS=2: both
# requests in flight, decode steps shared in one R=2 megakernel launch). Identity: every tok line of
# P equals S's for the same id, and S's first 64 equal the frozen reference (REF/lines.txt, the job-4
# receipts). Cost: t1 = S decode_s / (n-1) per request; t2 = P slot-1 decode_s / (n-1) (slot 1 is
# admitted second, so its decode window is pair steps only); c = t2 / t1. REPS alternating pairs.
#   gpu-wait run --vram 23 --timeout 3600 -- env N=200 REPS=3 bench/mega2-pair-gate.sh ENGINE1 ENGINE2 OUT
# env: PACK, SHORT, LONG (token files), N (tokens per request, 64), REPS (1), REF (receipt dir with
# lines.txt), KNOBS (extra engine env). Every arm reads its knobs back from the engine log (P1) and
# the fail word on every request; a non-zero fail word or a missing read-back FAILs the arm.
set -euo pipefail
export LC_ALL=C
here=$(cd "$(dirname "$0")/.." && pwd)
e1=$(realpath "${1:?usage: mega2-pair-gate.sh ENGINE1 ENGINE2 OUT [ENGINE0]}") e2=$(realpath "${2:?}") out=$(realpath -m "${3:?}")
e0=${4:-}; [[ -z $e0 ]] || e0=$(realpath "$e0")
pack=${PACK:-$HOME/Projects/mojo/mojo-baro/.work/moe-ngram/pack}
short=${SHORT:-$HOME/Projects/mojo/mojo-baro/.work/l5x/p13951.tokens}
long=${LONG:-$HOME/Projects/mojo/mojo-baro/.work/l5x/p16146.tokens}
n=${N:-64} reps=${REPS:-1} ref=${REF:-$here/exchange/receipts/l5x2/p2t/exact}
mkdir -p "$out"
export MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=${MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT:-2}
card=$(grep -l . /sys/class/drm/card*/device/mem_info_vram_total | xargs -n1 dirname \
	| while read -r d; do echo "$(cat "$d/mem_info_vram_total") $d"; done | sort -n | tail -1 | cut -d' ' -f2)
{ echo "checkout $(git -C "$here" rev-parse --short HEAD)"
  echo "engine1 $e1 sha256 $(sha256sum "$e1" | cut -c1-16)"
  echo "engine2 $e2 sha256 $(sha256sum "$e2" | cut -c1-16)"
  [[ -z $e0 ]] || echo "engine0 $e0 sha256 $(sha256sum "$e0" | cut -c1-16) (main's R=1 kernel, t1 baseline)"
  echo "pack $pack short $short long $long n $n reps $reps"
  echo "clock caps: sclk $(cat "$card/pp_dpm_sclk" 2>/dev/null | tr '\n' ' ')"
  echo "power cap: $(cat "$card/hwmon/hwmon*/power1_cap" 2>/dev/null) uW"
  echo "load at start $(cut -d' ' -f1-3 /proc/loadavg) vram_used $(( $(cat "$card/mem_info_vram_used") >> 20 )) MiB gtt $(( $(cat "$card/mem_info_gtt_used") >> 20 )) MiB"; } > "$out/arm.txt"
( while :; do echo "$(date +%s) $(( $(cat "$card/mem_info_vram_used") >> 20 )) $(( $(cat "$card/mem_info_gtt_used") >> 20 )) $(cut -d' ' -f1 /proc/loadavg)"; sleep 5; done ) > "$out/vram.tsv" &
sampler=$!
trap 'kill $sampler 2>/dev/null || :' EXIT
req() { printf '{"id":%d,"prompt":[%s],"n":%d}\n' "$1" "$(tr -s ' \n' ',' < "$2" | sed 's/^,//; s/,$//')" "$n"; }
line1=$(req 1 "$short"); line2=$(req 2 "$long")
knobs="BARO_SERVE=1 BARO_PACK=$pack BARO_TMAX=26624 BARO_MEGA=1 BARO_PREFILL=1 BARO_SPEC=0 BARO_CKPT=32 BARO_KV_SHARE=2 BARO_PF_GEMM=${PF_GEMM:-exact} ${KNOBS:-}"
run_arm() {  # run_arm NAME ENGINE EXTRA_ENV -> $out/NAME.log
	local name=$1 eng=$2 extra=$3 log=$out/$1.log fifo=$out/$1.in
	rm -f "$fifo"; mkfifo "$fifo"
	# shellcheck disable=SC2086
	env $knobs $extra "$eng" < "$fifo" > "$log" 2>&1 &
	local pid=$!
	exec 4>"$fifo"
	printf '%s\n%s\n' "$line1" "$line2" >&4
	local t=0
	local nd
	while nd=$({ grep -c '"done":true' "$log" 2>/dev/null || :; }); (( ${nd:-0} < 2 )); do
		kill -0 $pid 2>/dev/null || { echo "FAIL $name: engine exited, see $log"; exec 4>&-; return 1; }
		sleep 1; t=$((t + 1)); (( t < 3000 )) || { echo "FAIL $name: 3000 s without two done lines"; kill $pid; exec 4>&-; return 1; }
	done
	exec 4>&-; wait $pid || :
	for w in "BARO_MEGA: True" "\"tmax\":26624" "BARO_KV_SHARE: pages [1-9]" "checkpoints: cap 32" "BARO_SPEC: False" "BARO_PF_GEMM: ${PF_GEMM:-exact}" "$4"; do
		grep -q "$w" "$log" || { echo "FAIL $name: read-back '$w' missing in $log"; return 1; }
	done
	local fails; fails=$(grep -c "mega fail word: [1-9]" "$log" || :)
	[[ $fails == 0 ]] || { echo "FAIL $name: $fails non-zero fail words in $log"; return 1; }
	(( $(grep -c "mega fail word: 0" "$log") == 2 )) || { echo "FAIL $name: fail word not read on both requests"; return 1; }
	echo "ok $name"
}
ids() { { grep "\"id\":$2,\"tok\":" "$1" || :; } | sed 's/.*"tok":\([0-9-]*\).*/\1/' | tr '\n' ' '; }
dec() { { grep "\"id\":$2,\"done\":true" "$1" || :; } | sed 's/.*"decode_s":\([0-9.e-]*\).*/\1/'; }
pst() { { grep "\"id\":$2,\"done\":true" "$1" || :; } | { grep -o "\"pair_steps\":[0-9]*" || echo 0; } | sed "s/.*://"; }
status=0
: > "$out/summary.tsv"; echo -e "rep\tarm\tid\tdecode_s\tpair_steps\tt_step_ms" >> "$out/summary.tsv"
for ((r = 1; r <= reps; r++)); do
	[[ -z $e0 ]] || run_arm "O$r" "$e0" "BARO_SEQS=1" "BARO_MEGA: True" || status=1
	run_arm "S$r" "$e1" "BARO_SEQS=1" "BARO_SEQS: 1" || status=1
	run_arm "P$r" "$e2" "BARO_SEQS=2" "BARO_SEQS: 2" || status=1
	for id in 1 2; do
		if [[ -n $e0 ]]; then
			o=$(ids "$out/O$r.log" $id); s=$(ids "$out/S$r.log" $id)
			[[ "$o" == "$s" ]] && echo "rep $r id $id: S == O ($(echo "$s" | wc -w) ids)" || { echo "FAIL rep $r id $id: S != O (R=1 kernel changed bytes)"; status=1; }
			d=$(dec "$out/O$r.log" $id)
			printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$r" "O" "$id" "$d" 0 "$(awk -v d="$d" -v n="$n" 'BEGIN{printf "%.3f", 1000*d/(n-1)}')" >> "$out/summary.tsv"
		fi
		s=$(ids "$out/S$r.log" $id); p=$(ids "$out/P$r.log" $id)
		[[ "$s" == "$p" ]] && echo "rep $r id $id: P == S ($(echo "$s" | wc -w) ids)" || { echo "FAIL rep $r id $id: P != S"; echo "S: $s"; echo "P: $p"; status=1; }
		for arm in S P; do
			d=$(dec "$out/$arm$r.log" $id); ps=$(pst "$out/$arm$r.log" $id || echo 0)
			printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$r" "$arm" "$id" "$d" "${ps:-0}" "$(awk -v d="$d" -v n="$n" 'BEGIN{printf "%.3f", 1000*d/(n-1)}')" >> "$out/summary.tsv"
		done
	done
	if [[ -f $ref/lines.txt ]]; then
		for id in 1 2; do
			f=short; [[ $id == 2 ]] && f=long
			want=$({ grep "$f.log:GENERATED:" "$ref/lines.txt" || :; } | sed 's/.*GENERATED: //' | tr -s ' ' | cut -d' ' -f1-64 | sed 's/ *$//')
			got=$(ids "$out/S$r.log" $id | tr -s ' ' | cut -d' ' -f1-64 | sed 's/ *$//')
			[[ "$want" == "$got" ]] && echo "rep $r id $id: S first 64 == reference" || { echo "FAIL rep $r id $id: S != reference"; echo "ref: $want"; echo "got: $got"; status=1; }
		done
	fi
	p2=$(pst "$out/P$r.log" 2 || echo 0)
	(( ${p2:-0} > 0 )) || { echo "FAIL rep $r: P slot 1 took 0 pair steps (pair path not exercised)"; status=1; }
done
awk -F'\t' 'NR>1 && $2=="O" {o+=$6; no++} NR>1 && $2=="S" {s+=$6; ns++} NR>1 && $2=="P" && $3==2 {p+=$6; np++} END{if(no) printf "t0 %.3f ms/step (O = main R=1, mean of both ids)\n", o/no; if(ns&&np) printf "t1 %.3f ms/step (S = new tree R=1), t2 %.3f ms/step (P slot 1), c_new = %.3f", s/ns, p/np, (p/np)/(s/ns); if(no&&np) printf ", c = t2/t0 = %.3f", (p/np)/(o/no); printf "\n"}' "$out/summary.tsv" | tee -a "$out/arm.txt"
echo "load at end $(cut -d' ' -f1-3 /proc/loadavg) max vram $(awk 'BEGIN{m=0} $2>m{m=$2} END{print m}' "$out/vram.tsv") MiB max gtt $(awk 'BEGIN{m=0} $3>m{m=$3} END{print m}' "$out/vram.tsv") MiB" >> "$out/arm.txt"
cat "$out/summary.tsv"
[[ $status == 0 ]] && echo "PASS mega2-pair-gate" || echo "FAIL mega2-pair-gate"
exit $status
