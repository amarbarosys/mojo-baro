#!/usr/bin/env bash
# One GPU job: serve RegesCore-35B with the vero persona on one engine, replay the recorded
# vero author loop against it (bench/vero-replay.sh), stop the server.
#
#   gpu-wait run --vram 23 --timeout 3600 -- bench/vero-ab.sh llama OUT
#   gpu-wait run --vram 23 --timeout 3600 -- env ENGINE=.work/l5x/engine KNOBS="BARO_MEGA=1 ..." bench/vero-ab.sh baro OUT
#   ... bench/vero-ab.sh llama,baro OUT     (in order, one job, each into OUT/<arm>)
#
# LIVE=TOPICS runs newera's tools/vero_author.sh (sampled, real tools) instead of the replay:
# OUT/live/summary.tsv, wall and gate PASS count in OUT/wall.txt.
# RUN (default newera run7) is the recorded run, PAR (2) conversations in flight, as
# vero_author.sh against llama-server -np 2. llama.cpp gets newera's flags (-c 65536 -np 2,
# q8_0 KV, flash attention). baro: ENGINE + KNOBS (env), resident pack, BARO_TMAX in KNOBS;
# every KNOBS entry is read back from the engine log (P1) or the arm FAILs. WARM_TOOLS=FILE (L5X-2
# a8f703d) renders the warm prompt with that tool list so the persona checkpoint matches requests. The resident arm
# FAILs when GTT passes 4 GB (spilled pack, P13) or the megakernel sets its fail word.
# A running job reads this file by byte offset: queue a copy (both scripts into one dir, REPO=
# the checkout), never the file you are editing (m.ledger 2026-10-16).
set -euo pipefail
export LC_ALL=C
here=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
arm=${1:?usage: vero-ab.sh llama|baro[,..] OUT} out=$(realpath -m "${2:?}")
if [[ $arm == *,* ]]; then
	IFS=, read -ra arms <<<"$arm"
	for a in "${arms[@]}"; do "$0" "$a" "$out/$a" || { echo "FAIL $a arm -> $out/$a"; exit 1; }; done
	exit 0
fi
mkdir -p "$out"
models=$HOME/Models/RegesCore-1.0-35
gguf=${GGUF:-$models/RegesCore-1.0-35B-UD-Q4_K_S-BARO-7d42feb.gguf}
persona=${PERSONA:-$models/personas/persona-vero.jinja}   # pin it: bench/vero-persona-pin.sh
run=${RUN:-$HOME/Projects/newera/vero/corpus/regescore/run7}
card=$(grep -l . /sys/class/drm/card*/device/mem_info_vram_total | xargs -n1 dirname \
	| while read -r d; do echo "$(cat "$d/mem_info_vram_total") $d"; done | sort -n | tail -1 | cut -d' ' -f2)
port=$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
log=$out/server.log
{ echo "checkout $(git -C "$here" rev-parse --short HEAD) arm $arm"
  echo "gguf $gguf"; echo "persona $persona sha256 $(sha256sum "$persona" | cut -c1-16)"; echo "run $run par ${PAR:-2}"
  echo "load at start $(cut -d' ' -f1-3 /proc/loadavg) vram_used $(( $(cat "$card/mem_info_vram_used") >> 20 )) MiB"; } > "$out/arm.txt"
case $arm in
llama)
	bin=${LLAMA:-$HOME/llama.cpp/build/bin/llama-server}
	echo "llama.cpp $(git -C "$(dirname "$bin")/../.." log -1 --format='%h %cs') $bin" >> "$out/arm.txt"
	args=(-m "$gguf" --jinja --chat-template-file "$persona" -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c 65536 -np 2)
	echo "args ${args[*]}" >> "$out/arm.txt"
	"$bin" "${args[@]}" --host 127.0.0.1 --port "$port" > "$log" 2>&1 < /dev/null &
	;;
baro)
	engine=$(realpath "${ENGINE:?ENGINE=path/to/engine}")
	read -ra knobs <<<"${KNOBS:?KNOBS='BARO_TMAX=... ...'}"
	pack=${PACK:-$here/.work/moe-ngram/pack}; tok=$here/.work/latent-spec/pack-tier/tokenizer.json
	echo "engine $engine sha256 $(sha256sum "$engine" | cut -c1-16)" >> "$out/arm.txt"
	echo "baro-serve sha256 $(sha256sum "$here/serve/target/release/baro-serve" | cut -c1-16) pack $pack knobs ${knobs[*]} ${WARM:+warm}${WARM_TOOLS:+ warm-tools $WARM_TOOLS sha256 $(sha256sum "$WARM_TOOLS" | cut -c1-16)}" >> "$out/arm.txt"
	warmargs=(${WARM:+--warm} ${WARM_TOOLS:+--warm-tools "$WARM_TOOLS"})
	if [[ -n ${WARM_STATE_DIR:-} ]]; then   # warm once, save the shared-prefix checkpoint, load it in later arms
		mkdir -p "$WARM_STATE_DIR"
		wst=$WARM_STATE_DIR/$( { sha256sum "$engine" "$persona" ${WARM_TOOLS:+"$WARM_TOOLS"} | cut -c1-16; echo "${knobs[*]}"; } | sha256sum | cut -c1-16).bst
		if [[ -f $wst ]]; then warmargs=(--engine-env "BARO_STATE_LOAD=$wst"); echo "warm state load $wst" >> "$out/arm.txt"
		else warmargs+=(--warm-save "$wst"); echo "warm state save $wst" >> "$out/arm.txt"; fi
	fi
	env BARO_CKPT_DIR="$out/ckpt" "${knobs[@]}" "$here/serve/target/release/baro-serve" --engine "$engine" --pack "$pack" \
		--tokenizer "$tok" --chat-template-file "$persona" --port "$port" "${warmargs[@]}" > "$log" 2>&1 < /dev/null &
	;;
*) echo "FAIL usage: arm $arm"; exit 2 ;;
esac
spid=$!
( while :; do echo "$(date +%s) $(( $(cat "$card/mem_info_vram_used") >> 20 )) $(( $(cat "$card/mem_info_gtt_used") >> 20 )) $(cut -d' ' -f1 /proc/loadavg)"; sleep 5; done ) > "$out/vram.tsv" &
sampler=$!
trap 'pkill -TERM -P $spid 2>/dev/null || :; kill $spid $sampler 2>/dev/null || :; wait $spid 2>/dev/null || :' EXIT
for _ in $(seq 900); do
	curl -sf --max-time 2 "localhost:$port/health" > /dev/null && break
	kill -0 $spid 2>/dev/null || { echo "FAIL serve: $arm server exited, see $log"; exit 1; }
	sleep 1
done
curl -sf --max-time 2 "localhost:$port/health" > /dev/null || { echo "FAIL serve: no /health in 900 s, see $log"; exit 1; }
if [[ $arm == baro ]]; then
	for kv in "${knobs[@]}"; do   # P1: the engine prints each knob it took; the name must appear with the value
		k=${kv%%=*} v=${kv#*=}
		case $k in
		BARO_TMAX) w="tmax: $v," ;;
		BARO_MEGA) w="BARO_MEGA: $([[ $v == 1 ]] && echo True || echo False)" ;;
		BARO_PREFILL) w="BARO_PREFILL: $([[ $v == 1 ]] && echo True || echo False)" ;;
		BARO_SPEC) w="BARO_SPEC: $([[ $v == 1 ]] && echo True || echo False)" ;;
		BARO_CKPT) w="checkpoints: cap $v" ;;
		BARO_KV_SHARE) w="BARO_KV_SHARE: pages $([[ $v == 0 ]] && echo 0 || echo '[1-9]')" ;;
		BARO_NGRAM) w="BARO_SPEC: $([[ $v == 1 ]] && echo True || echo False)" ;;
		BARO_SPEC_K) w="spec k: $v " ;;
		BARO_NGRAM_MIN) w="ngram min: $v" ;;
		MODULAR_*) echo "not read back (MAX prints nothing) $kv" >> "$out/arm.txt"; continue ;;
		BARO_DEFAULT_TOP_K) w="sampling defaults: top_k $v " ;;
		BARO_DEFAULT_TOP_P) w="sampling defaults: .* top_p $v " ;;
		BARO_DEFAULT_MIN_P) w="sampling defaults: .* min_p $v\$" ;;
		BARO_PF_GEMM) w="BARO_PF_GEMM: $v\$" ;;
		BARO_PROFILE) echo "read back after the run (profile: lines) $kv" >> "$out/arm.txt"; continue ;;
		*) w="$k: $v" ;;   # every other knob the engine prints as "NAME: parsed value" (BARO_SEQS, BARO_PF_ATT, ...)
		esac
		grep -qE -- "$w" "$log" || { echo "FAIL readback: $kv ('$w') not in $log"; exit 1; }
		echo "readback ok $kv: $(grep -m1 -E -- "$w" "$log")" >> "$out/arm.txt"
	done
	if grep -q '^warm state load' "$out/arm.txt"; then
		grep -q 'state loaded:' "$log" || { echo "FAIL readback: warm state not loaded, see $log"; exit 1; }
	elif grep -q '^warm state save' "$out/arm.txt"; then
		grep -q 'state saved:' "$log" || { echo "FAIL readback: warm state not saved, see $log"; exit 1; }
	fi
	( while kill -0 $spid 2>/dev/null; do
		g=$(( $(cat "$card/mem_info_gtt_used") >> 20 )); ((g > 4096)) && { echo "gtt_used $g MiB" > "$out/SPILL"; kill $spid; break; }
		sleep 2; done ) &
fi
rc=0
[[ $arm != llama ]] || curl -sf "localhost:$port/props" | jq -c '.default_generation_settings.params | {temperature, top_k, top_p, min_p}' >> "$out/arm.txt"
if [[ -n ${LIVE:-} ]]; then   # the real loop: newera's vero_author.sh, sampled, real tools
	t0=$EPOCHREALTIME
	URL=http://127.0.0.1:$port PAR=${PAR:-2} "$HOME/Projects/newera/tools/vero_author.sh" "$LIVE" "$out/live" > "$out/live.log" 2>&1 || rc=1
	awk -v a="$t0" -v b="$EPOCHREALTIME" 'BEGIN { printf "wall_s %.1f\n", b - a }' > "$out/wall.txt"
	awk -F'\t' 'NR > 1 { n++; if ($4 == "PASS") p++ } END { printf "samples %d gate PASS %d\n", n, p }' "$out/live/summary.tsv" >> "$out/wall.txt"
else
	"$(dirname "$0")/vero-replay.sh" "$run" "http://127.0.0.1:$port" "$out" "${PAR:-2}" || rc=1
fi
echo "load at end $(cut -d' ' -f1-3 /proc/loadavg) max gtt $(awk 'BEGIN{m=0} $3>m{m=$3} END{print m}' "$out/vram.tsv") MiB" >> "$out/arm.txt"
[[ ! -f $out/SPILL ]] || { echo "FAIL spill: $(cat "$out/SPILL"), run INVALID"; exit 1; }
! grep -qE "fail word: [1-9]" "$log" || { echo "FAIL fail word: $(grep -m1 -E 'fail word: [1-9]' "$log"), run INVALID"; exit 1; }
[[ " ${knobs[*]:-} " != *" BARO_PROFILE="* ]] || grep -q "profile: attn" "$log" || { echo "FAIL readback: BARO_PROFILE set, no profile lines in $log"; exit 1; }
((rc == 0)) || { echo "FAIL replay: see $out"; exit 1; }
echo "PASS $arm -> $out ($(tail -1 "$out/wall.txt"))"
