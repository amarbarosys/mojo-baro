#!/usr/bin/env bash
# Decode tok/s of N engine arms, the champion's harness method (one engine process per prompt,
# bench/mtp-prompts, default GEN), in ONE gpu-wait job. For bisects and config probes: every arm
# is `label=ENGINE[,K=V...]` (K=V pairs go into that arm's environment only). The harness arm of
# bench/decode-probe.sh generalised to N binaries.
#
#   gpu-wait run --vram 23 --timeout 600 -- bench/decode-arms.sh OUT \
#     a=.work/s1/engine-x b=.work/s1/engine-y,MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=2
#
# Common env: BARO_PACK=$PACK (default .work/moe-ngram/pack) BARO_MEGA=0 BARO_SPEC=0 BARO_TMAX=8192.
# PROMPTS="bench/mtp-prompts/p0[1-3]-*.tokens" (default p01 only; a glob that matches nothing FAILs).
# Read-back per run (P1): the log must say `BARO_MEGA: False` and `BARO_SPEC: False`, a set fail
# word voids the run, max vram_used and gtt_used during the run are recorded (MAX's pool cap shows
# in vram; gtt over 4 GiB = spilled pack, run INVALID), and arm.txt names each binary's sha256, the
# power cap and the 1-min load average. The job waits up to 60 s for the load average to reach
# LOAD_START_MAX (4) and FAILs otherwise (the S0 load gate).
# Runs interleave: every arm on prompt 1, then every arm on prompt 2 (the 2026-09-24 config job ran
# arms in blocks and the load average climbed from 4 to 9 between the first arm and the last).
# results.tsv: one row per run; summary.tsv: median per arm and GENERATED identity vs the first arm.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
out=$(realpath -m "${1:?usage: decode-arms.sh OUT label=ENGINE[,K=V...] ...}"); shift
(($#)) || { echo "FAIL usage: no arms"; exit 2; }
mkdir -p "$out"
pack=${PACK:-$here/.work/moe-ngram/pack}
common=(BARO_PACK="$pack" BARO_MEGA=0 BARO_SPEC=0 BARO_TMAX=8192)
read -ra prompts <<<"$(cd "$here" && echo ${PROMPTS:-bench/mtp-prompts/p01-*.tokens})"
for tf in "${prompts[@]}"; do [[ -f $here/$tf ]] || { echo "FAIL setup: prompt $tf does not exist"; exit 1; }; done
card=$(grep -l . /sys/class/drm/card*/device/mem_info_vram_total | xargs -n1 dirname \
  | while read -r d; do echo "$(cat "$d/mem_info_vram_total") $d"; done | sort -n | tail -1 | cut -d' ' -f2)
load1() { cut -d' ' -f1 /proc/loadavg; }
gt() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a > b) }'; }
for _ in $(seq 12); do gt "$(load1)" "${LOAD_START_MAX:-4}" || break; sleep 5; done
gt "$(load1)" "${LOAD_START_MAX:-4}" && { echo "FAIL load: 1-min load average $(load1) > ${LOAD_START_MAX:-4} after 60 s, run refused"; exit 1; }
trap 'pkill -P $$ 2>/dev/null || :' EXIT
{
  echo "checkout $(git -C "$here" rev-parse --short HEAD) pack $pack $(jq -r .pack_sha256 "$pack/identity.json" 2>/dev/null | cut -c1-16)"
  echo "common ${common[*]}"
  echo "power_cap_uW=$(cat "$card"/hwmon/hwmon*/power1_cap) load_at_start=$(load1) vram_used_at_start=$(( $(cat "$card/mem_info_vram_used") >> 20 )) MiB"
} > "$out/arm.txt"
labels=()
for a in "$@"; do
  label=${a%%=*}; IFS=, read -ra parts <<<"${a#*=}"; eng=$(realpath "${parts[0]}")
  [[ -x $eng ]] || { echo "FAIL setup: arm $label engine $eng not executable"; exit 1; }
  echo "arm $label engine $eng sha256 $(sha256sum "$eng" | cut -c1-16) env ${parts[*]:1}" >> "$out/arm.txt"
  labels+=("$label")
done
printf 'arm\tprompt\tn_prompt\ttok_s\tgenerated_md5\tfail_word\tvram_max_mib\tgtt_max_mib\tload1\tpack_load\n' > "$out/results.tsv"
fails=()
for tf in "${prompts[@]}"; do   # prompt outer, arms inner: load drift during the job hits every arm alike
  for a in "$@"; do
    label=${a%%=*}; IFS=, read -ra parts <<<"${a#*=}"; eng=$(realpath "${parts[0]}")
    p=$(basename "$tf" .tokens); log=$out/$label.$p.log; l=$(load1)
    ( while :; do echo "$(( $(cat "$card/mem_info_vram_used") >> 20 )) $(( $(cat "$card/mem_info_gtt_used") >> 20 ))"; sleep 0.5; done ) > "$out/$label.$p.vram" &
    vs=$!
    rc=0; env "${common[@]}" "${parts[@]:1}" BARO_PROMPT="$here/$tf" "$eng" > "$log" 2>&1 < /dev/null || rc=$?
    kill $vs 2>/dev/null || :; wait $vs 2>/dev/null || :
    vmax=$(awk 'BEGIN{m=0} $1>m{m=$1} END{print m}' "$out/$label.$p.vram"); gmax=$(awk 'BEGIN{m=0} $2>m{m=$2} END{print m}' "$out/$label.$p.vram")
    t=$(grep -oE 'tok/s_gen: [0-9.]+' "$log" | cut -d' ' -f2 || :)
    fw=$(grep -oE 'mega fail word: [0-9]+' "$log" | grep -oE '[0-9]+$' | sort -u | tr '\n' ' ' || :)
    g=$(grep -m1 '^GENERATED' "$log" | md5sum | cut -c1-12)
    pl=$(grep -oE 'pack loaded in [0-9.]+ ?[a-z]*' "$log" | head -1 | sed 's/pack loaded in //;s/ //g' || :)
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$label" "$p" "$(wc -w < "$here/$tf")" "${t:-nan}" "$g" "${fw:--}" "$vmax" "$gmax" "$l" "${pl:--}" >> "$out/results.tsv"
    why=
    ((rc == 0)) || why="engine exit $rc"
    [[ -n $t ]] || why="${why:+$why, }no tok/s_gen"
    for w in "BARO_MEGA: False" "BARO_SPEC: False"; do grep -qF "$w" "$log" || why="${why:+$why, }no '$w'"; done
    [[ ! $fw =~ [1-9] ]] || why="${why:+$why, }fail word $fw"
    ((gmax <= 4096)) || why="${why:+$why, }gtt ${gmax} MiB, pack spilled"
    [[ -z $why ]] || { echo "FAIL $label $p: $why, see $log"; fails+=("$label.$p"); }
  done
done
python3 - "$out/results.tsv" "${labels[@]}" > "$out/summary.tsv" <<'PY'
import sys, statistics as st
rows = [l.split("\t") for l in open(sys.argv[1]).read().splitlines()[1:]]
first = sys.argv[2]
ref = {r[1]: r[4] for r in rows if r[0] == first}
print(f"arm\tmedian_tok_s\tmin\tmax\tn\tidentity_vs_{first}\tvram_max_mib\tload1_max")
for lab in sys.argv[2:]:
    rs = [r for r in rows if r[0] == lab]
    ts = [float(r[3]) for r in rs if r[3] != "nan"]
    same = sum(1 for r in rs if ref.get(r[1]) == r[4])
    print(f"{lab}\t{st.median(ts) if ts else 'nan'}\t{min(ts) if ts else 'nan'}\t{max(ts) if ts else 'nan'}\t{len(ts)}"
          f"\t{same}/{len(rs)}\t{max(int(r[6]) for r in rs)}\t{max(float(r[8]) for r in rs)}")
PY
column -t "$out/summary.tsv"
((${#fails[@]} == 0)) || { echo "FAIL ${#fails[@]}/$(( $# * ${#prompts[@]} )) ${fails[*]} -> $out"; exit 1; }
echo "PASS $# arms x ${#prompts[@]} prompts -> $out/summary.tsv"
