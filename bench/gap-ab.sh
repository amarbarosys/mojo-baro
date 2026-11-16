#!/usr/bin/env bash
# usage: bench/gap-ab.sh ENGINE OUT : exploratory (3 prompts, GEN 256): what the megakernel saves where it exists.
# Arms: plain launch (MEGA=0), plain mega (MEGA=1), MTP k=2 launch, MTP k=2 mega_win. Resident (one load per arm),
# fail word read on every arm (as bench/mega-g-ab.sh), path and knob lines recorded. Run inside one gpu-wait job.
set -euo pipefail
cd "$(dirname "$0")/.."
e=$(realpath "$1"); out=$(realpath -m "$2"); mkdir -p "$out"
pack=$(realpath ../nerkyor-vs-forge/.work/nerkyor/pack-q4)
for spec in false true; do
  : > "$out/req-$spec.jsonl"; i=0
  for tf in bench/mtp-prompts/p0[1-3]-*.tokens; do i=$((i + 1))
    printf '{"id":%d,"prompt":[%s],"n":256,"spec":%s}\n' $i "$(tr -s ' \n' ',,' < "$tf" | sed 's/,$//')" $spec >> "$out/req-$spec.jsonl"; done
done
run() {  # label spec env...
  local l=$1 s=$2; shift 2
  env BARO_PACK="$pack" BARO_SERVE=1 BARO_TMAX=16384 "$@" "$e" < "$out/req-$s.jsonl" > "$out/$l.log" 2>&1 || { echo "FAIL $l: see $out/$l.log"; exit 1; }
  if grep -qE 'mega fail word: [1-9]' "$out/$l.log"; then echo "FAIL $l: fail word set"; exit 1; fi
  printf '%-7s median %s | %s\n' "$l" "$(grep -oE 'tok/s_gen: [0-9.]+' "$out/$l.log" | cut -d' ' -f2 | sort -g | sed -n 2p)" "$(grep -E '^(BARO_MEGA|BARO_MEGA_WIN|path):' "$out/$l.log" | sort -u | tr '\n' ' ')"
}
run plain0 false BARO_MEGA=0 BARO_SPEC=0
run plain1 false BARO_MEGA=1 BARO_SPEC=0
run mtp0 true BARO_MEGA=0 BARO_MEGA_WIN=0 BARO_SPEC=1 BARO_SPEC_K=2
run mtp1 true BARO_MEGA=1 BARO_MEGA_WIN=1 BARO_SPEC=1 BARO_SPEC_K=2
