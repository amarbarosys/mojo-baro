#!/usr/bin/env bash
# usage: bench/dflash2-prof.sh ENGINE OUTDIR : live dflash2 kernel split (BARO_PROFILE 2 = ssm/attn, 4 = ffn), 3 prompts
set -euo pipefail
cd "$(dirname "$0")/.."
e=$(realpath "$1"); out=$(realpath -m "$2"); mkdir -p "$out"
pack=$(realpath ../nerkyor-vs-forge/.work/nerkyor/pack-q4); DF=$(realpath ${DF:-.work/dflash2/draft-pack})
: > "$out/req.jsonl"; i=0
for tf in bench/mtp-prompts/p0[1-3]-*.tokens; do i=$((i+1))
  printf '{"id":%d,"prompt":[%s],"n":%d,"spec":false}\n' $i "$(tr -s ' \n' ',,' < "$tf" | sed 's/,$//')" ${GEN:-128} >> "$out/req.jsonl"; done
for p in 2 4; do
  env BARO_PACK="$pack" BARO_SERVE=1 BARO_TMAX=16384 BARO_MEGA=0 BARO_DFLASH_PACK="$DF" BARO_SPEC=dflash2 BARO_PROFILE=$p "$e" < "$out/req.jsonl" > "$out/p$p.log" 2>&1 || { echo "FAIL p$p: see $out/p$p.log"; exit 1; }
done
grep -hE 'kernel:|profile|tok/s_gen|^dflash2:' "$out"/p*.log | sort | uniq -c | sort -rn | head -60
