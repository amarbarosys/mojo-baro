#!/usr/bin/env bash
# usage: bench/dflash2-d3.sh ENGINE_FW1 ENGINE_FW0 PACK OUTDIR   (run inside ONE gpu-wait job)
# dflash2 D3 gate (docs/lanes/2026-11-16-dflash2-report.md "D3 gate", bars frozen there before the run).
# Every arm resident (BARO_SERVE=1, one pack load per arm), BARO_MEGA=0, greedy, the same prompts:
#   ref   FW1 BARO_SPEC=0 BARO_PREFILL=0: m=1 greedy ids per prompt = the force references; nospec tok/s
#   f1    FW1 BARO_SPEC=dflash2 + force: forced agreement of the 8-row verify window (WMMA route), draft acceptance
#   f0    FW0 same: the same-arm control on the shipped q4rowb route
#   f1s   FW1 as f1 + BARO_DF_SHIFT=1: reverse arm (features one position late): acceptance must drop, predicted ids
#         must equal f1's (in a forced run the window carries the reference, so the draft cannot change them)
#   live  FW1 BARO_SPEC=dflash2: tok/s_gen, tokens per cycle, ids vs ref (reported)
#   mtp2  FW1 MTP k=2 (request spec true): tok/s_gen
# SMOKE=1 runs ref, f1, live only and judges nothing. Env: PROMPTS (glob, default all 20 mtp-prompts), GEN (256),
# BARO_TMAX (16384), DF (draft pack, .work/dflash2/draft-pack). Exit 1 on any FAIL.
set -euo pipefail
cd "$(dirname "$0")/.."
[ $# -eq 4 ] || { echo "FAIL args: ENGINE_FW1 ENGINE_FW0 PACK OUTDIR"; exit 1; }
e1=$(realpath "$1"); e0=$(realpath "$2"); pack=$(realpath "$3"); out=$(realpath -m "$4")
GEN=${GEN:-256}; TMAX=${BARO_TMAX:-16384}; DF=$(realpath "${DF:-.work/dflash2/draft-pack}"); SMOKE=${SMOKE:-0}
[ "$TMAX" -ge 16384 ] || { echo "FAIL tmax: $TMAX < 16384"; exit 1; }
read -r -a prompts <<< "$(echo ${PROMPTS:-bench/mtp-prompts/p[0-9][0-9]-*.tokens})"
np=${#prompts[@]}
mkdir -p "$out/ref"
echo "fw1 $(sha256sum "$e1" | cut -c1-16) fw0 $(sha256sum "$e0" | cut -c1-16) pack $pack draft $DF gen $GEN tmax $TMAX prompts $np smoke $SMOKE tree $(git rev-parse --short HEAD)" > "$out/arm.txt"

req() {  # $1 file  $2 request spec (true|false)  $3 force (1 = ref ids)
  : > "$1"
  local i=0 f
  for tf in "${prompts[@]}"; do
    i=$((i + 1)); f=""
    [ "$3" = 1 ] && f=",\"force\":[$(tr -s ' ' ',' < "$out/ref/$i.ids" | sed 's/^,//;s/,$//')]"
    printf '{"id":%d,"prompt":[%s],"n":%d,"spec":%s%s}\n' "$i" "$(tr -s ' \n' ',,' < "$tf" | sed 's/,$//')" "$GEN" "$2" "$f" >> "$1"
  done
}
run() {  # $1 label  $2 engine  $3 request file  $4.. env
  local l=$1 bin=$2 r=$3 n
  shift 3
  env BARO_PACK="$pack" BARO_SERVE=1 BARO_TMAX="$TMAX" BARO_MEGA=0 BARO_DFLASH_PACK="$DF" "$@" "$bin" < "$r" > "$out/$l.log" 2>&1 \
    || { echo "FAIL $l: engine exit, see $out/$l.log"; exit 1; }
  n=$(grep -c '"done": *true' "$out/$l.log" || true)
  [ "$n" = "$np" ] || { echo "FAIL $l: $n of $np requests done, see $out/$l.log"; exit 1; }
  if grep -qE 'mega fail word: [1-9]' "$out/$l.log"; then echo "FAIL $l: fail word set"; exit 1; fi
  { echo "$l env: $*"; grep -E '^(BARO_MEGA|BARO_SPEC|BARO_FORCE|dflash2 draft):' "$out/$l.log" | sort -u | sed "s/^/$l /"; } >> "$out/arm.txt"
  grep -oE 'tok/s_gen: [0-9.]+' "$out/$l.log" | cut -d' ' -f2 > "$out/$l.toks"
}
med() { sort -g "$1" | awk '{ v[NR] = $1 } END { print (NR % 2) ? v[(NR + 1) / 2] : (v[NR / 2] + v[NR / 2 + 1]) / 2 }'; }
agree() { grep '^forced agreement:' "$out/$1.log" | awk '{ a += $3; n += $5; p = $3 / $5; if (NR == 1 || p < mn) mn = p } END { printf "%.5f %d %d %.5f\n", a / n, a, n, mn }'; }
tpc() { grep '^dflash2:' "$out/$1.log" | awk '{ c += $4; a += $6; d += $11; for (i = 0; i < 8; i++) h[i] += $(13 + i) } END { printf "%.3f cycles %d accepted %d draft_s %.2f hist", (a + c) / c, c, a, d; for (i = 0; i < 8; i++) printf " %d", h[i]; print "" }'; }

req "$out/req-plain.jsonl" false 0
run ref "$e1" "$out/req-plain.jsonl" BARO_SPEC=0 BARO_PREFILL=0  # prompts replay as in every dflash2 arm
i=0
grep '^GENERATED:' "$out/ref.log" | sed 's/^GENERATED: *//' | while read -r ids; do i=$((i + 1)); echo "$ids" > "$out/ref/$i.ids"; done
[ "$(ls "$out/ref" | wc -l)" = "$np" ] || { echo "FAIL ref: GENERATED lines != $np"; exit 1; }
req "$out/req-force.jsonl" false 1
run f1 "$e1" "$out/req-force.jsonl" BARO_SPEC=dflash2
run live "$e1" "$out/req-plain.jsonl" BARO_SPEC=dflash2
if [ "$SMOKE" != 1 ]; then
  run f0 "$e0" "$out/req-force.jsonl" BARO_SPEC=dflash2
  run f1s "$e1" "$out/req-force.jsonl" BARO_SPEC=dflash2 BARO_DF_SHIFT=1
  req "$out/req-spec.jsonl" true 0
  run mtp2 "$e1" "$out/req-spec.jsonl" BARO_SPEC=1 BARO_SPEC_K=2
fi

ids_eq() { diff <(grep '^GENERATED' "$out/$1.log") <(grep '^GENERATED' "$out/$2.log") | grep -c '^>' || true; }
{
  echo "ref   tok/s_gen median $(med "$out/ref.toks")"
  echo "f1    agreement pooled/agree/n/min $(agree f1)  tokens/cycle $(tpc f1)"
  echo "live  tok/s_gen median $(med "$out/live.toks")  tokens/cycle $(tpc live)  ids != ref on $(ids_eq ref live) of $np"
  if [ "$SMOKE" != 1 ]; then
    echo "f0    agreement pooled/agree/n/min $(agree f0)  tokens/cycle $(tpc f0)"
    echo "f1s   agreement pooled/agree/n/min $(agree f1s)  tokens/cycle $(tpc f1s)"
    echo "f1s   predicted != f1 on $(diff <(grep '^predicted:' "$out/f1.log") <(grep '^predicted:' "$out/f1s.log") | grep -c '^>' || true) of $np"
    echo "mtp2  tok/s_gen median $(med "$out/mtp2.toks")  $(grep '^mtp:' "$out/mtp2.log" | awk '{ d += $3; a += $5 } END { printf "drafted %d accepted %d", d, a }')"
  fi
} | tee "$out/summary.txt"
[ "$SMOKE" = 1 ] && { echo "SMOKE dflash2-d3 done (no verdict)"; exit 0; }
a1=$(agree f1 | cut -d' ' -f1); a0=$(agree f0 | cut -d' ' -f1)
t1=$(tpc f1 | cut -d' ' -f1); ts=$(tpc f1s | cut -d' ' -f1)
pd=$(diff <(grep '^predicted:' "$out/f1.log") <(grep '^predicted:' "$out/f1s.log") | grep -c '^>' || true)
lv=$(med "$out/live.toks"); m2=$(med "$out/mtp2.toks")
v() { if awk "BEGIN { exit !($2) }"; then echo "$1 PASS ($3)"; else echo "$1 FAIL ($3)"; fi; }
{
  v G1 "$a1 >= $a0 - 0.005" "f1 $a1 vs control f0 $a0 - 0.005"
  v R1 "$ts <= 0.75 * $t1 && $pd == 0" "f1s tokens/cycle $ts vs 0.75 x $t1; predicted differ on $pd"
  v G3 "$lv > $m2" "live $lv vs mtp2 $m2 tok/s_gen, ratio $(awk "BEGIN { printf \"%.3f\", $lv / $m2 }")"
} | tee -a "$out/summary.txt"
grep -q FAIL "$out/summary.txt" && { echo "FAIL dflash2-d3"; exit 1; }
echo "PASS dflash2-d3"
