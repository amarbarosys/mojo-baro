#!/usr/bin/env bash
# usage: bench/kdefault-gate.sh ENGINE_FW0 ENGINE_FW1 PACK OUTDIR   (one gpu-wait job)
# Gate for a new speculation default (bars frozen in docs/lanes/2026-11-16-kdefault-report.md before the run),
# modelled on bench/dflash2-d3.sh: every arm resident (BARO_SERVE=1), BARO_MEGA default, greedy, 20 mtp-prompts.
#   ref  FW0 BARO_SPEC=0 BARO_PREFILL=0: m=1 greedy ids = force references
#   a    FW0 MTP k=2 live (today's default)      a2  same again (same-arm control)
#   b    FW1 MTP k=7 live (candidate: FEWROW build, k=7)
# BARO_FORCE refuses spec, so identity = prompts whose greedy ids differ from ref (GEN 256), speed = tok/s_gen.
set -euo pipefail
cd "$(dirname "$0")/.."
[ $# -eq 4 ] || { echo "FAIL args: ENGINE_FW0 ENGINE_FW1 PACK OUTDIR"; exit 1; }
e0=$(realpath "$1"); e1=$(realpath "$2"); pack=$(realpath "$3"); out=$(realpath -m "$4")
GEN=${GEN:-256}; read -r -a prompts <<< "$(echo bench/mtp-prompts/p[0-9][0-9]-*.tokens)"; np=${#prompts[@]}
mkdir -p "$out/ref"
echo "fw0 $(sha256sum "$e0" | cut -c1-16) fw1 $(sha256sum "$e1" | cut -c1-16) pack $pack gen $GEN prompts $np tree $(git rev-parse --short HEAD)" > "$out/arm.txt"
req() {  # file spec force
  : > "$1"; local i=0 f
  for tf in "${prompts[@]}"; do i=$((i + 1)); f=""
    [ "$3" = 1 ] && f=",\"force\":[$(tr -s ' ' ',' < "$out/ref/$i.ids" | sed 's/^,//;s/,$//')]"
    printf '{"id":%d,"prompt":[%s],"n":%d,"spec":%s%s}\n' "$i" "$(tr -s ' \n' ',,' < "$tf" | sed 's/,$//')" "$GEN" "$2" "$f" >> "$1"
  done
}
run() {  # label engine reqfile env...
  local l=$1 bin=$2 r=$3 n; shift 3
  env BARO_PACK="$pack" BARO_SERVE=1 BARO_TMAX=16384 "$@" "$bin" < "$r" > "$out/$l.log" 2>&1 || { echo "FAIL $l: engine exit, see $out/$l.log"; exit 1; }
  n=$(grep -c '"done": *true' "$out/$l.log" || true); [ "$n" = "$np" ] || { echo "FAIL $l: $n of $np done"; exit 1; }
  if grep -qE 'mega fail word: [1-9]' "$out/$l.log"; then echo "FAIL $l: fail word set"; exit 1; fi
  { echo "$l env: $*"; grep -E '^(BARO_MEGA|BARO_SPEC|spec k):' "$out/$l.log" | sort -u | sed "s/^/$l /"; } >> "$out/arm.txt"
  grep -oE 'tok/s_gen: [0-9.]+' "$out/$l.log" | cut -d' ' -f2 > "$out/$l.toks"
}
med() { sort -g "$1" | awk '{ v[NR] = $1 } END { print (NR % 2) ? v[(NR + 1) / 2] : (v[NR / 2] + v[NR / 2 + 1]) / 2 }'; }
agree() { grep '^forced agreement:' "$out/$1.log" | awk '{ a += $3; n += $5 } END { printf "%.5f %d %d\n", a / n, a, n }'; }
req "$out/req-plain.jsonl" false 0
run ref "$e0" "$out/req-plain.jsonl" BARO_SPEC=0 BARO_PREFILL=0 BARO_MEGA=0
i=0; grep '^GENERATED:' "$out/ref.log" | sed 's/^GENERATED: *//' | while read -r ids; do i=$((i + 1)); echo "$ids" > "$out/ref/$i.ids"; done
req "$out/req-spec.jsonl" true 0
run a "$e0" "$out/req-spec.jsonl" BARO_SPEC=1 BARO_SPEC_K=2
run a2 "$e0" "$out/req-spec.jsonl" BARO_SPEC=1 BARO_SPEC_K=2
run b "$e1" "$out/req-spec.jsonl" BARO_SPEC=1 BARO_SPEC_K=${B_K:-7}
diffn() { diff <(grep '^GENERATED' "$out/ref.log") <(grep '^GENERATED' "$out/$1.log") | grep -c '^>' || true; }
da=$(diffn a); da2=$(diffn a2); db=$(diffn b); sa=$(med "$out/a.toks"); sa2=$(med "$out/a2.toks"); sb=$(med "$out/b.toks")
{ echo "a  ids != ref $da/$np  tok/s_gen median $sa"; echo "a2 ids != ref $da2/$np  tok/s_gen median $sa2"; echo "b  ids != ref $db/$np  tok/s_gen median $sb"; } | tee "$out/summary.txt"
awk -v da="$da" -v da2="$da2" -v db="$db" -v sa="$sa" -v sb="$sb" 'BEGIN { m = (da > da2) ? da : da2; g1 = db <= m + 2; g2 = sb > sa; printf "G1 ids b <= max(a,a2)+2: %s\nG2 speed b > a: %s (%.3fx)\n", g1 ? "PASS" : "FAIL", g2 ? "PASS" : "FAIL", sb / sa; exit !(g1 && g2) }'
