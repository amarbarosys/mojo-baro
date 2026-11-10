#!/usr/bin/env bash
# usage: bench/ab-resident.sh ENGINE OUTDIR "ENV_A" "ENV_B" [label_A label_B]
# Two arms, same binary, one RESIDENT engine process per arm (BARO_SERVE=1, prompts as JSONL over
# stdin): two pack loads instead of one per prompt, so a 5-prompt A/B is ~2 min of card, not 9.
# AB_ENGINE_B=path runs arm B on a second binary (same env strings allowed), for a whole-build A/B such as a loop
# candidate against its champion; both sha256 go into arm.txt as engA=/engB=, and an identical pair is refused.
# Numbers are the served-path shape (bench/served-prompts.sh), comparable within this run only,
# never to the champion's one-process-per-prompt figures. PROMPTS glob (default p01-p05), GEN (64).
# Read-back per arm: BARO_MEGA / BARO_SPEC / BARO_MOE_DOT lines from the log (P1), fail word,
# 1-min load before and max during (refused above LOAD_START_MAX, 4), max vram/gtt. Identity =
# GENERATED ids equal per prompt across arms. Prints medians and the ratio; exit 1 on any FAIL.
# SEQ=1: sequential A/B (P4 amended 2026-10-11). Prompts (default all 20 mtp-prompts) run in blocks of
# BLOCK=5, arm order alternating AB, BA, ...; after every block from the 2nd, iTool seq-decide
# bootstraps the paired median ratio and stops once its CI excludes 1.00 (99% at interim looks, 95% at
# the last). Prints DECIDED or UNDECIDED with n; UNDECIDED at 20 = the plain 20-prompt result.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd); cd "$here"; export LC_ALL=C
eng=$1; out=$(realpath -m "$2"); envA=$3; envB=$4; la=${5:-A}; lb=${6:-B}; engB=${AB_ENGINE_B:-$eng}
mkdir -p "$out"; gen=${GEN:-64}; tmax=${BARO_TMAX:-8192}
seq=${SEQ:-0}; block=${BLOCK:-5}
if [ "$seq" = 1 ]; then dflt='bench/mtp-prompts/p[0-9][0-9]-*.tokens'; else dflt='bench/mtp-prompts/p0[1-5]-*.tokens'; fi
mapfile -t prompts < <(ls ${PROMPTS:-$dflt})
((${#prompts[@]})) || { echo "FAIL setup: PROMPTS matched nothing"; exit 1; }
[[ -x $eng ]] || { echo "FAIL setup: engine $eng not executable"; exit 1; }
[[ -x $engB ]] || { echo "FAIL setup: AB_ENGINE_B $engB not executable"; exit 1; }
[[ "$(sha256sum < "$eng")" == "$(sha256sum < "$engB")" && "$envA" == "$envB" ]] && { echo "REFUSED: both arms are the same binary with the same env (champion vs itself). Set AB_ENGINE_B or differ the env strings." >&2; exit 2; }
load1() { cut -d' ' -f1 /proc/loadavg; }
awk -v m="${LOAD_START_MAX:-4}" '{exit !($1 <= m)}' /proc/loadavg || { echo "FAIL load: 1-min load $(load1) > ${LOAD_START_MAX:-4}, timed run refused"; exit 1; }
card=$(for d in /sys/class/drm/card*/device; do [[ -f $d/mem_info_vram_total ]] && echo "$(cat $d/mem_info_vram_total) $d"; done | sort -n | tail -1 | cut -d' ' -f2)
: > "$out/req.jsonl"; i=0
for tf in "${prompts[@]}"; do i=$((i+1)); printf '{"id":%d,"prompt":[%s],"n":%d,"spec":%s}\n' "$i" "$(tr -s ' \n' ',,' < "$tf" | sed 's/,$//')" "$gen" "${REQ_SPEC:-false}" >> "$out/req.jsonl"; done
{ echo "engA=$eng sha256 $(sha256sum "$eng" | cut -c1-16)"; echo "engB=$engB sha256 $(sha256sum "$engB" | cut -c1-16)"; echo "pack ${BARO_PACK:-.work/moe-ngram/pack}"; echo "armA='$envA' armB='$envB' gen $gen prompts ${#prompts[@]} seq $seq block $block"; echo "power_cap_uW=$(cat "$card"/hwmon/hwmon*/power1_cap 2>/dev/null | head -1) load_start $(load1)"; } > "$out/arm.txt"
run_arm() {  # $1 label $2 env string
  local label=$1 e=$2 log=$out/$1$sfx.log vram=$out/$1$sfx.vram bin=$eng
  [ "$label" = "$lb" ] && bin=$engB
  ( while :; do echo "$(( $(cat "$card/mem_info_vram_used") >> 20 )) $(( $(cat "$card/mem_info_gtt_used") >> 20 )) $(load1)"; sleep 1; done ) > "$vram" 2>/dev/null & local sp=$!
  local want; want=$(wc -l < "$req")
  env BARO_PACK="${BARO_PACK:-.work/moe-ngram/pack}" BARO_SERVE=1 BARO_TMAX="$tmax" $e "$bin" < "$req" > "$log" 2>&1 || { kill $sp 2>/dev/null; echo "FAIL $label: engine exit, see $log"; return 1; }
  kill $sp 2>/dev/null; wait $sp 2>/dev/null || true
  { grep -E '^(BARO_MEGA|BARO_SPEC|BARO_MOE_DOT|mega fail word):' "$log" | sort -u | sed "s/^/$label /"; } >> "$out/arm.txt"
  grep -qE 'mega fail word: [1-9]' "$log" && { echo "FAIL $label: fail word set, arm INVALID"; return 1; }
  awk -v l="$label" '{ if ($1>v) v=$1; if ($2>g) g=$2; if ($3>ld) ld=$3 } END { printf "%s vram_max %d gtt_max %d load_max %.2f\n", l, v, g, ld }' "$vram" >> "$out/arm.txt"
  local n; n=$(grep -c '"done": *true' "$log" || true); [[ $n == "$want" ]] || { echo "FAIL $label: $n of $want requests done, see $log"; return 1; }
  awk -v g="$(tail -1 "$vram" | cut -d' ' -f2)" 'BEGIN{ if (g+0 > 4096) { print "FAIL spill: gtt " g " MiB"; exit 1 } }' || return 1
  { grep -oE 'tok/s_gen: [0-9.]+' "$log" || true; } | cut -d' ' -f2 >> "$out/$label.toks"
  { grep '^GENERATED' "$log" || true; } | while read -r l; do echo "$l" | md5sum | cut -c1-12; done >> "$out/$label.gen"
}
rm -f "$out/$la.toks" "$out/$lb.toks" "$out/$la.gen" "$out/$lb.gen"
if [ "$seq" = 1 ]; then
  nb=$(( (${#prompts[@]} + block - 1) / block ))
  for b in $(seq 0 $((nb - 1))); do
    req=$out/req-b$b.jsonl; sfx=-b$b; sed -n "$((b * block + 1)),$(((b + 1) * block))p" "$out/req.jsonl" > "$req"
    if [ $((b % 2)) = 0 ]; then run_arm "$la" "$envA"; run_arm "$lb" "$envB"; else run_arm "$lb" "$envB"; run_arm "$la" "$envA"; fi
    [ "$b" -ge 1 ] || continue
    if "$HOME/iTools/bin/seq-decide" "$out/$la.toks" "$out/$lb.toks" $([ "$b" = $((nb - 1)) ] && echo final); then prompts=("${prompts[@]:0:$(((b + 1) * block))}"); break; fi
  done
else
  req=$out/req.jsonl; sfx=""; run_arm "$la" "$envA"; run_arm "$lb" "$envB"
fi
python3 - "$out" "$la" "$lb" "${prompts[@]}" <<'PY'
import sys, statistics as st, pathlib
out, la, lb, *prompts = sys.argv[1:]
o = pathlib.Path(out)
ta = [float(x) for x in (o / f"{la}.toks").read_text().split()]; tb = [float(x) for x in (o / f"{lb}.toks").read_text().split()]
ga = (o / f"{la}.gen").read_text().split(); gb = (o / f"{lb}.gen").read_text().split()
rows = ["prompt\t%s_tok_s\t%s_tok_s\tratio\tidentity" % (la, lb)]
fails = []
for i, p in enumerate(prompts):
    name = pathlib.Path(p).stem; idn = "PASS" if i < len(ga) and i < len(gb) and ga[i] == gb[i] else "FAIL"
    if idn == "FAIL": fails.append(name)
    rows.append(f"{name}\t{ta[i]:.2f}\t{tb[i]:.2f}\t{tb[i]/ta[i]:.3f}\t{idn}")
(o / "results.txt").write_text("\n".join(rows) + "\n"); print("\n".join(rows))
sp = lambda x: (max(x) - min(x)) / st.median(x) * 100
print(f"{la} median {st.median(ta):.2f} tok/s_gen spread {sp(ta):.1f}%  |  {lb} median {st.median(tb):.2f} spread {sp(tb):.1f}%  |  ratio {st.median(tb)/st.median(ta):.3f}  |  identity fails: {fails or 'none'}")
PY
cat "$out/arm.txt"
