#!/usr/bin/env bash
# usage: bench/force-ab.sh ENGINE_REF ENGINE_CAND OUTDIR [ENV] [ENV_CAND]
# ENV applies to both arms; ENV_CAND replaces it for the candidate arm only.
# Teacher-forced identity gate between two builds (CLAUDE.md: never greedy equality
# past ~256 ids). For every bench/mtp-prompts/*.tokens: ENGINE_REF runs greedy, its
# GENERATED ids become BARO_FORCE for ENGINE_CAND, and the candidate's
# "forced agreement: a / n" line is recorded. Fail word read on every run.
set -uo pipefail
cd "$(dirname "$0")/.."
ref=$1; cand=$2; out=$3; envx=${4:-}; envc=${5:-$envx}
mkdir -p "$out"
ha=$(sha256sum "$ref" | cut -c1-16); hb=$(sha256sum "$cand" | cut -c1-16)
if [ "$ha" = "$hb" ]; then
  echo "REFUSED: ref and cand are the same binary ($ha)" >&2; exit 2
fi
echo "ref=$ref cand=$cand shaRef=$ha shaCand=$hb env='$envx' envCand='$envc'" | tee "$out/arm.txt"
echo "prompt agree checked pct ref_gen cand_gen" > "$out/results.txt"
for tf in bench/mtp-prompts/p*.tokens; do
  p=$(basename "$tf" .tokens)
  env BARO_PROMPT="$tf" BARO_SPEC=0 $envx "$ref" > "$out/$p.ref.log" 2>&1
  grep '^GENERATED:' "$out/$p.ref.log" | sed 's/^GENERATED://' > "$out/$p.ids"
  if [ ! -s "$out/$p.ids" ]; then echo "$p 0 0 VOID(ref)" >> "$out/results.txt"; continue; fi
  env BARO_PROMPT="$tf" BARO_SPEC=0 BARO_FORCE="$out/$p.ids" $envc "$cand" > "$out/$p.cand.log" 2>&1
  fa=$(grep -oE 'forced agreement: [0-9]+ / [0-9]+' "$out/$p.cand.log" | grep -oE '[0-9]+' | tr '\n' ' ')
  set -- $fa
  if [ -z "${1:-}" ] || [ "${2:-0}" = 0 ]; then echo "$p 0 0 VOID(cand)" >> "$out/results.txt"; continue; fi
  gr=$(grep -oE 'mega barrier gen: [0-9]+' "$out/$p.ref.log" | grep -oE '[0-9]+$'); gc=$(grep -oE 'mega barrier gen: [0-9]+' "$out/$p.cand.log" | grep -oE '[0-9]+$')
  echo "$p $1 $2 $(python3 -c "print(f'{100*$1/$2:.1f}')") ${gr:--} ${gc:--}" >> "$out/results.txt"
done
column -t "$out/results.txt"
python3 - "$out/results.txt" <<'PY'
import sys
rows=[l.split() for l in open(sys.argv[1]).read().splitlines()[1:]]
void=[r[0] for r in rows if r[3].startswith("VOID")]
ok=[r for r in rows if not r[3].startswith("VOID")]
pct=[float(r[3]) for r in ok]
print(f"prompts {len(ok)}/{len(rows)}  min {min(pct) if pct else 'nan'}%  mean {f'{sum(pct)/len(pct):.1f}%' if pct else 'nan'}  void: {void or 'none'}")
if void or len(ok) != len(rows) or (pct and min(pct) < 100.0):
    sys.exit(1)
PY
