#!/usr/bin/env bash
# usage: bench/force-ab-serve.sh ENGINE_REF ENGINE_CAND OUTDIR [ENV]
# Same gate as bench/force-ab.sh (teacher-forced identity between two builds),
# but the REFERENCE side runs as ONE resident engine (BARO_SERVE=1, JSON lines
# on stdin, serve/PROTOCOL.md) instead of one cold process per prompt.
#
# Why only the reference side: BARO_FORCE is read once at startup
# (serve/engine.mojo:487) while the request loop begins at 527, so the forced
# ids are process-global and cannot vary per request. The candidate therefore
# still runs one process per prompt. Adding a "force":[INT,...] field to the
# request line would make both sides resident and cut this gate by about 60%
# rather than 30%; that is an engine plus PROTOCOL.md change, not a harness one.
#
# Measured baseline (2026-09-12, q4 pack, 20 prompts): bench/force-ab.sh is
# 54 s wall for 40 runs, of which about 0.43 s pack load plus 0.4 s process
# start per run. This script removes 19 of the 20 reference loads.
set -uo pipefail
cd "$(dirname "$0")/.."
ref=$1; cand=$2; out=$3; envx=${4:-}
mkdir -p "$out"
ha=$(sha256sum "$ref" | cut -c1-16); hb=$(sha256sum "$cand" | cut -c1-16)
if [ "$ha" = "$hb" ]; then
  echo "REFUSED: ref and cand are the same binary ($ha)" >&2; exit 2
fi
echo "ref=$ref cand=$cand shaRef=$ha shaCand=$hb env='$envx' mode=resident-ref" | tee "$out/arm.txt"

# --- reference side: one process, every prompt, ids straight from the protocol
: > "$out/ref.jsonl"
id=0
for tf in bench/mtp-prompts/p*.tokens; do
  id=$((id + 1))
  p=$(tr -s ' \n' ',' < "$tf" | sed 's/,$//')
  echo "{\"id\":$id,\"prompt\":[$p],\"n\":64,\"spec\":false}"
done > "$out/ref-requests.jsonl"
env BARO_SERVE=1 BARO_SPEC=0 $envx "$ref" < "$out/ref-requests.jsonl" > "$out/ref.out" 2>"$out/ref.err"
grep '^{' "$out/ref.out" > "$out/ref.jsonl" || true
if ! grep -q '"ready":true' "$out/ref.jsonl"; then
  echo "VOID: reference engine never printed its ready line; see $out/ref.err" >&2; exit 3
fi

# one ids file per prompt, in request order, from the {"id","tok"} lines
python3 - "$out" <<'PY'
import json, sys, pathlib
out = pathlib.Path(sys.argv[1])
toks = {}
for line in (out / "ref.jsonl").read_text().splitlines():
    try: d = json.loads(line)
    except Exception: continue
    if "tok" in d and "id" in d:
        toks.setdefault(d["id"], []).append(d["tok"])
for i, ids in sorted(toks.items()):
    (out / f"ref-{i:02d}.ids").write_text(" ".join(str(t) for t in ids) + "\n")
print(f"reference: {len(toks)} prompts, {sum(len(v) for v in toks.values())} tokens")
PY

# --- candidate side: one process per prompt, because BARO_FORCE is global
echo "prompt agree checked pct" > "$out/results.txt"
id=0
for tf in bench/mtp-prompts/p*.tokens; do
  id=$((id + 1)); p=$(basename "$tf" .tokens); ids="$out/ref-$(printf '%02d' $id).ids"
  if [ ! -s "$ids" ]; then echo "$p 0 0 VOID(ref)" >> "$out/results.txt"; continue; fi
  env BARO_PROMPT="$tf" BARO_SPEC=0 BARO_FORCE="$ids" $envx "$cand" > "$out/$p.cand.log" 2>&1
  fa=$(grep -oE 'forced agreement: [0-9]+ / [0-9]+' "$out/$p.cand.log" | grep -oE '[0-9]+' | tr '\n' ' ')
  set -- $fa
  if [ -z "${1:-}" ] || [ "${2:-0}" = 0 ]; then echo "$p 0 0 VOID(cand)" >> "$out/results.txt"; continue; fi
  echo "$p $1 $2 $(python3 -c "print(f'{100*$1/$2:.1f}')")" >> "$out/results.txt"
done
column -t "$out/results.txt"
python3 - "$out/results.txt" <<'PY'
import sys
rows = [l.split() for l in open(sys.argv[1]).read().splitlines()[1:]]
void = [r[0] for r in rows if r[3].startswith("VOID")]
ok = [r for r in rows if not r[3].startswith("VOID")]
pct = [float(r[3]) for r in ok]
print(f"prompts {len(ok)}/{len(rows)}  min {min(pct) if pct else 'nan'}%  mean {sum(pct)/len(pct) if pct else float('nan'):.1f}%  void: {void or 'none'}")
# A void is a FAILED arm, not a skipped row. Averaging over the survivors
# printed "prompts 1/20 min 100.0% mean 100.0%" on this script's first ever
# run, which reads as a pass while 19 of 20 requests had been eaten.
if void:
    print(f"FAIL: {len(void)} void arm(s) -- this is not a pass")
    sys.exit(1)
PY
