#!/usr/bin/env bash
# usage: bench/ailab/own-identity-c0.sh OUT      env: DRY=1 (stand-in engine, CPU), SEL, TMAX, SEQS, COND (1 = run Q0 and M0 when S and Q differ), ARMS
#        (override list, e.g. "S Q Q0 M0"); QUEUE=stub (dry only: re-exec through a fake queue)
# P4d, a DIAGNOSTIC (no preflight, never a gate verdict, label UNVERIFIED): P4c saw HEAD seqs vs HEAD single differ on 14 of 16 units at equal `cached`.
# Candidate C0: equal `cached` is not equal checkpoint state, because the burst's creator chunks from order-dependent hints. This job drives each
# engine DIRECTLY on its wire, ONE request in flight, the same creator request first (n=1, ckpt hint 3569), then the scored units in order, so both
# engines restore the same prefix state. S = HEAD single-session engine, Q = HEAD seqs engine (BARO_SEQS=$SEQS, KV_SHARE 1, KV_POOL 3).
# Conditional arms, only when S vs Q still differ: Q0 = Q with BARO_SEQS_ROWS=0 (C1), M0 = S with BARO_MEGA=0 (C2).
# Verdict: ownid-compare lines S vs Q (and S vs Q0, S vs M0, b vs S, a1 vs Q against the P4c burst receipts).
set -euo pipefail
cd "$(dirname "$0")/../.."
out=${1:?usage: own-identity-c0.sh OUT}; dry=${DRY:-0}; export DRY=$dry
case $out in /*) ;; *) out=$PWD/$out ;; esac
GW=$HOME/.local/bin/gpu-wait; self=$(realpath "$0")
sel=${SEL:-.work/ailab/p4c}; tmax=${TMAX:-24576}; seqs=${SEQS:-20}; cond=${COND:-1}; arms=${ARMS:-}
fwd=(); for k in DRY SEL TMAX SEQS COND ARMS STUB_ENGINE STUB_FLIP STUB_DEAD STUB_BADSEQS; do if [ -n "${!k+x}" ]; then fwd+=("$k=${!k}"); fi; done
if [ "${GPU_WAITING_ROOM_JOB:-}" = stub ] && [ "$dry" != 1 ]; then echo "FAIL queue: the fake queue marker without DRY=1 would run a real GPU job bare"; exit 1; fi
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ] && [ "${QUEUE:-}" = stub ] && [ "$dry" = 1 ]; then
  exec env -i PATH="$PATH" HOME="$HOME" GPU_WAITING_ROOM_JOB=stub "${fwd[@]}" "$self" "$out"
fi
if [ "$dry" != 1 ] && [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
  exec "$GW" run --timeout 1500 --vram 21 --label ailab-own-identity-c0 -- env "${fwd[@]}" "$self" "$out"
fi
mkdir -p "$out"
pack=${PACK:-.work/ailab/pack-kq}
if [ "$dry" = 1 ]; then
  eng_s=${STUB_ENGINE:-bench/stub-engine.py}; eng_q=$eng_s; export FAKE_OUT=$out/stub-engine-requests.jsonl STUB_CACHED=3569
else
  eng_s=.work/ailab/engine-main-head; eng_q=.work/ailab/engine-seqs32s2-head
  [ -f .work/ailab/engines-head.txt ] || { echo "FAIL input: no .work/ailab/engines-head.txt (run build-engines-head.sh)"; exit 1; }
  bc=$(sed -n 's/^commit=\([0-9a-f]*\).*/\1/p' .work/ailab/engines-head.txt)
  git diff --quiet "$bc" HEAD -- serve kernels uregex minja || { echo "FAIL input: engine sources changed since the build at $bc"; exit 1; }
  for n in seqs32s2 main; do
    want=$(sed -n "s/^engine-$n-head sha256=//p" .work/ailab/engines-head.txt); got=$(sha256sum ".work/ailab/engine-$n-head" | cut -c1-16)
    [ "$want" = "$got" ] || { echo "FAIL input: engine-$n-head sha $got is not the recorded $want"; exit 1; }
  done
fi
for f in "$sel/scored.txt" "$sel/units" "$sel/units/s01-r1.ref.json" "$sel/roles.txt"; do [ -e "$f" ] || { echo "FAIL input: missing $f"; exit 1; }; done
{ echo "gate=own-identity-c0 (P4d diagnostic, UNVERIFIED, no preflight) dry=$dry commit=$(git rev-parse --short HEAD) engines_built_at=${bc:-stub} job=${GPU_WAITING_ROOM_JOB:-none} tmax=$tmax seqs=$seqs cond=$cond"
  echo "engine_single=$eng_s sha=$(sha256sum "$eng_s" | cut -c1-16) engine_seqs=$eng_q sha=$(sha256sum "$eng_q" | cut -c1-16) pack=$pack"
  echo "units: $(wc -l < "$sel/scored.txt") scored, creator s01-r1 hint 3569 n=1, then scored.txt order, one request in flight"; } | tee "$out/arm.txt"
common=(BARO_PACK="$pack" BARO_TMAX="$tmax" BARO_SPEC=0 BARO_MOE_DOT=1 BARO_TIER_ZC=1 BARO_TIER_PINNED=1 BARO_PREFILL=1 MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=2)
if [ "$dry" = 1 ]; then common+=(FAKE_OUT="$out/stub-engine-requests.jsonl" STUB_CACHED=3569 STUB_FLIP="${STUB_FLIP:-0}" STUB_DEAD="${STUB_DEAD:-0}" STUB_BADSEQS="${STUB_BADSEQS:-0}"); fi
declare -A ENG=([S]=$eng_s [Q]=$eng_q [Q0]=$eng_q [M0]=$eng_s)
declare -A EXTRA=([S]="BARO_SERVE=1" [M0]="BARO_SERVE=1 BARO_MEGA=0"
  [Q]="BARO_SERVE=1 BARO_SEQS=$seqs BARO_KV_SHARE=1 BARO_KV_POOL=3" [Q0]="BARO_SERVE=1 BARO_SEQS=$seqs BARO_KV_SHARE=1 BARO_KV_POOL=3 BARO_SEQS_ROWS=0")
PL="python3 bench/ailab/parity_llama.py"
cur=""; wire0=""
cleanup() { if [ -n "$cur" ]; then kill -TERM "$cur" 2> /dev/null || :; sleep 1; kill -KILL "$cur" 2> /dev/null || :; fi; }
trap cleanup EXIT
run_arm() {  # name: drive the engine over its wire, read every knob back from its own echo
  local a=$1 e=() kv
  for kv in "${common[@]}" ${EXTRA[$a]}; do e+=(--env "$kv"); done
  echo "arm $a: engine=${ENG[$a]} env: ${common[*]} ${EXTRA[$a]}" | tee -a "$out/arm.txt"
  python3 bench/ailab/wire-drive.py --engine "${ENG[$a]}" --units "$sel/units" --order "$sel/scored.txt" --creator s01-r1 --out "$out/$a" "${e[@]}" > "$out/drive-$a.log" 2>&1 \
    || { tail -5 "$out/drive-$a.log"; echo "FAIL drive-$a: see $out/drive-$a.log"; exit 1; }
  tail -1 "$out/drive-$a.log"
  local echo_f=$out/$a/engine-echo.txt
  if true; then  # read-back (dry run too, so a stand-in reaches the same checks): the engine's own lines must show the arm
    case $a in
      S | M0) if grep -q 'BARO_SEQS' "$echo_f"; then grep -q 'BARO_SEQS: *1$' "$echo_f" || { echo "FAIL readback-$a: single-session arm echoes a BARO_SEQS other than 1 ($echo_f)"; exit 1; }; fi ;;
      Q | Q0) grep -q "BARO_SEQS: *$seqs" "$echo_f" || { echo "FAIL readback-$a: no 'BARO_SEQS: $seqs' echo in $echo_f"; exit 1; } ;;
    esac
    case $a in M0) grep -q 'BARO_MEGA: *False' "$echo_f" || { echo "FAIL readback-$a: no 'BARO_MEGA: False' echo in $echo_f"; exit 1; } ;; esac
    grep -q "\"tmax\": *$tmax" "$echo_f" || { echo "FAIL readback-$a: ready line tmax is not $tmax ($echo_f)"; exit 1; }
  fi
  wsha=$(cat "$out/$a/wire.sha"); echo "wire $a: request lines sha $wsha" | tee -a "$out/arm.txt"
  if [ -n "${wire0:-}" ] && [ "$wsha" != "$wire0" ]; then echo "FAIL wire-$a: request lines differ from the first arm ($wire0 vs $wsha)"; exit 1; fi
  wire0=$wsha
  { echo "engine environ $a: $(tr '\n' ' ' < "$out/$a/engine-environ.txt")"; } | tee -a "$out/arm.txt"
  { echo "readback $a:"; sed 's/^/  /' "$echo_f" | cut -c1-200; } >> "$out/arm.txt"
}
[ -n "$arms" ] || arms="S Q"
for a in $arms; do run_arm "$a"; done
cmp_args() { local s="a1=.work/ailab/p4c-run/a1,b=.work/ailab/p4c-run/b"; for a in S Q Q0 M0; do if [ -d "$out/$a" ]; then s="$s,$a=$out/$a"; fi; done; echo "$s"; }
$PL ownid-compare --scored "$sel/scored.txt" --arms "$(cmp_args)" 2>&1 | grep -E '^(S|Q|b|a1) vs ' | tee "$out/verdict.txt" || :
if [ "$cond" = 1 ] && [ -z "${ARMS:-}" ]; then
  m=$(sed -n 's/^S vs Q: \([0-9]*\) mismatches.*/\1/p' "$out/verdict.txt")
  [ -n "$m" ] || { echo "FAIL compare: no 'S vs Q' line in $out/verdict.txt"; exit 1; }
  comp=$(sed -n 's/^S vs Q: [0-9]* mismatches of \([0-9]*\) comparable.*/\1/p' "$out/verdict.txt")
  if [ "${comp:-0}" -eq 0 ]; then
    echo "S vs Q NOT COMPARABLE: no unit has equal restore points in both arms (see the table); Q0 and M0 not run" | tee -a "$out/verdict.txt"
  elif [ "$m" -gt 0 ]; then
    echo "S vs Q still differs ($m): running the conditional arms Q0 and M0" | tee -a "$out/verdict.txt"
    for a in Q0 M0; do run_arm "$a"; done
  else
    echo "S vs Q equal at every comparable unit: P4c was a gate artifact (pin the creator in both arms from now on); Q0 and M0 not run" | tee -a "$out/verdict.txt"
  fi
fi
$PL ownid-compare --scored "$sel/scored.txt" --arms "$(cmp_args)" > "$out/verdict-full.txt" 2>&1
{ echo "== final pairs"; grep -E '^(S|Q|b|a1) vs ' "$out/verdict-full.txt"; } | tee -a "$out/verdict.txt"
echo "UNVERIFIED: diagnostic, no preflight. per-unit table: $out/verdict-full.txt"
echo "gpu-wait stats: $("$GW" stats --days 1 2> /dev/null | head -3 | tr '\n' ' ' || :)" | tee -a "$out/verdict.txt"
