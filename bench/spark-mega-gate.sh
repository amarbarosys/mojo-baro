#!/usr/bin/env bash
# spark-mega-gate: the spark megakernel decode path (BARO_SP_MEGA=1, kernels/spark_mega.mojo) against the launch path.
#   bench/spark-mega-gate.sh ENGINE PACKDIR REFDIR OUT [MUTANT_ENGINE]        (one gpu-wait job; the script queues itself)
# REFDIR holds pNN-*.ids (prompt ids) and pNN-*.llama-tokens.txt (reference tokens of the pinned llama.cpp oracle, same GGUF).
# Per prompt, real engine, 64 tokens:
# 1 bytes     teacher-forced run of both paths with BARO_SP_BYTES: every K and V row of all layers, the last logits row and
#             the last x must be byte-equal (cmp), and the predicted ids equal.
# 2 agreement forced agreement vs the oracle tokens; the launch path on the same prompt is the same-arm control. Bar frozen
#             before the first run: megakernel agreement == launch agreement on every prompt (it follows from 1).
# 3 speed     no-spec tok/s_gen of both paths, order flipped per prompt; median, min, max over the prompts (P4).
# 4 mutant    MUTANT_ENGINE (-D BARO_SM_MUTANT=1) on the first prompt must FAIL the byte test (stops there, P28).
# Every megakernel run: read-back "BARO_SP_MEGA: 1", launches >= 1, fail word 0; else FAIL VOID.
# QUICK=N: first N prompts (a signal, never a verdict). MEGA_N: BARO_SP_MEGA_N (0 = all tokens in one launch).
# Writes OUT/arm.txt, OUT/rows.tsv "prompt bytes agree_mega agree_launch toks_mega toks_launch", OUT/summary.txt.
set -euo pipefail
fail() { echo "FAIL $1: $2" | tee -a "$O/summary.txt"; exit 1; }
E=$(realpath "${1:?ENGINE}"); P=$(realpath "${2:?PACKDIR}"); REF=$(realpath "${3:?REFDIR}"); O=$(realpath -m "${4:?OUT}"); MU=${5:+$(realpath "$5")}
quick=${QUICK:-0}; megan=${MEGA_N:-0}
mkdir -p "$O"
if [ "${INJOB:-0}" = 0 ]; then
  np=$(ls "$REF"/p*.ids | wc -l); [ "$quick" -gt 0 ] && np=$quick
  echo "job: $np prompts x 4 engine runs (about 2 s each) + mutant"
  : > "$O/summary.txt"
  gpu-wait run --vram 4 --mem 8 --cores 4 --timed --timeout ${TIMEOUT:-600} --label spark-mega-gate -- env INJOB=1 QUICK="$quick" MEGA_N="$megan" \
    bash "$(realpath "$0")" "$E" "$P" "$REF" "$O" ${MU:+"$MU"} > "$O/job.log" 2>&1 || { tail -5 "$O/job.log"; fail job "see $O/job.log"; }
  cat "$O/summary.txt"; exit 0
fi
{ echo "engine=$E sha=$(sha256sum "$E" | cut -c1-16) pack=$P pack_sha=$(sha256sum "$P/pack.bin" | cut -c1-16) mutant=${MU:-none}"
  echo "quick=$quick mega_n=$megan gen=64 ref=$REF"
  echo "power_cap_uW=$(cat /sys/class/drm/card1/device/hwmon/hwmon*/power1_cap 2>/dev/null || echo unread)"; } > "$O/arm.txt"
run() { # run NAME MEGA PROMPT [K=V...]: one engine process, log OUT/NAME.log
  local n=$1 m=$2 p=$3; shift 3
  env BARO_PROMPT="$REF/$p.ids" BARO_PACK="$P" BARO_GEN=64 BARO_SPEC=0 BARO_SP_MEGA="$m" BARO_SP_MEGA_N="$megan" "$@" "${BIN:-$E}" > "$O/$n.log" 2>&1 || fail run "$n exit $? (see $O/$n.log)"
  if [ "$m" = 1 ]; then
    grep -q '^BARO_SP_MEGA: 1 ' "$O/$n.log" || fail VOID "$n: megakernel not active (read-back)"
    local l; l=$(sed -n 's/^spark mega launches: \([0-9]*\) .*/\1/p' "$O/$n.log"); [ "${l:-0}" -ge 1 ] || fail VOID "$n: 0 megakernel launches"
    grep -q 'fail word: 0 *$' "$O/$n.log" || fail residency "$n: fail word not 0 (see $O/$n.log)"
  else
    grep -q '^BARO_SP_MEGA: 0 ' "$O/$n.log" || fail VOID "$n: launch arm ran with the megakernel on"
  fi
}
: > "$O/rows.tsv"; bad=0; i=0
for f in "$REF"/p*.ids; do p=$(basename "$f" .ids); i=$((i + 1)); if [ "$quick" -gt 0 ] && [ $i -gt "$quick" ]; then break; fi
  [ -s "$REF/$p.llama-tokens.txt" ] || fail ref "$p: no reference tokens"
  run "$p.launch-forced" 0 "$p" BARO_FORCE="$REF/$p.llama-tokens.txt" BARO_SP_BYTES="$O/$p.launch.bin"
  run "$p.mega-forced" 1 "$p" BARO_FORCE="$REF/$p.llama-tokens.txt" BARO_SP_BYTES="$O/$p.mega.bin"
  if cmp -s "$O/$p.launch.bin" "$O/$p.mega.bin" && [ -s "$O/$p.mega.bin" ] && [ "$(grep '^predicted:' "$O/$p.launch-forced.log")" = "$(grep '^predicted:' "$O/$p.mega-forced.log")" ]; then b="equal:$(stat -c %s "$O/$p.mega.bin")"; else b=DIFF; bad=1; fi
  am=$(sed -n 's/^forced agreement: //p' "$O/$p.mega-forced.log" | tr -d ' '); al=$(sed -n 's/^forced agreement: //p' "$O/$p.launch-forced.log" | tr -d ' ')
  if [ -z "$am" ] || [ "$am" != "$al" ]; then bad=1; fi
  if [ $((i % 2)) = 1 ]; then run "$p.mega" 1 "$p"; run "$p.launch" 0 "$p"; else run "$p.launch" 0 "$p"; run "$p.mega" 1 "$p"; fi
  tm=$(sed -n 's/^tok\/s_gen: \([0-9.]*\) .*/\1/p' "$O/$p.mega.log"); tl=$(sed -n 's/^tok\/s_gen: \([0-9.]*\) .*/\1/p' "$O/$p.launch.log")
  if [ "$(grep '^generated:' "$O/$p.mega.log")" != "$(grep '^generated:' "$O/$p.launch.log")" ]; then b="$b,GEN-DIFF"; bad=1; fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$p" "$b" "$am" "$al" "$tm" "$tl" >> "$O/rows.tsv"
  rm -f "$O/$p.launch.bin" "$O/$p.mega.bin"
done
n=$(wc -l < "$O/rows.tsv"); [ "$n" -ge 1 ] || fail cases "0 prompts ran"
mres="not run"
if [ -n "$MU" ]; then p=$(basename "$(ls "$REF"/p*.ids | sed -n 1p)" .ids)
  run "$p.launch-forced" 0 "$p" BARO_FORCE="$REF/$p.llama-tokens.txt" BARO_SP_BYTES="$O/mut.launch.bin"
  BIN=$MU run "mutant" 1 "$p" BARO_FORCE="$REF/$p.llama-tokens.txt" BARO_SP_BYTES="$O/mut.mega.bin"
  if cmp -s "$O/mut.launch.bin" "$O/mut.mega.bin"; then mres="FAIL mutant: a changed FFN epilogue left the bytes equal (the byte test is blind)"; bad=1; else mres="PASS mutant differs from the launch path"; fi
  rm -f "$O/mut.launch.bin" "$O/mut.mega.bin"
fi
med() { sort -n | awk '{a[NR] = $1} END {if (NR == 0) {print "none"; exit}; printf "median %.0f min %.0f max %.0f", (a[int((NR + 1) / 2)] + a[int(NR / 2) + 1]) / 2, a[1], a[NR]}'; }
{ cat "$O/arm.txt"
  echo "bytes: $(awk -F'\t' '$2 ~ /^equal:[0-9]+$/' "$O/rows.tsv" | wc -l) of $n prompts byte-equal (K and V rows of all layers, last logits, last x; predicted ids)"
  echo "agreement vs the oracle tokens, megakernel == launch on $(awk -F'\t' '$3 != "" && $3 == $4' "$O/rows.tsv" | wc -l) of $n prompts"
  echo "tok/s_gen megakernel: $(cut -f5 "$O/rows.tsv" | med) | launch path: $(cut -f6 "$O/rows.tsv" | med) | $n prompts, 64 tokens, mega_n=$megan"
  echo "mutant: $mres"; } >> "$O/summary.txt"
cat "$O/summary.txt"
if [ $bad = 0 ]; then if [ "$quick" -gt 0 ]; then echo "QUICK ok ($n prompts): a signal, not a verdict" | tee -a "$O/summary.txt"; else echo "PASS spark-mega-gate" | tee -a "$O/summary.txt"; fi; else fail gate "see $O/rows.tsv"; fi
