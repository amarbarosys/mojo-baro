#!/usr/bin/env bash
# usage: bench/check-cpu-engine.sh [OUT]    env QUICK=N prompts (default 20), NTOK=K forced ids per prompt (default 64),
#        MIN_PCT=P bar per prompt (default 100). Step 3 = QUICK=3 NTOK=1, step 5 = defaults (docs/lanes/2026-11-10-cpu-tier.md).
# Forced agreement of cpu/engine_cpu.mojo against the GPU engine on bench/mtp-prompts/*.tokens. GPU reference: built with
# -D BARO_SHARED_EXP=1 (read back from its ISA: no v_rsq_f32 in the SSM conv kernel), run with BARO_MEGA=0 BARO_PREFILL=0
# BARO_REPLAY_M=1 BARO_SPEC=0 (read back from its log). Arms: GPU greedy ids (ref), GPU forced on its own ids (control, must
# read 100%), CPU forced (candidate, every prompt >= MIN_PCT), CPU mutant with gate and up swapped in the FFN (must fail).
# PASS only if all four hold. Receipts in OUT: arm.txt, *.log, agree.txt.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-.work/cpu-engine}; QUICK=${QUICK:-20}; NTOK=${NTOK:-64}; MIN_PCT=${MIN_PCT:-100}; TMAX=16384
MOJO=${MOJO:-./.venv/bin/mojo}
PACK=$(readlink -f "${PACK:-$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.work/engine-pack-q4}")
CAP="systemd-run --user --scope -q -p CPUQuota=150% -p MemoryMax=10G -p MemorySwapMax=0"
export MODULAR_CRASH_REPORTING_ENABLED=false
fail() { echo "FAIL $1: $2"; exit 1; }
cool() { while [ "$(sensors | awk '/Tctl/{gsub(/[^0-9.]/,"",$2); print int($2)}')" -ge 84 ]; do echo "Tctl >= 84 C, waiting"; sleep 30; done; }
mkdir -p "$OUT/mut"; [ -f "$PACK/pack.bin" ] || fail pack "$PACK/pack.bin missing"
sed 's/swiglu(self.f\[OG + j\], self.f\[OU + j\])/swiglu(self.f[OU + j], self.f[OG + j])/' cpu/engine_cpu.mojo >"$OUT/mut/engine_cpu.mojo"
[ "$(diff cpu/engine_cpu.mojo "$OUT/mut/engine_cpu.mojo" | grep -c '^>')" -eq 1 ] || fail mutant "sed did not change exactly 1 line"
rm -f "$OUT/eng-gpu" "$OUT/engine_cpu" "$OUT/mut/engine_cpu"
cool; $CAP timeout 1500 "$MOJO" build -D BARO_SHARED_EXP=1 serve/engine.mojo -I . -I kernels -o "$OUT/eng-gpu" >"$OUT/build-gpu.log" 2>&1 || fail build "$OUT/build-gpu.log"
cool; $CAP timeout 900 "$MOJO" build -I . -I kernels -I cpu -I serve cpu/engine_cpu.mojo -o "$OUT/engine_cpu" >"$OUT/build-cpu.log" 2>&1 || fail build "$OUT/build-cpu.log"
cool; $CAP timeout 900 "$MOJO" build -I . -I kernels -I cpu -I serve "$OUT/mut/engine_cpu.mojo" -o "$OUT/mut/engine_cpu" >"$OUT/build-mut.log" 2>&1 || fail build "$OUT/build-mut.log"
rm -rf "$OUT/isa"; timeout 900 python3 tools/isa-receipt.py "$OUT/eng-gpu" --dump "$OUT/isa" >"$OUT/isa.txt" 2>&1 || fail isa "$OUT/isa.txt"
cs=$(grep -l '<ssm_amar_ssm_conv_l2g' "$OUT"/isa/*.s | head -1) || fail isa "no conv_l2g kernel in the GPU engine"
nrsq=$(awk '/<ssm_amar_ssm_conv_l2g/{p=1} /^$/{if(p)n++} p&&n<2' "$cs" | grep -c 'v_rsq_f32' || true)
[ "$nrsq" -eq 0 ] || fail shared_exp "GPU engine conv_l2g has $nrsq v_rsq_f32: not a BARO_SHARED_EXP=1 build"
ls bench/mtp-prompts/*.tokens | head -n "$QUICK" >"$OUT/prompts.txt"
[ "$(wc -l <"$OUT/prompts.txt")" -eq "$QUICK" ] || fail prompts "fewer than $QUICK prompt files"
k=0; : >"$OUT/req-ref.jsonl"
while read -r f; do k=$((k + 1)); echo "{\"id\":$k,\"prompt\":[$(tr -s ' \n' ',' <"$f" | sed 's/^,//; s/,$//')],\"n\":$NTOK,\"spec\":false}" >>"$OUT/req-ref.jsonl"; done <"$OUT/prompts.txt"
{ echo "pack=$PACK sha16=$(head -c 1048576 "$PACK/pack.bin" | sha256sum | cut -c1-16) quick=$QUICK ntok=$NTOK min_pct=$MIN_PCT tmax=$TMAX"
  for b in eng-gpu engine_cpu mut/engine_cpu; do echo "$b sha256=$(sha256sum "$OUT/$b" | cut -c1-16)"; done
  echo "gpu env: BARO_SERVE=1 BARO_PACK=$PACK BARO_CKPT=0 BARO_MEGA=0 BARO_PREFILL=0 BARO_REPLAY_M=1 BARO_SPEC=0 BARO_TMAX=$TMAX; build -D BARO_SHARED_EXP=1 (ISA: conv_l2g v_rsq_f32 $nrsq)"
  echo "commit=$(git rev-parse --short HEAD) dirty='$(git status --short -- cpu kernels serve bench | tr '\n' ' ')'"; } | tee "$OUT/arm.txt"
GENV="env BARO_SERVE=1 BARO_PACK=$PACK BARO_CKPT=0 BARO_MEGA=0 BARO_PREFILL=0 BARO_REPLAY_M=1 BARO_SPEC=0 BARO_TMAX=$TMAX MODULAR_CRASH_REPORTING_ENABLED=false"
gpu() { gpu-wait run --timeout 900 -- bash -c "$GENV $PWD/$OUT/eng-gpu < $PWD/$OUT/$1.jsonl > $PWD/$OUT/$1.log 2>&1" || true; }
readback() {
  grep -q '^BARO_SERVE: True' "$1" || fail readback "$1: no 'BARO_SERVE: True'"
  grep -q "\"pack\":\"$PACK\"" "$1" || fail readback "$1: ready line does not name $PACK"
  grep -q '^BARO_MEGA: False' "$1" || fail readback "$1: no 'BARO_MEGA: False'"
  grep -q '^BARO_PREFILL: False' "$1" || fail readback "$1: no 'BARO_PREFILL: False'"
  grep -q '^BARO_REPLAY_M: 1 ' "$1" || fail readback "$1: no 'BARO_REPLAY_M: 1'"
  grep -q "\"ready\":true,\"tmax\":$TMAX," "$1" || fail readback "$1: no ready line with tmax $TMAX"
  grep -q 'mtp: drafted [1-9]' "$1" && fail readback "$1: spec drafted tokens on a no-spec arm" || true
}
gpu req-ref; readback "$OUT/req-ref.log"
: >"$OUT/req-force.jsonl"
for i in $(seq 1 "$QUICK"); do
  ids=$(grep -o "^{\"id\":$i,\"tok\":[0-9]*}" "$OUT/req-ref.log" | sed 's/.*"tok"://; s/}//' | paste -sd, -)
  [ "$(echo "$ids" | tr ',' '\n' | grep -c .)" -eq "$NTOK" ] || fail ref "prompt $i: expected $NTOK GPU ids, got '$ids'"
  sed -n "${i}p" "$OUT/req-ref.jsonl" | sed "s/,\"spec\":false}/,\"spec\":false,\"force\":[$ids]}/" >>"$OUT/req-force.jsonl"
done
gpu req-force; readback "$OUT/req-force.log"
cpu() { cool; BARO_TMAX=$TMAX timeout 7200 "$OUT/$1" "$PACK" <"$OUT/req-force.jsonl" >"$OUT/$2.log" 2>&1 || true; }
cpu engine_cpu cpu
grep -q "\"ready\":true,\"tmax\":$TMAX,\"mrows\":1,\"kmax\":0,\"spec_k\":0,\"kv\":\"float32\",\"device\":\"cpu\"" "$OUT/cpu.log" || fail readback "cpu.log: no cpu ready line"
# P28: the mutant arm answers one question (is it caught?), so it runs one prompt at a time and stops at the first miss.
: >"$OUT/cpu-mut.log"; m=0
while read -r req; do
  m=$((m + 1)); cool
  echo "$req" | BARO_TMAX=$TMAX timeout 7200 "$OUT/mut/engine_cpu" "$PACK" >"$OUT/cpu-mut.one" 2>&1 || true
  cat "$OUT/cpu-mut.one" >>"$OUT/cpu-mut.log"
  grep '^forced agreement:' "$OUT/cpu-mut.one" | awk -v min="$MIN_PCT" '{exit !(100 * $3 / $5 < min)}' && break
done <"$OUT/req-force.jsonl"
grep '^forced agreement:' "$OUT/cpu-mut.log" | awk -v m="$m" '{pct = 100 * $3 / $5; printf "cpu-mut prompt %d agree %d / %d %.2f%% %s\n", NR, $3, $5, pct, (pct >= '"$MIN_PCT"' ? "ok" : "UNDER")}' >"$OUT/mut.txt"
: >"$OUT/agree.txt"
for arm in req-force cpu; do
  got=$(grep -c '^forced agreement:' "$OUT/$arm.log" || true)
  [ "$got" -eq "$QUICK" ] || fail incomplete "$arm: $got of $QUICK forced agreement lines, see $OUT/$arm.log"
  grep '^forced agreement:' "$OUT/$arm.log" | awk -v arm="$arm" -v min="$([ "$arm" = req-force ] && echo 100 || echo "$MIN_PCT")" '{pct = 100 * $3 / $5; printf "%s prompt %d agree %d / %d %.2f%% %s\n", arm, NR, $3, $5, pct, (pct >= min ? "ok" : "UNDER")}' >>"$OUT/agree.txt"
done
cat "$OUT/mut.txt" >>"$OUT/agree.txt"
cat "$OUT/agree.txt"
grep -q '^req-force .* UNDER' "$OUT/agree.txt" && fail control "GPU forced on its own ids is under 100%: instrument void"
grep -q '^cpu-mut .* UNDER' "$OUT/agree.txt" || fail mutant "mutant engine passed every prompt"
grep -q '^cpu .* UNDER' "$OUT/agree.txt" && fail agreement "$(grep '^cpu .* UNDER' "$OUT/agree.txt" | tr '\n' ';')"
echo "PASS cpu-engine: $QUICK prompts x $NTOK forced ids, cpu == gpu >= $MIN_PCT% on every prompt, control 100%, mutant caught"
