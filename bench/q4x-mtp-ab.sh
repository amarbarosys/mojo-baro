#!/usr/bin/env bash
# q4x MTP verify window, same-job A/B (lane q4x-mtp, prereg bench/q4x-mtp-prereg.md): ONE binary, one gpu-wait job.
# A = Q4X_SPEC off (no MTP head loaded), B_k = Q4X_MTP + Q4X_SPEC=k for each k in KS (default "2 4"), Q4X_SPEC_P default 0.5.
# Arms rotate per prompt (A K2 K4 / K2 K4 A / K4 A K2), one process per run (resident Q4X_STDIN refuses Q4X_SPEC).
# Read-back (P1) per run: `kv 16384` and the expert-cache slots from the log, `spec rounds ... k K pmin P` for B arms.
# Identity on EVERY run: the `ids` line of each B_k equals A's. A run that fails is retried once and says so.
# usage: bench/q4x-mtp-ab.sh OUT_DIR      env ENGINE=bin KS="2 4" NP=20 NG=64 TMAX=16384 Q4X_SPEC_P=0.5
set -euo pipefail
OUT=${1:?usage: q4x-mtp-ab.sh OUT_DIR}; REPO=$(cd "$(dirname "$0")/.." && pwd); . "$REPO/bench/min16k.sh"
ENGINE=${ENGINE:?ENGINE}; KS=${KS:-2 4}; NP=${NP:-20}; NG=${NG:-64}; TMAX=${TMAX:-16384}; PMIN=${Q4X_SPEC_P:-0.5}
GGUF=${Q4X_GGUF:-$HOME/Models/qwen3.8-flash-next-gsq/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf}
MTP=${MTP_PATH:-$HOME/Models/qwen3.8-flash-next/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf}
WARM=${Q4X_WARM:-$REPO/exchange/receipts/q4x-strata/r3a-warm-l01.txt}
min16k "$TMAX" 16000
if [ -z "${MTP_AB_INNER:-}" ]; then
  mkdir -p "$OUT"
  exec gpu-wait run --vram 20 --timeout "${GPU_TIMEOUT:-5400}" -- env MTP_AB_INNER=1 ENGINE="$ENGINE" KS="$KS" NP="$NP" NG="$NG" TMAX="$TMAX" Q4X_SPEC_P="$PMIN" Q4X_GGUF="$GGUF" MTP_PATH="$MTP" Q4X_WARM="$WARM" HOME="$HOME" PATH="$PATH" bash "$REPO/bench/q4x-mtp-ab.sh" "$OUT"
fi
cd "$REPO"; export MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=2
R=$OUT/receipt.txt
fail() { echo "FAIL $1: $2 (receipt $R)" | tee -a "$R"; exit 1; }
echo "date $(date -Is) commit $(git rev-parse --short HEAD) dirty=$(git status --porcelain -- serve kernels | wc -l) engine=$ENGINE sha256 $(sha256sum "$ENGINE" | cut -c1-16) tmax=$TMAX ng=$NG np=$NP ks='$KS' pmin=$PMIN warm=$(sha256sum "$WARM" | cut -c1-16) mtp=$MTP" | tee "$R"
ARMS="A"; for k in $KS; do ARMS="$ARMS K$k"; done; read -ra AR <<< "$ARMS"
run() { # log prompt-file arm
  local log=$1 ids; ids=$(jq -r 'join(" ")' "$2"); local -a e=(Q4X_TMAX="$TMAX" Q4X_WARM="$WARM")
  [ "$3" = A ] || e+=(Q4X_MTP="$MTP" Q4X_SPEC="${3#K}" Q4X_SPEC_P="$PMIN")
  env "${e[@]}" "$ENGINE" "$GGUF" "$NG" $ids > "$log" 2>&1 && grep -q '^ids ' "$log" && ! grep -qiE 'not resident|exception|assert' "$log" && return 0
  echo "RETRY $(basename "$log"): $(grep -m1 -iE 'not resident|exception|assert|error' "$log" | cut -c1-120)" | tee -a "$R"; mv "$log" "$log.failed"
  env "${e[@]}" "$ENGINE" "$GGUF" "$NG" $ids > "$log" 2>&1 && grep -q '^ids ' "$log" || fail run "$log failed twice"
}
i=0; bad=0
for pf in $(ls bench/q4x-prompts/*.prompt | head -"$NP"); do
  b=$(basename "$pf" .prompt)
  for j in "${!AR[@]}"; do run "$OUT/$b.${AR[$(((j + i) % ${#AR[@]}))]}.log" "$pf" "${AR[$(((j + i) % ${#AR[@]}))]}"; done
  ia=$(grep '^ids ' "$OUT/$b.A.log"); line="$b A $(grep -o 'gen ms/token [0-9.]*' "$OUT/$b.A.log" | awk '{print $3}')"
  for a in "${AR[@]:1}"; do
    l=$OUT/$b.$a.log; grep -q '^spec rounds' "$l" || fail spec "$b $a printed no spec line (MTP path not taken)"
    [ "$(grep '^ids ' "$l")" = "$ia" ] && eq=EQUAL || { eq=DIFF; bad=$((bad + 1)); }
    line="$line | $a $eq $(grep -o 'gen ms/token [0-9.]*' "$l" | awk '{print $3}') $(grep '^spec rounds' "$l" | sed 's/spec rounds \([0-9]*\) k \([0-9]*\) pmin \([0-9.]*\) drafted \([0-9]*\) accepted \([0-9]*\) tokens\/round \([0-9.]*\).*/r\1 k\2 p\3 drafted \4 accepted \5 t\/r \6/')"
  done
  echo "$line | $(grep -m1 -o 'expert cache [0-9]* slots' "$OUT/$b.A.log") A, $(grep -m1 -o 'expert cache [0-9]* slots' "$OUT/$b.${AR[1]}.log") B; kv $(grep -c "kv $TMAX" "$OUT/$b.A.log")" | tee -a "$R"
  i=$((i + 1))
done
awk -v ars="$ARMS" '
function med(arr, n,   t, i) { n = asort(arr, t); return n % 2 ? t[(n + 1) / 2] : (t[n / 2] + t[n / 2 + 1]) / 2 }
/^p[0-9][0-9]-/ { n++; ma[n] = $3; for (f = 4; f <= NF; f++) if ($f == "|") { arm = $(f + 1); v = $(f + 3); mb[arm][n] = v; for (g = f; g <= f + 12; g++) if ($g == "t/r") tr[arm][n] = $(g + 1); for (g = f; g <= f + 12; g++) { if ($g == "drafted") d[arm] += $(g + 1); if ($g == "accepted") ac[arm] += $(g + 1) } } }
END { split(ars, AR, " "); mA = med(ma); printf "A median %.3f ms/token over %d prompts\n", mA, n
  for (k = 2; k in AR; k++) { a = AR[k]; split("", x); split("", y); split("", z); for (i = 1; i <= n; i++) { x[i] = mb[a][i]; y[i] = mb[a][i] / ma[i]; z[i] = tr[a][i] }
    printf "%s median %.3f ms/token | B/A %.4f | paired median ratio %.4f | tokens/round median %.2f | acceptance %.3f (accepted %d of drafted %d)\n", a, med(x), med(x) / mA, med(y), med(z), ac[a] / d[a], ac[a], d[a] } }' "$R" | tee -a "$R"
[ "$bad" -eq 0 ] || fail identity "$bad B runs differ from A"
echo "IDENTITY PASS: every B run equals A on $NP prompts" | tee -a "$R"; echo "job PASS" | tee -a "$R"
