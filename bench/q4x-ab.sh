#!/usr/bin/env bash
# q4x same-job A/B (lane q4x-serve16k): one binary, two env strings, one RESIDENT process per run (Q4X_STDIN), order A B B A.
# usage: bench/q4x-ab.sh OUT_DIR "ENV_A" "ENV_B"      (runs itself under gpu-wait)
#   env  PROMPTS="glob of JSON id arrays" (default bench/q4x-prompts/*.prompt, 20 prompts)  NG=64  ENGINE=bin  TMAX=16384
#        ENGINE_A=bin  arm A runs this binary instead of ENGINE (two-binary A/B: main vs lane, env strings may then be equal)
#        EXTRA="K=V ..." shared by both arms (for 16K prompts: Q4X_CHUNK=8192 Q4X_TMAX=16640 through TMAX/EXTRA)
# Read-back (P1): every arm's `q4x: attsplit` and `kv N` lines are checked against the env string; both binaries' sha256 are printed from
# inside the run (P7); identical env strings are refused. Identity on EVERY run: the ids of each prompt must equal across all four runs.
# Value per prompt = median of that arm's two runs of `gen ms/token`; arm figure = median over prompts with the min-max range (P4).
set -euo pipefail
OUT=${1:?usage: q4x-ab.sh OUT_DIR ENV_A ENV_B}; EA=${2:-}; EB=${3:-}
REPO=$(cd "$(dirname "$0")/.." && pwd); . "$REPO/bench/min16k.sh"; . "$REPO/bench/pcie-link.sh"
GGUF=${Q4X_GGUF:-$HOME/Models/qwen3.8-flash-next-gsq/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf}
ENGINE=${ENGINE:?ENGINE}; ENGINE_A=${ENGINE_A:-$ENGINE}; NG=${NG:-64}; TMAX=${TMAX:-16384}; EXTRA=${EXTRA:-}; PROMPTS=${PROMPTS:-bench/q4x-prompts/*.prompt}
min16k "$TMAX" 16000
[ "$EA" != "$EB" ] || [ "$ENGINE_A" != "$ENGINE" ] || { echo "REFUSED: both arms have the same env string (champion against itself)"; exit 1; }
if [ -z "${Q4X_AB_INNER:-}" ]; then
  mkdir -p "$OUT"
  exec gpu-wait run --vram 20 --timeout "${GPU_TIMEOUT:-3000}" -- env Q4X_AB_INNER=1 ENGINE="$ENGINE" ENGINE_A="$ENGINE_A" NG="$NG" TMAX="$TMAX" EXTRA="$EXTRA" PROMPTS="$PROMPTS" Q4X_GGUF="$GGUF" HOME="$HOME" PATH="$PATH" bash "$REPO/bench/q4x-ab.sh" "$OUT" "$EA" "$EB"
fi
cd "$REPO"
export MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=2
R=$OUT/receipt.txt
fail() { echo "FAIL $1: $2 (receipt $R)" | tee -a "$R"; exit 1; }
{ echo "date $(date -Is) commit $(git rev-parse --short HEAD) dirty=$(git status --porcelain -- serve kernels | wc -l)"
  echo "engineA=$ENGINE_A sha256 $(sha256sum "$ENGINE_A" | cut -c1-16)"
  echo "engine=$ENGINE sha256 $(sha256sum "$ENGINE" | cut -c1-16) tmax=$TMAX ng=$NG extra='$EXTRA'"
  echo "armA='$EA' armB='$EB' order A B B A"; pcie_link; } | tee "$R"
P=$(ls $PROMPTS); NP=$(echo "$P" | wc -l)
for pf in $P; do echo "prompt $(basename "$pf") sha256 $(sha256sum "$pf" | cut -c1-16) ids $(jq length "$pf")" >> "$R"; done
for pf in $P; do echo "$(basename "$pf" .prompt) $NG $(jq -r 'join(" ")' "$pf")"; done > "$OUT/requests.txt"
run() {  # label envstring
  local log=$OUT/$1.log eng=$ENGINE
  [ "${1:0:1}" = A ] && eng=$ENGINE_A
  env Q4X_STDIN=1 Q4X_PREFILL=1 Q4X_TMAX="$TMAX" $EXTRA $2 "$eng" "$GGUF" < "$OUT/requests.txt" > "$log" 2>&1 || fail "$1" "engine failed: $(grep -m1 -iE 'error|fault' "$log" | cut -c1-160)"
  [ "$(grep -c '^out ' "$log")" = "$NP" ] || fail "$1" "$(grep -c '^out ' "$log") of $NP answers"
  local want; want=$(sed -n 's/.*Q4X_ATTSPLIT=\([0-9]*\).*/\1/p' <<< "$2")
  if [ -n "$want" ]; then [ "$(grep -m1 '^q4x: attsplit' "$log" | awk '{print $3}')" = "$want" ] || fail "$1" "read-back attsplit $(grep -m1 '^q4x: attsplit' "$log") != $want"; fi
  for k in misspar devarg; do want=$(sed -n "s/.*Q4X_$(tr a-z A-Z <<< $k)=\([0-9]*\).*/\1/p" <<< "$2"); [ -z "$want" ] || [ "$(grep -m1 "^q4x: $k" "$log" | awk '{print $3}')" = "$want" ] || fail "$1" "read-back $k $(grep -m1 "^q4x: $k" "$log") != $want"; done
  grep -q "kv $TMAX" "$log" || fail "$1" "read-back kv not $TMAX: $(grep -m1 'expert cache' "$log" | cut -c1-120)"
  echo "read-back $1 '$2': $(grep -m1 '^q4x: attsplit' "$log") | $(grep -m1 'expert cache' "$log" | sed 's/^q4x: //' | cut -c1-110) | gtt_gb $(awk '{printf "%.2f", $1/2^30}' /sys/class/drm/card1/device/mem_info_gtt_used 2>/dev/null || echo ?)" | tee -a "$R"
  grep '^gen ms/token' "$log" | awk '{print $3}' > "$OUT/$1.ms"
  grep '^out ' "$log" | awk '{ $1=""; print }' | sed 's/^ //' > "$OUT/$1.ids"
}
run A1 "$EA"; run B1 "$EB"; run B2 "$EB"; run A2 "$EA"
python3 - "$OUT" "$NP" <<'PY' | tee -a "$R"
import sys, statistics as st, pathlib
o = pathlib.Path(sys.argv[1]); n = int(sys.argv[2])
ids = {k: (o / f"{k}.ids").read_text().splitlines() for k in ("A1", "B1", "B2", "A2")}
ms = {k: [float(x) for x in (o / f"{k}.ms").read_text().split()] for k in ids}
bad = [i for i in range(n) if len({ids[k][i] for k in ids}) != 1]
print(f"IDENTITY {'PASS' if not bad else 'FAIL'}: {n - len(bad)} of {n} prompts equal across A1 B1 B2 A2" + (f", differing prompt indices {bad}" if bad else ""))
a = [st.median([ms["A1"][i], ms["A2"][i]]) for i in range(n)]; b = [st.median([ms["B1"][i], ms["B2"][i]]) for i in range(n)]
sp = lambda x: (max(x) - min(x)) / st.median(x) * 100
print(f"A median {st.median(a):.3f} ms/token (min {min(a):.3f} max {max(a):.3f}) | B median {st.median(b):.3f} (min {min(b):.3f} max {max(b):.3f}) | B/A {st.median(b) / st.median(a):.4f} | paired median ratio {st.median([y / x for x, y in zip(a, b)]):.4f}")
run_spread = {k: sp(ms[k]) for k in ms}
print("per-run spread %: " + " ".join(f"{k} {v:.1f}" for k, v in run_spread.items()) + f" | A1/A2 medians {st.median(ms['A1']):.3f}/{st.median(ms['A2']):.3f} B1/B2 {st.median(ms['B1']):.3f}/{st.median(ms['B2']):.3f}")
sys.exit(1 if bad else 0)
PY
echo "job PASS$(pcie_flag)" | tee -a "$R"
