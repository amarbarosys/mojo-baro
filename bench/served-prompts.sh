#!/usr/bin/env bash
# usage: bench/served-prompts.sh ENGINE PACK OUTDIR "ENV" [PORT]
#   e.g. bench/served-prompts.sh .work/moe-served/engine .work/moe-w1/pack \
#        .work/moe-served/run "BARO_SPEC=0 BARO_MEGA=0"
#
# P4 receipt for the served path against the one-shot path, SAME binary, same
# stint: arm A is one process per prompt (BARO_PROMPT, the shape every tok/s
# number in bench/*-protocol.md was measured in), arm B is the same binary kept
# resident by serve/src (baro-serve) and driven over HTTP, one POST
# /v1/completions per prompt with the prompt as token ids.
#
# The two arms cannot be interleaved per prompt the way bench/ab-prompts.sh
# interleaves its two binaries: a resident engine holds the whole pack (21 GB
# for the MoE), so a one-shot process cannot allocate while the server is up.
# Arm A therefore runs to completion, the server starts, arm B runs. Wrap the
# whole thing in bench/clock-probe.sh so one clock receipt covers both arms.
#
# Not a two-binary A/B, so bench/ab-prompts.sh's same-sha refusal does not
# apply: one binary on two paths IS the experiment. The sha is printed once and
# the server's own /health read-back (pack, tmax, spec_k) is the arm receipt for
# the resident side (P1); each one-shot log carries its own parameter echo.
#
# Reported per prompt: tok/s_gen from each arm (arm B's from the response
# "timings".tok_s, which the engine computes as (n-1)/decode_s exactly as the
# one-shot line does), arm B's client-side wall-clock tok/s over the whole HTTP
# round trip, and token identity between the arms.
set -uo pipefail
cd "$(dirname "$0")/.."
eng=$1; pack=$2; out=$3; envx=${4:-}; port=${5:-8099}
mkdir -p "$out"
sha=$(sha256sum "$eng" | cut -c1-16)
serve=serve/target/release/baro-serve
[ -x "$serve" ] || { echo "no $serve; cargo build --release --manifest-path serve/Cargo.toml" >&2; exit 2; }
ssha=$(sha256sum "$serve" | cut -c1-16)
{
  echo "engine=$eng shaEngine=$sha serve=$serve shaServe=$ssha pack=$pack env='$envx' port=$port"
  echo "power_cap_uW=$(cat /sys/class/drm/card1/device/hwmon/hwmon*/power1_cap) vddgfx=$(grep -A1 OD_VDDGFX_OFFSET /sys/class/drm/card1/device/pp_od_clk_voltage | tail -1)"
  echo "commit=$(git rev-parse --short HEAD) dirty='$(git status --porcelain | tr '\n' ';')'"
} | tee "$out/arm.txt"

# ---- arm A: one process per prompt, the shape the 93.46 receipt was taken in
echo "arm A (one-shot) ..." >&2
for tf in bench/mtp-prompts/p*.tokens; do
  p=$(basename "$tf" .tokens)
  env BARO_PROMPT="$tf" BARO_PACK="$pack" $envx "$eng" > "$out/$p.oneshot.log" 2>&1
done

# ---- arm B: the same binary resident behind baro-serve, one HTTP request each
echo "arm B (served) ..." >&2
env BARO_PACK="$pack" $envx "$serve" --engine "$eng" --pack "$pack" --port "$port" \
  > "$out/serve.out" 2>"$out/serve.err" &
spid=$!
ready=0
for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$port/health" -o "$out/health.json" 2>/dev/null; then ready=1; break; fi
  sleep 1
done
if [ "$ready" != 1 ]; then
  echo "VOID: server never answered /health on port $port; see $out/serve.err" >&2
  kill $spid 2>/dev/null; exit 3
fi
echo "health: $(cat "$out/health.json")" | tee -a "$out/arm.txt"

python3 - "$out" "$port" <<'PY'
import json, pathlib, sys, time, urllib.request
out, port = pathlib.Path(sys.argv[1]), sys.argv[2]
rows = []
for tf in sorted(pathlib.Path("bench/mtp-prompts").glob("p*.tokens")):
    ids = [int(x) for x in tf.read_text().split()]
    body = json.dumps({"prompt": ids, "max_tokens": 64, "spec": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as r:
        resp = json.load(r)
    wall = time.perf_counter() - t0
    (out / f"{tf.stem}.served.json").write_text(json.dumps(resp, indent=1))
    rows.append((tf.stem, resp, wall))
(out / "served.jsonl").write_text("".join(json.dumps({"p": p, "wall_s": w, "timings": r["timings"],
    "tokens": r["choices"][0]["tokens"]}) + "\n" for p, r, w in rows))
print(f"served: {len(rows)} prompts")
PY
rc=$?
curl -sf -X POST "http://127.0.0.1:$port/v1/cancel" -d '{}' -o /dev/null 2>/dev/null
kill -INT $spid 2>/dev/null; wait $spid 2>/dev/null
[ $rc = 0 ] || { echo "VOID: served arm failed (rc $rc)" >&2; exit 4; }

# ---- receipt
python3 - "$out" <<'PY'
import json, pathlib, statistics as st, sys
out = pathlib.Path(sys.argv[1])
served = {json.loads(l)["p"]: json.loads(l) for l in (out / "served.jsonl").read_text().splitlines()}
rows = []
for log in sorted(out.glob("*.oneshot.log")):
    p = log.name[: -len(".oneshot.log")]
    txt = log.read_text()
    one = next((float(l.split(":")[1]) for l in txt.splitlines() if l.startswith("tok/s_gen:")), None)
    gen = next((l.split(":", 1)[1].split() for l in txt.splitlines() if l.startswith("GENERATED:")), None)
    s = served.get(p)
    if one is None or gen is None or s is None:
        rows.append((p, one, None, None, "VOID")); continue
    ident = "PASS" if [int(x) for x in gen] == s["tokens"] else "FAIL"
    wall_tok_s = (len(s["tokens"]) - 1) / s["wall_s"] if s["wall_s"] > 0 else 0.0
    rows.append((p, one, s["timings"]["tok_s"], wall_tok_s, ident))
hdr = f"{'prompt':<17}{'oneshot':>10}{'served':>10}{'servedWall':>12}  identity"
print(hdr); print("-" * len(hdr))
for p, a, b, w, i in rows:
    print(f"{p:<17}{a if a is None else f'{a:10.2f}'}{b if b is None else f'{b:10.2f}'}"
          f"{w if w is None else f'{w:12.2f}'}  {i}")
(out / "results.txt").write_text(hdr + "\n" + "\n".join(
    f"{p} {a} {b} {w} {i}" for p, a, b, w, i in rows) + "\n")
void = [r[0] for r in rows if r[4] == "VOID"]
ok = [r for r in rows if r[4] != "VOID"]
A = [r[1] for r in ok]; B = [r[2] for r in ok]; W = [r[3] for r in ok]
sp = lambda x: (max(x) - min(x)) / st.median(x) * 100
fails = [r[0] for r in ok if r[4] != "PASS"]
print(f"one-shot median {st.median(A):.2f} tok/s_gen spread {sp(A):.1f}%  |  "
      f"served median {st.median(B):.2f} spread {sp(B):.1f}%  |  ratio {st.median(B)/st.median(A):.3f}")
print(f"served wall-clock median {st.median(W):.2f} tok/s (HTTP round trip included)  |  "
      f"identity fails: {fails or 'none'}")
# A void is a failed arm, not a skipped row (P10).
if void:
    print(f"FAIL: {len(void)} void row(s): {void}"); sys.exit(1)
if fails:
    print(f"FAIL: served tokens differ from one-shot on {fails}"); sys.exit(1)
PY
