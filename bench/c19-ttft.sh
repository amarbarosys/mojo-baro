#!/usr/bin/env bash
# usage: bench/c19-ttft.sh ENG_F32 ENG_INT8 PACK OUT
# C19 stage 3 (AMDHQ 06 "## C19:", stage 3 amendment): time to first token at 32k
# for a checkpoint shipped over B4's shaped link, per arm:
#   cg-L2   = transfer(CacheGen payload) + decode_s + fork(rebuilt BAROST01, f32 engine)
#   int8-03 = transfer(BAROST03) + fork(BARO_KVQ=int8 engine)
#   int8-02 = transfer(BAROST02) + fork(f32 engine)            reported
#   recipe  = cold prefill, f32 engine, BARO_CKPT=0            reported
# fork_s: one engine process per (arm, suffix), request write -> first tok line.
# Phases: forks in ONE gpu-wait job; transfers on the veth rig with no GPU held;
# decode_s read from tools/c19_rc.mojo's log (median of its reps).
# Inputs under C19_DIR (default .work/c19): states/{int803,int8}-lat_state_pos31999_*.baro,
# s1-32k/cg-L2.baro, cg-L2-32k.payload, rc-L2.log, P-32000.tokens.
# GATE_DRYRUN=1: stub engines and 8 MB slices of the files through the real rig.
# C19_LINK_ONLY=1: no GPU; reuse OUT/forks.json from a finished run, repeat each
# transfer C19_REPS times (default 1) over C19_PROFILES (default 100mbit 1gbit 10gbit;
# also 25gbit, unshaped), medians per cell.
set -euo pipefail
cd "$(dirname "$0")/.."
engf=$1; engi=$2; pack=$3; out=$4
dir=${C19_DIR:-.work/c19}
fail() { echo "FAIL $1: $2"; exit 1; }
mkdir -p "$out"
f03=$(ls "$dir"/states/int803-lat_state_pos31999_*.baro)
f02=$(ls "$dir"/states/int8-lat_state_pos31999_*.baro)
fcg="$dir/s1-32k/cg-L2.baro"
fpay="$dir/cg-L2-32k.payload"
fkv4=${C19_KV4:-}   # C20 stage C: BAROKV4 payload, loaded by the (kv4-capable) int8 engine as arm kv4-03
Q="bench/mtp-prompts/p02-python-fib.tokens bench/mtp-prompts/p05-math.tokens bench/mtp-prompts/p11-email.tokens"

gpu_phase() {
    trap 'pkill -TERM -P $$ 2>/dev/null || :' EXIT
    python3 - "$out" "$dir/P-32000.tokens" "$engf" "$engi" "$pack" "$f03" "$f02" "$fcg" "$fkv4" $Q <<'PY'
import json, os, re, subprocess, sys, time
out, Pf, engf, engi, pack, f03, f02, fcg, fkv4, *qs = sys.argv[1:]
P = [int(x) for x in open(Pf).read().split()]
arms = [("int8-03", engi, f03), ("int8-02", engf, f02), ("cg-L2", engf, fcg), ("recipe", engf, None)] + ([("kv4-03", engi, fkv4)] if fkv4 else [])
rows = []
for arm, eng, st in arms:
    for qi, qf in enumerate(qs):
        env = dict(os.environ, BARO_SERVE="1", BARO_SPEC="0", BARO_PACK=pack, BARO_TMAX="32768")
        if st is None:
            env["BARO_CKPT"] = "0"
        log = open(f"{out}/{arm}-q{qi + 1}.out", "w")
        p = subprocess.Popen(eng.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=open(f"{out}/{arm}-q{qi + 1}.err", "w"), text=True, env=env)
        lines = []
        for line in p.stdout:
            log.write(line); lines.append(line)
            if '"ready":true' in line:
                break
        else:
            sys.exit(f"FAIL engine {arm} q{qi + 1}: no ready line")
        req = {"id": 1, "prompt": P + [int(x) for x in open(qf).read().split()], "n": 1, "spec": False}
        if st:
            req["state_load"] = os.path.abspath(st)
        t0 = time.perf_counter()
        p.stdin.write(json.dumps(req) + "\n"); p.stdin.flush()
        first = None
        for line in p.stdout:
            log.write(line); lines.append(line)
            if first is None and line.startswith("{") and '"tok"' in line:
                first = (time.perf_counter() - t0, json.loads(line)["tok"])
            if '"done":true' in line or '"error"' in line:
                break
        p.stdin.close(); p.wait(timeout=120); log.close()
        if first is None:
            sys.exit(f"FAIL engine {arm} q{qi + 1}: no tok line, see {out}/{arm}-q{qi + 1}.out")
        loads = [l for l in lines if l.startswith("state loaded:")]
        pos = [int(re.search(r"pos (\d+)", l).group(1)) for l in loads]
        if st and pos != [len(P) - 1]:
            sys.exit(f"FAIL void: {arm} q{qi + 1} state loaded pos {pos}, want [{len(P) - 1}]")
        rows.append({"arm": arm, "q": qi + 1, "fork_s": first[0], "tok": first[1], "load_line": loads[0].strip() if loads else ""})
        print(f"{arm} q{qi + 1}: first token {first[1]} after {first[0]:.3f} s", flush=True)
json.dump(rows, open(f"{out}/forks.json", "w"), indent=1)
PY
}

if [ "${C19_GPU_PHASE:-0}" = 1 ]; then gpu_phase; exit 0; fi

REPS=${C19_REPS:-1}
PROFILES=${C19_PROFILES:-100mbit 1gbit 10gbit}
if [ "${C19_LINK_ONLY:-0}" = 1 ]; then
    [ -s "$out/forks.json" ] || fail setup "C19_LINK_ONLY needs $out/forks.json"
    x03=$f03; x02=$f02; xpay=$fpay; xkv4=$fkv4
elif [ "${GATE_DRYRUN:-0}" = 1 ]; then
    stub="$out/stub-engine.py"
    cat > "$stub" <<'PY'
import json, struct, sys
print('{"ready":true}', flush=True)
for line in sys.stdin:
    r = json.loads(line)
    if r.get("state_load"):
        pos = struct.unpack("<q", open(r["state_load"], "rb").read(16)[8:16])[0]
        print(f"state loaded: {r['state_load']}  pos {pos}", flush=True)
    print(json.dumps({"id": 1, "tok": 42}), flush=True)
    print('{"id":1,"done":true}', flush=True)
PY
    engf="python3 $stub"; engi="python3 $stub"
    mkdir -p "$out/slices"
    for f in "$f03" "$f02" "$fpay" $fkv4; do head -c 8000000 "$f" > "$out/slices/$(basename "$f")"; done
    gpu_phase
    x03="$out/slices/$(basename "$f03")"; x02="$out/slices/$(basename "$f02")"; xpay="$out/slices/$(basename "$fpay")"; xkv4=${fkv4:+$out/slices/$(basename "$fkv4")}
else
    for f in "$f03" "$f02" "$fcg" "$fpay" $fkv4 "$dir/rc-L2.log" "$dir/P-32000.tokens"; do [ -s "$f" ] || fail setup "$f missing"; done
    [ -x "$engf" ] && [ -x "$engi" ] || fail setup "engines missing"
    grep -q '^round_trip PASS' "$dir/rc-L2.log" || fail setup "coder round trip not PASS in $dir/rc-L2.log"
    {
        echo "engf=$engf sha=$(sha256sum "$engf" | cut -c1-16) engi=$engi sha=$(sha256sum "$engi" | cut -c1-16)"
        echo "commit=$(git rev-parse --short HEAD) dirty='$(git status --porcelain | tr '\n' ' ')' pack=$pack"
        for f in "$f03" "$f02" "$fcg" "$fpay" $fkv4; do echo "file $(sha256sum "$f" | cut -c1-64) $(stat -c %s "$f") $f"; done
        grep -E '^(payload_bytes|decode_s_median|round_trip)' "$dir/rc-L2.log"
    } > "$out/arm.txt"
    cat "$out/arm.txt"
    gpu-wait run --priority 20 --vram 22 --timeout 900 -- env C19_DIR="$dir" C19_KV4="$fkv4" C19_GPU_PHASE=1 "$0" "$engf" "$engi" "$pack" "$out" \
        > "$out/gpu-phase.log" 2>&1 || fail gpu "$out/gpu-phase.log"
    x03=$f03; x02=$f02; xpay=$fpay; xkv4=$fkv4
fi
[ -s "$out/forks.json" ] || fail gpu "no forks.json"

# ---- link phase: B4's rig, no GPU held
NS=c19peer; VA=veth-c19a; VB=veth-c19b; HOST_IP=10.99.8.1; PEER_IP=10.99.8.2; XP=9912
cleanup() {
    sudo firewall-cmd --zone=trusted --remove-interface="$VA" >/dev/null 2>&1 || :
    sudo ip netns del "$NS" 2>/dev/null || :
    sudo ip link del "$VA" 2>/dev/null || :
}
trap cleanup EXIT
cleanup
sudo ip netns add "$NS"
sudo ip link add "$VA" type veth peer name "$VB"
sudo ip link set "$VB" netns "$NS"
sudo ip addr add "$HOST_IP/24" dev "$VA"
sudo ip link set "$VA" up
sudo firewall-cmd --zone=trusted --add-interface="$VA" >/dev/null
sudo ip netns exec "$NS" ip addr add "$PEER_IP/24" dev "$VB"
sudo ip netns exec "$NS" ip link set "$VB" up
sudo ip netns exec "$NS" ip link set lo up
: > "$out/transfers.tsv"
for name in $PROFILES; do
    case $name in
        100mbit) spec="tbf rate 100mbit burst 32kbit latency 400ms" ;;
        1gbit) spec="tbf rate 1gbit burst 320kbit latency 100ms" ;;
        10gbit) spec="tbf rate 10gbit burst 3200kbit latency 50ms" ;;
        25gbit) spec="tbf rate 25gbit burst 8000kbit latency 50ms" ;;
        unshaped) spec="" ;;
        *) fail setup "unknown profile $name" ;;
    esac
    if [ -n "$spec" ]; then sudo tc qdisc replace dev "$VA" root $spec; else sudo tc qdisc del dev "$VA" root 2>/dev/null || :; fi
    for r in $(seq "$REPS"); do
    for pair in "int8-03:$x03" "int8-02:$x02" "cg-L2:$xpay" ${xkv4:+"kv4-03:$xkv4"}; do
        arm=${pair%%:*}; f=${pair#*:}
        dst="$out/rx-$name-$arm.bin"
        sudo rm -f "$dst"
        sudo ip netns exec "$NS" bash -c "timeout $(( 60 + $(stat -c%s "$f") / 5000000 )) nc -l -p $XP > '$dst'" &
        np=$!
        sleep 0.3
        t0=$(date +%s.%N)
        nc -N "$PEER_IP" "$XP" < "$f"
        wait "$np" || fail "transfer $name $arm" "listener exit"
        t1=$(date +%s.%N)
        a=$(sha256sum < "$f" | cut -c1-64); b=$(sha256sum < "$dst" | cut -c1-64)
        sudo rm -f "$dst"
        [ "$a" = "$b" ] || fail "transfer $name $arm" "sha mismatch"
        printf '%s\t%s\t%s\t%s\n' "$name" "$arm" "$(stat -c%s "$f")" "$(python3 -c "print($t1 - $t0)")" | tee -a "$out/transfers.tsv"
    done
    done
done
sudo tc qdisc del dev "$VA" root 2>/dev/null || :

dec=$(sed -n 's/^decode_s_median \([0-9.e-]*\).*/\1/p' "$dir/rc-L2.log")
[ -n "$dec" ] || fail decode "no decode_s_median in $dir/rc-L2.log"
python3 - "$out" "$dec" "$PROFILES" <<'PY' | tee "$out/summary.md"
import json, statistics, sys
out, dec, profiles = sys.argv[1], float(sys.argv[2]), sys.argv[3].split()
forks = json.load(open(f"{out}/forks.json"))
kv4 = any(r["arm"] == "kv4-03" for r in forks)
fk = {a: statistics.median(r["fork_s"] for r in forks if r["arm"] == a) for a in ("int8-03", "int8-02", "cg-L2", "recipe") + (("kv4-03",) if kv4 else ())}
tok = {(r["arm"], r["q"]): r["tok"] for r in forks}
void = [(a, q) for q in (1, 2, 3) for a in ("int8-03", "int8-02") + (("kv4-03",) if kv4 else ()) if tok[(a, q)] != tok[("recipe", q)]]
cgdiff = [q for q in (1, 2, 3) if tok[("cg-L2", q)] != tok[("recipe", q)]]
xs = {}
for line in open(f"{out}/transfers.tsv"):
    p, a, b, s = line.split("\t"); xs.setdefault((p, a), []).append(float(s))
xf = {k: statistics.median(v) for k, v in xs.items()}
print("fork_s medians: " + ", ".join(f"{a} {v:.3f}" for a, v in fk.items()) + f"; decode_s {dec:.3f}")
print(f"first tokens vs recipe: int8 mismatches {void or 'none'}; cg-L2 differs on {cgdiff or 'none'}")
print()
print("| profile | n | int8-03 | int8-02 | cg-L2 | recipe | int8-03 / cg-L2 | verdict | xfer range int8-03 / cg-L2 |" + (" kv4-03 | int8-03 / kv4-03 |" if kv4 else ""))
print("|---|---|---|---|---|---|---|---|---|" + ("---|---|" if kv4 else ""))
for p in profiles:
    i3 = xf[(p, "int8-03")] + fk["int8-03"]
    i2 = xf[(p, "int8-02")] + fk["int8-02"]
    cg = xf[(p, "cg-L2")] + dec + fk["cg-L2"]
    v = "WIN" if cg <= i3 / 1.5 else "LOSE" if i3 <= cg / 1.5 else "TIE"
    n = len(xs[(p, "cg-L2")])
    spread = " / ".join(f"{min(xs[(p, a)]):.3f}-{max(xs[(p, a)]):.3f}" for a in ("int8-03", "cg-L2"))
    k4 = f" {xf[(p, 'kv4-03')] + fk['kv4-03']:.3f} s | {i3 / (xf[(p, 'kv4-03')] + fk['kv4-03']):.2f}x |" if kv4 else ""
    print(f"| {p} | {n} | {i3:.3f} s | {i2:.3f} s | {cg:.3f} s | {fk['recipe']:.3f} s | {i3 / cg:.2f}x | {v} | {spread} |" + k4)
print()
print("VOID" if void else "valid")
PY
if [ "${GATE_DRYRUN:-0}" = 1 ]; then echo "PASS gate-dryrun: c19-ttft ran stub forks and 8 MB slices through the rig"; exit 0; fi
grep -qx valid "$out/summary.md" || fail void "$out/summary.md"
echo "DONE c19-ttft: $out/summary.md"
