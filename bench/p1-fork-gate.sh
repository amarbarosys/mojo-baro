#!/usr/bin/env bash
# usage: bench/p1-fork-gate.sh FORMAT [OUT]        FORMAT = f32 | int8
#
# P1 gate 2, IDENTITY HALF (bench/p1-fork-protocol.md): POST /v1/fork with "target" between two
# live 9B nodes of ours on one XTX (BARO_TMAX=4096, MAX memory cap 10), across the B4-mini veth
# link at 100 Mbit, 1 Gbit and 10 Gbit. The 32k timing half is NOT here: two engines do not fit at
# 32k (docs/P1-FORK-TARGET.md).
#
#   root namespace                      peer namespace b4fork
#   node A (holder)  --veth-f4a, tbf--> tools/fork-link-forwarder.py --veth-f4b--> node B (0.0.0.0)
#
# Both nodes are ordinary user processes in the root namespace. The state crosses the shaped hop
# exactly once (tbf on veth-f4a egress); the return hop is unshaped. Node B is restarted COLD
# before every rate and before the falsifiers: a node that kept a prefix from an earlier rate would
# report cached == pos from its own checkpoint and the reuse check would prove nothing.
#
# The orchestrator builds and tears down the link (sudo, runtime only, nothing persistent) and
# scores on the CPU. The GPU part is this script re-run as ONE blocking gpu-wait job (PHASE=nodes);
# servers are its direct children, reaped child-first with a bounded wait.
# env: QUICK=N prompts (a claim needs 20, P18)   NFALS=N falsifier prompts (default 5)
set -euo pipefail
cd "$(dirname "$0")/.."
fmt=${1:?usage: bench/p1-fork-gate.sh f32|int8 [OUT]}
case "$fmt" in f32|int8) ;; *) echo "FAIL setup: FORMAT must be f32 or int8"; exit 1 ;; esac
out=${2:-.work/fork/g2-gate/$fmt}
quick=${QUICK:-20}; nfals=${NFALS:-5}; phase=${PHASE:-all}
engine=${BARO_ENGINE:-.work/fork/engine}
pack=${BARO_PACK:-$HOME/Projects/mojo/mojo-baro/.work/engine-pack-q4}
serve=serve/target/release/baro-serve
py=./.venv/bin/python
NS=b4fork; VA=veth-f4a; VB=veth-f4b; HOST_IP=10.99.8.1; PEER_IP=10.99.8.2
bport=${BPORT:-18472}; fbase=${FBASE:-19470}
mkdir -p "$out"
# The nodes phase runs inside the orchestrator's gpu-wait job, whose output the orchestrator already
# tees into gate.log; teeing there too doubled every line.
[ "${PHASE:-all}" = nodes ] || exec > >(tee -a "$out/gate.log") 2>&1
fail() { echo "FAIL $1: $2 (log $out/gate.log)"; exit 1; }

if [ "$phase" = nodes ]; then
  pids=()
  reap() {  # PID...: the engine child first (baro-serve outlives TERM while it runs), bounded wait
    for p in "$@"; do pkill -TERM -P "$p" 2>/dev/null || true; kill -TERM "$p" 2>/dev/null || true; done
    for _ in $(seq 1 20); do
      alive=0; for p in "$@"; do kill -0 "$p" 2>/dev/null && alive=1; done
      [ "$alive" = 0 ] && return 0; sleep 0.5
    done
    for p in "$@"; do pkill -KILL -P "$p" 2>/dev/null || true; kill -KILL "$p" 2>/dev/null || true; done
  }
  trap 'reap "${pids[@]}"' EXIT
  start() {  # NAME PACKPATH EXTRA_ARGS... ; sets $url and $spid. Never call in a command substitution.
    local name=$1 pk=$2; shift 2
    : > "$out/$name.stdout"
    env MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=10 BARO_TMAX=4096 BARO_SPEC=0 \
      BARO_STATE_INT8="$([ "$fmt" = int8 ] && echo 1 || echo 0)" BARO_CKPT_DIR="$out/ckpts-$name" \
      "$serve" --engine "$engine" --pack "$pk" "$@" > "$out/$name.stdout" 2>> "$out/$name.stderr" &
    spid=$!; pids+=("$spid")
    for _ in $(seq 1 600); do
      grep -q '^listening on' "$out/$name.stdout" && break
      kill -0 "$spid" 2>/dev/null || fail "start-$name" "server exited: $(tail -2 "$out/$name.stderr" | tr '\n' ' ')"
      sleep 0.5
    done
    url=$(grep -m1 -oE 'http://[0-9.:]+' "$out/$name.stdout") || fail "start-$name" "no listening line in 300 s"
    # Guards run as statements: `fail` inside $(...) exits only the subshell, and `grep | tail || fail`
    # is masked by tail, so neither may live inside the echo.
    local st lim idn cold
    st=$(curl -fsS "$url/v1/state" 2>&1) || fail "readback-$name" "GET $url/v1/state: $st"
    cold=$(echo "$st" | $py -c "import json,sys;d=json.load(sys.stdin);print('portable',d['portable'],'resident_states',len(d['states']));sys.exit(0 if d['portable'] and not d['states'] else 1)") || fail "readback-$name" "node is not portable and cold at start: $cold"
    lim=$(grep -o 'limits Limits { tmax: [0-9]*' "$out/$name.stderr" | tail -1 || true)
    [ "$lim" = "limits Limits { tmax: 4096" ] || fail "readback-$name" "engine limits line says '$lim', wanted tmax 4096"
    idn=$(grep -o 'identity Identity { pack: "[0-9a-f]\{16\}' "$out/$name.stderr" | tail -1 || true)
    echo "readback $name: $cold | $lim | $idn" | tee -a "$out/arm.txt"
  }
  cold_b() { [ -z "${bpid:-}" ] || reap "$bpid"; rm -rf "$out/ckpts-b"; start b "$out/pack-b" --port "$bport" --host 0.0.0.0; bpid=$spid; }

  start a "$pack" --port 0; ua=$url
  $py tools/p1-fork-drive.py refs "$out" "$ua" "$quick" || fail refs "driver"
  for prof in 100mbit 1gbit 10gbit; do
    case $prof in 100mbit) spec="rate 100mbit burst 32kbit latency 400ms" ;; 1gbit) spec="rate 1gbit burst 320kbit latency 100ms" ;; *) spec="rate 10gbit burst 3200kbit latency 50ms" ;; esac
    # shellcheck disable=SC2086
    sudo -n tc qdisc replace dev "$VA" root tbf $spec || fail "tc-$prof" "qdisc replace"
    echo "link $prof: $(tc qdisc show dev "$VA" | head -1)" | tee -a "$out/arm.txt"
    $py tools/p1-fork-drive.py rate "$out" "$PEER_IP" "$((fbase + 3))" "$prof" | tee -a "$out/arm.txt" || fail "rate-$prof" "driver"
    cold_b
    $py tools/p1-fork-drive.py forks "$out" "$ua" "$PEER_IP:$fbase" "$prof" "$quick" || fail "forks-$prof" "driver"
  done
  cold_b
  $py tools/p1-fork-drive.py falsify "$out" "$ua" "$PEER_IP:$((fbase + 1))" "$PEER_IP:$((fbase + 2))" "$nfals" || fail falsify "driver"
  echo "readback engines: A state-saved $(grep -c 'state saved:.*format' "$out/a.stderr" || true) formats [$(grep -o 'format BAROST0[12]' "$out/a.stderr" | sort | uniq -c | tr -s ' ' | tr '\n' ' ')] ; B state-loaded $(grep -c 'state loaded:' "$out/b.stderr" || true)" | tee -a "$out/arm.txt"
  exit 0
fi

# ---- orchestrator: link up, one GPU job, score, link down ------------------------------------
# PHASE=linktest: the same link, no GPU and no engine. A stand-in HTTP server takes node B's port;
# proves root -> shaped veth -> forwarder -> back to the host through firewalld, reads the three
# rates back at the receiver, and proves teardown leaves nothing (P15).
for f in "$engine" "$serve" "$pack/pack.bin" "$pack/identity.json" tools/fork-link-forwarder.py; do [ -e "$f" ] || fail setup "missing $f"; done
sudo -n true || fail setup "sudo -n is required for the veth link"
rm -f "$out"/forks-*.json "$out"/refs.json "$out"/falsify.json "$out"/rate-*.json "$out"/a.stderr "$out"/b.stderr
ln -sfn "$(readlink -f "$pack")" "$out/pack-b"
teardown() {
  [ -s "$out/forwarder.pid" ] && sudo -n kill "$(cat "$out/forwarder.pid")" 2>/dev/null || true
  sudo -n firewall-cmd --zone=trusted --remove-interface="$VA" >/dev/null 2>&1 || true
  sudo -n ip netns del "$NS" 2>/dev/null || true
  sudo -n ip link del "$VA" 2>/dev/null || true
  rm -f "$out/forwarder.pid"
}
trap teardown EXIT
teardown
sudo -n ip netns add "$NS"
sudo -n ip link add "$VA" type veth peer name "$VB"
sudo -n ip link set "$VB" netns "$NS"
sudo -n ip addr add "$HOST_IP/24" dev "$VA"
sudo -n ip link set "$VA" up
# firewalld's default zone drops unsolicited inbound on the veth (bench/b4-cross-host.sh found TCP
# "no route to host" while ICMP passed); runtime only, removed in teardown
sudo -n firewall-cmd --zone=trusted --add-interface="$VA" >/dev/null || fail link "firewall-cmd"
sudo -n ip netns exec "$NS" ip addr add "$PEER_IP/24" dev "$VB"
sudo -n ip netns exec "$NS" ip link set "$VB" up
sudo -n ip netns exec "$NS" ip link set lo up
sudo -n ip netns exec "$NS" sh -c "echo \$\$ > '$PWD/$out/forwarder.pid'; exec python3 '$PWD/tools/fork-link-forwarder.py' $PEER_IP $fbase $HOST_IP $bport" > "$out/forwarder.log" 2>&1 &
for _ in $(seq 1 40); do grep -q '^forwarder ready' "$out/forwarder.log" && break; sleep 0.25; done
grep -q '^forwarder ready' "$out/forwarder.log" || fail link "forwarder did not start: $(tail -2 "$out/forwarder.log" | tr '\n' ' ')"
ping -c1 -W2 "$PEER_IP" >/dev/null || fail link "peer $PEER_IP unreachable"
echo "arm: gate2-identity format=$fmt prompts=$quick falsify=$nfals engine_sha=$(sha256sum "$engine" | cut -c1-16) serve_sha=$(sha256sum "$serve" | cut -c1-16) pack=$pack commit=$(git rev-parse --short HEAD) dirty='$(git status --porcelain | grep -v '^?? .work' | tr '\n' ';')' link=$HOST_IP<->$PEER_IP $(head -1 "$out/forwarder.log")" | tee "$out/arm.txt"

if [ "$phase" = linktest ]; then
  mkdir -p "$out/www"; echo linktest-ok > "$out/www/probe.txt"
  ( cd "$out/www" && exec python3 -m http.server "$bport" --bind 0.0.0.0 ) > "$out/standin.log" 2>&1 &
  wpid=$!
  trap 'kill "$wpid" 2>/dev/null || true; teardown' EXIT
  sleep 1
  got=$(curl -fsS --max-time 10 "http://$PEER_IP:$fbase/probe.txt") || fail linktest "no answer through the forwarder: $(tail -2 "$out/forwarder.log" | tr '\n' ' ')"
  [ "$got" = linktest-ok ] || fail linktest "wrong body through the forwarder: $got"
  echo "linktest: root -> $PEER_IP:$fbase -> forwarder -> $HOST_IP:$bport answered"
  for prof in 100mbit 1gbit 10gbit; do
    case $prof in 100mbit) spec="rate 100mbit burst 32kbit latency 400ms" ;; 1gbit) spec="rate 1gbit burst 320kbit latency 100ms" ;; *) spec="rate 10gbit burst 3200kbit latency 50ms" ;; esac
    # shellcheck disable=SC2086
    sudo -n tc qdisc replace dev "$VA" root tbf $spec || fail "tc-$prof" "qdisc replace"
    $py tools/p1-fork-drive.py rate "$out" "$PEER_IP" "$((fbase + 3))" "$prof" || fail "rate-$prof" "driver"
  done
  echo "PASS linktest"
  exit 0
fi

rc=0
gpu-wait run --priority 50 --vram 23 --timeout 2400 -- env HOME="$HOME" PATH="$PATH" PHASE=nodes QUICK="$quick" NFALS="$nfals" BPORT="$bport" FBASE="$fbase" \
  BARO_ENGINE="$engine" BARO_PACK="$pack" "$PWD/bench/p1-fork-gate.sh" "$fmt" "$PWD/$out" || rc=$?
[ "$rc" = 0 ] || fail job "gpu-wait job exited $rc"
grep -h 'rewrote' "$out/forwarder.log" | sort | uniq -c | tr -s ' ' | tr '\n' ';' | sed 's/^/forwarder: /' | tee -a "$out/arm.txt"; echo
$py tools/p1-fork-score.py "$out" "$fmt" "$quick" "$nfals"
