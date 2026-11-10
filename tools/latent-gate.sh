#!/usr/bin/env bash
# L1 gate (bench/chat-protocol.md): the engine exports prefix checkpoints to an
# out-of-process receiver over a unix socket, without changing what it
# generates.
#
#   tools/latent-gate.sh [OUTDIR]
#
# Checks, each fatal:
#   P-L1a  socket unset -> teacher-forced identity 20/20 against a main-built
#          reference engine, i.e. the export path is not reached at all.
#   P-L1b  with a receiver listening, it gets exactly the number of handles the
#          engine says it exported, each with a valid LatentHeader.
#   P-L1c  an exported-then-ingested checkpoint is byte-identical to the one the
#          engine minted, compared against BARO_STATE_SAVE's own payload.
#
# NOT proven here: interop with the production latentos-agent. The receiver is
# tools/latent-recv.mojo, which speaks the same ipc/proto wire and nothing else.
set -uo pipefail
cd "$(dirname "$0")/.."
out=${1:-.work/latent-gate}
mkdir -p "$out"
rc=0
fail() { echo "FAIL: $*" | tee -a "$out/gate.log"; rc=1; }
ok()   { echo "ok:   $*" | tee -a "$out/gate.log"; }
: > "$out/gate.log"

MOJO=./.venv/bin/mojo
PACK=${BARO_PACK:-.work/engine-pack-q4}
REF=${LATENT_REF_ENGINE:-.work/engine}

echo "== build from the committed tree" | tee -a "$out/gate.log"
needs_build_engine=0
needs_build_recv=0
[ ! -x "$out/engine" ] && needs_build_engine=1
[ ! -x "$out/recv" ] && needs_build_recv=1
if [ $needs_build_engine -eq 0 ] && [ $(find serve kernels latentos -newer "$out/engine" 2>/dev/null | head -1 | wc -l) -gt 0 ]; then
  needs_build_engine=1
fi
if [ $needs_build_recv -eq 0 ] && [ $(find tools/latent-recv.mojo latentos -newer "$out/recv" 2>/dev/null | head -1 | wc -l) -gt 0 ]; then
  needs_build_recv=1
fi

if [ $needs_build_engine -eq 1 ]; then
  $MOJO build serve/engine.mojo -I . -I kernels -I serve -o "$out/engine" > "$out/build-engine.log" 2>&1 \
    || { fail "engine build: $(grep -m1 'error:' "$out/build-engine.log" | cut -c1-160)"; exit 1; }
fi
if [ $needs_build_recv -eq 1 ]; then
  $MOJO build tools/latent-recv.mojo -I . -I kernels -I serve -o "$out/recv" > "$out/build-recv.log" 2>&1 \
    || { fail "recv build: $(grep -m1 'error:' "$out/build-recv.log" | cut -c1-160)"; exit 1; }
fi
echo "engine $(sha256sum "$out/engine" | cut -c1-16)  recv $(sha256sum "$out/recv" | cut -c1-16)" | tee -a "$out/gate.log"

# ---- P-L1a: socket unset changes nothing ------------------------------------
echo "== P-L1a identity with BARO_LATENT_SOCK unset" | tee -a "$out/gate.log"
if [ ! -x "$REF" ]; then
  fail "P-L1a SKIPPED: no reference engine at $REF (build main's engine first)"
else
  if bench/force-ab.sh "$REF" "$out/engine" "$out/ab" "BARO_PACK=$PACK" > "$out/ab.log" 2>&1; then
    tail -1 "$out/ab.log" | tee -a "$out/gate.log"
    ok "P-L1a identity gate passed"
  else
    fail "P-L1a identity: $(tail -1 "$out/ab.log")"
  fi
fi

# ---- P-L1b / P-L1c: export over a real socket -------------------------------
echo "== P-L1b/c export to a listening receiver" | tee -a "$out/gate.log"
sock="$out/latent.sock"
rm -f "$sock" "$out"/recv-*.bin
prompt=$(head -1 bench/mtp-prompts/p01-water.tokens | tr -s ' \n' ',,' | sed 's/,$//')

"$out/recv" "$sock" 1 "$out" > "$out/recv.log" 2>&1 &
recv_pid=$!
for _ in $(seq 1 200); do [ -S "$sock" ] && break; sleep 0.01; done
if [ ! -S "$sock" ]; then
  fail "P-L1b: receiver never created $sock"; kill $recv_pid 2>/dev/null; exit 1
fi

# BARO_CKPT=1 caps the chain at one slot, so the engine exports exactly one
# checkpoint and BARO_STATE_SAVE writes that same slot: P-L1c then compares
# like with like instead of guessing which of several a file holds.
echo "{\"id\":1,\"prompt\":[$prompt],\"n\":8,\"spec\":false}" \
  | env BARO_SERVE=1 BARO_SPEC=0 BARO_CKPT=1 BARO_PACK="$PACK" \
        BARO_LATENT_SOCK="$sock" BARO_STATE_SAVE="$out/minted.bin" \
        "$out/engine" > "$out/engine.log" 2>&1
eng_rc=$?
wait $recv_pid; recv_rc=$?

exported=$(grep -oE 'latent: exported [0-9]+' "$out/engine.log" | grep -oE '[0-9]+$' | tail -1)
received=$(grep -c '^recv: handle' "$out/recv.log")
echo "engine rc=$eng_rc exported=${exported:-none}  recv rc=$recv_rc received=$received" | tee -a "$out/gate.log"

[ "$eng_rc" -eq 0 ] || fail "P-L1b: engine exited $eng_rc: $(grep -m1 -i error "$out/engine.log" | cut -c1-160)"
[ "$recv_rc" -eq 0 ] || fail "P-L1b: receiver exited $recv_rc: $(tail -1 "$out/recv.log")"
[ -n "${exported:-}" ] && [ "$exported" -ge 1 ] || fail "P-L1b: engine exported nothing"
[ "$received" = "${exported:-0}" ] || fail "P-L1b: engine exported ${exported:-0}, receiver got $received"
[ "$received" = "${exported:-0}" ] && [ "$eng_rc" -eq 0 ] && ok "P-L1b $received handles, each with a valid header"

# P-L1c: the ingested payload against the engine's own saved state payload.
# BARO_STATE_SAVE writes (serve/engine.mojo:295) "BAROST01" + int64 pos, conv_n,
# ssm_n, kv_n = 40 B, then a 32 B pack salt, then int32 tokens[pos], and only
# then conv and ssm. The token block is variable, so the offset is read from
# the file's own pos rather than hardcoded.
if [ -f "$out/recv-0.bin" ] && [ -f "$out/minted.bin" ]; then
  if python3 - "$out/minted.bin" "$out/recv-0.bin" <<'PY'
import struct, sys
src, recv_file = sys.argv[1], sys.argv[2]
with open(src, "rb") as f1, open(recv_file, "rb") as f2:
    hdr = f1.read(40)
    assert hdr[:6] == b"BAROST", f"bad magic: {hdr[:8]}"
    pos, conv_n, ssm_n, kv_n = struct.unpack_from("<qqqq", hdr, 8)
    f1.seek(40 + 32 + 4 * pos)
    payload = f1.read(4 * (conv_n + ssm_n))
    recv = f2.read()
    if payload != recv:
        print(f"differs: minted {len(payload)}B vs recv {len(recv)}B")
        sys.exit(1)
PY
  then
    ok "P-L1c ingested payload byte-identical to the minted one ($(stat -c%s "$out/recv-0.bin") bytes)"
  else
    fail "P-L1c: payloads differ"
  fi
else
  fail "P-L1c: missing $out/recv-0.bin or $out/minted.bin"
fi

rm -f "$sock"
[ $rc -eq 0 ] && echo "L1 GATE PASS" | tee -a "$out/gate.log" || echo "L1 GATE FAIL" | tee -a "$out/gate.log"
exit $rc
