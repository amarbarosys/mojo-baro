#!/usr/bin/env bash
# P4 trainer checks (ORACLE_PY and MOJO override the interpreter paths). Receipts: exchange/receipts/latent-bridge/p4/.
#   checks.sh build            build .work/bridge/bridge (CPU)
#   checks.sh tiny             GPU: oracle dumps (ALL, noxattn, single, identical, A, B, C, A@gate0.8, D), gate-0, smoke + rerun sha
#   checks.sh synth-real       GPU: real-dim synthetic set (bridge_oracle.py gen DIR 2 real): gate-0, 200-step rerun sha, speed
#   checks.sh tiny-oracle      CPU: torch recompute of every tiny dump -> rel L2 per tensor <= 1e-4
#   checks.sh real-grad        GPU: dump one real micro-batch per REAL_GRAD spec (default "A:A:lam=0 C:C:", the test400 root)
#   checks.sh real-oracle      CPU: torch check of those dumps (RAM ~6 GB).  BRIDGE_DATA=root, REAL_EXTRA=flags for overfit/rerun/speed
#   checks.sh real-gate0 | real-overfit | real-rerun | real-speed      GPU, each one gpu-wait job
# GPU stages are run INSIDE one gpu-wait job each:  gpu-wait run --shared --vram N --timeout S -- bash miniagi/bridge/checks.sh STAGE
# Every line printed is also appended to the stage receipt; exit non-zero on the first failed check.
set -euo pipefail
cd "$(dirname "$0")/../.."
BIN=${BRIDGE_BIN:-.work/bridge/bridge}
PY="${ORACLE_PY:-.work/oracle-venv/bin/python} -P"
MOJO=${MOJO:-./.venv/bin/mojo}
ROOT=${BRIDGE_DATA:-$HOME/.cache/baro/latent-bridge}
TINY=$ROOT/tiny
R=exchange/receipts/latent-bridge/p4
W=.work/bridge
mkdir -p $R $W
stage=${1:-}
rc_file=$R/$stage.txt
log() { echo "$*" | tee -a "$rc_file"; }
run() { log "\$ $*"; "$@" 2>&1 | tee -a "$rc_file"; return "${PIPESTATUS[0]}"; }
hdr() { : > "$rc_file"; log "# $stage · $(date -Is) · HEAD $(git rev-parse --short HEAD) · bin sha $(sha256sum $BIN 2>/dev/null | cut -c1-12)"; }
vram() { for c in /sys/class/drm/card*/device/mem_info_vram_used; do [ -r "$c" ] && echo "$(( $(cat "$c") >> 20 ))" && return; done; echo -1; }

case "$stage" in
build)
  hdr
  run $MOJO build miniagi/bridge/bridge.mojo -I miniagi/bridge -I miniagi/latent -I miniagi/train -o $BIN
  log "built $BIN sha $(sha256sum $BIN | cut -c1-12)"
  ;;
tiny)
  hdr
  [ -f $TINY/cfg.txt ] || run $PY tools/bridge_oracle.py gen $TINY 1
  log "vram_used_MB_at_start $(vram)"
  for spec in ALL:ALL:0.3 noxattn:noxattn:0.3 single:single:0.3 identical:identical:0.3 A:A:0.3 B:B:0.3 C:C:0.3 Ag:A:0.8 D:D:0.3; do
    IFS=: read -r name cfg ginit <<<"$spec"
    d=$W/case-tiny-$name; rm -rf $d; mkdir -p $d
    run env HOME=$HOME $BIN $cfg data=$TINY b=8 ginit=$ginit check=$d
  done
  run env HOME=$HOME $BIN ALL data=$TINY b=8 gate0=1
  for s in 1 2; do
    rm -rf $W/tiny-rerun$s
    run env HOME=$HOME $BIN ALL data=$TINY b=8 steps=20 every=10 ckpt=10 out=$W/tiny-rerun$s seed=3
  done
  a=$(sha256sum $W/tiny-rerun1/step-10.ckpt | cut -d' ' -f1); b=$(sha256sum $W/tiny-rerun2/step-10.ckpt | cut -d' ' -f1)
  log "tiny rerun step-10 sha256 $a $b"
  [ "$a" = "$b" ] && log "tiny rerun PASS" || { log "FAIL tiny rerun: checkpoints differ"; exit 1; }
  ;;
tiny-oracle)
  hdr
  fail=()
  for cfg in ALL noxattn single identical A B C Ag D; do
    run $PY tools/bridge_oracle.py check $W/case-tiny-$cfg $TINY 1e-4 || fail+=("$cfg")
  done
  run $PY tools/tests/bridge_oracle_mutants.py $W/case-tiny-A $TINY || fail+=(mutants)
  [ ${#fail[@]} -eq 0 ] && log "tiny-oracle PASS 9/9 + mutants" || { log "FAIL tiny-oracle ${#fail[@]}/9 ${fail[*]}"; exit 1; }
  ;;
synth-real)
  hdr
  SR=$ROOT/synth-real
  [ -f $SR/cfg.txt ] || run $PY tools/bridge_oracle.py gen $SR 2 real
  log "vram_used_MB_at_start $(vram)"
  run env HOME=$HOME $BIN ALL data=$SR b=32 gate0=1
  for s in 1 2; do
    rm -rf $W/srerun$s
    run env HOME=$HOME $BIN A data=$SR b=32 steps=200 every=100 ckpt=200 out=$W/srerun$s seed=5
  done
  a=$(sha256sum $W/srerun1/step-200.ckpt | cut -d' ' -f1); b=$(sha256sum $W/srerun2/step-200.ckpt | cut -d' ' -f1)
  log "synth-real rerun step-200 sha256 $a $b"
  [ "$a" = "$b" ] && log "synth-real rerun PASS" || { log "FAIL synth-real rerun: checkpoints differ"; exit 1; }
  for cfg in A ALL; do
    rm -rf $W/sspeed-$cfg
    run env HOME=$HOME $BIN $cfg data=$SR b=32 steps=60 every=60 ckpt=0 out=$W/sspeed-$cfg seed=1
  done
  ;;
real-grad)
  hdr
  log "vram_used_MB_at_start $(vram)"
  for spec in ${REAL_GRAD:-A:A:lam=0 C:C:}; do
    IFS=: read -r name cfg extra <<<"$spec"
    d=$W/case-real-$name; rm -rf $d; mkdir -p $d
    run env HOME=$HOME $BIN $cfg data=$ROOT b=32 ginit=0.3 check=$d $extra
  done
  ;;
real-oracle)
  hdr
  fail=()
  for name in ${REAL_ORACLE:-A C}; do
    run $PY tools/bridge_oracle.py check $W/case-real-$name $ROOT 1e-4 || fail+=("$name")
  done
  [ ${#fail[@]} -eq 0 ] && log "real-oracle PASS" || { log "FAIL real-oracle ${fail[*]} (see act bf16_flips lines)"; exit 1; }
  ;;
real-gate0)
  hdr
  run env HOME=$HOME $BIN ALL data=$ROOT b=32 gate0=1 split=${GATE0_SPLIT:-train}
  ;;
real-overfit)
  hdr
  rm -rf $W/overfit
  run env HOME=$HOME $BIN A data=$ROOT b=32 steps=300 every=25 ckpt=0 items=32 evalsplit=train out=$W/overfit seed=1 ${REAL_EXTRA:-lam=0}
  acc=$(grep '^step 300 ' $rc_file | sed 's/.*| acc \([0-9.]*\) \/ \([0-9.]*\).*/\2/')
  log "overfit eval acc at step 300: $acc"
  awk -v a="$acc" 'BEGIN { if (a + 0 >= 0.95) { print "overfit PASS"; exit 0 } else { print "FAIL overfit: acc " a " < 0.95"; exit 1 } }' | tee -a $rc_file
  ;;
real-rerun)
  hdr
  for s in 1 2; do
    rm -rf $W/rerun$s
    run env HOME=$HOME $BIN A data=$ROOT b=32 steps=200 every=50 ckpt=200 out=$W/rerun$s seed=5 ${REAL_EXTRA:-lam=0}
  done
  a=$(sha256sum $W/rerun1/step-200.ckpt | cut -d' ' -f1); b=$(sha256sum $W/rerun2/step-200.ckpt | cut -d' ' -f1)
  log "rerun step-200 sha256 $a $b"
  [ "$a" = "$b" ] && log "rerun PASS" || { log "FAIL rerun: checkpoints differ"; exit 1; }
  ;;
real-speed)
  hdr
  log "vram_used_MB_at_start $(vram)"
  for cfg in ${REAL_SPEED:-A C}; do
    rm -rf $W/speed-$cfg
    run env HOME=$HOME $BIN $cfg data=$ROOT b=32 steps=60 every=60 ckpt=0 out=$W/speed-$cfg seed=1 ${REAL_EXTRA:-lam=0}
  done
  grep -E "^RESULT" $rc_file | tee -a $rc_file.summary
  ;;
*)
  echo "usage: checks.sh build|tiny|tiny-oracle|real-grad|real-oracle|real-gate0|real-overfit|real-rerun|real-speed"; exit 2
  ;;
esac
