#!/usr/bin/env bash
# fsm-wire-stub: a fake serve-mode engine standing in for serve/engine.mojo so bench/fsm-wire-gate.sh runs end to end on
# the CPU. It proves the GATE SCRIPT (probe logic, verdict, exit codes), not the engine: its answers are scripted.
#   bench/fsm-wire-stub.sh --make DIR   writes DIR/stub-cand (fixed behaviour) and DIR/stub-base (old behaviour), prints both paths
#   GPU_WAITING_ROOM_JOB=stub EXPLORE=1 bench/fsm-wire-gate.sh DIR/stub-cand DIR/stub-base OUT      -> UNVERIFIED, exit 3, all ok
#   the same with the two arguments swapped                                                       -> FAIL (the gate can fail)
# run as an engine (no argument) it reads STUB_MODE=cand|base and speaks the protocol on stdin/stdout.
set -euo pipefail
if [ "${1:-}" = --make ]; then
  dir=${2:?DIR}; mkdir -p "$dir"; self=$(readlink -f "$0")
  for m in cand base; do
    printf '#!/usr/bin/env bash\nSTUB_MODE=%s exec bash %s\n' "$m" "$self" > "$dir/stub-$m"; chmod +x "$dir/stub-$m"; echo "$dir/stub-$m"
  done
  exit 0
fi
mode=${STUB_MODE:-cand}
if [ "$mode" = cand ]; then echo '{"ready":true,"tmax":16384,"mrows":4,"kmax":3,"spec_k":0,"kv":"f32","pack":"stub","tier":"off","seqs":1}'
else echo '{"ready":true,"tmax":16384,"mrows":4,"kmax":3,"spec_k":0,"kv":"f32","pack":"stub"}'; fi
while IFS= read -r line; do
  id=$(sed -n 's/.*"id": *\([0-9]*\).*/\1/p' <<< "$line"); id=${id:-0}
  case "$line" in
    *'"cancel"'*)
      if [ "$mode" = cand ]; then echo "cancel: id 99 is not decoding, dropped"; else echo '{"id":0,"error":"missing or non-integer id"}'; fi
      continue;;
    *latent_resume*)
      if [ "$mode" = cand ]; then echo "{\"id\":$id,\"error\":\"latent_resume: unknown latent_id lat_nope\"}"; continue; fi
      exit 1;;
    *latent_fork*)
      if [ "$mode" = cand ]; then echo "{\"id\":$id,\"error\":\"latent_fork: no checkpoint in the chain\"}"
      else echo '{"latent_id": "lat_1", "tokens": 0, "bytes": 0}'; fi
      continue;;
    *state_prefix*)
      if [ "$mode" = cand ]; then echo "{\"id\":$id,\"error\":\"prompt+n exceeds TMAX 16384 (after prefix extension)\"}"; continue; fi
      exit 2;;
    *slens_ctx*)
      if [ "$mode" = cand ]; then
        echo "{\"id\":$id,\"error\":\"slens: no base for context 5\"}"; echo "{\"id\":$id,\"done\":true,\"n\":1,\"finish\":\"error\",\"fail\":0}"; continue
      fi
      exit 1;;
  esac
  echo "{\"id\":$id,\"tok\":11}"; echo "{\"id\":$id,\"tok\":12}"
  if [ "$mode" = cand ]; then echo "{\"id\":$id,\"done\":true,\"n\":2,\"finish\":\"length\",\"fail\":0,\"path\":\"launch\",\"grammar\":\"none\"}"
  else echo "{\"id\":$id,\"done\":true,\"n\":2,\"finish\":\"length\"}"; fi
done
