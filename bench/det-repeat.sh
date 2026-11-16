#!/usr/bin/env bash
# Fresh-process repeats of one long decode (spec off/on), first divergent token per run: ENG, MODES, REPS, NTOK, EXTRA (e.g. BARO_SUMPROBE=1). Run under gpu-wait.
cd ~/Projects/mojo/mojo-baro
E=${ENG:-.work/fewrow/eng-turbo}
for m in ${MODES:-off on}; do for r in ${REPS:-1 2 3}; do
  if [ $m = off ]; then rs=false; ev="BARO_SPEC=0"; else rs=true; ev="BARO_SPEC=1 BARO_SPEC_K=2"; fi
  env GPU_WAITING_ROOM_JOB=1 REQ_SPEC=$rs bench/energy-run.sh .work/det $E ${NTOK:-4096} "$ev HSA_ENABLE_MWAITX=1 ${EXTRA:-}" $m$r 2>&1 | tail -1 | cut -d' ' -f1-3
done; done
