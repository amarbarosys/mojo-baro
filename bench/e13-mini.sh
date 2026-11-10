#!/usr/bin/env bash
# E13-mini end to end (AMDHQ 06-experiments.md E13-mini): dump k=8 for 1,000
# GSM8K train items, train the projector 150 batched steps, evaluate arms 0 and
# L8-proj on the 100 round-5 math items, verdict. One gpu-wait job.
#   bench/e13-mini.sh [OUT_DIR]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=$(readlink -f "${1:-.work/e13-mini}")
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
  bench/preflight.sh --check
  mkdir -p "$OUT"
  ./.venv/bin/mojo build bench/e13_engine_dump.mojo -o .work/e13/e13_engine_dump -I . -I kernels -I serve -I bench > "$OUT/build-dump.log" 2>&1
  ./.venv/bin/mojo build bench/bench_latent_handoff.mojo -o .work/bench_latent_handoff -I . -I kernels -I serve -I bench > "$OUT/build-eval.log" 2>&1
  exec gpu-wait run --priority 20 --vram 22 --timeout 3600 -- "$0" "$OUT"
fi
t0=$(date +%s); stamp() { echo "[$(( $(date +%s) - t0 ))s] $*" | tee -a "$OUT/timeline.txt"; }
: > "$OUT/timeline.txt"
echo "head=$(git rev-parse --short HEAD) amdhq=$(git -C $HOME/AMDHQ rev-parse --short HEAD)" | tee "$OUT/arm.txt"

stamp dump start
env BARO_PACK=.work/engine-pack-q4 E13_DUMP_LIMIT=1000 E13_SKIP_K32=1 E13_DUMP_DIR="$OUT" \
  .work/e13/e13_engine_dump --mode dump > "$OUT/dump.log" 2>&1
stamp "dump done: $(grep run_full_dump "$OUT/dump.log")"

stamp train start
$HOME/AMDHQ/.venv/bin/python $HOME/AMDHQ/tools/latent-os/e13_train.py --mode smoke --smoke-steps 150 \
  --dump "$OUT/train-k8.bin" --k 8 --batch-mode batched --micro-batch 8 --eval-every 25 \
  --out "$OUT/projector-k8.bin" --report "$OUT/train-report.json" > "$OUT/train.log" 2>&1
stamp "train done: $(python3 -c "import json;d=json.load(open('$OUT/train-report.json'));print(d['steps_run'],'steps',d['step_time_s_mean'],'s/step last',d['last_loss'],'holdout',d['holdout_last'])")"

stamp eval start
ids=$(python3 -c "import json;print(','.join(t['id'] for t in json.load(open('bench/data/e8_tasks.json')) if t['type']=='math'))")
env BARO_PACK=.work/engine-pack-q4 BARO_E8_RECV_MAX=32 BARO_E13_PROJ_K8="$OUT/projector-k8.bin" \
  .work/bench_latent_handoff --ids "$ids" --arms 0,L8-proj --out "$OUT/eval" > "$OUT/eval.log" 2>&1
python3 bench/e8_score.py "$OUT/eval.raw.json" bench/data/e8_tasks.json "$OUT/eval.json" "$OUT/eval.md" > "$OUT/score.log" 2>&1
stamp eval done

python3 bench/e13_mini_verdict.py "$OUT/eval.json" "$OUT/train-report.json" | tee "$OUT/verdict.txt"
stamp total
