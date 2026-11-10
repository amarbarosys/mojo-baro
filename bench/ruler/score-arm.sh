#!/usr/bin/env bash
# bench/ruler/score-arm.sh: collect one arm's per-size RULER responses into
# a single all/ dir and score them (bench/ruler-protocol.md item E1).
#
# Usage: bench/ruler/score-arm.sh ARM_DIR [SIZES] [--out OUT_DIR] [--prompts DIR]
#   ARM_DIR = .../ruler-baseline/<arm>  (layout: ARM_DIR/<size>/responses/<task>_<size>)
#   SIZES   = comma-separated sizes to link/score (default: every size dir present)
#   OUT_DIR = where all/ and table.json are written (default: ARM_DIR itself;
#             pass this to keep a read-only ARM_DIR untouched)
#   --prompts DIR = passed through to score.py (default: bench/ruler/prompts,
#             gitignored/generated; point at the arm's own source repo's
#             prompts/ when scoring a foreign arm dir)
set -euo pipefail
cd "$(dirname "$0")/../.."

ARM_DIR=""
SIZES=""
OUT_DIR=""
PROMPTS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT_DIR=$2; shift 2 ;;
    --prompts) PROMPTS=$2; shift 2 ;;
    *)
      if [ -z "$ARM_DIR" ]; then ARM_DIR=$1
      elif [ -z "$SIZES" ]; then SIZES=$1
      else echo "score-arm: unexpected argument: $1" >&2; exit 1
      fi
      shift ;;
  esac
done
[ -n "$ARM_DIR" ] || { echo "usage: score-arm.sh ARM_DIR [SIZES] [--out OUT_DIR]" >&2; exit 1; }
[ -d "$ARM_DIR" ] || { echo "score-arm: ARM_DIR not found: $ARM_DIR" >&2; exit 1; }
OUT_DIR=${OUT_DIR:-$ARM_DIR}

all_dir="$OUT_DIR/all"
mkdir -p "$all_dir"

if [ -n "$SIZES" ]; then
  IFS=',' read -ra size_names <<< "$SIZES"
else
  size_names=()
  for d in "$ARM_DIR"/*/; do
    d=${d%/}
    [ -d "$d/responses" ] && size_names+=("$(basename "$d")")
  done
fi

linked=0
for size in "${size_names[@]}"; do
  resp_dir="$ARM_DIR/$size/responses"
  [ -d "$resp_dir" ] || continue
  for task_dir in "$resp_dir"/*/; do
    [ -d "$task_dir" ] || continue
    name=$(basename "$task_dir")
    ln -sfn "$(cd "$task_dir" && pwd)" "$all_dir/$name"
    linked=$((linked + 1))
  done
done
[ "$linked" -gt 0 ] || { echo "score-arm: no <size>/responses/<task>_<size> dirs found under $ARM_DIR" >&2; exit 1; }

score_args=(bench/ruler/score.py "$all_dir" --json "$OUT_DIR/table.json")
if [ -n "$SIZES" ]; then
  score_args+=(--sizes "$SIZES")
fi
if [ -n "$PROMPTS" ]; then
  score_args+=(--prompts "$PROMPTS")
fi
./.venv/bin/python3 "${score_args[@]}"
