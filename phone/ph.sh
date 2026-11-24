#!/usr/bin/env bash
# ph.sh: locked adb on the Xiaomi (Adreno 840). Every call under timeout, corewait phone lock held.
#   ph.sh push SECS LOCAL... (to $T)    ph.sh run SECS 'CMD' (in $T; GPU bins: 'env LD_LIBRARY_PATH=/vendor/lib64 ./bin')
#   ph.sh pull SECS REMOTE LOCAL        ph.sh clean 10  (kill leftovers, verify none run)
set -euo pipefail
model=25128PNA1C T=/data/local/tmp/baro
serial=$(timeout 10 adb devices -l | awk -v m="model:$model" '$0 ~ m && $2 == "device" {print $1; exit}')
[ -n "$serial" ] || { echo "FAIL adb: xiaomi not attached" >&2; exit 1; }
mkdir -p ~/.cache/corewait
exec 9> ~/.cache/corewait/phone-"$serial".lock
flock -w 120 9 || { echo "FAIL lock: xiaomi busy" >&2; exit 1; }
[ "${BARO_PHONE_LOCK_HELD:-0}" = 1 ] || { exec 8> ~/Android/baro/.work/phone-"$serial".lock; flock -w 120 8 || { echo "FAIL lock: xiaomi busy (baro lock)" >&2; exit 1; }; }
A() { timeout "$1" adb -s "$serial" "${@:2}"; }
op=$1; s=$2; shift 2
case $op in
  push) A 20 shell "mkdir -p $T"; for f in "$@"; do A "$s" push "$f" $T/ | tail -1; done ;;
  run)  A "$s" shell "cd $T && $1; echo rc=\$?" ;;  # prefix GPU bins with \$CL (a global LD_LIBRARY_PATH breaks /system bins)
  pull) A "$s" pull "$1" "$2" | tail -1 ;;
  cache) M=/data/local/tmp/baro-models  # ph.sh cache SECS SHA NAME: move $T/NAME into gate3's sha cache, link back
         A "$s" shell "mkdir -p $M && { [ -L $T/$2 ] || mv $T/$2 $M/$1.gguf; } && ln -sf $M/$1.gguf $T/$2 && stat -c %s $M/$1.gguf" ;;
  clean) A "$s" shell "for p in llama-bench llama_ref llama_ocl engine wwk; do for q in \$(pidof \$p); do kill \$q; done; done; sleep 1; pidof llama-bench llama_ref llama_ocl engine wwk || echo clean" ;;  # pidof: pkill -f matched this shell itself (exit 143)
  *) echo "FAIL op: $op" >&2; exit 2 ;;
esac
