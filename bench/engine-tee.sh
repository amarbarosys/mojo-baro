#!/usr/bin/env bash
# Engine wrapper that records the wire: baro-serve --engine DIR/engine (a copy of this file) runs
# the engine named on line 1 of DIR/engine.real under the same PID (exec) and appends its stdin to
# <prefix>.in and its stdout to <prefix>.out, prefix = line 2. Config in a file, not env, because
# bench/vero-ab.sh reads back every knob. Input of bench/vero-force-agree.sh.
set -euo pipefail
real=$(sed -n 1p "$0.real") prefix=$(sed -n 2p "$0.real")
exec "$real" "$@" < <(tee -a "$prefix.in") > >(tee -a "$prefix.out")
