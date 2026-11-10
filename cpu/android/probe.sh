#!/usr/bin/env bash
# CPU backend steps 0a/0b: build with the mojo package only (no max), cross-compile
# to aarch64 Android, link the KGEN runtime shim with the NDK, run on an adb device.
# Usage: [ANDROID_SERIAL=<serial>] cpu/android/probe.sh [MOJO_BIN]   (default .work/mojo-only/bin/mojo;
# ANDROID_SERIAL picks the device when adb sees more than one)
set -euo pipefail
cd "$(dirname "$0")/../.."
MOJO=${1:-.work/mojo-only/bin/mojo}
NDK=${NDK:-/opt/android-sdk/ndk/29.0.13113456/toolchains/llvm/prebuilt/linux-x86_64/bin}
OUT=.work/cpu-android-probe; mkdir -p "$OUT"
fail() { echo "FAIL $1: $2 (log $OUT/log)"; exit 1; }
[ -x "$MOJO" ] || fail setup "no mojo at $MOJO (python -m venv .work/mojo-only && .work/mojo-only/bin/pip install mojo==1.2.0)"
adb get-state >/dev/null 2>&1 || fail adb "no device"
for p in hello pthread; do
  "$MOJO" build --emit object --target-triple aarch64-unknown-linux-android --target-cpu generic \
    cpu/android/probe_$p.mojo -o "$OUT/$p.o" >"$OUT/log" 2>&1 || fail "build $p" mojo
  "$NDK/aarch64-linux-android29-clang" -O2 "$OUT/$p.o" cpu/android/mojo_rt_shim.c -o "$OUT/$p" >>"$OUT/log" 2>&1 || fail "link $p" ndk
  adb push "$OUT/$p" /data/local/tmp/baro-probe-$p >/dev/null
done
got=$(adb shell 'cd /data/local/tmp && chmod +x baro-probe-* && ./baro-probe-hello && ./baro-probe-pthread')
echo "$got"
[ "$got" = $'hello android 22.5\npthread sum 360 expect 360' ] || fail run "unexpected output"
echo "PASS android probe $(adb shell getprop ro.product.model) abi=$(adb shell getprop ro.product.cpu.abi)"
