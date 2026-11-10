#!/usr/bin/env bash
# Stub whisper-server for the baro-serve audio tests: takes the real argument list, opens --port only after a 1 s
# delay (a slow model load), then accepts and discards connections.
set -u
while [ $# -gt 0 ]; do [ "$1" = --port ] && port=$2; shift; done
sleep 1
exec socat "TCP-LISTEN:${port:?no --port},reuseaddr,fork,bind=127.0.0.1" "SYSTEM:cat >/dev/null"
