#!/usr/bin/env bash
# Stub engine for the baro-serve unit tests: speaks the serve/PROTOCOL.md wire (ready line, token lines, done or
# error line, cancel line) with no model. STUB_MODE picks the behaviour, STUB_LOG gets one line per event.
#   ok         3 tokens then done
#   slow       a token every 50 ms (up to 400) until {"cancel":ID} arrives, then done finish=cancelled
#   error      {"id":N,"error":"boom"} for every request
#   die_idle   ready, then exit 3 after 300 ms with no request
#   die_mid    one token for the first request, then exit 1 (no done line)
#   drop_done  3 tokens, no done line, then exit 0
set -u
mode=${STUB_MODE:-ok}
log=${STUB_LOG:-/dev/null}
echo '{"ready":true,"tmax":4096,"mrows":1,"kmax":1,"spec_k":0,"pack":"stub","kv":"f32"}'
echo "ready mode=$mode" >>"$log"
trap 'echo "stdin-eof" >>"$log"' EXIT
if [ "$mode" = die_idle ]; then sleep 0.3; echo "exit 3" >>"$log"; exit 3; fi
done_line() { echo "{\"id\":$1,\"done\":true,\"n\":${3:-3},\"prefill_s\":0.0,\"decode_s\":0.0,\"tok_s\":0.0,\"finish\":\"$2\"}"; }
while IFS= read -r line; do
  case "$line" in *'"cancel":'*) echo "idle-cancel" >>"$log"; continue ;; esac
  id=${line#*\"id\":}; id=${id%%,*}
  echo "request $id" >>"$log"
  echo "line $line" >>"$log"
  case "$mode" in
    ok) for t in 1 2 3; do echo "{\"id\":$id,\"tok\":$t}"; done; done_line "$id" length ;;
    error) echo "{\"id\":$id,\"error\":\"boom\"}" ;;
    die_mid) echo "{\"id\":$id,\"tok\":1}"; echo "exit 1" >>"$log"; exit 1 ;;
    drop_done) for t in 1 2 3; do echo "{\"id\":$id,\"tok\":$t}"; done; exit 0 ;;
    slow)
      i=0
      while [ $i -lt 400 ]; do
        if read -t 0.05 -r l; then
          case "$l" in *'"cancel":'*) echo "cancel $id" >>"$log"; done_line "$id" cancelled "$i"; continue 2 ;; esac
        fi
        echo "{\"id\":$id,\"tok\":$((i % 50))}"
        i=$((i + 1))
      done
      echo "finished $id" >>"$log"; done_line "$id" length 400 ;;
  esac
done
