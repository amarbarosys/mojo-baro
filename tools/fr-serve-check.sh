#!/usr/bin/env bash
# Live check: tools/baro serve on the Qwythos bake answers one chat request; the engine log must read BARO_FR: True.
# usage: gpu-wait run --timeout 600 -- tools/fr-serve-check.sh   (log .work/fr-serve.log)
cd ~/Projects/mojo/mojo-baro
G=$(ls ~/Models/qwythos-9b-v2-mtp-q6_k/*BARO-eeb082e.gguf); L=.work/fr-serve.log
setsid timeout 300 tools/baro serve "$G" --port 8791 > $L 2>&1 & sp=$!
for i in $(seq 150); do curl -s -m 2 127.0.0.1:8791/health | grep -qiE "\"ok\"|status.:.ok|^ok" && break; sleep 1; done
curl -s -m 90 127.0.0.1:8791/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hi"}],"max_tokens":16000,"stream":false}' | head -c 160; echo
kill -- -$sp 2>/dev/null; sleep 3; ps -p $sp >/dev/null && echo still-alive || echo stopped
