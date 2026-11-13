#!/usr/bin/env bash
# Install the CoiOS embedding endpoint on :8081 as a user service: baro-embed.service (baro-serve --embed-only, CPU),
# the COIOS_EMBED_* drop-in for cois-hub, and coios-embed.service (BGE-M3, GPU) disabled. Idempotent.
# Needs embed/build.sh run first (it fills ~/.cache/baro/embed). Does not restart cois-hub (the shared MCP endpoint of
# every local session): the drop-in applies at its next restart. Machine-state note: ~/Brain/OS/coios-embed-service.md.
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
U=$HOME/.config/systemd/user
for f in baro-serve serve_embed norm.tsv; do [ -s "$HOME/.cache/baro/embed/$f" ] || { echo "FAIL prereq: $HOME/.cache/baro/embed/$f missing (run embed/build.sh)"; exit 1; }; done
[ -d "$HOME/Models/bge-small-en-v1.5" ] || { echo "FAIL prereq: ~/Models/bge-small-en-v1.5 missing"; exit 1; }
mkdir -p "$U/cois-hub.service.d"
install -m 644 systemd/baro-embed.service "$U/baro-embed.service"
install -m 644 systemd/cois-hub-embed.conf "$U/cois-hub.service.d/embed.conf"
systemctl --user daemon-reload
systemctl --user disable coios-embed.service 2>/dev/null || true
systemctl --user enable --now baro-embed.service
for _ in $(seq 25); do curl -sf localhost:8081/health > /dev/null && break; sleep 0.4; done
curl -sf localhost:8081/v1/embeddings -H 'content-type: application/json' -d '{"input":"CoiOS sovereign memory kernel","model":"bge-small-en-v1.5"}' \
  | python3 -c 'import sys,json; d=json.load(sys.stdin); n=len(d["data"][0]["embedding"]); assert n==384; print("OK baro-embed :8081 dim", n)' \
  || { echo "FAIL baro-embed: no 384-dim answer on :8081, see journalctl --user -u baro-embed"; exit 1; }
echo "coios-embed.service: $(systemctl --user is-enabled coios-embed.service 2>&1 || true) / $(systemctl --user is-active coios-embed.service 2>&1 || true)"
