#!/usr/bin/env bash
# usage: tools/spark-chat-ref.sh MODEL.gguf OUTDIR CHAT.json [PORT]
# Oracle for the chat path: llama-server renders CHAT.json's messages with the
# GGUF's own template (/apply-template, honouring chat_template_kwargs), tokenizes
# the result (/tokenize), and greedy-decodes 64 tokens from those ids.
# Writes prompt.txt, prompt-tokens.txt, ref-tokens-64.txt.
set -euo pipefail
model=$1; out=$2; chat=$3; port=${4:-8098}
mkdir -p "$out"
~/llama.cpp-master/build/bin/llama-server -m "$model" -c 4096 -ngl 99 -fa off -ctk f32 -ctv f32 -t 8 \
  --host 127.0.0.1 --port "$port" > "$out/server.log" 2>&1 &
pid=$!
trap 'kill $pid 2>/dev/null; wait $pid 2>/dev/null' EXIT
for i in $(seq 1 180); do curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break; sleep 1; done
python3 - "$chat" "$port" "$out" <<'PY'
import json, sys, urllib.request
chat, port, out = sys.argv[1:]
case = json.load(open(chat))
def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))
kw = {k: v for k, v in case.items() if k not in ("messages", "tools", "add_generation_prompt")}
body = {"messages": case["messages"], "chat_template_kwargs": kw}
if "tools" in case: body["tools"] = case["tools"]
prompt = post("/apply-template", body)["prompt"]
open(f"{out}/prompt.txt", "w").write(prompt)
ids = post("/tokenize", {"content": prompt, "add_special": False})["tokens"]
open(f"{out}/prompt-tokens.txt", "w").write("\n".join(map(str, ids)) + "\n")
d = post("/completion", {"prompt": ids, "n_predict": 64, "temperature": 0, "top_k": 1, "cache_prompt": False, "return_tokens": True})
open(f"{out}/ref-tokens-64.txt", "w").write("\n".join(map(str, d["tokens"])) + "\n")
print("prompt chars:", len(prompt), "ids:", len(ids), "gen:", len(d["tokens"]), "tg tok/s:", d.get("timings", {}).get("predicted_per_second"))
print(repr(prompt[:160]))
print(repr(d["content"][:160]))
PY
