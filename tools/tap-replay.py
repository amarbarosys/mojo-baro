#!/usr/bin/env python3
"""Replay the chat requests of an openai-tap log against a baro-serve URL and
print the prefix-checkpoint receipt per request (bench/chat-protocol.md M1a
P-F3): prompt_tokens, baro.cached_tokens, baro.prefill_rows, engine prefill_s /
restore_s and the client-side wall TTFT (request start -> first byte, the
request is non-streaming with max_tokens tokens so wall ~ prefill + n tokens).

usage: tools/tap-replay.py URL TAP.jsonl [--rows 0,1,2] [--max-tokens 4] [--repeat N]
"""
import argparse, json, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("url"); ap.add_argument("tap")
ap.add_argument("--rows", default="0,1,2"); ap.add_argument("--max-tokens", type=int, default=4)
ap.add_argument("--repeat", type=int, default=1)
a = ap.parse_args()
rows = [json.loads(l) for l in open(a.tap)]
sel = [int(x) for x in a.rows.split(",")]
print("row prompt_tokens cached prefill_rows prefill_s restore_s wall_s decode_s")
for rep in range(a.repeat):
    for i in sel:
        req = rows[i]["request"]
        body = {"messages": req["messages"], "max_tokens": a.max_tokens, "stream": False, "spec": False}
        t0 = time.time()
        r = urllib.request.Request(a.url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                   headers={"content-type": "application/json"})
        d = json.load(urllib.request.urlopen(r, timeout=600))
        wall = time.time() - t0
        u, tm = d["usage"], d["timings"]
        print(f'{i} {u["prompt_tokens"]} {u["baro"]["cached_tokens"]} {u["baro"]["prefill_rows"]} '
              f'{tm["prefill_s"]:.4f} {tm.get("restore_s") or 0:.4f} {wall:.4f} {tm["decode_s"]:.4f}')
