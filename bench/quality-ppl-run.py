#!/usr/bin/env python3
"""Perplexity, ours arm. bench/quality-protocol.md (amendment 2).

Token stream: the text tokenized by a CPU llama-server on the reference GGUF
(`/tokenize`, add_special true), chunked n_ctx at a time, token 0 of each
chunk replaced by BOS when the vocab adds BOS: exactly
tools/perplexity/perplexity.cpp. Scored positions: logits at 256..510
predicting tokens 257..511 (first = n_ctx/2), 255 per chunk.

Engine: serve/engine.mojo over its BARO_SERVE=1 stdin protocol. First request
replays the bake's baro.run.prompt.tokens and must reproduce
baro.run.ref.tokens (else VOID). Then one teacher-forced request per chunk
("force", top_logprobs:1 so the engine dumps each step's raw logits row to
BARO_DUMP_LOGITS_DIR as row-<step>.bin); rows are read and the dump dir
cleared before the next request is written.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np
import requests


def post(url, path, body):
    r = requests.post(url.rstrip("/") + path, json=body, timeout=600)
    r.raise_for_status()
    return r.json()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--tok-url", required=True, help="CPU llama-server on the reference GGUF")
    ap.add_argument("--ref-prompt", required=True, help="baro.run.prompt.tokens (space separated)")
    ap.add_argument("--ref-tokens", required=True, help="baro.run.ref.tokens (space separated)")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dumpdir = os.path.join(args.out, "dump")
    shutil.rmtree(dumpdir, ignore_errors=True)
    os.makedirs(dumpdir)

    text = open(args.text, encoding="utf-8").read()
    ids = post(args.tok_url, "/tokenize", {"content": text, "add_special": True})["tokens"]
    bos_only = post(args.tok_url, "/tokenize", {"content": "", "add_special": True})["tokens"]
    add_bos = len(bos_only) == 1
    bos = bos_only[0] if add_bos else None
    need = args.ctx * args.chunks
    if len(ids) < need:
        print(f"VOID: text tokenizes to {len(ids)} ids, need {need}", file=sys.stderr)
        sys.exit(3)
    chunks = []
    for c in range(args.chunks):
        ch = ids[c * args.ctx:(c + 1) * args.ctx]
        if add_bos:
            ch = [bos] + ch[1:]
        chunks.append(ch)
    first = args.ctx // 2

    env = dict(os.environ)
    env.update(BARO_SERVE="1", BARO_SPEC="0", BARO_PACK=args.pack, BARO_DUMP_LOGITS_DIR=dumpdir)
    proc = subprocess.Popen([args.engine], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=open(os.path.join(args.out, "engine.err"), "w"),
                            env=env, text=True, bufsize=1)
    ready_line = None
    while True:
        line = proc.stdout.readline()
        if not line:
            print("VOID: engine exited before ready line", file=sys.stderr)
            sys.exit(4)
        if line.startswith('{"ready":true'):
            ready_line = line.strip()
            break

    def request(req):
        proc.stdin.write(json.dumps(req) + "\n")
        proc.stdin.flush()
        toks = []
        while True:
            line = proc.stdout.readline()
            if not line:
                print(f"VOID: engine exited during request {req['id']}", file=sys.stderr)
                sys.exit(5)
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("id") != req["id"]:
                continue
            if "error" in d:
                print(f"VOID: engine error on request {req['id']}: {d}", file=sys.stderr)
                sys.exit(5)
            if "tok" in d:
                toks.append(d["tok"])
            if d.get("done"):
                return toks, d

    ref_prompt = [int(x) for x in args.ref_prompt.split()]
    ref = [int(x) for x in args.ref_tokens.split()]
    got, _ = request({"id": 1, "prompt": ref_prompt, "n": len(ref), "spec": False})
    match = sum(1 for a, b in zip(got, ref) if a == b)
    identity = {"ref_n": len(ref), "got_n": len(got), "prefix_match": next((i for i, (a, b) in enumerate(zip(got, ref)) if a != b), min(len(got), len(ref)))}
    print(f"identity: {identity}", file=sys.stderr)
    shutil.rmtree(dumpdir)
    os.makedirs(dumpdir)
    if got != ref:
        print(f"VOID: engine+pack do not reproduce baro.run.ref.tokens ({match}/{len(ref)})", file=sys.stderr)
        sys.exit(8)

    total_nll = 0.0
    total_n = 0
    vocab = None
    per_chunk = []
    for c, ch in enumerate(chunks):
        _, done = request({"id": 10 + c, "prompt": [ch[0]], "n": args.ctx - 1, "spec": False,
                           "top_logprobs": 1, "force": ch[1:]})
        rows = [fn for fn in os.listdir(dumpdir) if fn.startswith("row-") and fn.endswith(".bin")]
        if len(rows) != args.ctx - 1:
            print(f"VOID: chunk {c} dumped {len(rows)} rows, expected {args.ctx - 1}", file=sys.stderr)
            sys.exit(6)
        nll = 0.0
        for pos in range(first, args.ctx - 1):
            x = np.fromfile(os.path.join(dumpdir, f"row-{pos}.bin"), dtype=np.float32).astype(np.float64)
            if vocab is None:
                vocab = x.size
            m = x.max()
            lse = m + math.log(np.exp(x - m).sum())
            nll += lse - x[ch[pos + 1]]
        shutil.rmtree(dumpdir)
        os.makedirs(dumpdir)
        n = args.ctx - 1 - first
        total_nll += nll
        total_n += n
        per_chunk.append({"chunk": c, "nll": nll, "n": n, "ppl": math.exp(nll / n), "done": done})
        print(f"chunk {c}: ppl {math.exp(nll / n):.4f} running {math.exp(total_nll / total_n):.4f}", file=sys.stderr)

    proc.stdin.close()
    proc.wait(timeout=60)
    ppl = math.exp(total_nll / total_n)
    result = {"engine": args.engine, "pack": args.pack, "ctx": args.ctx, "chunks": args.chunks,
              "token_count": len(ids), "add_bos": add_bos, "bos": bos, "chunk0_head": chunks[0][:8],
              "scored_from": first, "vocab": vocab, "total_nll": total_nll, "total_n": total_n,
              "ppl": ppl, "identity": identity, "ready": ready_line, "per_chunk": per_chunk}
    with open(os.path.join(args.out, "ppl-result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"PPL(ours) = {ppl:.4f} ({total_n} scored tokens, vocab {vocab}, add_bos {add_bos})")


if __name__ == "__main__":
    main()
