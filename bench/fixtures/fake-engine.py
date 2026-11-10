#!/usr/bin/env python3
"""CPU-only baro-serve engine fixture for the P6 PWA gate."""

import json
import os
import sys


TOKENS = [1, 2, 3]
TOOL_TOKENS = [4, 5, 6, 7, 8, 9, 10]


def emit(value):
    print(json.dumps(value, separators=(",", ":")), flush=True)


def main():
    pack = os.environ.get("BARO_PACK", "fake-pack")
    emit({"ready": True, "tmax": 4096, "mrows": 8, "kmax": 0, "spec_k": 0, "pack": pack})
    for line in sys.stdin:
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue
        request_id = int(request.get("id", 0))
        tokens = TOOL_TOKENS if request.get("n") == len(TOOL_TOKENS) else TOKENS
        for token in tokens:
            emit({"id": request_id, "tok": token})
            if request.get("hidden"):
                emit({"id": request_id, "hidden": [0.1, 0.2]})
            if request.get("logits_topk"):
                emit({"id": request_id, "logits_topk": [{"id": 7, "logit": 3.5}]})
        emit({
            "id": request_id,
            "done": True,
            "n": len(tokens),
            "prefill_s": 0.0,
            "decode_s": 0.001,
            "tok_s": 3000.0,
            "finish": "length",
        })


if __name__ == "__main__":
    main()
