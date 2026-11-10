#!/usr/bin/env python3
"""CPU checker for a BARO gguf: seconds, no GPU, no build.

  tools/gguf-check.py MODEL-BARO.gguf [CHALLENGE.json]   (default MODEL.gguf.challenge.json)

Claims, each printed PASS / FAIL / NOT CHECKED:
  tensor-bytes   sha256 of the data region == baro.data.sha256
  challenge      every key in the challenge (kept OUTSIDE the file) equals the file's,
                 and its baro.data.sha256 equals the recomputed hash
Exit 1 on any FAIL. NOT CHECKED is not a pass: sources-complete and token identity
need the GPU closure (tools/gguf-closure.sh).
"""
import hashlib
import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(name):
    spec = spec_from_file_location(name.replace("-", "_"), ROOT / "tools" / f"{name}.py")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    model = sys.argv[1]
    chpath = Path(sys.argv[2] if len(sys.argv) > 2 else model + ".challenge.json")
    f, _, data_start, kv = load("gguf-extract").parse(model)
    f.seek(data_start)
    got = hashlib.file_digest(f, "sha256").hexdigest()
    fail = False

    def say(claim, ok, why):
        nonlocal fail
        fail |= ok is False
        print(f"{'NOT CHECKED' if ok is None else 'PASS' if ok else 'FAIL'} {claim}: {why}")

    want = kv.get("baro.data.sha256")
    if want is None:
        say("tensor-bytes", None, "file has no baro.data.sha256 (baked before it existed)")
    else:
        say("tensor-bytes", got == want, f"data {got[:16]} vs file {want[:16]}")
    if not chpath.is_file():
        say("challenge", None, f"no {chpath}")
    else:
        ch = json.loads(chpath.read_text())
        bad = [k for k, v in ch.items() if kv.get(k) != v]
        if ch.get("baro.data.sha256", got) != got:
            bad.append("data recomputed")
        say("challenge", not bad, f"{len(ch)} keys, mismatch: {', '.join(bad) or 'none'}")
    say("sources-complete", None, "needs tools/gguf-closure.sh")
    say("token-identity", None, "needs tools/gguf-closure.sh")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
