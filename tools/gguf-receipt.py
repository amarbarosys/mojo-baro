#!/usr/bin/env python3
"""The receipt ledger of a self-describing BARO gguf (B1).

  tools/gguf-receipt.py MODEL.gguf --list
  tools/gguf-receipt.py MODEL.gguf --append '<json object>' [--in-file [--out DST]]

A receipt records that one card verified this exact file: its gfx target, card
name, driver and ROCm version, power cap, the tok/s the rebuilt engine reached
on that card, and the commit the harness was checked against. `baro.hw.*` says
what the baking card measured; the ledger says what other cards found.

Only `tools/baro verify` writes here, and every record carries `mode`, which it
sets to "verified". A run-mode number is self-reported and never becomes a
receipt: the file supplied its own stopwatch in that mode.

Two stores, because a GGUF KV cannot be extended in place:

- default, a sidecar `MODEL.gguf.baro-receipts.jsonl` next to the model. One
  JSON object per line, append-only, costs nothing.
- `--in-file`, which rewrites the whole container with the merged ledger in
  `baro.hw.receipts` (a JSON array as a string). That is a full copy of the
  file, 21 GB for the MoE bake, so it is opt-in and writes to `--out` (default
  `MODEL.receipts.gguf`), never over the original.

`--list` merges both stores and prints them oldest first, sidecar records
marked `(sidecar)`.
"""
import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KEY = "baro.hw.receipts"


def load(name):
    spec = spec_from_file_location(name.replace("-", "_"), ROOT / "tools" / f"{name}.py")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sidecar_path(model):
    return model.with_name(model.name + ".baro-receipts.jsonl")


def read_kv(model):
    ge = load("gguf-extract")
    _, _, _, kv = ge.parse(str(model))
    return {(k.decode("utf-8") if isinstance(k, bytes) else k): v for k, v in kv.items()}


def in_file_records(kv):
    raw = kv.get(KEY)
    if not raw:
        return []
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        recs = json.loads(raw)
    except json.JSONDecodeError:
        print(f"warning: {KEY} is not JSON, ignoring it", file=sys.stderr)
        return []
    return recs if isinstance(recs, list) else [recs]


def sidecar_records(model):
    p = sidecar_path(model)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def main():
    if len(sys.argv) < 3:
        print(__doc__.strip().splitlines()[2], file=sys.stderr)
        return 2
    model = Path(sys.argv[1]).resolve()
    args = sys.argv[2:]

    if args[0] == "--list":
        kv = read_kv(model)
        recs = [(r, "") for r in in_file_records(kv)] + [(r, " (sidecar)") for r in sidecar_records(model)]
        if not recs:
            print(f"{model.name}: no receipts yet")
            print("a verified card adds one with: tools/baro verify MODEL.gguf --append")
            return 0
        print(f"{model.name}: {len(recs)} receipt(s)")
        for r, tag in recs:
            print(f"  {r.get('date', '?')}  {r.get('mode', '?')}  {r.get('gfx', '?')} "
                  f"{r.get('card', '?')}  tok/s_gen {r.get('tok_s_gen_one_prompt', '?')}  "
                  f"vs commit {r.get('checked_against_commit', '?')}{tag}")
        return 0

    if args[0] != "--append":
        print(f"unknown option {args[0]}", file=sys.stderr)
        return 2

    rec = json.loads(args[1])
    if rec.get("mode") != "verified":
        print("refused: a receipt must carry mode=verified; run-mode numbers are self-reported "
              "and do not become receipts", file=sys.stderr)
        return 1
    if not rec.get("checked_against_commit"):
        print("refused: a receipt must name the commit its harness was checked against",
              file=sys.stderr)
        return 1

    in_file = "--in-file" in args
    if not in_file:
        p = sidecar_path(model)
        with p.open("a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        print(f"receipt appended to {p} ({len(sidecar_records(model))} total)")
        return 0

    dst = Path(args[args.index("--out") + 1]) if "--out" in args else \
        model.with_name(model.name.replace(".gguf", "") + ".receipts.gguf")
    if dst.exists():
        print(f"refused: {dst} exists", file=sys.stderr)
        return 1
    kv = read_kv(model)
    merged = in_file_records(kv) + sidecar_records(model) + [rec]
    embed = load("gguf-embed")
    # Keep every other baro.* key: only the ledger changes here, so the drop
    # list is the ledger key alone and the rest are carried through as-is.
    keep = [(k, v.decode("utf-8") if isinstance(v, bytes) else v)
            for k, v in kv.items() if k.startswith("baro.") and k not in (KEY, embed.PAD_KEY)]
    added, dropped = embed.rewrite(str(model), str(dst),
                                   keep + [(KEY, json.dumps(merged))])
    print(f"wrote {dst} with {len(merged)} receipt(s) in {KEY} "
          f"(+{added} baro kv, {dropped} replaced)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
