#!/usr/bin/env python3
"""B4 stage 2: replay an LRU of resident experts over a BARO_EXPERTS trace.

usage: bench/moe-locality.py TRACE [--bandwidth GBps] [--bytes-per-expert B]

The trace is what `BARO_EXPERTS=<path>` wrote (bench/moe-locality-protocol.md):
blocks of `# trace n_gen=... layers=... topk=... prompt_tokens=...` followed by
`token layer id0 .. id7` rows. Each block is one prompt, and the LRU is reset
between blocks, because a host-resident tier starts cold for a new request and
a warm carry-over would flatter the hit rate.

Reported per capacity: hit rate, expert bytes moved per token at that capacity,
resident VRAM, and milliseconds per token at the measured PCIe rate. The
default bytes-per-expert is this pack's measured figure, not a guess: 589,824
bytes for a q4_k gate/up/down third (exchange/2026-09-15-p3-b4-stage1-report.md
derives it from .work/moe-w1/pack/index.txt, three of the 40 layers store
`down` as q6_k and are ignored in this averaging, which biases the bytes
slightly LOW and is stated rather than hidden).
"""
import argparse
import pathlib
import sys
from collections import OrderedDict

BYTES_PER_EXPERT = 3 * 589824  # gate + up + down, q4_k, one expert, one layer
DEFAULT_GBPS = 28.5


def read_trace(path):
    """-> list of blocks; each block is a list of (token, layer, [ids])."""
    blocks, cur, meta = [], [], []
    for line in pathlib.Path(path).read_text().splitlines():
        if line.startswith("#"):
            if cur:
                blocks.append(cur)
                cur = []
            meta.append(dict(kv.split("=", 1) for kv in line[1:].split() if "=" in kv))
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        cur.append((int(parts[0]), int(parts[1]), [int(x) for x in parts[2:]]))
    if cur:
        blocks.append(cur)
    return blocks, meta


def replay(blocks, capacity, n_layers):
    """LRU per layer, reset per block. -> (hits, refs, cold_misses)."""
    hits = refs = cold = 0
    for block in blocks:
        cache = {ly: OrderedDict() for ly in range(n_layers)}
        for _tok, layer, ids in block:
            c = cache.setdefault(layer, OrderedDict())
            for e in ids:
                if e < 0:
                    continue
                refs += 1
                if e in c:
                    hits += 1
                    c.move_to_end(e)
                else:
                    cold += 1
                    c[e] = True
                    if len(c) > capacity:
                        c.popitem(last=False)
    return hits, refs, cold


def sequential_repeat(blocks, n_layers):
    """Fraction of (layer, expert) picks that were also picked at token-1."""
    same = total = 0
    for block in blocks:
        prev = {}
        by_token = {}
        for tok, layer, ids in block:
            by_token.setdefault(tok, []).append((layer, [e for e in ids if e >= 0]))
        for tok in sorted(by_token):
            for layer, ids in by_token[tok]:
                p = prev.get(layer)
                if p is not None:
                    for e in ids:
                        total += 1
                        if e in p:
                            same += 1
                prev[layer] = set(ids)
    return same, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--bandwidth", type=float, default=DEFAULT_GBPS,
                    help="sustained H2D GB/s, measured, default %(default)s")
    ap.add_argument("--bytes-per-expert", type=int, default=BYTES_PER_EXPERT)
    a = ap.parse_args()

    blocks, meta = read_trace(a.trace)
    if not blocks:
        print(f"{a.trace}: no trace rows", file=sys.stderr)
        return 2
    n_layers = int(meta[0].get("layers", 40))
    topk = int(meta[0].get("topk", 8))
    tokens = sum(len({t for t, _l, _i in b}) for b in blocks)
    rows = sum(len(b) for b in blocks)
    expected = tokens * n_layers
    print(f"trace: {len(blocks)} prompts, {tokens} decoded tokens, {rows} rows "
          f"({n_layers} layers x top-{topk})")
    if rows != expected:
        # P1: a truncated trace would quietly produce a hit rate over fewer
        # tokens than claimed, so it fails here instead.
        print(f"FAIL: expected {expected} rows for {tokens} tokens, got {rows}")
        return 1

    same, total = sequential_repeat(blocks, n_layers)
    print(f"sequential repeat (same expert, same layer, consecutive tokens): "
          f"{100 * same / total:.1f}% of {total} picks")

    total_bytes_uncached = a.bytes_per_expert * topk * n_layers
    print(f"\nper token, everything uncached: {total_bytes_uncached / 1e9:.3f} GB, "
          f"{1e3 * total_bytes_uncached / (a.bandwidth * 1e9):.1f} ms at {a.bandwidth} GB/s\n")
    hdr = f"{'resident/layer':>15}{'resident VRAM':>15}{'hit rate':>10}{'GB/token':>10}{'ms/token':>10}"
    print(hdr)
    print("-" * len(hdr))
    for cap in (8, 16, 32, 64, 96, 128, 192, 256):
        hits, refs, _cold = replay(blocks, cap, n_layers)
        rate = hits / refs if refs else 0.0
        moved = a.bytes_per_expert * topk * n_layers * (1 - rate)
        vram = a.bytes_per_expert * cap * n_layers
        print(f"{cap:>15}{vram / 1e9:>13.1f} GB{100 * rate:>9.1f}%"
              f"{moved / 1e9:>10.3f}{1e3 * moved / (a.bandwidth * 1e9):>10.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
