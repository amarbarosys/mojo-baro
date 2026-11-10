#!/usr/bin/env python3
"""Differential oracle for MoE expert cache simulation (LRU, Belady, Pin+LRU, CacheOnlyDraft)."""
import struct
import sys
from collections import OrderedDict
from pathlib import Path


def load_trace_text(path: str):
    trace = []
    topk = 8
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                for tok in line.split():
                    if tok.startswith("topk="):
                        topk = int(tok.split("=", 1)[1])
                continue
            parts = line.split()
            if len(parts) < 2 + topk:
                continue
            ly = int(parts[1])
            for j in range(topk):
                ex = int(parts[2 + j])
                trace.append((ly << 10) | ex)
    return trace, topk


def load_trace_bin(path: str):
    data = Path(path).read_bytes()
    n_tok, n_layer, topk, _ = struct.unpack_from("<IIII", data, 0)
    off = 16
    trace = []
    for _ in range(n_tok):
        for ly in range(n_layer):
            ids = struct.unpack_from(f"<{topk}H", data, off)
            off += 2 * topk
            for ex in ids:
                trace.append((ly << 10) | int(ex))
    return trace, topk


def load_trace(path: str):
    with open(path, "rb") as f:
        head = f.read(4)
    if head.startswith(b"#"):
        return load_trace_text(path)
    return load_trace_bin(path)


def rank_unique_keys(trace):
    freq = {}
    for k in trace:
        freq[k] = freq.get(k, 0) + 1
    return sorted(freq.keys(), key=lambda k: (-freq[k], k))


def sim_lru(trace, cap: int) -> int:
    if cap <= 0:
        return 0
    cache = OrderedDict()
    hits = 0
    for k in trace:
        if k in cache:
            hits += 1
            cache.move_to_end(k)
        else:
            if len(cache) >= cap:
                cache.popitem(last=False)
            cache[k] = None
    return hits


def sim_belady(trace, cap: int) -> int:
    if cap <= 0:
        return 0
    n = len(trace)
    nxt = [n] * n
    last_pos = {}
    for i in range(n - 1, -1, -1):
        k = trace[i]
        nxt[i] = last_pos.get(k, n)
        last_pos[k] = i
    cache = {}
    hits = 0
    for i, k in enumerate(trace):
        ni = nxt[i]
        if k in cache:
            hits += 1
            cache[k] = ni
        elif len(cache) < cap:
            cache[k] = ni
        else:
            far_k = max(cache, key=lambda x: (cache[x], x))
            far_n = cache[far_k]
            if (far_n, far_k) > (ni, k):
                del cache[far_k]
                cache[k] = ni
    return hits


def sim_pin_lru(trace, topk: int, cap: int, npin: int, unique_keys):
    actual_pin = min(npin, cap // 2, len(unique_keys))
    hot = set(unique_keys[:actual_pin])
    lru_cap = max(0, cap - actual_pin)
    hot_loaded = set()
    lru = OrderedDict()
    hits = 0
    draft_ge1 = 0
    n_groups = len(trace) // topk
    for g in range(n_groups):
        base = g * topk
        any_hit = False
        for j in range(topk):
            k = trace[base + j]
            if (k in hot and k in hot_loaded) or (k not in hot and k in lru):
                any_hit = True
                break
        if any_hit:
            draft_ge1 += 1
        for j in range(topk):
            k = trace[base + j]
            if k in hot:
                if k in hot_loaded:
                    hits += 1
                else:
                    hot_loaded.add(k)
            else:
                if k in lru:
                    hits += 1
                    lru.move_to_end(k)
                elif lru_cap > 0:
                    if len(lru) >= lru_cap:
                        lru.popitem(last=False)
                    lru[k] = None
    return actual_pin, hits, draft_ge1, n_groups


def write_pin_bin(path: str, unique_keys, npin: int):
    actual_pin = min(npin, len(unique_keys))
    out = bytearray(struct.pack("<4sI", b"PIN1", actual_pin))
    for k in unique_keys[:actual_pin]:
        ly = k >> 10
        ex = k & 0x3FF
        out += struct.pack("<HH", ly, ex)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(out)
    return actual_pin, len(out)


def run_selftest() -> int:
    # 1. Compulsory miss check: pinning all keys on a strictly unique sequence must yield 0 hits
    uniq_trace = [(0 << 10) | i for i in range(64)]
    u_keys = rank_unique_keys(uniq_trace)
    _, plru_hits, _, _ = sim_pin_lru(uniq_trace, 8, 64, 32, u_keys)
    assert plru_hits == 0, f"expected 0 hits on cold unique stream, got {plru_hits}"

    # 2. Belady optimality check: Belady >= LRU and Belady >= Pin+LRU on skewed trace
    skew_trace = []
    for t in range(200):
        for j in range(8):
            ex = j if (t % 3 != 0) else (j + (t % 24))
            skew_trace.append((0 << 10) | ex)
    s_keys = rank_unique_keys(skew_trace)
    for cap in (8, 16, 24):
        lh = sim_lru(skew_trace, cap)
        bh = sim_belady(skew_trace, cap)
        _, ph, _, _ = sim_pin_lru(skew_trace, 8, cap, cap // 2, s_keys)
        assert bh >= lh, f"Belady ({bh}) < LRU ({lh}) at cap={cap}"
        assert bh >= ph, f"Belady ({bh}) < Pin+LRU ({ph}) at cap={cap}"
    print("SELFTEST PASS: compulsory_miss=0, belady>=lru, belady>=pin_lru")
    return 0


def main():
    args = sys.argv[1:]
    if not args or "--self-test" in args:
        sys.exit(run_selftest())
    path = args[0]
    caps = [128, 256, 512, 1024]
    npin = 128
    pin_out = ""
    i = 1
    while i < len(args):
        if args[i] == "--caps":
            caps = [int(x) for x in args[i + 1].split(",")]
            i += 2
        elif args[i] == "--npin":
            npin = int(args[i + 1])
            i += 2
        elif args[i] == "--pin-out":
            pin_out = args[i + 1]
            i += 2
        else:
            raise SystemExit(f"unknown arg: {args[i]}")

    trace, topk = load_trace(path)
    unique_keys = rank_unique_keys(trace)
    n_req = len(trace)
    print(f"trace: {path} requests: {n_req} unique-(layer,expert): {len(unique_keys)} topk: {topk}")
    print("cap\tnpin\tlru_hit\tbelady_hit\tpin_lru_hit\tdraft_ge1_cov")
    for cap in caps:
        lh = sim_lru(trace, cap)
        bh = sim_belady(trace, cap)
        ap, ph, dg, ng = sim_pin_lru(trace, topk, cap, npin, unique_keys)
        assert bh >= lh and bh >= ph, f"Belady optimality violated at cap={cap}"
        print(
            f"{cap}\t{ap}\t{lh / n_req:.4f}\t{bh / n_req:.4f}\t{ph / n_req:.4f}\t{dg / max(ng, 1):.4f}"
        )
    if pin_out:
        ap, nbytes = write_pin_bin(pin_out, unique_keys, npin)
        print(f"pinned_binary: {pin_out} entries: {ap} bytes: {nbytes}")


if __name__ == "__main__":
    main()
