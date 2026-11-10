#!/usr/bin/env python3
"""BAROKV6: BAROKV5's nibble stream entropy coded with rANS, one stream per (block, 4 channels).
usage: kvc_rans.py KV5_PAYLOAD KV6_OUT [--check N]
Input = a BAROKV5 payload (kvc_int4.py --kv5). Output layout:
  fixed part as BAROKV5 (magic BAROKV6) | for K then V:
    tails f16 [blocks][128]                       (as BAROKV5)
    freq  u16 [8 layers][4 heads][256 ch][16]     15 symbols (q+8 in 1..15 stored at index 0..14)
                                                  + one pad, each table sums to 4096
    off   u32 [blocks][64] + 1                    byte offset of every stream, then the end
    streams                                       rANS, 32-bit state in [2^16, 2^24), byte renorm,
                                                  512 symbols (128 rows x 4 ch, row-major); the
                                                  decoder reads 4 state bytes big-end first at
                                                  off[i], then renorm bytes forward
Every stream of one pool is encoded in lockstep with numpy. --check N decodes N random streams
per pool in pure Python and compares them with the source nibbles (the oracle).
"""
import sys
import numpy as np

NAT, NKVH, HD, KVPAGE = 8, 4, 256, 128
NB_PER_PAGE = NAT * NKVH
TAIL16 = KVPAGE * 2
PACKED = KVPAGE * HD // 2
NSYM, PROB_BITS, L = 15, 12, 1 << 16
GROUP = 4
STREAMS_PER_BLOCK = HD // GROUP
SYMS = KVPAGE * GROUP


def freq_tables(sym):
    """sym [blocks, 128, 256] uint8 in 0..14 -> [32 tables (layer, head)][256][16] uint16."""
    blocks = sym.shape[0]
    pages = blocks // NB_PER_PAGE
    s = sym.reshape(pages, NB_PER_PAGE, KVPAGE, HD).transpose(1, 3, 0, 2).reshape(NB_PER_PAGE, HD, pages * KVPAGE)
    idx = (np.arange(HD)[None, :, None] * NSYM + s).reshape(NB_PER_PAGE, -1)
    counts = np.stack([np.bincount(row, minlength=HD * NSYM) for row in idx]).reshape(NB_PER_PAGE, HD, NSYM).astype(np.float64)
    counts += 1e-3
    f = np.floor(counts / counts.sum(-1, keepdims=True) * (1 << PROB_BITS)).astype(np.int64)
    f = np.maximum(f, 1)
    rem = (1 << PROB_BITS) - f.sum(-1)
    top = counts.argmax(-1)
    np.put_along_axis(f, top[..., None], np.take_along_axis(f, top[..., None], -1) + rem[..., None], -1)
    assert (f > 0).all() and (f.sum(-1) == 1 << PROB_BITS).all()
    out = np.zeros((NB_PER_PAGE, HD, 16), dtype=np.uint16)
    out[..., :NSYM] = f
    return out


def encode_pool(sym, tables):
    """sym [blocks, 128, 256] uint8 0..14 -> (offsets u32 [blocks*64 + 1], bytes)."""
    blocks = sym.shape[0]
    f = tables[..., :NSYM].astype(np.uint64)                     # [32][256][15]
    cum = np.concatenate([np.zeros_like(f[..., :1]), np.cumsum(f, -1)], -1)[..., :NSYM]
    st = sym.reshape(blocks, KVPAGE, STREAMS_PER_BLOCK, GROUP).transpose(0, 2, 1, 3).reshape(-1, SYMS)
    nstreams = st.shape[0]
    tab = np.repeat(np.tile(np.arange(NB_PER_PAGE), blocks // NB_PER_PAGE), STREAMS_PER_BLOCK)
    chan0 = np.tile(np.arange(STREAMS_PER_BLOCK) * GROUP, blocks)
    cap = SYMS * 2 + 8
    out = np.zeros((nstreams, cap), dtype=np.uint8)
    wp = np.full(nstreams, cap, dtype=np.int64)
    x = np.full(nstreams, L, dtype=np.uint64)
    rows = np.arange(nstreams)
    for j in range(SYMS - 1, -1, -1):
        s = st[:, j].astype(np.int64)
        ch = chan0 + (j % GROUP)
        fs = f[tab, ch, s]
        cs = cum[tab, ch, s]
        x_max = np.uint64((L >> PROB_BITS) << 8) * fs
        for _ in range(2):
            m = x >= x_max
            if not m.any():
                break
            wp[m] -= 1
            out[rows[m], wp[m]] = (x[m] & np.uint64(0xFF)).astype(np.uint8)
            x[m] >>= np.uint64(8)
        assert (x < x_max).all()
        x = (x // fs) * np.uint64(1 << PROB_BITS) + (x % fs) + cs
    for k in range(4):
        wp -= 1
        out[rows, wp] = ((x >> np.uint64(8 * k)) & np.uint64(0xFF)).astype(np.uint8)
    lengths = cap - wp
    offsets = np.zeros(nstreams + 1, dtype=np.uint32)
    offsets[1:] = np.cumsum(lengths)
    buf = np.empty(int(lengths.sum()), dtype=np.uint8)
    starts = offsets[:-1].astype(np.int64)
    for k in range(int(lengths.max())):
        m = lengths > k
        buf[starts[m] + k] = out[rows[m], wp[m] + k]
    return offsets, buf


def decode_stream(buf, off, end, table, chan0):
    f = table[..., :NSYM].astype(np.int64)
    cum = np.concatenate([np.zeros_like(f[:, :1]), np.cumsum(f, -1)], -1)
    p = off
    x = int(buf[p]) << 24 | int(buf[p + 1]) << 16 | int(buf[p + 2]) << 8 | int(buf[p + 3])
    p += 4
    syms = np.zeros(SYMS, dtype=np.uint8)
    for j in range(SYMS):
        ch = chan0 + (j % GROUP)
        slot = x & ((1 << PROB_BITS) - 1)
        s = int(np.searchsorted(cum[ch], slot, side="right") - 1)
        x = int(f[ch, s]) * (x >> PROB_BITS) + slot - int(cum[ch, s])
        while x < L:
            x = (x << 8) | int(buf[p]); p += 1
        syms[j] = s
    assert p == end and x == L, (p, end, x)
    return syms


def main():
    src, dst = sys.argv[1], sys.argv[2]
    check = int(sys.argv[sys.argv.index("--check") + 1]) if "--check" in sys.argv else 0
    raw = np.fromfile(src, dtype=np.uint8)
    assert raw[:8].tobytes() == b"BAROKV5\0", raw[:8].tobytes()
    pos, conv_n, ssm_n, kv_n = (int(x) for x in raw[8:40].view(np.int64))
    fixed = 72 + 4 * pos + 2 * (conv_n + ssm_n)
    pages = -(-pos // KVPAGE)
    blocks = pages * NB_PER_PAGE
    assert kv_n == blocks * (KVPAGE * HD + KVPAGE * 4)
    per_pool = blocks * (TAIL16 + PACKED)
    assert raw.size == fixed + 2 * per_pool, (raw.size, fixed + 2 * per_pool)
    head = raw[:fixed].copy(); head[:8] = np.frombuffer(b"BAROKV6\0", dtype=np.uint8)
    parts = [head.tobytes()]
    rng = np.random.default_rng(0)
    for kv in range(2):
        base = fixed + kv * per_pool
        tails = raw[base: base + blocks * TAIL16]
        packed = raw[base + blocks * TAIL16: base + per_pool].reshape(blocks, PACKED)
        sym = np.empty((blocks, KVPAGE * HD), dtype=np.uint8)
        sym[:, 0::2] = packed & 15; sym[:, 1::2] = packed >> 4
        sym = sym.reshape(blocks, KVPAGE, HD) - 1
        assert sym.max() <= 14
        tables = freq_tables(sym)
        offsets, buf = encode_pool(sym, tables)
        parts += [tails.tobytes(), tables.tobytes(), offsets.tobytes(), buf.tobytes()]
        for i in (rng.choice(offsets.size - 1, size=check, replace=False) if check else []):
            b, g = divmod(int(i), STREAMS_PER_BLOCK)
            got = decode_stream(buf, int(offsets[i]), int(offsets[i + 1]), tables[b % NB_PER_PAGE], g * GROUP)
            want = sym[b, :, g * GROUP:(g + 1) * GROUP].reshape(-1)
            assert np.array_equal(got, want), f"FAIL oracle: pool {kv} stream {i}"
        print(f"pool {kv}: tails {tails.size} tables {tables.nbytes} offsets {offsets.nbytes} streams {buf.size} ({buf.size * 8 / sym.size:.3f} bits/val){f', oracle {check} streams OK' if check else ''}")
    pay = b"".join(parts)
    open(dst, "wb").write(pay)
    print(f"BAROKV6 payload {len(pay)} B ({raw.size / len(pay):.3f}x of BAROKV5 {raw.size}, {585299012 / len(pay):.3f}x of BAROST03), pos {pos}")


if __name__ == "__main__":
    main()
