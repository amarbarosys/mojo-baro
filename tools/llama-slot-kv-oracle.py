#!/usr/bin/env python3
"""Oracle for tools/state-to-llama-slot.mojo on attention-only models (qwen2, llama).

usage: llama-slot-kv-oracle.py SLOT_FILE OUT_STATE --hd N

Reads a llama-server slot file (seq state v3, one stream, non-transposed V, no cell ext,
no recurrent section) and writes the kv-only BAROST01 that serve/spark.mojo's state_save
writes: conv_n = ssm_n = 0, zero salt, the first ceil(pos/128) pages of the page-major pool.
The Mojo reverse tool (tools/llama-slot-to-state.mojo) is qwen35-only, so this is the
reverse leg of bench/bridge-roundtrip.sh for the other two E15 models. It is a verifier:
it shares no code with the tool it checks, and nothing at serve time uses it.
"""
import argparse
import struct
import sys

import numpy as np

KVPAGE = 128
GGSQ = 0x67677371


def fail(msg):
    print(f"FAIL oracle: {msg}")
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slot")
    ap.add_argument("out")
    ap.add_argument("--hd", type=int, required=True)
    a = ap.parse_args()
    d = open(a.slot, "rb").read()
    magic, ver, ntok = struct.unpack_from("<III", d, 0)
    if magic != GGSQ or ver != 3:
        fail(f"not a seq state v3 file (magic {magic:#x} version {ver})")
    o = 12
    raw = struct.unpack_from(f"<{ntok}i", d, o)
    o += 4 * ntok
    if ntok < 4 or raw[0] != -1 or raw[1] != 1 or 3 + raw[2] + 1 != ntok or raw[3 + raw[2]] != 0:
        fail(f"unrecognised server_tokens framing {raw[:4]}")
    tokens = raw[3 : 3 + raw[2]]
    (n_stream,) = struct.unpack_from("<I", d, o)
    o += 4
    if n_stream != 1:
        fail(f"expected one KV stream, got {n_stream}")
    (cells,) = struct.unpack_from("<I", d, o)
    o += 4
    for t in range(cells):
        pos, nseq = struct.unpack_from("<iI", d, o)
        if pos != t or nseq != 1:
            fail(f"cell {t}: pos {pos} n_seq_id {nseq} (a cell ext would land here; this oracle is for models without one)")
        o += 12
    v_trans, n_layer = struct.unpack_from("<II", d, o)
    o += 8
    if v_trans != 0:
        fail("transposed V cache; run llama-server with -fa on")
    pos = cells
    pages = -(-pos // KVPAGE)
    pools = []
    nkvh = None
    for side in range(2):
        layers = []
        for _ in range(n_layer):
            ty, rsz = struct.unpack_from("<iQ", d, o)
            o += 12
            es = {0: 4, 1: 2}.get(ty) or fail(f"unsupported KV dtype {ty}")
            row = rsz // es
            if row % a.hd:
                fail(f"row width {row} is not a multiple of --hd {a.hd}")
            nkvh = row // a.hd
            x = np.frombuffer(d, dtype="<f4" if es == 4 else "<f2", count=cells * row, offset=o).astype(np.float32)
            o += cells * rsz
            layers.append(x.reshape(cells, nkvh, a.hd))
        # ours: [page][layer][kvh][t % KVPAGE][hd], zero beyond pos
        pool = np.zeros((pages, n_layer, nkvh, KVPAGE, a.hd), dtype=np.float32)
        for li, x in enumerate(layers):
            for t in range(pos):
                pool[t // KVPAGE, li, :, t % KVPAGE, :] = x[t]
        pools.append(pool)
    if o != len(d):
        fail(f"trailing bytes: parsed {o} of {len(d)} (a recurrent section would land here)")
    kvn = pools[0].size
    with open(a.out, "wb") as f:
        f.write(b"BAROST01")
        f.write(struct.pack("<qqqq", pos, 0, 0, kvn))
        f.write(bytes(32))
        f.write(struct.pack(f"<{pos}i", *tokens[:pos]))
        f.write(pools[0].tobytes())
        f.write(pools[1].tobytes())
    print(f"oracle: {cells} cells, {n_layer} layers, nkvh {nkvh}, hd {a.hd}, kv_n {kvn} -> {a.out}")
    print(f"geometry: --att {n_layer} --nkvh {nkvh} --hd {a.hd} --ext 0")


if __name__ == "__main__":
    main()
