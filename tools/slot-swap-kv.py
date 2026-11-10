#!/usr/bin/env python3
"""usage: slot-swap-kv.py IN.slot OUT.slot

P11 falsifier for bench/p1-bridge-gate.sh: a copy of a llama-server slot file with its K and V
sections exchanged. Cell count, framing and row sizes are untouched, so llama.cpp accepts and
reuses it; only the generated ids can show that the state is wrong. Harness-side on purpose:
the tool under test has no flag that produces a bad file."""
import struct
import sys

d = bytearray(open(sys.argv[1], "rb").read())
magic, ver, ntok = struct.unpack_from("<III", d, 0)
if magic != 0x67677371 or ver != 3:
    print("FAIL slot-swap-kv: not a seq state v3 file"); sys.exit(1)
o = 12 + 4 * ntok
n_stream, cells = struct.unpack_from("<II", d, o)
o += 8
if n_stream != 1:
    print("FAIL slot-swap-kv: expected one stream"); sys.exit(1)
# a cell is pos(4) n_seq_id(4) [ext 12] seq_id(4); find which by checking where v_trans/n_layer land
for cell in (12, 24):
    p = o + cells * cell
    v_trans, n_layer = struct.unpack_from("<II", d, p)
    ty, rsz = struct.unpack_from("<iQ", d, p + 8)
    if v_trans == 0 and 0 < n_layer < 512 and ty in (0, 1) and 0 < rsz < (1 << 20):
        o = p + 8
        break
else:
    print("FAIL slot-swap-kv: cannot locate the KV data section"); sys.exit(1)
sec = n_layer * (12 + cells * rsz)
k, v = bytes(d[o : o + sec]), bytes(d[o + sec : o + 2 * sec])
if len(v) != sec or k == v:
    print("FAIL slot-swap-kv: K and V sections are not two equal-sized distinct blocks"); sys.exit(1)
d[o : o + sec], d[o + sec : o + 2 * sec] = v, k
open(sys.argv[2], "wb").write(d)
print(f"slot-swap-kv: {n_layer} layers, {cells} cells, swapped 2 x {sec} bytes")
