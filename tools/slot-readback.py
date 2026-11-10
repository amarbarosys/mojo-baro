#!/usr/bin/env python3
"""usage: slot-readback.py SLOT...   (slot files written BY the running llama-server)

P1 read-back for bench/p1-bridge-gate.sh. This build's llama-server log says nothing about KV
type or flash attention at default verbosity, and a flag passed is not evidence it took effect.
The slot file the server itself writes does carry them: k/v type code, row bytes, and v_trans,
which llama.cpp sets to 0 only when flash attention is on. Every file must agree, or FAIL."""
import json
import statistics
import struct
import sys

if len(sys.argv) > 3 and sys.argv[1] == "--decode-rate":
    # usage: slot-readback.py --decode-rate OUT PROMPT...   The offload receipt: this log names no
    # device, but a 7B to 9B model decoding near 100 tok/s is on the GPU; on this CPU it is near 10. Read from
    # control L, which runs every time; the cold arm may be a refcache hit with stale timings.
    out = sys.argv[2]
    r = [json.load(open(f"{out}/ids/{p}.ctrlL.json"))["timings"]["predicted_per_second"] for p in sys.argv[3:]]
    print(f"control L decode median {statistics.median(r):.1f} tok/s over {len(r)} prompts")
    sys.exit(0)

seen = set()
for path in sys.argv[1:]:
    d = open(path, "rb").read()
    magic, ver, ntok = struct.unpack_from("<III", d, 0)
    if magic != 0x67677371 or ver != 3:
        print(f"FAIL readback: {path} is not a seq state v3 file"); sys.exit(1)
    o = 12 + 4 * ntok
    n_stream, cells = struct.unpack_from("<II", d, o)
    o += 8
    for cell in (12, 24):
        p = o + cells * cell
        v_trans, n_layer = struct.unpack_from("<II", d, p)
        ty, rsz = struct.unpack_from("<iQ", d, p + 8)
        if v_trans in (0, 1) and 0 < n_layer < 512 and ty in (0, 1) and 0 < rsz < (1 << 20):
            break
    else:
        print(f"FAIL readback: cannot locate the KV data section in {path}"); sys.exit(1)
    seen.add((n_stream, v_trans, n_layer, ty, rsz, cell == 24))
if len(seen) != 1:
    print(f"FAIL readback: slot files disagree: {sorted(seen)}"); sys.exit(1)
n_stream, v_trans, n_layer, ty, rsz, ext = seen.pop()
if n_stream != 1 or v_trans != 0 or ty != 1:
    print(f"FAIL readback: need one stream, flash attention on (v_trans 0), f16 KV; got n_stream {n_stream} v_trans {v_trans} k_type {ty}"); sys.exit(1)
print(f"n_stream=1 flash_attn=on(v_trans=0) kv=f16(type 1) kv_layers={n_layer} row_bytes={rsz} cell_ext={int(ext)} files={len(sys.argv) - 1}")
