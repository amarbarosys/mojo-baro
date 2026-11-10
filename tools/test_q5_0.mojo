# Q5_0 dequantiser check: every Q5_0 tensor of a GGUF, all rows, written as f32 for tools/test_q5_0.py to compare
# against gguf-py bit for bit. usage: test_q5_0 MODEL.gguf OUT.bin
from std.sys import argv
from shard_index import ShardIndex
from ggml_quant import T_Q5_0, row_f32

def main() raises:
    var idx = ShardIndex(String(argv()[1]))
    var out = List[UInt8]()
    var n = 0
    for name in idx.gguf.tensor_names:
        var t = idx[name]
        if t.gtype != T_Q5_0:
            continue
        var ne0 = t.dims[0]
        var rows = t.nbytes // (ne0 // 32 * 22)
        var buf = List[Float32](length=ne0, fill=0)
        for r in range(rows):
            row_f32(T_Q5_0, idx.maps[t.shard] + t.off + r * (ne0 // 32 * 22), ne0, buf)
            var p = buf.unsafe_ptr().unsafe_bitcast[UInt8]()
            for i in range(ne0 * 4):
                out.append(p[unsafe_offset=i])
        n += 1
    with open(String(argv()[2]), "w") as f:
        f.write_bytes(Span(out))
    print("q5_0 tensors", n, "floats", len(out) // 4)
