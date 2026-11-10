"""Build the spark2_5 engine pack from a GGUF (Q8_0, or K-quant Q4_K / Q5_K / Q6_K). Mojo port of tools/spark-pack.py,
which stays as the oracle (byte-identical pack.bin and index.txt, checked by tools/spark-pack-check.sh).

Same pack format as tools/engine-pack.py --q8: q8 = int8 [out, in] followed by fp16 block-32 scales [out, in/32].
Q8_0 tensors are copied 1:1 from the GGUF's own blocks. K-quant tensors are dequantised with gguf-py's arithmetic
(f32, no fused ops) and every matrix is requantised to Q8_0 (d = amax / 127, ggml's round-half-away-from-zero, d
stored as f16), the same as llama-quantize --allow-requantize, which refuses arch spark2_5 (2026-10-12).
token_embd is stored dequantised f32 for the lookup and again as q8 under "output.weight" for the tied LM head.

Order per layer: attn_norm f32, attn_qkv q8, attn_gate q8, attn_output q8, ffn_norm f32, ffn_gate q8, ffn_up q8,
ffn_down q8; then output_norm, output.

Usage: .work/spark-pack MODEL.gguf OUTDIR
Build: ./.venv/bin/mojo build --fp-mode contract=off tools/spark-pack.mojo -I serve -I tools -o .work/spark-pack
"""
from std.sys import argv, exit, stderr
from shard_index import ShardIndex, TensorRef
from ggml_quant import T_F32, T_Q8_0, T_Q4_K, T_Q5_K, T_Q6_K, QK_K, u8, f16, dequant_block, block_bytes, release

comptime N_LAYERS = 36
comptime CHUNK_ROWS = 256


def roundf(x: Float32) -> Float32:
    var a = abs(x)
    var fl = Float32(Int(a)) if a < Float32(1 << 30) else a
    var b = fl + Float32(Int((a - fl) * 2))
    return -b if x < 0 else b


def emit(mut f: FileHandle, buf: List[UInt8]):
    f.write_bytes(Span(buf))


def main() raises:
    var av = argv()
    if len(av) != 3:
        print("usage: spark-pack MODEL.gguf OUTDIR", file=stderr)
        exit(2)
    var src = String(av[1])
    var outdir = String(av[2])
    var idx = ShardIndex(src)
    var order = List[String]()
    order.append("token_embd.weight")
    for i in range(N_LAYERS):
        var b = "blk." + String(i) + "."
        for n in ["attn_norm.weight", "attn_qkv.weight", "attn_gate.weight", "attn_output.weight", "ffn_norm.weight",
                  "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"]:
            order.append(b + n)
    order.append("output_norm.weight")
    order.append("output.weight")
    var lines = String()
    var off = 0
    var f = open(outdir + "/pack.bin", "w")
    for name in order:
        var t = idx["token_embd.weight" if name == "output.weight" else name]
        var ne0 = t.dims[0]
        var n_elem = t.nbytes if t.gtype != T_F32 else t.nbytes // 4
        var dt: String
        var written = 0
        if t.gtype == T_F32:
            n_elem = t.nbytes // 4
            var buf = List[UInt8](length=t.nbytes, fill=0)
            for k in range(t.nbytes):
                buf[k] = UInt8(u8(t.addr + k))
            emit(f, buf)
            written = t.nbytes
            dt = "f32"
        elif t.gtype == T_Q8_0 or t.gtype == T_Q4_K or t.gtype == T_Q5_K or t.gtype == T_Q6_K:
            var rows: Int
            if t.gtype == T_Q8_0:
                rows = t.nbytes // (ne0 // 32 * 34)
            else:
                rows = t.nbytes // (ne0 // QK_K * block_bytes(t.gtype))
            n_elem = rows * ne0
            var as_f32 = name == "token_embd.weight"
            dt = "f32" if as_f32 else "q8"
            var dbuf = List[UInt8]()
            var row = List[Float32](length=ne0, fill=0)
            var r0 = 0
            while r0 < rows:
                var r1 = min(rows, r0 + CHUNK_ROWS)
                var buf = List[UInt8]()
                for r in range(r0, r1):
                    if t.gtype == T_Q8_0:
                        var rb = t.addr + r * (ne0 // 32 * 34)
                        for g in range(ne0 // 32):
                            var bb = rb + g * 34
                            if as_f32:
                                var dg = f16(bb)
                                for l in range(32):
                                    var q = Float32(Int(Pointer[Int8, MutAnyOrigin](unsafe_from_address=bb + 2 + l)[]))
                                    var x = q * dg
                                    var bits = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(Pointer(to=x)))
                                    for k in range(4):
                                        buf.append(bits[unsafe_offset=k])
                            else:
                                for l in range(32):
                                    buf.append(UInt8(u8(bb + 2 + l)))
                                dbuf.append(UInt8(u8(bb)))
                                dbuf.append(UInt8(u8(bb + 1)))
                        continue
                    var bs = block_bytes(t.gtype)
                    var rb = t.addr + r * (ne0 // QK_K * bs)
                    for kb in range(ne0 // QK_K):
                        dequant_block(t.gtype, rb + kb * bs, row, kb * QK_K)
                    if as_f32:
                        for k in range(ne0):
                            var x = row[k]
                            var bits = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(Pointer(to=x)))
                            for z in range(4):
                                buf.append(bits[unsafe_offset=z])
                        continue
                    for g in range(ne0 // 32):
                        var amax = Float32(0)
                        for l in range(32):
                            amax = max(amax, abs(row[g * 32 + l]))
                        var d = amax / Float32(127)
                        var id = Float32(0) if d == 0 else Float32(1) / d
                        for l in range(32):
                            var q = roundf(row[g * 32 + l] * id)
                            buf.append(UInt8(Int(Int8(Int(q))) & 0xFF))
                        var dh = Float16(d)
                        var hb = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(Pointer(to=dh)))
                        dbuf.append(hb[unsafe_offset=0])
                        dbuf.append(hb[unsafe_offset=1])
                emit(f, buf)
                written += len(buf)
                r0 = r1
            if not as_f32:
                emit(f, dbuf)
                written += len(dbuf)
        else:
            raise Error(name + ": unsupported ggml type " + String(t.gtype))
        lines += name + " " + dt + " " + String(off) + " " + String(n_elem) + "\n"
        off += written
        release(t.addr, t.nbytes)
    f.close()
    with open(outdir + "/index.txt", "w") as fi:
        fi.write(lines)
    var gib = Float64(off) / Float64(1 << 30)
    var c = Int(gib * 100 + 0.5)
    print("packed", len(order), "tensors,", String(c // 100) + "." + ("0" if c % 100 < 10 else "") + String(c % 100), "GiB")
