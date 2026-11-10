"""ggml tensor rows as f32, the way gguf-py's `gguf.quants.dequantize` computes them (same per-element operation
order, f32 throughout, no fused ops), plus ggml's bf16 rounding. Shared by tools/spark-pack.mojo and
tools/engine-pack.mojo; their Python oracles are byte-checked by tools/spark-pack-check.sh and
tools/engine-pack-check.sh.
"""
from std.memory import bitcast
from std.ffi import external_call

comptime T_F32 = 0
comptime T_F16 = 1
comptime T_Q4_0 = 2
comptime T_Q5_0 = 6
comptime T_Q8_0 = 8
comptime T_Q4_K = 12
comptime T_Q5_K = 13
comptime T_Q6_K = 14
comptime T_IQ4_XS = 23
comptime T_BF16 = 30
comptime QK_K = 256


def release(addr: Int, nbytes: Int):
    """Drop a tensor's mmap pages from this process once written (MADV_DONTNEED; the page cache keeps them): without
    it every byte read counts toward peak RSS, 7.2 GB on a 7.3 GB GGUF."""
    var lo = addr - addr % 4096
    _ = external_call["madvise", Int32](lo, addr + nbytes - lo, Int32(4))


def u8(a: Int) -> Int:
    return Int(Pointer[UInt8, MutAnyOrigin](unsafe_from_address=a)[])


def f16(a: Int) -> Float32:
    return Float32(Pointer[Float16, MutAnyOrigin](unsafe_from_address=a)[])


def scale_min(sb: Int, j: Int) -> Tuple[Float32, Float32]:
    if j < 4:
        return (Float32(u8(sb + j) & 63), Float32(u8(sb + 4 + j) & 63))
    var sc = (u8(sb + 8 + j - 4) & 0x0F) | ((u8(sb + j - 4) >> 2) & 0x30)
    var mn = (u8(sb + 8 + j - 4) >> 4) | ((u8(sb + 4 + j - 4) >> 2) & 0x30)
    return (Float32(sc), Float32(mn))


def dequant_block(gtype: Int, b: Int, mut out: List[Float32], o: Int):
    """One QK_K block at address b into out[o : o + 256], gguf-py's operation order."""
    if gtype == T_Q6_K:
        var d = f16(b + 128 + 64 + 16)
        for p in range(QK_K):
            var c = p // 128
            var qlv = (u8(b + c * 64 + p % 64) >> (4 * ((p // 64) % 2))) & 0x0F
            var qhv = (u8(b + 128 + c * 32 + p % 32) >> (2 * ((p // 32) % 4))) & 0x03
            var q = Float32(Int(Int8((qlv | (qhv << 4)) - 32)))
            var sc = Float32(Int(Pointer[Int8, MutAnyOrigin](unsafe_from_address=b + 192 + p // 16)[]))
            var ds = d * sc
            out[o + p] = ds * q
        return
    var d = f16(b)
    var dmin = f16(b + 2)
    var sb = b + 4
    for j in range(8):
        var sm = scale_min(sb, j)
        var dj = d * sm[0]
        var mj = dmin * sm[1]
        var c = j // 2
        var h = j % 2
        for l in range(32):
            var q: Int
            if gtype == T_Q4_K:
                q = (u8(b + 16 + c * 32 + l) >> (4 * h)) & 0x0F
            else:
                var ql = (u8(b + 48 + c * 32 + l) >> (4 * h)) & 0x0F
                var qh = (u8(b + 16 + l) >> j) & 0x01
                q = ql | (qh << 4)
            var t = dj * Float32(q)
            out[o + j * 32 + l] = t - mj


def iq4_xs_block(b: Int, mut out: List[Float32], o: Int):
    """IQ4_XS: f16 d, u16 scales_h, u8 scales_l[4], qs[128]; 8 sub-blocks of 32, 6-bit scales, non-linear 4-bit grid."""
    comptime kv = SIMD[DType.int32, 16](-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)
    var d = f16(b)
    var sh = u8(b + 2) | (u8(b + 3) << 8)
    for ib in range(8):
        var sc = ((u8(b + 4 + ib // 2) >> (4 * (ib % 2))) & 0x0F) | (((sh >> (2 * ib)) & 0x03) << 4)
        var dl = d * Float32(sc - 32)
        for j in range(16):
            var v = u8(b + 8 + ib * 16 + j)
            out[o + ib * 32 + j] = dl * Float32(kv[v & 0x0F])
            out[o + ib * 32 + 16 + j] = dl * Float32(kv[v >> 4])


def block_bytes(gtype: Int) raises -> Int:
    if gtype == T_Q4_K:
        return 144
    if gtype == T_Q5_K:
        return 176
    if gtype == T_Q6_K:
        return 210
    raise Error("not a K-quant type " + String(gtype))


def row_f32(gtype: Int, a: Int, ne0: Int, mut out: List[Float32]) raises:
    """One row of ne0 elements at address a, dequantised into out[0 : ne0]."""
    if gtype == T_F32:
        var p = Pointer[Float32, MutAnyOrigin](unsafe_from_address=a)
        for k in range(ne0):
            out[k] = p[unsafe_offset=k]
    elif gtype == T_F16:
        var p = Pointer[Float16, MutAnyOrigin](unsafe_from_address=a)
        for k in range(ne0):
            out[k] = Float32(p[unsafe_offset=k])
    elif gtype == T_BF16:
        var p = Pointer[UInt16, MutAnyOrigin](unsafe_from_address=a)
        for k in range(ne0):
            out[k] = from_bf16(p[unsafe_offset=k])
    elif gtype == T_Q4_0:
        for g in range(ne0 // 32):
            var b = a + g * 18
            var d = f16(b)
            for j in range(16):
                var v = u8(b + 2 + j)
                out[g * 32 + j] = d * Float32((v & 0x0F) - 8)
                out[g * 32 + 16 + j] = d * Float32((v >> 4) - 8)
    elif gtype == T_Q5_0:
        for g in range(ne0 // 32):
            var b = a + g * 22
            var d = f16(b)
            var qh = u8(b + 2) | (u8(b + 3) << 8) | (u8(b + 4) << 16) | (u8(b + 5) << 24)
            for j in range(16):
                var v = u8(b + 6 + j)
                out[g * 32 + j] = d * Float32(((v & 0x0F) | (((qh >> j) & 1) << 4)) - 16)
                out[g * 32 + 16 + j] = d * Float32(((v >> 4) | (((qh >> (j + 16)) & 1) << 4)) - 16)
    elif gtype == T_Q8_0:
        for g in range(ne0 // 32):
            var b = a + g * 34
            var d = f16(b)
            for l in range(32):
                var q = Float32(Int(Pointer[Int8, MutAnyOrigin](unsafe_from_address=b + 2 + l)[]))
                out[g * 32 + l] = q * d
    elif gtype == T_Q4_K or gtype == T_Q5_K or gtype == T_Q6_K:
        var bs = block_bytes(gtype)
        for kb in range(ne0 // QK_K):
            dequant_block(gtype, a + kb * bs, out, kb * QK_K)
    elif gtype == T_IQ4_XS:
        for kb in range(ne0 // QK_K):
            iq4_xs_block(a + kb * 136, out, kb * QK_K)
    else:
        raise Error("ggml type " + type_name(gtype) + " has no Mojo dequantiser (the Python oracle may)")


def to_bf16(x: Float32) -> UInt16:
    """Round-to-nearest-even, ggml_fp32_to_bf16 without its NaN case (as tools/engine-pack.py)."""
    var u = bitcast[DType.uint32](x)
    var bias = ((u >> 16) & 1) + 0x7FFF
    return UInt16((u + bias) >> 16)


def from_bf16(h: UInt16) -> Float32:
    return bitcast[DType.float32](UInt32(h) << 16)


def type_name(gtype: Int) -> String:
    """gguf-py's GGMLQuantizationType name, lower case."""
    var names: List[String] = ["f32", "f16", "q4_0", "q4_1", "", "", "q5_0", "q5_1", "q8_0", "q8_1", "q2_k", "q3_k",
        "q4_k", "q5_k", "q6_k", "q8_k", "iq2_xxs", "iq2_xs", "iq3_xxs", "iq1_s", "iq4_nl", "iq3_s", "iq2_s", "iq4_xs",
        "i8", "i16", "i32", "i64", "f64", "iq1_m", "bf16", "", "", "", "tq1_0", "tq2_0", "", "", "", "mxfp4", "nvfp4",
        "q1_0"]
    if gtype >= 0 and gtype < len(names) and names[gtype] != "":
        return names[gtype]
    return "type" + String(gtype)
