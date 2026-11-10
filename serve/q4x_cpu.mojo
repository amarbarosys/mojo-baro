"""CPU compute for routed experts that miss the VRAM slot cache (q4x round 3, Strata's miss balance).

A missed expert is either copied to VRAM (DMA) or computed here from the registered mmap of its
GGUF shard while the GPU runs the hits. Weights are Q2_0 (18 bytes per 64: f16 d, 16 bytes of 2-bit
q, w = d * (q - 1), byte j holding elements 4j..4j+3). x (bf16-rounded, as on the GPU) and the
swiglu activation h are quantized to int8 per 64-block and stored in the weight's lane order, so one
AVX-512 VNNI vpdpbusd covers a block; the -1 offset is the exact integer block sum. Close to the GPU
path (serve/expert_cpu.mojo shape), not bit-exact.
"""
from std.memory import bitcast
from std.math import exp, round
from std.sys import llvm_intrinsic
from max.algorithm import parallelize

comptime f32 = DType.float32
comptime bf16 = DType.bfloat16
comptime u8 = DType.uint8
comptime i8 = DType.int8
comptime Q2B = 18
comptime WORKERS = 8
comptime FPtr = MutPointer[Scalar[f32], MutAnyOrigin]
comptime BPtr = MutPointer[Scalar[u8], MutAnyOrigin]
comptime QPtr = MutPointer[Scalar[i8], MutAnyOrigin]


@always_inline
def q2_quant(x: FPtr, n: Int, xq: QPtr, xd: FPtr, xs: FPtr):
    for b in range(n // 64):
        var v = x.unsafe_offset(b * 64).load[width=64]().cast[bf16]().cast[f32]()
        var amax = abs(v).reduce_max()
        var inv = Float32(127) / amax if amax > 0 else Float32(0)
        var q = round(v * inv).cast[i8]()
        var eo = q.deinterleave()
        var e = eo[0].deinterleave()
        var o = eo[1].deinterleave()
        xq.unsafe_offset(b * 64).store(e[0].join(o[0]).join(e[1].join(o[1])))
        xd[b] = amax / 127
        xs[b] = Float32(q.cast[DType.int32]().reduce_add())


@always_inline
def q2_row(w: BPtr, xq: QPtr, xd: FPtr, xs: FPtr, nb: Int) -> Float32:
    var acc = SIMD[f32, 16](0)
    var corr = Float32(0)
    for b in range(nb):
        var bp = w.unsafe_offset(b * Q2B)
        var d = bitcast[DType.float16, 1](SIMD[DType.uint16, 1](UInt16(bp[0]) | (UInt16(bp[1]) << 8))).cast[f32]()[0]
        var q = bp.unsafe_offset(2).load[width=16]()
        var a = (q & 3).join((q >> 2) & 3).join(((q >> 4) & 3).join(q >> 6))
        var r = llvm_intrinsic["llvm.x86.avx512.vpdpbusd.512", SIMD[DType.int32, 16]](SIMD[DType.int32, 16](0), a, xq.unsafe_offset(b * 64).load[width=64]())
        var sc = d * xd[b]
        acc = r.cast[f32]().fma(SIMD[f32, 16](sc), acc)
        corr += sc * xs[b]
    return acc.reduce_add() - corr


def q2_row_ref(w: BPtr, x: FPtr, k: Int) -> Float32:
    var acc = Float32(0)
    for b in range(k // 64):
        var bp = w.unsafe_offset(b * Q2B)
        var d = bitcast[DType.float16, 1](SIMD[DType.uint16, 1](UInt16(bp[0]) | (UInt16(bp[1]) << 8))).cast[f32]()[0]
        for j in range(16):
            var q = Int(bp[2 + j])
            for s in range(4):
                acc += d * Float32(((q >> (2 * s)) & 3) - 1) * x[b * 64 + 4 * j + s].cast[bf16]().cast[f32]()
    return acc


def cpu_experts(n: Int, gw: List[Int], uw: List[Int], dw: List[Int], xs: List[Int], ys: List[Int], scratch: FPtr, h: Int, ff: Int):
    """Expert FFN for n (expert, token) pairs: gw/uw/dw = expert row-0 addresses, xs = f32 input rows (h floats),
    ys = f32 output rows (h floats). scratch holds n * (h + 2 * ff) floats."""
    var gb = (h // 64) * Q2B
    var db = (ff // 64) * Q2B
    var sw = h + 2 * ff
    var nbh = h // 64
    var nbf = ff // 64
    for i in range(n):
        var s = scratch.unsafe_offset(i * sw)
        q2_quant(FPtr(unsafe_from_address=xs[i]), h, QPtr(unsafe_from_address=Int(s)), s.unsafe_offset(h // 4), s.unsafe_offset(h // 4 + nbh))
    var gp = gw.unsafe_ptr()
    var upp = uw.unsafe_ptr()
    var rows = n * ff
    def up(k: Int) {imm gp, imm upp, imm scratch, imm h, imm ff, imm gb, imm sw, imm nbh, imm rows}:
        for t in range(k * rows // WORKERS, (k + 1) * rows // WORKERS):
            var i = t // ff
            var r = t % ff
            var s = scratch.unsafe_offset(i * sw)
            var xq = QPtr(unsafe_from_address=Int(s))
            var g = q2_row(BPtr(unsafe_from_address=gp[i] + r * gb), xq, s.unsafe_offset(h // 4), s.unsafe_offset(h // 4 + nbh), nbh)
            var u = q2_row(BPtr(unsafe_from_address=upp[i] + r * gb), xq, s.unsafe_offset(h // 4), s.unsafe_offset(h // 4 + nbh), nbh)
            s[h + ff + r] = ((g / (1 + exp(-g))) * u).cast[bf16]().cast[f32]()
    parallelize(up, WORKERS, WORKERS)
    for i in range(n):
        var s = scratch.unsafe_offset(i * sw)
        var hq = s.unsafe_offset(h // 4 + 2 * nbh)
        q2_quant(s.unsafe_offset(h + ff), ff, QPtr(unsafe_from_address=Int(hq)), hq.unsafe_offset(ff // 4), hq.unsafe_offset(ff // 4 + nbf))
    var dp = dw.unsafe_ptr()
    var yp = ys.unsafe_ptr()
    var orows = n * h
    def down(k: Int) {imm dp, imm yp, imm scratch, imm h, imm ff, imm db, imm sw, imm nbh, imm nbf, imm orows}:
        for t in range(k * orows // WORKERS, (k + 1) * orows // WORKERS):
            var i = t // h
            var c = t % h
            var hq = scratch.unsafe_offset(i * sw + h // 4 + 2 * nbh)
            var y = FPtr(unsafe_from_address=yp[i])
            y[c] = q2_row(BPtr(unsafe_from_address=dp[i] + c * db), QPtr(unsafe_from_address=Int(hq)), hq.unsafe_offset(ff // 4), hq.unsafe_offset(ff // 4 + nbf), nbf)
    parallelize(down, WORKERS, WORKERS)
    _ = (gw.unsafe_ptr(), uw.unsafe_ptr(), dw.unsafe_ptr(), ys.unsafe_ptr())
