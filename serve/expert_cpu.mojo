"""BARO_TIER_CPU: routed experts the VRAM tier missed, computed on the CPU
straight from the pinned store while the GPU runs the hits (Strata's split).

x and the swiglu activation are quantized to int8 per 32-group (scale
max|v|/127) and dotted against the 4-bit weights with AVX-512 VNNI vpdpbusd,
the llama.cpp CPU shape; the min term uses exact float group sums. The f32
dequant version was compute-bound (24.6 GB/s cache-hot, 17.9 cold). Close to
the GPU m=1 path, not bit-exact.
"""
from max.algorithm import parallelize
from std.memory import bitcast
from std.math import exp, round
from std.sys import llvm_intrinsic

comptime f32 = DType.float32
comptime f16 = DType.float16
comptime bf16 = DType.bfloat16
comptime u16 = DType.uint16
comptime Q4K = 256
comptime Q4K_BYTES = 144
comptime WORKERS = 8


@inline(.always)
def _half(lo: UInt8, hi: UInt8) -> Float32:
    return bitcast[f16, 1](SIMD[u16, 1](UInt16(lo) | (UInt16(hi) << 8)))[0].cast[f32]()


@inline(.always)
def q4k_row_dot_cpu(
    w: MutPointer[UInt8, MutUntrackedOrigin], row_base: Int, nb: Int,
    xq: MutPointer[Int8, MutUntrackedOrigin], xd: MutPointer[Float32, MutUntrackedOrigin],
    xs: MutPointer[Float32, MutUntrackedOrigin],
) -> Float32:
    """sum_g d*sc_g * xd_g * (q . xq)_g - dm*mn_g * xs_g, one vpdpbusd per 64 weights."""
    var acc = SIMD[f32, 16](0)
    var macc = SIMD[f32, 8](0)
    for b in range(nb):
        var base = row_base + b * Q4K_BYTES
        var hdr = w.unsafe_offset(base).unsafe_load[width=16]()
        var d = _half(hdr[0], hdr[1])
        var dm = _half(hdr[2], hdr[3])
        var s4 = hdr.slice[4, offset=4]()
        var m4 = hdr.slice[4, offset=8]()
        var t4 = hdr.slice[4, offset=12]()
        var sc = (s4 & 0x3F).join((t4 & 0x0F) | ((s4 >> 2) & 0x30)).cast[f32]()
        var mn = (m4 & 0x3F).join((t4 >> 4) | ((m4 >> 2) & 0x30)).cast[f32]()
        var gi = b * 8
        var scv = sc * xd.unsafe_offset(gi).unsafe_load[width=8]() * d
        macc = (mn * xs.unsafe_offset(gi).unsafe_load[width=8]()).fma(SIMD[f32, 8](dm), macc)
        comptime for p in range(4):
            var qb = w.unsafe_offset(base + 16 + p * 32).unsafe_load[width=32]()
            var r = llvm_intrinsic["llvm.x86.avx512.vpdpbusd.512", SIMD[DType.int32, 16]](
                SIMD[DType.int32, 16](0), (qb & 0x0F).join(qb >> 4), xq.unsafe_offset((gi + 2 * p) * 32).unsafe_load[width=64]()
            )
            var sv = SIMD[f32, 8](scv[2 * p]).join(SIMD[f32, 8](scv[2 * p + 1]))
            acc = r.cast[f32]().fma(sv, acc)
    return acc.reduce_add() - macc.reduce_add()


@inline(.always)
def _quant(
    x: MutPointer[Float32, MutUntrackedOrigin], n: Int, xq: MutPointer[Int8, MutUntrackedOrigin],
    xd: MutPointer[Float32, MutUntrackedOrigin], xs: MutPointer[Float32, MutUntrackedOrigin],
):
    for g in range(n // 32):
        var v = x.unsafe_offset(g * 32).unsafe_load[width=32]()
        var amax = abs(v).reduce_max()
        xs[unsafe_offset=g] = v.reduce_add()
        xd[unsafe_offset=g] = amax / 127
        var inv = Float32(127) / amax if amax > 0 else Float32(0)
        xq.unsafe_offset(g * 32).unsafe_store(round(v * inv).cast[DType.int8]())


def expert_ffn_cpu[
    H: Int, FFN: Int
](
    gate: MutPointer[UInt8, MutUntrackedOrigin],
    up: MutPointer[UInt8, MutUntrackedOrigin],
    down: MutPointer[UInt8, MutUntrackedOrigin],
    ids: MutPointer[Int32, MutUntrackedOrigin],
    wts: MutPointer[Float32, MutUntrackedOrigin],
    n: Int,
    x: MutPointer[Float32, MutUntrackedOrigin],
    hbuf: MutPointer[Float32, MutUntrackedOrigin],
    xq: MutPointer[Int8, MutUntrackedOrigin],
    aux: MutPointer[Float32, MutUntrackedOrigin],
    dst: MutPointer[Float32, MutUntrackedOrigin],
):
    """dst[c] = sum_j wts[j] * down_e(swiglu(gate_e x, up_e x))[c], e = ids[j], j < n.
    gate/up/down point at the layer's tensor in the store; x is bf16-exact f32;
    hbuf holds n * FFN floats, xq H + n * FFN bytes, aux 2 * (H + n * FFN) / 32 floats."""
    comptime NBH = H // Q4K
    comptime NBF = FFN // Q4K
    comptime ROW_GU = NBH * Q4K_BYTES
    comptime ROW_D = NBF * Q4K_BYTES

    comptime GH = H // 32
    var gs = (n * FFN) // 32
    _quant(x, H, xq, aux, aux.unsafe_offset(GH))

    var rows = n * FFN

    def gu(k: Int) {var gate, var up, var ids, var xq, var aux, var hbuf, var rows}:
        for t in range(k * rows // WORKERS, (k + 1) * rows // WORKERS):
            var rb = Int(ids[unsafe_offset=t // FFN]) * FFN * ROW_GU + (t % FFN) * ROW_GU
            var g = q4k_row_dot_cpu(gate, rb, NBH, xq, aux, aux.unsafe_offset(GH))
            var u = q4k_row_dot_cpu(up, rb, NBH, xq, aux, aux.unsafe_offset(GH))
            hbuf[unsafe_offset=t] = (g / (Float32(1) + exp(-g)) * u).cast[bf16]().cast[f32]()

    parallelize(gu, WORKERS, WORKERS)
    var hq = xq.unsafe_offset(H)
    var hd = aux.unsafe_offset(2 * GH)
    var hs = hd.unsafe_offset(gs)
    _quant(hbuf, n * FFN, hq, hd, hs)

    def dn(k: Int) {var down, var ids, var wts, var n, var hq, var hd, var hs, var dst}:
        for c in range(k * H // WORKERS, (k + 1) * H // WORKERS):
            var acc = Float32(0)
            for j in range(n):
                acc += wts[unsafe_offset=j] * q4k_row_dot_cpu(down, Int(ids[unsafe_offset=j]) * H * ROW_D + c * ROW_D, NBF, hq.unsafe_offset(j * FFN), hd.unsafe_offset(j * (FFN // 32)), hs.unsafe_offset(j * (FFN // 32)))
            dst[unsafe_offset=c] = acc

    parallelize(dn, WORKERS, WORKERS)
