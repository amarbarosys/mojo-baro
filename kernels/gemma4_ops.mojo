"""Small kernels of the gemma4 decode step (serve/gemma4.mojo): f16-weight GEMV, row RMSNorm, NEOX rope, embed, PLE input.

All activations are f32, weights are the GGUF's f16 bytes read in place. Pointers are raw device pointers; a
flag word says which optional inputs a launch uses (an unused pointer may be any valid address).
Spec of each op: docs/provenance/gemma4.md.
"""
from max.gpu import block_idx, thread_idx, lane_id, WARP_SIZE
from max.gpu.primitives import warp
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.math import tanh, sqrt, cos, sin
from layout import row_major, stack_allocation
from matmul_skinny import ROW_WAVES

comptime f32 = DType.float32
comptime F16P = Pointer[Float16, MutAnyOrigin]
comptime F32P = Pointer[Float32, MutAnyOrigin]
comptime I32P = Pointer[Int32, MutAnyOrigin]
comptime NT = 256  # threads of the norm / PLE kernels
comptime RMS_W = 1  # g4_rmsnorm flags: multiply by w
comptime RMS_RES = 2  # add res after the norm
comptime RMS_SC = 4  # multiply by sc[0] last


@always_inline
def gelu(g: Float32) -> Float32:
    return 0.5 * g * (1 + tanh(0.7978845608028654 * g * (1 + 0.044715 * g * g)))


def g4_gemv[EPI: Int](W: F16P, x: F32P, y: F32P, aux: F32P, n: Int32, k: Int32):
    """y[r] = W[r, :] . x for W [n, k] f16 row-major, one wave per row, k a multiple of 256.
    EPI 0: y = dot. EPI 1: y = gelu(y) * dot (y holds the gate rows). EPI 2: y = gelu(dot) * aux[r]."""
    var lane = Int(lane_id())
    var row = Int(block_idx.x) * ROW_WAVES + Int(thread_idx.x) // WARP_SIZE
    if row >= Int(n):
        return
    comptime V = 8
    comptime STEP = WARP_SIZE * V
    comptime U = 4
    var K = Int(k)
    var wr = W.unsafe_offset(row * K + lane * V)
    var xr = x.unsafe_offset(lane * V)
    var acc = SIMD[f32, V](0)
    var kk = 0
    while kk + U * STEP <= K:
        var ws = Array[SIMD[DType.float16, V], U](uninitialized=True)
        comptime for u in range(U):
            ws[u] = wr.unsafe_load[width=V, alignment=16](kk + u * STEP)
        comptime for u in range(U):
            acc += ws[u].cast[f32]() * xr.unsafe_load[width=V, alignment=16](kk + u * STEP)
        kk += U * STEP
    while kk < K:
        acc += wr.unsafe_load[width=V, alignment=16](kk).cast[f32]() * xr.unsafe_load[width=V, alignment=16](kk)
        kk += STEP
    var d = warp.sum(acc.reduce_add())
    if lane == 0:
        comptime if EPI == 0:
            y[unsafe_offset=row] = d
        elif EPI == 1:
            y[unsafe_offset=row] = gelu(y[unsafe_offset=row]) * d
        else:
            y[unsafe_offset=row] = gelu(d) * aux[unsafe_offset=row]


@always_inline
def block_sum(v: Float32) -> Float32:
    """Sum over the NT threads of a block; every thread gets the total."""
    var red = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[NT // WARP_SIZE]())
    var s = warp.sum(v)
    var tid = Int(thread_idx.x)
    if tid % WARP_SIZE == 0:
        red[tid // WARP_SIZE] = s
    barrier()
    var t = Float32(0)
    comptime for i in range(NT // WARP_SIZE):
        t += rebind[Float32](red[i])
    return t


def g4_rmsnorm(x: F32P, w: F32P, res: F32P, dst: F32P, sc: F32P, n: Int32, eps: Float32, flags: Int32):
    """One block of NT threads per row of n: dst = (x / sqrt(mean(x^2) + eps)) [* w] [+ res] [* sc[0]].
    Rows are contiguous (row r at r * n in x, res and out); w is one row shared by all rows."""
    var N = Int(n)
    var r = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var xr = x.unsafe_offset(r * N)
    var ss = Float32(0)
    for i in range(tid, N, NT):
        var v = xr[unsafe_offset=i]
        ss += v * v
    var inv = 1 / sqrt(block_sum(ss) / Float32(N) + eps)
    var f = Int(flags)
    for i in range(tid, N, NT):
        var v = xr[unsafe_offset=i] * inv
        if f & RMS_W:
            v *= w[unsafe_offset=i]
        if f & RMS_RES:
            v += res[unsafe_offset=r * N + i]
        if f & RMS_SC:
            v *= sc[unsafe_offset=0]
        dst[unsafe_offset=r * N + i] = v


def g4_rope(x: F32P, dst: F32P, hd: Int32, pos: Int32, base: Float32, ff: F32P, use_ff: Int32):
    """NEOX rope over whole heads (n_rot == hd): block r = head r, thread j rotates the pair (j, j + hd/2) by
    pos * base^(-2j/hd) [/ ff[j]]. dst may be x (in place) or another place (the KV pool row)."""
    var H = Int(hd)
    var half = H // 2
    var r = Int(block_idx.x)
    var j = Int(thread_idx.x)
    if j >= half:
        return
    var theta = Float32(Int(pos)) * (base ** (Float32(-2 * j) / Float32(H)))
    if use_ff != 0:
        theta = theta / ff[unsafe_offset=j]
    var c = cos(theta)
    var s = sin(theta)
    var x0 = x[unsafe_offset=r * H + j]
    var x1 = x[unsafe_offset=r * H + j + half]
    dst[unsafe_offset=r * H + j] = x0 * c - x1 * s
    dst[unsafe_offset=r * H + j + half] = x0 * s + x1 * c


def g4_embed(tab: F16P, toks: I32P, pos: Int32, dst: F32P, n: Int32, scale: Float32):
    """dst = tab[toks[pos]] * scale (the token comes from the device token row: no host sync per step)."""
    var i = Int(block_idx.x) * NT + Int(thread_idx.x)
    if i < Int(n):
        dst[unsafe_offset=i] = tab[unsafe_offset=Int(toks[unsafe_offset=Int(pos)]) * Int(n) + i].cast[f32]() * scale


def g4_ple(p: F32P, tab: F16P, toks: I32P, pos: Int32, w: F32P, dst: F32P, n: Int32, row: Int32, eps: Float32, sw: Float32, se: Float32):
    """Per-layer inputs, one block per layer slice of n (= NT): dst = rmsnorm(p, eps) * (w * sw) + tab[tok] * se,
    with tab the per-layer token table [vocab, row = n_layers * n] f16 and tok = toks[pos]."""
    var tok = toks[unsafe_offset=Int(pos)]
    var N = Int(n)
    var b = Int(block_idx.x)
    var i = Int(thread_idx.x)
    var o = b * N + i
    var v = p[unsafe_offset=o]
    var inv = 1 / sqrt(block_sum(v * v) / Float32(N) + eps)
    var e = tab[unsafe_offset=Int(tok) * Int(row) + o]
    dst[unsafe_offset=o] = v * inv * (w[unsafe_offset=i] * sw) + e.cast[f32]() * se
