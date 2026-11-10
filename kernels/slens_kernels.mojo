# Micro-batch kernels for the steer lens (serve/spark.mojo `op: slens`): R rows are B independent micro-sequences of
# T tokens, row r = u * T + t. No KV cache: K and V of the batch itself are the keys.
from max.gpu import block_idx, global_idx, lane_id, thread_idx, WARP_SIZE
from max.gpu.primitives import warp
from max.gpu.sync import barrier
from std.math import cos, sin
from bexp import exp, log
from max.gpu.memory import AddressSpace
from layout import row_major, stack_allocation

comptime f32 = DType.float32
comptime P32 = Pointer[Float32, MutAnyOrigin]


def amar_rope_micro[HD_: Int, NROT: Int, NEOX: Bool, FF: Bool](X: P32, T: Int32, nh: Int32, freq_base: Float32, ff: P32):
    """Rotary embedding of rows [R, nh, HD_] with position t = r mod T (amar_rope_plain with a per-sequence restart)."""
    var h = Int(block_idx.x)
    var r = Int(block_idx.y)
    var j = Int(thread_idx.x)
    var theta = Float32(r % Int(T)) * exp(Float32(-2 * j) / Float32(NROT) * log(freq_base))
    comptime if FF:
        theta = theta / ff[j]
    var c = cos(theta)
    var s = sin(theta)
    var p = X + (r * Int(nh) + h) * HD_
    comptime if NEOX:
        var x0 = p[j]
        var x1 = p[j + NROT // 2]
        p[j] = x0 * c - x1 * s
        p[j + NROT // 2] = x0 * s + x1 * c
    else:
        var x0 = p[2 * j]
        var x1 = p[2 * j + 1]
        p[2 * j] = x0 * c - x1 * s
        p[2 * j + 1] = x0 * s + x1 * c


def amar_attn_micro[HD_: Int, NQH_: Int, NKVH_: Int, ODT: DType](
    Q: P32, K: P32, V: P32, O: Pointer[Scalar[ODT], MutAnyOrigin], T: Int32, scale: Float32,
):
    """Causal attention inside each micro-sequence: row r = u * T + t attends to rows u * T .. r of the same batch."""
    var h = Int(block_idx.x)
    var r = Int(block_idx.y)
    var d = Int(thread_idx.x)
    var t = r % Int(T)
    var first = r - t
    var kvh = h // (NQH_ // NKVH_)
    var q = Q[(r * NQH_ + h) * HD_ + d]
    var sums = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[HD_ // WARP_SIZE]())
    var m: Float32 = -1e30
    var den: Float32 = 0
    var acc: Float32 = 0
    for tp in range(t + 1):
        var kv = (first + tp) * NKVH_ + kvh
        var w = warp.sum(q * K[kv * HD_ + d])
        if lane_id() == 0:
            sums[d // WARP_SIZE] = rebind[sums.ElementType](w)
        barrier()
        var tot: Float32 = 0
        comptime for i in range(HD_ // WARP_SIZE):
            tot += rebind[Scalar[f32]](sums[i])
        barrier()
        var s = tot * scale
        var mn = max(m, s)
        var a = exp(m - mn)
        var p = exp(s - mn)
        acc = acc * a + p * V[kv * HD_ + d]
        den = den * a + p
        m = mn
    O[r * (NQH_ * HD_) + h * HD_ + d] = (acc / den).cast[ODT]()


def amar_micro_init(xp: P32, xc: P32, bank: P32, scale: Float32, T: Int32, H: Int32, n: Int32):
    """xp[u*T + t, :] = xc[t, :] + scale * bank[u, :] for every row (the steered residual after layer L)."""
    var i = Int(global_idx.x)
    if i < Int(n):
        var h = i % Int(H)
        var row = i // Int(H)
        var t = row % Int(T)
        var u = row // Int(T)
        xp[i] = xc[t * Int(H) + h] + scale * bank[u * Int(H) + h]


def amar_rows_add_vec(x: P32, v: P32, n: Int32, H: Int32):
    """x[r, :] += v for every row r of an [n / H, H] block: the steering vector added after a layer (engine.mojo path)."""
    var i = Int(global_idx.x)
    if i < Int(n):
        x[i] += v[i % Int(H)]
