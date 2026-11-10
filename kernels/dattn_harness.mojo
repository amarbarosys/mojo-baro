from std.math import sqrt
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import TileTensor, row_major

from dattn import (
    amar_dattn_exact, amar_dattn_split, amar_dattn_combine, dkv_off, dspan,
    KVPAGE, DTHREADS, DMAXS, f32,
)

comptime FLAT = 1 << 26
comptime DROWS = 4
comptime flat_layout = row_major[FLAT]()


@always_inline
def hash_val(i: Int) -> Float32:
    return Float32(Int((UInt64(i) * UInt64(2654435761)) % UInt64(2001))) / 1000 - 1


def kv_pool_elems[HD: Int, NKVH: Int](T: Int) -> Int:
    return ((T + KVPAGE - 1) // KVPAGE) * NKVH * KVPAGE * HD


def fill_q(host: HostBuffer[f32], n: Int, qscale: Int):
    var p = host.unsafe_ptr()
    for i in range(n):
        p[unsafe_offset=i] = hash_val(i) * Float32(qscale)


def fill_kv[HD: Int, NKVH: Int, KVT: DType](host: HostBuffer[KVT], T: Int, seed: Int):
    var p = host.unsafe_ptr()
    for kvh in range(NKVH):
        for t in range(T):
            var base = dkv_off[HD, NKVH, 1](t, 0, kvh)
            var src = (kvh * T + t) * HD + seed
            for d in range(HD):
                p[unsafe_offset=base + d] = hash_val(src + d).cast[KVT]()


def eff_nsplit[HD: Int, NLD: Int](T: Int, ns: Int) -> Int:
    var NS = (T + dspan[HD, NLD]() - 1) // dspan[HD, NLD]()
    return min(min(max(ns, 1), max(NS, 1)), DMAXS)


def launch_dattn[
    HD: Int, NQH: Int, NKVH: Int, KVT: DType, NLD: Int, ROT: Bool
](
    ctx: DeviceContext,
    mut q_d: DeviceBuffer[f32],
    mut k_d: DeviceBuffer[KVT],
    mut v_d: DeviceBuffer[KVT],
    mut o_d: DeviceBuffer[f32],
    mut p_d: DeviceBuffer[f32],
    T: Int,
    ns: Int,
    exact: Bool,
    m: Int = 1,
) raises:
    comptime q_layout = row_major[DROWS * NQH, HD]()
    var Q = TileTensor(q_d, q_layout)
    var tab_h = ctx.enqueue_create_host_buffer[DType.int32](4096)
    for i in range(4096):
        tab_h[i] = Int32(i)
    var tab_d = ctx.enqueue_create_buffer[DType.int32](4096)
    ctx.enqueue_copy(dst_buf=tab_d, src_buf=tab_h)
    var Kc = TileTensor(k_d, flat_layout)
    var Vc = TileTensor(v_d, flat_layout)
    var O = TileTensor(o_d, q_layout)
    var Pg = TileTensor(p_d, flat_layout)
    var scale = Float32(1) / sqrt(Float32(HD))
    if exact:
        comptime k_exact = amar_dattn_exact[HD, NQH, NKVH, KVT, 1, type_of(q_layout), type_of(flat_layout), type_of(q_layout)]
        ctx.enqueue_function[k_exact](Q, Kc, Vc, O, tab_d.unsafe_ptr(), Int32(T), scale, Int32(0), grid_dim=(NQH, m), block_dim=HD)
    else:
        comptime k_split = amar_dattn_split[HD, NQH, NKVH, KVT, 1, NLD, ROT, type_of(q_layout), type_of(flat_layout), type_of(q_layout), type_of(flat_layout)]
        ctx.enqueue_function[k_split](Q, Kc, Vc, O, Pg, tab_d.unsafe_ptr(), Int32(T), Int32(ns), scale, Int32(0), grid_dim=(NKVH, ns, m), block_dim=DTHREADS)
        if ns > 1:
            comptime k_comb = amar_dattn_combine[HD, DMAXS, type_of(flat_layout), type_of(q_layout)]
            ctx.enqueue_function[k_comb](Pg, O, Int32(ns), grid_dim=m * NQH, block_dim=DTHREADS)
