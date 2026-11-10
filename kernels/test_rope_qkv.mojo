"""amar_rope_qkv (one launch) byte-equal to amar_rope_plain on q plus amar_rope_kv_append on k/v, the two-launch decode path.
Shapes (HD, NQH, NKVH) = (64, 32, 8), (128, 28, 4), (256, 16, 4); NEOX and adjacent pairs; with and without rope_freqs.
Compares q after rope and every byte of both cache pools.
"""
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from attn import KVT, KVPAGE, KVPAD
from spark_kernels import amar_rope_plain, amar_rope_kv_append, amar_rope_qkv

comptime f32 = DType.float32
comptime NAT = 2
comptime ATT_I = 1
comptime POS = 1299
comptime BASE = Float32(5e5)


def lcg(mut st: UInt64) -> Float32:
    st = st * 6364136223846793005 + 1442695040888963407
    return Float32(Int(st >> 40)) / Float32(1 << 23) - 1


def run_case[HD_: Int, NQH_: Int, NKVH_: Int, NEOX: Bool, FF: Bool](ctx: DeviceContext) raises -> Int:
    comptime NQ = NQH_ * HD_
    comptime NKV = NKVH_ * HD_
    comptime NPAGE = (POS + 1 + KVPAGE - 1) // KVPAGE
    comptime POOL = NPAGE * NAT * NKVH_ * (KVPAGE * HD_ + KVPAD)
    comptime q_l = row_major[NQH_, HD_]()
    comptime kv_l = row_major[NKVH_, HD_]()
    comptime c_l = row_major[POOL]()
    var st = UInt64(HD_ * 7 + NQH_ + Int(NEOX) * 3 + Int(FF))
    var q_h = ctx.enqueue_create_host_buffer[f32](NQ)
    var k_h = ctx.enqueue_create_host_buffer[f32](NKV)
    var v_h = ctx.enqueue_create_host_buffer[f32](NKV)
    var ff_h = ctx.enqueue_create_host_buffer[f32](HD_ // 2)
    var tab_h = ctx.enqueue_create_host_buffer[DType.int32](NPAGE)
    ctx.synchronize()
    for i in range(NQ):
        q_h[i] = lcg(st) * 2
    for i in range(NKV):
        k_h[i] = lcg(st) * 2
        v_h[i] = lcg(st)
    for i in range(HD_ // 2):
        ff_h[i] = 1 + (lcg(st) + 1) * 4
    for i in range(NPAGE):
        tab_h[i] = Int32(NPAGE - 1 - i)
    var ff_d = ctx.enqueue_create_buffer[f32](HD_ // 2)
    var tab_d = ctx.enqueue_create_buffer[DType.int32](NPAGE)
    var k_d = ctx.enqueue_create_buffer[f32](NKV)
    var v_d = ctx.enqueue_create_buffer[f32](NKV)
    ctx.enqueue_copy(dst_buf=ff_d, src_buf=ff_h)
    ctx.enqueue_copy(dst_buf=tab_d, src_buf=tab_h)
    ctx.enqueue_copy(dst_buf=k_d, src_buf=k_h)
    ctx.enqueue_copy(dst_buf=v_d, src_buf=v_h)
    var ffp = Pointer[Float32, MutAnyOrigin](unsafe_from_address=Int(ff_d.unsafe_ptr()))
    var K = TileTensor(k_d, kv_l)
    var V = TileTensor(v_d, kv_l)
    var out_q = List[List[Float32]]()
    var out_c = List[List[Float32]]()
    for arm in range(2):
        var q_d = ctx.enqueue_create_buffer[f32](NQ)
        var kc_d = ctx.enqueue_create_buffer[KVT](POOL)
        var vc_d = ctx.enqueue_create_buffer[KVT](POOL)
        ctx.enqueue_copy(dst_buf=q_d, src_buf=q_h)
        ctx.enqueue_memset(kc_d, 0)
        ctx.enqueue_memset(vc_d, 0)
        var Q = TileTensor(q_d, q_l)
        var Kc = TileTensor(kc_d, c_l)
        var Vc = TileTensor(vc_d, c_l)
        if arm == 0:
            ctx.enqueue_function[amar_rope_plain[HD_, type_of(q_l), NEOX=NEOX, FF=FF]](Q, Int32(POS), Int32(NQH_), BASE, ffp, grid_dim=(NQH_, 1), block_dim=HD_ // 2)
            ctx.enqueue_function[amar_rope_kv_append[HD_, NAT, type_of(c_l), type_of(kv_l), HD_, NKVH_, NEOX=NEOX, FF=FF]](
                Kc, Vc, K, V, tab_d.unsafe_ptr(), Int32(POS), BASE, Int32(ATT_I), ffp, grid_dim=(NKVH_, 2), block_dim=HD_)
        else:
            ctx.enqueue_function[amar_rope_qkv[HD_, NAT, type_of(q_l), type_of(c_l), type_of(kv_l), HD_, NQH_, NKVH_, NEOX=NEOX, FF=FF]](
                Q, Kc, Vc, K, V, tab_d.unsafe_ptr(), Int32(POS), BASE, Int32(ATT_I), ffp, grid_dim=NQH_ + 2 * NKVH_, block_dim=HD_)
        var qo = ctx.enqueue_create_host_buffer[f32](NQ)
        var ko = ctx.enqueue_create_host_buffer[KVT](POOL)
        var vo = ctx.enqueue_create_host_buffer[KVT](POOL)
        ctx.enqueue_copy(dst_buf=qo, src_buf=q_d)
        ctx.enqueue_copy(dst_buf=ko, src_buf=kc_d)
        ctx.enqueue_copy(dst_buf=vo, src_buf=vc_d)
        ctx.synchronize()
        var a = List[Float32]()
        for i in range(NQ):
            a.append(qo[i])
        var c = List[Float32]()
        for i in range(POOL):
            c.append(ko[i].cast[f32]())
            c.append(vo[i].cast[f32]())
        out_q.append(a^)
        out_c.append(c^)
    var bad = 0
    for i in range(NQ):
        if out_q[0][i].to_bits() != out_q[1][i].to_bits():
            bad += 1
    for i in range(2 * POOL):
        if out_c[0][i].to_bits() != out_c[1][i].to_bits():
            bad += 1
    print("rope_qkv HD", HD_, "NQH", NQH_, "NKVH", NKVH_, "neox", NEOX, "ff", FF, "differing", bad, "of", NQ + 2 * POOL)
    return bad


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var bad = 0
    bad += run_case[64, 32, 8, False, True](ctx)
    bad += run_case[64, 32, 8, True, False](ctx)
    bad += run_case[128, 28, 4, True, False](ctx)
    bad += run_case[128, 28, 4, False, True](ctx)
    bad += run_case[256, 16, 4, True, False](ctx)
    if bad != 0:
        raise Error("FAIL rope_qkv: " + String(bad) + " values differ")
    print("PASS rope_qkv: byte-equal on 5 cases")
