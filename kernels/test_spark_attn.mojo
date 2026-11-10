"""KATT head-dimension parity (bench/dense-protocol.md, KATT section).
The Spark attention path, amar_rope_kv_append then amar_attn_decode_swa_gated, runs at
(HD, NQH, NKVH) = (64, 32, 8), (128, 28, 4), (256, 16, 4) on synthetic data. Inputs, the
appended cache row and the bf16 outputs are dumped to KATT_OUT (default .work/katt) for
tools/spark-attn-ref.py, the numpy float64 oracle that decides pass or fail.
"""
from std.math import sqrt
from std.os import getenv
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from attn import KVT, KVPAGE, KVPAD, kv_off
from spark_kernels import amar_rope_kv_append, amar_attn_decode_swa_gated

comptime f32 = DType.float32
comptime bf16 = DType.bfloat16
comptime NAT = 2
comptime ATT_I = 1
comptime POS = 299
comptime R = 2
comptime TT = POS + R
comptime WIN = 100
comptime BASE = Float32(1e4)


def lcg(mut st: UInt64) -> Float32:
    st = st * 6364136223846793005 + 1442695040888963407
    return Float32(Int(st >> 40)) / Float32(1 << 23) - 1


def dump(path: String, p: MutPointer[UInt8, MutUntrackedOrigin], nbytes: Int) raises:
    with open(path, "w") as f:
        f.write_bytes(Span[UInt8](unsafe_ptr=p, length=nbytes))


def run_case[HD_: Int, NQH_: Int, NKVH_: Int](ctx: DeviceContext, odir: String) raises:
    comptime NROT_ = HD_
    comptime NQ = R * NQH_ * HD_
    comptime NKV = NKVH_ * HD_
    comptime ND = NKVH_ * TT * HD_
    comptime NPAGE = (TT + KVPAGE - 1) // KVPAGE
    comptime POOL = NPAGE * NAT * NKVH_ * (KVPAGE * HD_ + KVPAD)
    comptime q_l = row_major[R * NQH_, HD_]()
    comptime kv_l = row_major[NKVH_, HD_]()
    comptime c_l = row_major[POOL]()
    comptime g_l = row_major[R * NQH_]()  # one gate per query row, as the engine passes them since 2c936ea
    comptime k_app = amar_rope_kv_append[NROT_, NAT, type_of(c_l), type_of(kv_l), HD_, NKVH_]
    comptime k_att = amar_attn_decode_swa_gated[type_of(q_l), type_of(c_l), type_of(g_l), type_of(q_l), NAT, HD_, NQH_, NKVH_]

    var q_h = ctx.enqueue_create_host_buffer[f32](NQ)
    var kd_h = ctx.enqueue_create_host_buffer[f32](ND)
    var vd_h = ctx.enqueue_create_host_buffer[f32](ND)
    var kn_h = ctx.enqueue_create_host_buffer[f32](NKV)
    var vn_h = ctx.enqueue_create_host_buffer[f32](NKV)
    var g_h = ctx.enqueue_create_host_buffer[f32](R * NQH_)
    var kp_h = ctx.enqueue_create_host_buffer[KVT](POOL)
    var vp_h = ctx.enqueue_create_host_buffer[KVT](POOL)
    var o_h = ctx.enqueue_create_host_buffer[bf16](NQ)
    var ka_h = ctx.enqueue_create_host_buffer[f32](2 * NKV)
    ctx.synchronize()
    var st = UInt64(HD_ * 1000 + NQH_)
    for i in range(NQ):
        q_h[i] = lcg(st) * 2
    for i in range(POOL):
        kp_h[i] = 0
        vp_h[i] = 0
    for h in range(NKVH_):
        for t in range(TT):
            for d in range(HD_):
                var k = lcg(st) * 2
                var v = lcg(st)
                var i = (h * TT + t) * HD_ + d
                kd_h[i] = k
                vd_h[i] = v
                if t == POS:
                    kn_h[h * HD_ + d] = k
                    vn_h[h * HD_ + d] = v
                else:
                    var o = kv_off[NAT, HD_, NKVH_](t, ATT_I, h) + d
                    kp_h[o] = k.cast[KVT]()
                    vp_h[o] = v.cast[KVT]()
    for h in range(R * NQH_):
        g_h[h] = lcg(st) * 3

    var q_d = ctx.enqueue_create_buffer[f32](NQ)
    var kn_d = ctx.enqueue_create_buffer[f32](NKV)
    var vn_d = ctx.enqueue_create_buffer[f32](NKV)
    var g_d = ctx.enqueue_create_buffer[f32](R * NQH_)
    var kp_d = ctx.enqueue_create_buffer[KVT](POOL)
    var vp_d = ctx.enqueue_create_buffer[KVT](POOL)
    var o_d = ctx.enqueue_create_buffer[bf16](NQ)
    ctx.enqueue_copy(dst_buf=q_d, src_buf=q_h)
    ctx.enqueue_copy(dst_buf=kn_d, src_buf=kn_h)
    ctx.enqueue_copy(dst_buf=vn_d, src_buf=vn_h)
    ctx.enqueue_copy(dst_buf=g_d, src_buf=g_h)
    ctx.enqueue_copy(dst_buf=kp_d, src_buf=kp_h)
    ctx.enqueue_copy(dst_buf=vp_d, src_buf=vp_h)
    var Q = TileTensor(q_d, q_l)
    var Kn = TileTensor(kn_d, kv_l)
    var Vn = TileTensor(vn_d, kv_l)
    var G = TileTensor(g_d, g_l)
    var tab_h = ctx.enqueue_create_host_buffer[DType.int32](4096)
    for i in range(4096):
        tab_h[i] = Int32(i)
    var tab_d = ctx.enqueue_create_buffer[DType.int32](4096)
    ctx.enqueue_copy(dst_buf=tab_d, src_buf=tab_h)
    var Kc = TileTensor(kp_d, c_l)
    var Vc = TileTensor(vp_d, c_l)
    var O = TileTensor(o_d, q_l)
    ctx.enqueue_function[k_app](
        Kc, Vc, Kn, Vn, tab_d.unsafe_ptr(), Int32(POS), BASE, Int32(ATT_I), Pointer[Float32, MutAnyOrigin](unsafe_from_address=Int(tab_d.unsafe_ptr())), grid_dim=(NKVH_, 2), block_dim=HD_
    )
    var scale = Float32(1) / sqrt(Float32(HD_))
    var pre = odir + "/hd" + String(HD_) + "_"
    for w in range(2):
        var win = WIN if w == 1 else 0
        ctx.enqueue_memset(o_d, 0)
        ctx.enqueue_function[k_att](
            Q, Kc, Vc, G, O, tab_d.unsafe_ptr(), Int32(POS + 1), Int32(win), scale, Int32(ATT_I),
            grid_dim=(NQH_, R), block_dim=HD_,
        )
        ctx.enqueue_copy(dst_buf=o_h, src_buf=o_d)
        ctx.synchronize()
        var name = String("o_swa.bin") if w == 1 else String("o_full.bin")
        dump(pre + name, o_h.unsafe_ptr().unsafe_bitcast[UInt8](), NQ * 2)
    ctx.enqueue_copy(dst_buf=kp_h, src_buf=kp_d)
    ctx.enqueue_copy(dst_buf=vp_h, src_buf=vp_d)
    ctx.synchronize()
    for h in range(NKVH_):
        for d in range(HD_):
            var o = kv_off[NAT, HD_, NKVH_](POS, ATT_I, h) + d
            ka_h[h * HD_ + d] = kp_h[o].cast[f32]()
            ka_h[NKV + h * HD_ + d] = vp_h[o].cast[f32]()
    dump(pre + "q.bin", q_h.unsafe_ptr().unsafe_bitcast[UInt8](), NQ * 4)
    dump(pre + "kd.bin", kd_h.unsafe_ptr().unsafe_bitcast[UInt8](), ND * 4)
    dump(pre + "vd.bin", vd_h.unsafe_ptr().unsafe_bitcast[UInt8](), ND * 4)
    dump(pre + "gate.bin", g_h.unsafe_ptr().unsafe_bitcast[UInt8](), R * NQH_ * 4)
    dump(pre + "kvapp.bin", ka_h.unsafe_ptr().unsafe_bitcast[UInt8](), 2 * NKV * 4)
    print("KATT dump HD", HD_, "NQH", NQH_, "NKVH", NKVH_, "pool", POOL, "->", pre + "*.bin")


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var odir = getenv("KATT_OUT", ".work/katt")
    run_case[64, 32, 8](ctx, odir)
    run_case[128, 28, 4](ctx, odir)
    run_case[256, 16, 4](ctx, odir)
