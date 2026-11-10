"""Step 1: one attention layer and one SSM layer (step 1b) plus the greedy pick of the launch path (serve/window.mojo blk32_forward shape), layer 3 of the Qwythos pack.

usage: test_cpu_layer PACKDIR
Every stage runs the GPU kernel, then the CPU form of cpu/ops.mojo on the GPU's own input, and compares bits.
Exit 0 only if every stage of every step is bit-equal. Prints one line per stage and a final table.
"""
from std.memory import bitcast
from std.sys import argv
from std.os import getenv
from std.math import ceildiv
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major
from matmul_skinny import amar_matmul_skinny_q4rowb, amar_skinny_reduce, amar_skinny_reduce_add, amar_skinny_reduce_swiglu_bf16, ROW_WAVES, ROW_THREADS
from elementwise import amar_rmsnorm_cast
from attn import amar_head_rmsnorm, amar_rope_yarn, amar_kv_append, amar_attn_decode, amar_gate_mul_cast, amar_qgate_split, KVT, KVPAGE
from model import H, FFN, VOCAB, QF, KV, HD, NQH, NKVH, NROT, YARN_LOW, YARN_HIGH, FREQ_BASE, FREQ_SCALE, MSCALE
from ops import q4_row_dot, rmsnorm_cast, head_rmsnorm, rope_yarn, attn_head, swiglu, ssm_gates, ssm_conv, ssm_delta, ssm_gated
from ssm import amar_ssm_conv_l2g, amar_ssm_delta_step, amar_ssm_gated_out_bf16, NH_K, NH_V, SSTATE, CONV, SSM_EPS
from matmul_skinny import SPLITK
from elementwise import amar_argmax_pos
from sample_ref import sample_row_ref
from harness import load_pack
from pack import Pack
from bexp import exp

comptime bf16 = DType.bfloat16
comptime f32 = DType.float32
comptime ATT = NQH * HD
comptime NSTEP = 8
comptime CACHE = NKVH * KVPAGE * HD
comptime h_l = row_major[H]()
comptime xm_l = row_major[1, H]()
comptime qfm_l = row_major[1, QF]()
comptime qm_l = row_major[NQH, HD]()
comptime kvm_l = row_major[NKVH, HD]()
comptime kvf_l = row_major[1, KV]()
comptime attf_l = row_major[ATT]()
comptime attm_l = row_major[1, ATT]()
comptime ffnm_l = row_major[1, FFN]()
comptime hd_l = row_major[HD]()
comptime cache_l = row_major[CACHE]()
comptime p_h_l = row_major[1, 1, H]()
comptime p_qf_l = row_major[1, 1, QF]()
comptime p_kv_l = row_major[1, 1, KV]()
comptime p_ffn_l = row_major[1, 1, FFN]()


struct Q4W(Movable):
    var q: DeviceBuffer[DType.uint8]
    var s: DeviceBuffer[DType.float16]
    var qh: List[UInt8]
    var sh: List[UInt8]

    def __init__(out self, ctx: DeviceContext, dir: String, idx: String, name: String) raises:
        var off = 0
        var n = 0
        for l in idx.split("\n"):
            var p = l.split(" ")
            if len(p) == 4 and String(p[0]) == name:
                off = atol(String(p[2]))
                n = atol(String(p[3]))
        if n == 0:
            raise Error("FAIL index: no tensor " + name)
        self.qh = read_at(dir + "/pack.bin", off, n // 2)
        self.sh = read_at(dir + "/pack.bin", off + n // 2, n // 16)
        self.q = ctx.enqueue_create_buffer[DType.uint8](n // 2)
        self.s = ctx.enqueue_create_buffer[DType.float16](n // 32)
        ctx.enqueue_copy(self.q, self.qh.unsafe_ptr())
        ctx.enqueue_copy(self.s, self.sh.unsafe_ptr().unsafe_bitcast[Float16]())


def read_at(path: String, off: Int, n: Int) raises -> List[UInt8]:
    with open(path, "r") as f:
        _ = f.seek(off)
        return f.read_bytes(n)


def f32_at(dir: String, idx: String, name: String, n: Int) raises -> List[Float32]:
    for l in idx.split("\n"):
        var p = l.split(" ")
        if len(p) == 4 and String(p[0]) == name:
            var b = read_at(dir + "/pack.bin", atol(String(p[2])), n * 4)
            var out = List[Float32](length=n, fill=0)
            for i in range(n):
                out[i] = b.unsafe_ptr().unsafe_bitcast[Float32]().unsafe_offset(i).unsafe_load()
            return out^
    raise Error("FAIL index: no tensor " + name)


def up_f32(ctx: DeviceContext, v: List[Float32]) raises -> DeviceBuffer[f32]:
    var d = ctx.enqueue_create_buffer[f32](len(v))
    ctx.enqueue_copy(d, v.unsafe_ptr())
    return d^


def dn_f32(ctx: DeviceContext, d: DeviceBuffer[f32], n: Int) raises -> List[Float32]:
    var v = List[Float32](length=n, fill=0)
    ctx.enqueue_copy(v.unsafe_ptr(), d)
    ctx.synchronize()
    return v^


def dn_bf16(ctx: DeviceContext, d: DeviceBuffer[bf16], n: Int) raises -> List[BFloat16]:
    var v = List[BFloat16](length=n, fill=0)
    ctx.enqueue_copy(v.unsafe_ptr(), d)
    ctx.synchronize()
    return v^


def bits32(x: Float32) -> UInt32:
    return bitcast[DType.uint32, 1](SIMD[f32, 1](x))[0]


def digest(v: List[Float32], h: UInt64) -> UInt64:
    var r = h
    for x in v:
        r = (r ^ UInt64(bits32(x))) * 1099511628211
    return r


def diff_f32(name: String, step: Int, a: List[Float32], b: List[Float32], mut bad: Dict[String, Int]):
    var c = 0
    for i in range(len(a)):
        if bits32(a[i]) != bits32(b[i]):
            if c < 2 and bad.get(name, 0) < 2:
                print("  mismatch", name, "step", step, "i", i, "cpu", a[i], "gpu", b[i])
            c += 1
    bad[name] = bad.get(name, 0) + c


def diff_bf16(name: String, step: Int, a: List[BFloat16], b: List[BFloat16], mut bad: Dict[String, Int]):
    var c = 0
    for i in range(len(a)):
        if bitcast[DType.uint16, 1](SIMD[bf16, 1](a[i]))[0] != bitcast[DType.uint16, 1](SIMD[bf16, 1](b[i]))[0]:
            if c < 2 and bad.get(name, 0) < 2:
                print("  mismatch", name, "step", step, "i", i, "cpu", a[i], "gpu", b[i])
            c += 1
    bad[name] = bad.get(name, 0) + c


def rms(x: List[Float32], g: List[Float32]) -> List[BFloat16]:
    var o = List[BFloat16](length=len(x), fill=0)
    rmsnorm_cast(x.unsafe_ptr(), g.unsafe_ptr(), o.unsafe_ptr(), len(x), Float32(1e-6))
    return o^


def gemv[N: Int, K: Int](ctx: DeviceContext, a: DeviceBuffer[bf16], w: Q4W, p: DeviceBuffer[f32]) raises:
    comptime a_l = row_major[1, K]()
    comptime q_l = row_major[N, K // 2]()
    comptime s_l = row_major[N, K // 32]()
    comptime p_l = row_major[1, 1, N]()
    ctx.enqueue_function[amar_matmul_skinny_q4rowb[2, 1, type_of(a_l), type_of(q_l), type_of(s_l), type_of(p_l)]](
        TileTensor(a, a_l), TileTensor(w.q, q_l), TileTensor(w.s, s_l), TileTensor(p, p_l),
        Int32(1), Int32(N), Int32(K), grid_dim=ceildiv(N, ROW_WAVES), block_dim=ROW_THREADS,
    )


def cpu_gemv(a: List[BFloat16], w: Q4W, n: Int, k: Int, r0: Int = 0) -> List[Float32]:
    var out = List[Float32](length=n, fill=0)
    for r in range(n):
        out[r] = q4_row_dot(
            a.unsafe_ptr(), w.qh.unsafe_ptr().unsafe_offset((r0 + r) * (k // 2)),
            w.sh.unsafe_ptr().unsafe_bitcast[Float16]().unsafe_offset((r0 + r) * (k // 32)), k // 32,
        )
    return out^


def main() raises:
    var a = argv()
    if len(a) != 2:
        raise Error("usage: test_cpu_layer PACKDIR")
    var dir = String(a[1])
    var idx: String
    with open(dir + "/index.txt", "r") as f:
        idx = f.read()
    var ctx = DeviceContext()
    var bad = Dict[String, Int]()
    pack_check(ctx, dir, idx, bad)
    var wq = Q4W(ctx, dir, idx, "blk.3.attn_q.weight")
    var wk = Q4W(ctx, dir, idx, "blk.3.attn_k.weight")
    var wv = Q4W(ctx, dir, idx, "blk.3.attn_v.weight")
    var wo = Q4W(ctx, dir, idx, "blk.3.attn_output.weight")
    var wg = Q4W(ctx, dir, idx, "blk.3.ffn_gate.weight")
    var wu = Q4W(ctx, dir, idx, "blk.3.ffn_up.weight")
    var wd = Q4W(ctx, dir, idx, "blk.3.ffn_down.weight")
    var g_att = f32_at(dir, idx, "blk.3.attn_norm.weight", H)
    var g_post = f32_at(dir, idx, "blk.3.post_attention_norm.weight", H)
    var g_qn = f32_at(dir, idx, "blk.3.attn_q_norm.weight", HD)
    var g_kn = f32_at(dir, idx, "blk.3.attn_k_norm.weight", HD)
    var d_gatt = up_f32(ctx, g_att)
    var d_gpost = up_f32(ctx, g_post)
    var d_gqn = up_f32(ctx, g_qn)
    var d_gkn = up_f32(ctx, g_kn)
    var x_d = ctx.enqueue_create_buffer[f32](H)
    var curb = ctx.enqueue_create_buffer[bf16](H)
    var p_qf = ctx.enqueue_create_buffer[f32](QF)
    var p_kv = ctx.enqueue_create_buffer[f32](KV)
    var p_h = ctx.enqueue_create_buffer[f32](H)
    var p_g = ctx.enqueue_create_buffer[f32](FFN)
    var p_u = ctx.enqueue_create_buffer[f32](FFN)
    var qf = ctx.enqueue_create_buffer[f32](QF)
    var q_d = ctx.enqueue_create_buffer[f32](ATT)
    var gate_d = ctx.enqueue_create_buffer[f32](ATT)
    var k_d = ctx.enqueue_create_buffer[f32](KV)
    var v_d = ctx.enqueue_create_buffer[f32](KV)
    var ao_d = ctx.enqueue_create_buffer[f32](ATT)
    var aob = ctx.enqueue_create_buffer[bf16](ATT)
    var fgb = ctx.enqueue_create_buffer[bf16](FFN)
    comptime assert KVT == f32
    var kc = ctx.enqueue_create_buffer[f32](CACHE)
    var vc = ctx.enqueue_create_buffer[f32](CACHE)
    var tab = ctx.enqueue_create_buffer[DType.int32](1 << 16)
    ctx.enqueue_memset(kc, 0)
    ctx.enqueue_memset(vc, 0)
    ctx.enqueue_memset(tab, 0)
    ctx.synchronize()
    var X = TileTensor(x_d, xm_l)
    var CurB = TileTensor(curb, xm_l)
    var Gatt = TileTensor(d_gatt, h_l)
    var Gpost = TileTensor(d_gpost, h_l)
    var Gqn = TileTensor(d_gqn, hd_l)
    var Gkn = TileTensor(d_gkn, hd_l)
    var Qfm = TileTensor(qf, qfm_l)
    var Q = TileTensor(q_d, qm_l)
    var Gate = TileTensor(gate_d, attf_l)
    var Kf = TileTensor(k_d, kvf_l)
    var Khd = TileTensor(k_d, kvm_l)
    var Vf = TileTensor(v_d, kvf_l)
    var Vhd = TileTensor(v_d, kvm_l)
    var Kc = TileTensor(kc, cache_l)
    var Vc = TileTensor(vc, cache_l)
    var Ao = TileTensor(ao_d, qm_l)
    var Aof = TileTensor(ao_d, attf_l)
    var Aob = TileTensor(aob, attf_l)
    var Aobm = TileTensor(aob, attm_l)
    var Fgb = TileTensor(fgb, ffnm_l)
    var Pqf = TileTensor(p_qf, p_qf_l)
    var Pkv = TileTensor(p_kv, p_kv_l)
    var Ph = TileTensor(p_h, p_h_l)
    var Pg = TileTensor(p_g, p_ffn_l)
    var Pu = TileTensor(p_u, p_ffn_l)
    var toks: List[Int] = [1000, 5000, 20000, 100000, 300, 77, 4242, 99999]
    var poss = [0, 1, 2, 3, 4, 5, 300, 5000]
    var dg = UInt64(14695981039346656037)
    var emb_cache = Dict[Int, Int]()
    print("arm gpu=launch-path kernels (amar_rmsnorm_cast, q4rowb, head_rmsnorm, rope_yarn, attn_decode, reduce*) cpu=cpu/ops.mojo pack=" + dir)
    for st in range(NSTEP):
        var eb = read_at(dir + "/pack.bin", toks[st] * H * 2, H * 2)
        var x0 = List[Float32](length=H, fill=0)
        for i in range(H):
            x0[i] = eb.unsafe_ptr().unsafe_bitcast[BFloat16]().unsafe_offset(i).unsafe_load().cast[f32]()
        ctx.enqueue_copy(x_d, x0.unsafe_ptr())
        var pos = poss[st]
        ctx.enqueue_function[amar_rmsnorm_cast[type_of(xm_l), type_of(h_l), type_of(xm_l)]](X, Gatt, CurB, Int32(H), Float32(1e-6), grid_dim=1, block_dim=256)
        var cur_g = dn_bf16(ctx, curb, H)
        diff_bf16("rms1", st, rms(x0, g_att), cur_g, bad)
        gemv[QF, H](ctx, curb, wq, p_qf)
        ctx.enqueue_function[amar_skinny_reduce[type_of(p_qf_l), type_of(qfm_l), 1]](Pqf, Qfm, Int32(1), Int32(QF), grid_dim=ceildiv(QF, 256), block_dim=256)
        var qf_g = dn_f32(ctx, qf, QF)
        diff_f32("gemv_q", st, cpu_gemv(cur_g, wq, QF, H), qf_g, bad)
        gemv[KV, H](ctx, curb, wk, p_kv)
        ctx.enqueue_function[amar_skinny_reduce[type_of(p_kv_l), type_of(kvf_l), 1]](Pkv, Kf, Int32(1), Int32(KV), grid_dim=ceildiv(KV, 256), block_dim=256)
        var k_g = dn_f32(ctx, k_d, KV)
        diff_f32("gemv_k", st, cpu_gemv(cur_g, wk, KV, H), k_g, bad)
        gemv[KV, H](ctx, curb, wv, p_kv)
        ctx.enqueue_function[amar_skinny_reduce[type_of(p_kv_l), type_of(kvf_l), 1]](Pkv, Vf, Int32(1), Int32(KV), grid_dim=ceildiv(KV, 256), block_dim=256)
        var v_g = dn_f32(ctx, v_d, KV)
        diff_f32("gemv_v", st, cpu_gemv(cur_g, wv, KV, H), v_g, bad)
        ctx.enqueue_function[amar_qgate_split[type_of(qfm_l), type_of(qm_l), type_of(attf_l)]](Qfm, Q, Gate, grid_dim=(NQH, 1), block_dim=HD)
        var q_in = dn_f32(ctx, q_d, ATT)
        ctx.enqueue_function[amar_head_rmsnorm[type_of(qm_l), type_of(hd_l)]](Q, Gqn, Float32(1e-6), grid_dim=NQH, block_dim=HD)
        ctx.enqueue_function[amar_head_rmsnorm[type_of(kvm_l), type_of(hd_l)]](Khd, Gkn, Float32(1e-6), grid_dim=NKVH, block_dim=HD)
        var q_n = dn_f32(ctx, q_d, ATT)
        var k_n = dn_f32(ctx, k_d, KV)
        var q_c = q_in.copy()
        for h in range(NQH):
            head_rmsnorm(q_c.unsafe_ptr().unsafe_offset(h * HD), HD, g_qn.unsafe_ptr(), Float32(1e-6))
        diff_f32("hrms_q", st, q_c, q_n, bad)
        var k_c = k_g.copy()
        for h in range(NKVH):
            head_rmsnorm(k_c.unsafe_ptr().unsafe_offset(h * HD), HD, g_kn.unsafe_ptr(), Float32(1e-6))
        diff_f32("hrms_k", st, k_c, k_n, bad)
        ctx.enqueue_function[amar_rope_yarn[type_of(qm_l)]](Q, Int32(pos), Int32(NQH), grid_dim=(NQH, 1), block_dim=32)
        ctx.enqueue_function[amar_rope_yarn[type_of(kvm_l)]](Khd, Int32(pos), Int32(NKVH), grid_dim=(NKVH, 1), block_dim=32)
        var q_r = dn_f32(ctx, q_d, ATT)
        var k_r = dn_f32(ctx, k_d, KV)
        var q_rc = q_n.copy()
        for h in range(NQH):
            rope_yarn(q_rc.unsafe_ptr().unsafe_offset(h * HD), pos, NROT, FREQ_BASE, FREQ_SCALE, MSCALE, YARN_LOW, YARN_HIGH)
        diff_f32("rope_q", st, q_rc, q_r, bad)
        var k_rc = k_n.copy()
        for h in range(NKVH):
            rope_yarn(k_rc.unsafe_ptr().unsafe_offset(h * HD), pos, NROT, FREQ_BASE, FREQ_SCALE, MSCALE, YARN_LOW, YARN_HIGH)
        diff_f32("rope_k", st, k_rc, k_r, bad)
        ctx.enqueue_function[amar_kv_append[type_of(cache_l), type_of(kvm_l), 1]](Kc, Khd, tab.unsafe_ptr(), Int32(pos), Int32(0), grid_dim=(NKVH, 1), block_dim=HD)
        ctx.enqueue_function[amar_kv_append[type_of(cache_l), type_of(kvm_l), 1]](Vc, Vhd, tab.unsafe_ptr(), Int32(pos), Int32(0), grid_dim=(NKVH, 1), block_dim=HD)
        ctx.enqueue_function[amar_attn_decode[type_of(qm_l), type_of(cache_l), type_of(qm_l), 1]](Q, Kc, Vc, Ao, tab.unsafe_ptr(), Int32(pos + 1), Float32(0.0625), Int32(0), grid_dim=(NQH, 1), block_dim=HD)
        var kc_h = dn_f32(ctx, kc, CACHE)
        var vc_h = dn_f32(ctx, vc, CACHE)
        var ao_g = dn_f32(ctx, ao_d, ATT)
        var ao_c = List[Float32](length=ATT, fill=0)
        for h in range(NQH):
            var kvb = (h // (NQH // NKVH)) * KVPAGE * HD
            attn_head(q_r.unsafe_ptr().unsafe_offset(h * HD), kc_h.unsafe_ptr().unsafe_offset(kvb), vc_h.unsafe_ptr().unsafe_offset(kvb), pos + 1, HD, KVPAGE, Float32(0.0625), ao_c.unsafe_ptr().unsafe_offset(h * HD))
        diff_f32("attn", st, ao_c, ao_g, bad)
        ctx.enqueue_function[amar_gate_mul_cast[type_of(attf_l), type_of(attf_l), type_of(attf_l)]](Aof, Gate, Aob, Int32(ATT), grid_dim=ceildiv(ATT, 256), block_dim=256)
        var gate_g = dn_f32(ctx, gate_d, ATT)
        var aob_g = dn_bf16(ctx, aob, ATT)
        var aob_c = List[BFloat16](length=ATT, fill=0)
        for i in range(ATT):
            aob_c[i] = (ao_g[i] * (Float32(1) / (1 + exp(-gate_g[i])))).cast[bf16]()
        diff_bf16("gate_mul", st, aob_c, aob_g, bad)
        gemv[H, ATT](ctx, aob, wo, p_h)
        ctx.enqueue_function[amar_skinny_reduce_add[type_of(p_h_l), type_of(xm_l), 1]](Ph, X, Int32(1), Int32(H), grid_dim=ceildiv(H, 256), block_dim=256)
        var x1_g = dn_f32(ctx, x_d, H)
        var o_c = cpu_gemv(aob_g, wo, H, ATT)
        var x1_c = List[Float32](length=H, fill=0)
        for i in range(H):
            x1_c[i] = x0[i] + (Float32(0) + o_c[i])
        diff_f32("o_add", st, x1_c, x1_g, bad)
        ctx.enqueue_function[amar_rmsnorm_cast[type_of(xm_l), type_of(h_l), type_of(xm_l)]](X, Gpost, CurB, Int32(H), Float32(1e-6), grid_dim=1, block_dim=256)
        var cur2 = dn_bf16(ctx, curb, H)
        diff_bf16("rms2", st, rms(x1_g, g_post), cur2, bad)
        gemv[FFN, H](ctx, curb, wg, p_g)
        gemv[FFN, H](ctx, curb, wu, p_u)
        ctx.enqueue_function[amar_skinny_reduce_swiglu_bf16[type_of(p_ffn_l), type_of(ffnm_l), 1]](Pg, Pu, Fgb, Int32(1), Int32(FFN), grid_dim=ceildiv(FFN, 256), block_dim=256)
        var fgb_g = dn_bf16(ctx, fgb, FFN)
        var gc = cpu_gemv(cur2, wg, FFN, H)
        var uc = cpu_gemv(cur2, wu, FFN, H)
        var fgb_c = List[BFloat16](length=FFN, fill=0)
        for i in range(FFN):
            fgb_c[i] = swiglu(gc[i], uc[i])
        diff_bf16("gateup_swiglu", st, fgb_c, fgb_g, bad)
        gemv[H, FFN](ctx, fgb, wd, p_h)
        ctx.enqueue_function[amar_skinny_reduce_add[type_of(p_h_l), type_of(xm_l), 1]](Ph, X, Int32(1), Int32(H), grid_dim=ceildiv(H, 256), block_dim=256)
        var x2_g = dn_f32(ctx, x_d, H)
        var d_c = cpu_gemv(fgb_g, wd, H, FFN)
        var x2_c = List[Float32](length=H, fill=0)
        for i in range(H):
            x2_c[i] = x1_g[i] + (Float32(0) + d_c[i])
        diff_f32("down_add", st, x2_c, x2_g, bad)
        for xr in [x0.copy(), x1_g.copy(), x2_g.copy()]:
            ctx.enqueue_copy(sub(p_h, 0, H), xr.unsafe_ptr())
            ctx.enqueue_function[amar_rmsnorm_cast[type_of(xm_l), type_of(h_l), type_of(xm_l), f32]](TileTensor(p_h, xm_l), Gpost, TileTensor(qf, xm_l), Int32(H), Float32(1e-6), grid_dim=1, block_dim=256)
            var rf = List[Float32](length=H, fill=0)
            rmsnorm_cast[f32](xr.unsafe_ptr(), g_post.unsafe_ptr(), rf.unsafe_ptr(), H, Float32(1e-6))
            diff_f32("rms_f32", st, rf, dn_f32(ctx, sub(qf, 0, H), H), bad)
        dg = digest(x2_g, digest(ao_g, digest(k_r, digest(q_r, dg))))
        print("step", st, "pos", pos, "ao", ao_g[0], ao_g[1], ao_g[300], "x", x2_g[0], x2_g[1])
    dg = ssm_layer(ctx, dir, idx, toks, bad, dg)
    if getenv("TEST_FFN_X", "") != "":
        ffn_row(ctx, dir, idx, getenv("TEST_FFN_X", ""), Int(getenv("TEST_FFN_LAYER", "0")), bad)
    print("gpu-digest", dg)
    var total = 0
    for name in ["rms1", "gemv_q", "gemv_k", "gemv_v", "hrms_q", "hrms_k", "rope_q", "rope_k", "attn", "gate_mul", "o_add", "rms2", "gateup_swiglu", "down_add", "s_rms1", "gemv_qkv", "gemv_z", "gemv_ab", "conv", "conv_state", "gates", "delta_o", "delta_s", "gated", "s_out_add", "argmax", "pack", "rms_f32"]:
        print("stage", name, "mismatches", bad.get(name, 0))
        total += bad.get(name, 0)
    if total != 0:
        raise Error("MISMATCH " + String(total))
    print("PASS test_cpu_layer: cpu == gpu bitwise on 28 stages x " + String(NSTEP) + " steps")


def sub(d: DeviceBuffer[f32], off: Int, n: Int) raises -> DeviceBuffer[f32]:
    return d.create_sub_buffer[f32](off, n)


def ssm_layer(ctx: DeviceContext, dir: String, idx: String, toks: List[Int], mut bad: Dict[String, Int], dg0: UInt64) raises -> UInt64:
    comptime INNER = NH_V * SSTATE
    comptime SS = INNER * SSTATE
    var wqkv = Q4W(ctx, dir, idx, "blk.0.attn_qkv.weight")
    var wz = Q4W(ctx, dir, idx, "blk.0.attn_gate.weight")
    var wa = Q4W(ctx, dir, idx, "blk.0.ssm_alpha.weight")
    var wb = Q4W(ctx, dir, idx, "blk.0.ssm_beta.weight")
    var wout = Q4W(ctx, dir, idx, "blk.0.ssm_out.weight")
    var g_n = f32_at(dir, idx, "blk.0.attn_norm.weight", H)
    var cw = f32_at(dir, idx, "blk.0.ssm_conv1d.weight", CONV * 4)
    var sa = f32_at(dir, idx, "blk.0.ssm_a", NH_V)
    var db = f32_at(dir, idx, "blk.0.ssm_dt.bias", NH_V)
    var nw = f32_at(dir, idx, "blk.0.ssm_norm.weight", SSTATE)
    var d_gn = up_f32(ctx, g_n)
    var d_cw = up_f32(ctx, cw)
    var d_sa = up_f32(ctx, sa)
    var d_db = up_f32(ctx, db)
    var d_nw = up_f32(ctx, nw)
    var x_d = ctx.enqueue_create_buffer[f32](H)
    var curb = ctx.enqueue_create_buffer[bf16](H)
    var p_c = ctx.enqueue_create_buffer[f32](CONV)
    var p_i = ctx.enqueue_create_buffer[f32](INNER)
    var p_h = ctx.enqueue_create_buffer[f32](H)
    var pa = ctx.enqueue_create_buffer[f32](SPLITK * NH_V)
    var pb = ctx.enqueue_create_buffer[f32](SPLITK * NH_V)
    var qkv = ctx.enqueue_create_buffer[f32](CONV)
    var z = ctx.enqueue_create_buffer[f32](INNER)
    var cst = ctx.enqueue_create_buffer[f32](2 * 3 * CONV)
    var sst = ctx.enqueue_create_buffer[f32](2 * SS)
    var co = ctx.enqueue_create_buffer[f32](CONV)
    var eg = ctx.enqueue_create_buffer[f32](NH_V)
    var be = ctx.enqueue_create_buffer[f32](NH_V)
    var so = ctx.enqueue_create_buffer[f32](INNER)
    var resb = ctx.enqueue_create_buffer[bf16](INNER)
    for b in [pa, pb, cst, sst]:
        ctx.enqueue_memset(b, 0)
    comptime cm_l = row_major[1, CONV]()
    comptime im_l = row_major[1, INNER]()
    comptime g_l = row_major[1, NH_V]()
    comptime v_l = row_major[NH_V]()
    comptime o_l = row_major[1, NH_V, SSTATE]()
    comptime pab_l = row_major[SPLITK, 1, NH_V]()
    comptime cs_l = row_major[2, 1, 3, CONV]()
    comptime ss_l = row_major[2, 1, NH_V, SSTATE, SSTATE]()
    var X = TileTensor(x_d, xm_l)
    var Qkv = TileTensor(qkv, cm_l)
    var Z = TileTensor(z, im_l)
    var Co = TileTensor(co, cm_l)
    var Eg = TileTensor(eg, g_l)
    var Be = TileTensor(be, g_l)
    var So = TileTensor(so, o_l)
    var dg = dg0
    for st in range(NSTEP):
        var rs = st % 2
        var eb = read_at(dir + "/pack.bin", toks[st] * H * 2, H * 2)
        var x0 = List[Float32](length=H, fill=0)
        for i in range(H):
            x0[i] = eb.unsafe_ptr().unsafe_bitcast[BFloat16]().unsafe_offset(i).unsafe_load().cast[f32]()
        ctx.enqueue_copy(x_d, x0.unsafe_ptr())
        ctx.enqueue_function[amar_rmsnorm_cast[type_of(xm_l), type_of(h_l), type_of(xm_l)]](X, TileTensor(d_gn, h_l), TileTensor(curb, xm_l), Int32(H), Float32(1e-6), grid_dim=1, block_dim=256)
        var cur = dn_bf16(ctx, curb, H)
        diff_bf16("s_rms1", st, rms(x0, g_n), cur, bad)
        gemv[CONV, H](ctx, curb, wqkv, p_c)
        ctx.enqueue_function[amar_skinny_reduce[type_of(row_major[1, 1, CONV]()), type_of(cm_l), 1]](TileTensor(p_c, row_major[1, 1, CONV]()), Qkv, Int32(1), Int32(CONV), grid_dim=ceildiv(CONV, 256), block_dim=256)
        gemv[INNER, H](ctx, curb, wz, p_i)
        ctx.enqueue_function[amar_skinny_reduce[type_of(row_major[1, 1, INNER]()), type_of(im_l), 1]](TileTensor(p_i, row_major[1, 1, INNER]()), Z, Int32(1), Int32(INNER), grid_dim=ceildiv(INNER, 256), block_dim=256)
        gemv[NH_V, H](ctx, curb, wa, pa)
        gemv[NH_V, H](ctx, curb, wb, pb)
        var qkv_g = dn_f32(ctx, qkv, CONV)
        var z_g = dn_f32(ctx, z, INNER)
        var ab_g = dn_f32(ctx, sub(pa, 0, NH_V), NH_V) + dn_f32(ctx, sub(pb, 0, NH_V), NH_V)
        diff_f32("gemv_qkv", st, cpu_gemv(cur, wqkv, CONV, H), qkv_g, bad)
        diff_f32("gemv_z", st, cpu_gemv(cur, wz, INNER, H), z_g, bad)
        diff_f32("gemv_ab", st, cpu_gemv(cur, wa, NH_V, H) + cpu_gemv(cur, wb, NH_V, H), ab_g, bad)
        var cs_c = dn_f32(ctx, sub(cst, rs * 3 * CONV, 3 * CONV), 3 * CONV)
        var ss_c = dn_f32(ctx, sub(sst, rs * SS, SS), SS)
        ctx.enqueue_function[amar_ssm_conv_l2g[type_of(cm_l), type_of(cs_l), type_of(row_major[CONV, 4]()), type_of(cm_l), type_of(pab_l), type_of(g_l), type_of(v_l)]](
            Qkv, TileTensor(cst, cs_l), TileTensor(d_cw, row_major[CONV, 4]()), Co, TileTensor(pa, pab_l), TileTensor(pb, pab_l), Eg, Be,
            TileTensor(d_sa, v_l), TileTensor(d_db, v_l), Int32(st), Int32(0), Int32(2), Int32(1), grid_dim=max(CONV // 256, NH_V), block_dim=256)
        var co_g = dn_f32(ctx, co, CONV)
        var co_c = List[Float32](length=CONV, fill=0)
        ssm_conv(qkv_g.unsafe_ptr(), cs_c.unsafe_ptr(), cw.unsafe_ptr(), co_c.unsafe_ptr(), CONV, NH_K, SSTATE, SSM_EPS)
        diff_f32("conv", st, co_c, co_g, bad)
        diff_f32("conv_state", st, cs_c, dn_f32(ctx, sub(cst, (1 - rs) * 3 * CONV, 3 * CONV), 3 * CONV), bad)
        var gt_g = dn_f32(ctx, eg, NH_V) + dn_f32(ctx, be, NH_V)
        var gt_c = List[Float32](length=2 * NH_V, fill=0)
        for h in range(NH_V):
            var gb = ssm_gates(ab_g[h], ab_g[NH_V + h], sa[h], db[h])
            gt_c[h] = gb[0]
            gt_c[NH_V + h] = gb[1]
        diff_f32("gates", st, gt_c, gt_g, bad)
        ctx.enqueue_function[amar_ssm_delta_step[1, type_of(ss_l), type_of(cm_l), type_of(g_l), type_of(o_l)]](
            TileTensor(sst, ss_l), Co, Eg, Be, So, Int32(st), Int32(0), Int32(2), grid_dim=NH_V, block_dim=SSTATE)
        var so_g = dn_f32(ctx, so, INNER)
        var so_c = List[Float32](length=INNER, fill=0)
        ssm_delta(ss_c.unsafe_ptr(), co_g.unsafe_ptr(), gt_g.unsafe_ptr(), so_c.unsafe_ptr(), NH_V, NH_K, SSTATE)
        diff_f32("delta_o", st, so_c, so_g, bad)
        diff_f32("delta_s", st, ss_c, dn_f32(ctx, sub(sst, (1 - rs) * SS, SS), SS), bad)
        ctx.enqueue_function[amar_ssm_gated_out_bf16[type_of(o_l), type_of(im_l), type_of(row_major[SSTATE]()), type_of(im_l)]](
            So, Z, TileTensor(d_nw, row_major[SSTATE]()), TileTensor(resb, im_l), Int32(1), grid_dim=NH_V, block_dim=SSTATE)
        var rb_g = dn_bf16(ctx, resb, INNER)
        var so_n = so_g.copy()
        var rb_c = List[BFloat16](length=INNER, fill=0)
        ssm_gated(so_n.unsafe_ptr(), z_g.unsafe_ptr(), nw.unsafe_ptr(), rb_c.unsafe_ptr(), INNER, SSTATE, SSM_EPS)
        diff_bf16("gated", st, rb_c, rb_g, bad)
        gemv[H, INNER](ctx, resb, wout, p_h)
        ctx.enqueue_function[amar_skinny_reduce_add[type_of(p_h_l), type_of(xm_l), 1]](TileTensor(p_h, p_h_l), X, Int32(1), Int32(H), grid_dim=ceildiv(H, 256), block_dim=256)
        var x1_g = dn_f32(ctx, x_d, H)
        var o_c = cpu_gemv(rb_g, wout, H, INNER)
        for i in range(H):
            o_c[i] = x0[i] + (Float32(0) + o_c[i])
        diff_f32("s_out_add", st, o_c, x1_g, bad)
        dg = digest(x1_g, digest(so_g, digest(co_g, dg)))
        print("ssm step", st, "co", co_g[0], co_g[5000], "so", so_g[0], "x", x1_g[0])
    comptime NR = 3
    var lg = List[Float32](length=NR * VOCAB, fill=0)
    for i in range(NR * VOCAB):
        lg[i] = Float32((i * 2654435761) % 1000003) / 1000 if i < 2 * VOCAB else Float32(0)
    lg[123456] = 2000
    lg[7] = 2000
    lg[2 * VOCAB - 1] = 2000
    lg[VOCAB + 200000] = 2000
    var lg_d = up_f32(ctx, lg)
    var tk = ctx.enqueue_create_buffer[DType.int32](NR)
    ctx.enqueue_function[amar_argmax_pos[type_of(row_major[NR, VOCAB]()), type_of(row_major[NR]())]](TileTensor(lg_d, row_major[NR, VOCAB]()), TileTensor(tk, row_major[NR]()), Int32(VOCAB), Int32(0), grid_dim=NR, block_dim=256)
    var tk_g = List[Int32](length=NR, fill=0)
    ctx.enqueue_copy(tk_g.unsafe_ptr(), tk)
    ctx.synchronize()
    var c = 0
    for r in range(NR):
        var row = List[Float32](length=VOCAB, fill=0)
        for i in range(VOCAB):
            row[i] = lg[r * VOCAB + i]
        var t = sample_row_ref(row, 0, 0, 1, 0, 0, 0, 0)[0]
        print("argmax row", r, "cpu", t, "gpu", tk_g[r])
        c += Int(t != Int(tk_g[r]))
    bad["argmax"] = c
    return dg


def pack_check(ctx: DeviceContext, dir: String, idx: String, mut bad: Dict[String, Int]) raises:
    var cp = Pack(dir)
    var gp = load_pack(ctx, dir)
    var names = List[String]()
    for l in idx.split("\n"):
        if len(l.split(" ")) >= 4:
            names.append(String(l.split(" ")[0]))
    var c = 0
    for name in ["token_embd.weight", "blk.0.ssm_a", "blk.3.attn_q.weight", "output_norm.weight", "output.weight"]:
        var i = names.index(name)
        var n = cp.t[name].bytes()
        var g = List[UInt8](length=n, fill=0)
        ctx.enqueue_copy(g.unsafe_ptr(), gp.wbuf.create_sub_buffer[DType.uint8](gp.off[i], n))
        ctx.synchronize()
        var p = cp.ptr[DType.uint8](name)
        var h = UInt64(14695981039346656037)
        var m = 0
        for j in range(n):
            h = (h ^ UInt64(g[j])) * 1099511628211
            m += Int(g[j] != p[unsafe_offset=j])
        print("pack", name, "bytes", n, "fnv", h, "mismatching bytes", m)
        c += m
    bad["pack"] = c


def ffn_row(ctx: DeviceContext, dir: String, idx: String, xpath: String, l: Int, mut bad: Dict[String, Int]) raises:
    """Debug stage: layer l's FFN ops on one real residual row (H f32 at xpath), each on the GPU's own input."""
    var p = "blk." + String(l) + "."
    var wg = Q4W(ctx, dir, idx, p + "ffn_gate.weight")
    var wu = Q4W(ctx, dir, idx, p + "ffn_up.weight")
    var wd = Q4W(ctx, dir, idx, p + "ffn_down.weight")
    var gn = f32_at(dir, idx, p + "post_attention_norm.weight", H)
    var xb = read_at(xpath, 0, H * 4)
    var x = List[Float32](length=H, fill=0)
    for i in range(H):
        x[i] = xb.unsafe_ptr().unsafe_bitcast[Float32]().unsafe_offset(i).unsafe_load()
    var x_d = up_f32(ctx, x)
    var curb = ctx.enqueue_create_buffer[bf16](H)
    var p_g = ctx.enqueue_create_buffer[f32](FFN)
    var p_u = ctx.enqueue_create_buffer[f32](FFN)
    var fgb = ctx.enqueue_create_buffer[bf16](FFN)
    var p_h = ctx.enqueue_create_buffer[f32](H)
    ctx.enqueue_function[amar_rmsnorm_cast[type_of(xm_l), type_of(h_l), type_of(xm_l)]](TileTensor(x_d, xm_l), TileTensor(up_f32(ctx, gn), h_l), TileTensor(curb, xm_l), Int32(H), Float32(1e-6), grid_dim=1, block_dim=256)
    var cur = dn_bf16(ctx, curb, H)
    diff_bf16("ffn_rms", 0, rms(x, gn), cur, bad)
    gemv[FFN, H](ctx, curb, wg, p_g)
    gemv[FFN, H](ctx, curb, wu, p_u)
    var g_g = dn_f32(ctx, p_g, FFN)
    var u_g = dn_f32(ctx, p_u, FFN)
    diff_f32("ffn_gate", 0, cpu_gemv(cur, wg, FFN, H), g_g, bad)
    diff_f32("ffn_up", 0, cpu_gemv(cur, wu, FFN, H), u_g, bad)
    ctx.enqueue_function[amar_skinny_reduce_swiglu_bf16[type_of(p_ffn_l), type_of(ffnm_l), 1]](TileTensor(p_g, p_ffn_l), TileTensor(p_u, p_ffn_l), TileTensor(fgb, ffnm_l), Int32(1), Int32(FFN), grid_dim=ceildiv(FFN, 256), block_dim=256)
    var f_g = dn_bf16(ctx, fgb, FFN)
    var f_c = List[BFloat16](length=FFN, fill=0)
    for i in range(FFN):
        f_c[i] = swiglu(g_g[i], u_g[i])
    diff_bf16("ffn_swiglu", 0, f_c, f_g, bad)
    gemv[H, FFN](ctx, fgb, wd, p_h)
    var dn_g = dn_f32(ctx, p_h, H)
    diff_f32("ffn_down", 0, cpu_gemv(f_g, wd, H, FFN), dn_g, bad)
    for name in ["ffn_rms", "ffn_gate", "ffn_up", "ffn_swiglu", "ffn_down"]:
        print("probe", name, "mismatches", bad.get(name, 0))
