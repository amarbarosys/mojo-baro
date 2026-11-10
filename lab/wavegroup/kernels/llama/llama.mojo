# Llama forward over T rows (T <= TP) on mojo-baro's kernels. TP == 1 runs the decode GEMVs with fused epilogues;
# TP > 1 casts rows to f16, dequantises each weight into an f16 scratch with the GEMV's own unpack and runs the WMMA GEMM
# (build -D WMMA_TB=1 -D WMMA_C_F16=0). Attention is mojo-baro's GQA decode kernel, causal over rows (row r sees
# pos0 + r + 1 keys); its amar_attn_prefill assumes 256-wide heads. Buffers share one arena, lifetimes from the steps.
from std.math import ceildiv, fma
from std.sys import get_defined_int
from max.gpu import global_idx, block_idx, thread_idx, WARP_SIZE
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, TensorLayout, row_major, stack_allocation
from bexp import exp
from elementwise import amar_rmsnorm_cast, EW_THREADS
from gemv_kq import amar_gemv_kq, kq_lane16, kq_bytes
from gemv_split import wg_gemv_kq
from attn import amar_attn_decode_gqa, KVT
from moe import q6k_value, Q6K_BYTES
from q4x_kernels import UP
from matmul_wmma_pipe import amar_matmul_wmma_pipe
from rope import wg_rope_kv
from mega import wdot
from attn_split import wg_attn_split, wg_attn_merge, CH, NSMAX, PS
from weights import Weights
from plan import plan, from_steps
from model import H, FFN, VOCAB, HD, NQH, NKVH, N_LAYERS, QF, KV

comptime f32 = DType.float32
comptime f16 = DType.float16
comptime P32 = Pointer[Float32, MutAnyOrigin]
comptime ATT_SPLIT = get_defined_int["WG_ATT_SPLIT", 1]() == 1
comptime P16 = Pointer[Float16, MutAnyOrigin]

def tv[dt: DType, L: TensorLayout](ctx: DeviceContext, addr: Int, l: L) -> TileTensor[dt, L, MutAnyOrigin]:
    var b = DeviceBuffer[dt](ctx, Pointer[Scalar[dt], MutAnyOrigin](unsafe_from_address=addr), l.size(), owning=False)
    return rebind[TileTensor[dt, L, MutAnyOrigin]](TileTensor(b, l))

def embed_rows(w: UP, o: P32, toks: Pointer[Int32, MutAnyOrigin]):
    var i = Int(global_idx.x)
    var r = Int(block_idx.y)
    o[r * H + i] = q6k_value(w, Int(toks[r]) * (H // 256) * Q6K_BYTES, i)

def dequant[GT: Int](w: UP, o: P16, n32: Int32, k32: Int32):
    var t = Int(global_idx.x)
    var sbr = Int(k32) // 256
    if t < Int(n32) * sbr * 16:
        var sb = t // 16
        var r = kq_lane16[GT](w, sb * kq_bytes[GT](), t % 16)
        var v = fma(r[0].cast[f32](), SIMD[f32, 16](r[1]), SIMD[f32, 16](-r[2]))
        var at = (sb // sbr) * Int(k32) + (sb % sbr) * 256 + r[3]
        comptime for i in range(16):
            o[at + i] = v[i].cast[f16]()

def cast16(x: P32, o: P16, n: Int32):
    var i = Int(global_idx.x)
    if i < Int(n):
        o[i] = x[i].cast[f16]()

def epi[EPI: Int](t: P32, c: P32, o: P32, n: Int32):
    """The GEMV epilogues on GEMM output t: 1 adds into c, 3 writes silu(c) * t to o."""
    var i = Int(global_idx.x)
    if i < Int(n):
        comptime if EPI == 1:
            c[i] += t[i]
        else:
            o[i] = c[i] / (1 + exp(-c[i])) * t[i]

struct Pass[TP: Int](Movable):
    """The intermediates of one forward over up to TP rows."""
    var arena: DeviceBuffer[DType.uint8]
    var o: Dict[String, Int]

    def __init__(out self, ctx: DeviceContext) raises:
        comptime F = 4
        comptime MM = Self.TP > 1 # the GEMM path's f16 input, f16 weight scratch and f32 output
        var steps: List[List[String]] = [["x"], ["x", "xn"], ["xn", "qkv", "x16", "ws", "tmp"], ["xn", "qkv", "x16", "ws", "tmp"],
            ["xn", "qkv", "x16", "ws", "tmp"], ["qkv"], ["qkv", "att", "pg"], ["att", "x", "x16", "ws", "tmp"], ["x", "xn"],
            ["xn", "g", "x16", "ws", "tmp"], ["xn", "g", "act", "x16", "ws", "tmp"], ["act", "x", "x16", "ws", "tmp"], ["x", "xn"],
            ["xn", "logits"], ["logits", "am"]]
        var b = from_steps({"x": Self.TP * H * F, "xn": Self.TP * H * F, "qkv": Self.TP * (QF + 2 * KV) * F, "pg": NQH * NSMAX * PS * F, "am": AMB * 8,
            "att": Self.TP * QF * F, "g": Self.TP * FFN * F, "act": Self.TP * FFN * F, "logits": VOCAB * F, "x16": Self.TP * FFN * 2 if MM else 256,
            "ws": FFN * H * 2 if MM else 256, "tmp": Self.TP * FFN * F if MM else 256}, steps)
        self.arena = ctx.enqueue_create_buffer[DType.uint8](plan(b))
        self.o = Dict[String, Int]()
        for x in b:
            self.o[x.name] = Int(self.arena.unsafe_ptr()) + x.off
        self.o["q"] = self.o["qkv"] # q, k, v back to back: one GEMV fills all three at TP == 1
        self.o["k"] = self.o["qkv"] + Self.TP * QF * F
        self.o["v"] = self.o["k"] + Self.TP * KV * F

def gemm[M: Int, N: Int, K: Int](ctx: DeviceContext, a: Int, b: Int, c: Int) raises:
    comptime B128 = ceildiv(N, 128) * ceildiv(M, 128)
    comptime WM = 4 if B128 >= 64 else 2
    comptime WN = 4 if 64 <= B128 < 512 else 2
    comptime TN = 4 if B128 >= 512 else 2
    var A = tv[f16](ctx, a, row_major[M, K]())
    var Bm = tv[f16](ctx, b, row_major[N, K]())
    var C = tv[f32](ctx, c, row_major[M, N]())
    ctx.enqueue_function[amar_matmul_wmma_pipe[type_of(A).LayoutType, type_of(Bm).LayoutType, type_of(C).LayoutType, WM, WN, 2, TN]](
        A, Bm, C, Int32(M), Int32(N), Int32(K), grid_dim=(ceildiv(N, WN * TN * 16), ceildiv(M, WM * 2 * 16)), block_dim=WM * WN * 32)

def linear[EPI: Int, N: Int, K: Int, TP: Int](ctx: DeviceContext, w: Weights, p: Pass[TP], name: String, T: Int, x: String, c: String, o: String) raises:
    """Rows of x times W^T, then epilogue EPI: 0 stores to c, 1 adds into c, 3 writes silu(c) * result to o."""
    var r = w.ref(name)
    if r.dims[0] != K or r.dims[1] != N or (r.gtype != 12 and r.gtype != 14):
        raise Error("linear: " + name + " is not " + String(N) + "x" + String(K) + " q4_K/q6_K")
    var wp = UP(unsafe_from_address=w.addr(name))
    comptime if TP == 1:
        var X = tv[f32](ctx, p.o[x], row_major[1, K]())
        var C = tv[f32](ctx, p.o[c], row_major[N]())
        var O = tv[f32](ctx, p.o[o], row_major[N]())
        comptime XL = type_of(X).LayoutType
        comptime CL = type_of(C).LayoutType
        if r.gtype == 12:
            ctx.enqueue_function[amar_gemv_kq[12, 12, EPI, XL, CL, CL, f32]](X, wp, C, O, Int32(N), Int32(K), Int32(N), Int32(0), grid_dim=(N + 7) // 8, block_dim=256)
        else:
            ctx.enqueue_function[amar_gemv_kq[14, 14, EPI, XL, CL, CL, f32]](X, wp, C, O, Int32(N), Int32(K), Int32(N), Int32(0), grid_dim=(N + 7) // 8, block_dim=256)
    else:
        ctx.enqueue_function[cast16](P32(unsafe_from_address=p.o[x]), P16(unsafe_from_address=p.o["x16"]), Int32(T * K), grid_dim=ceildiv(T * K, 256), block_dim=256)
        var ws = P16(unsafe_from_address=p.o["ws"])
        if r.gtype == 12:
            ctx.enqueue_function[dequant[12]](wp, ws, Int32(N), Int32(K), grid_dim=ceildiv(N * K // 16, 256), block_dim=256)
        else:
            ctx.enqueue_function[dequant[14]](wp, ws, Int32(N), Int32(K), grid_dim=ceildiv(N * K // 16, 256), block_dim=256)
        gemm[TP, N, K](ctx, p.o["x16"], p.o["ws"], p.o[c] if EPI == 0 else p.o["tmp"])
        comptime if EPI != 0:
            ctx.enqueue_function[epi[EPI]](P32(unsafe_from_address=p.o["tmp"]), P32(unsafe_from_address=p.o[c]), P32(unsafe_from_address=p.o[o]),
                Int32(T * N), grid_dim=ceildiv(T * N, 256), block_dim=256)

def qkv(ctx: DeviceContext, w: Weights, p: Pass[1], b: String) raises:
    """q, k, v in one split-K GEMV (S=4) over the three tensors; v may be q6_K."""
    var t = [w.ref(b + n + ".weight").gtype for n in ["attn_q", "attn_k", "attn_v"]]
    if t[0] != 12 or t[1] != 12 or (t[2] != 12 and t[2] != 14):
        raise Error("qkv: types " + String(t[0]) + " " + String(t[1]) + " " + String(t[2]))
    var X = tv[f32](ctx, p.o["xn"], row_major[1, H]())
    var C = tv[f32](ctx, p.o["qkv"], row_major[QF + 2 * KV]())
    var W = [UP(unsafe_from_address=w.addr(b + n + ".weight")) for n in ["attn_q", "attn_k", "attn_v"]]
    comptime XL = type_of(X).LayoutType
    comptime CL = type_of(C).LayoutType
    comptime G = ((QF + 2 * KV) * 4 + 7) // 8
    if t[2] == 12:
        ctx.enqueue_function[wg_gemv_kq[12, 12, 12, 4, 0, f32, XL, CL, CL]](X, W[0], W[1], W[2], C, C, Int32(QF), Int32(KV), Int32(QF + 2 * KV), Int32(H), grid_dim=G, block_dim=256)
    else:
        ctx.enqueue_function[wg_gemv_kq[12, 12, 14, 4, 0, f32, XL, CL, CL]](X, W[0], W[1], W[2], C, C, Int32(QF), Int32(KV), Int32(QF + 2 * KV), Int32(H), grid_dim=G, block_dim=256)

def gateup[GT: Int](x: P32, wg: Int64, wu: Int64, act: P32):
    """One wave per row: act = silu(gate . x) * (up . x), the two GEMVs and their epilogue in one launch."""
    var row = Int(block_idx.x) * 8 + Int(thread_idx.x) // WARP_SIZE
    var lane = Int(thread_idx.x) % WARP_SIZE
    var g = wdot[GT, H, False](x, x, 0, Int(wg), row, lane)
    var u = wdot[GT, H, False](x, x, 0, Int(wu), row, lane)
    if lane == 0:
        act[row] = g / (1 + exp(-g)) * u

comptime AMB = 128 # argmax stage-1 blocks

def argmax_part(x: P32, n: Int32, bv: P32, bi: Pointer[Int32, MutAnyOrigin]):
    """Stage 1, grid AMB x 256: best (value, lowest index) of a slice per block."""
    var tid = Int(thread_idx.x)
    var lo = Int(block_idx.x) * ((Int(n) + AMB - 1) // AMB)
    var hi = min(Int(n), lo + (Int(n) + AMB - 1) // AMB)
    var v = Float32(-3.4e38)
    var k = Int32(0)
    var i = lo + tid
    while i < hi:
        if x[i] > v:
            v = x[i]
            k = Int32(i)
        i += 256
    var sv = stack_allocation[f32, address_space=AddressSpace.SHARED](row_major[256]())
    var si = stack_allocation[DType.int32, address_space=AddressSpace.SHARED](row_major[256]())
    sv[tid] = rebind[sv.ElementType](v)
    si[tid] = rebind[si.ElementType](k)
    barrier()
    if tid == 0:
        for t in range(1, 256):
            var tv_ = rebind[Float32](sv[t])
            var ti = rebind[Int32](si[t])
            if tv_ > v or (tv_ == v and ti < k):
                v = tv_
                k = ti
        bv[Int(block_idx.x)] = v
        bi[Int(block_idx.x)] = k

def argmax_fin(bv: P32, bi: Pointer[Int32, MutAnyOrigin], dst: Pointer[Int32, MutAnyOrigin]):
    """Stage 2, one thread: the same rule over the AMB block results."""
    var v = bv[0]
    var k = bi[0]
    for b in range(1, AMB):
        if bv[b] > v or (bv[b] == v and bi[b] < k):
            v = bv[b]
            k = bi[b]
    dst[0] = k

def norm[TP: Int](ctx: DeviceContext, w: Weights, p: Pass[TP], name: String, T: Int, eps: Float32) raises:
    var X = tv[f32](ctx, p.o["x"], row_major[TP, H]())
    var O = tv[f32](ctx, p.o["xn"], row_major[TP, H]())
    var G = tv[f32](ctx, w.addr(name), row_major[H]())
    ctx.enqueue_function[amar_rmsnorm_cast[type_of(X).LayoutType, type_of(G).LayoutType, type_of(O).LayoutType, f32]](X, G, O, Int32(H), eps, grid_dim=T, block_dim=EW_THREADS)

def forward[TP: Int, CL: TensorLayout](ctx: DeviceContext, w: Weights, p: Pass[TP], toks: Pointer[Int32, MutAnyOrigin], pos0: Int, T: Int,
        Kc: TileTensor[KVT, CL, MutAnyOrigin], Vc: TileTensor[KVT, CL, MutAnyOrigin], tab: Pointer[Int32, MutAnyOrigin], eps: Float32) raises:
    """Rows pos0 .. pos0+T-1 (token ids at toks) through every layer; their K/V land in the cache, the residual in x."""
    var Q = tv[f32](ctx, p.o["q"], row_major[TP * NQH, HD]())
    var K = tv[f32](ctx, p.o["k"], row_major[TP * NKVH, HD]())
    var V = tv[f32](ctx, p.o["v"], row_major[TP * NKVH, HD]())
    var ATT = tv[f32](ctx, p.o["att"], row_major[TP * NQH, HD]())
    var FF = tv[f32](ctx, w.addr("rope_freqs.weight"), row_major[HD // 2]())
    comptime QL = type_of(Q).LayoutType
    comptime KL = type_of(K).LayoutType
    ctx.enqueue_function[embed_rows](UP(unsafe_from_address=w.addr("token_embd.weight")), P32(unsafe_from_address=p.o["x"]), toks, grid_dim=(H // 256, T), block_dim=256)
    for l in range(N_LAYERS):
        var b = "blk." + String(l) + "."
        norm(ctx, w, p, b + "attn_norm.weight", T, eps)
        comptime if TP == 1:
            qkv(ctx, w, rebind[Pass[1]](p), b)
        else:
            linear[0, QF, H](ctx, w, p, b + "attn_q.weight", T, "xn", "q", "q")
            linear[0, KV, H](ctx, w, p, b + "attn_k.weight", T, "xn", "k", "k")
            linear[0, KV, H](ctx, w, p, b + "attn_v.weight", T, "xn", "v", "v")
        ctx.enqueue_function[wg_rope_kv[N_LAYERS, QL, KL, type_of(FF).LayoutType]](Q, K, V, FF, Pointer[Scalar[KVT], MutAnyOrigin](unsafe_from_address=Int(Kc.ptr)),
            Pointer[Scalar[KVT], MutAnyOrigin](unsafe_from_address=Int(Vc.ptr)), tab, Int32(pos0), Int32(l), grid_dim=(NQH + NKVH, T), block_dim=HD // 2)
        comptime if TP == 1 and ATT_SPLIT: # split over the context; one block per head leaves the GPU idle at decode
            var ns = (pos0 + 1 + CH - 1) // CH
            var pg = P32(unsafe_from_address=p.o["pg"])
            ctx.enqueue_function[wg_attn_split[N_LAYERS, QL, CL]](Q, Kc, Vc, tab, pg, Int32(pos0 + 1), Float32(0.125), Int32(l), grid_dim=(NQH, ns), block_dim=HD)
            ctx.enqueue_function[wg_attn_merge[QL]](pg, ATT, Int32(ns), grid_dim=NQH, block_dim=HD)
        else:
            ctx.enqueue_function[amar_attn_decode_gqa[QL, CL, QL, N_LAYERS]](Q, Kc, Vc, ATT, tab, Int32(pos0 + 1), Float32(0.125), Int32(l), grid_dim=(NKVH, T), block_dim=HD)
        linear[1, H, QF](ctx, w, p, b + "attn_output.weight", T, "att", "x", "x")
        norm(ctx, w, p, b + "ffn_norm.weight", T, eps)
        comptime if TP == 1:
            var tg = w.ref(b + "ffn_gate.weight").gtype
            if tg != w.ref(b + "ffn_up.weight").gtype or (tg != 12 and tg != 14):
                raise Error("gateup: types differ or not q4_K/q6_K")
            if tg == 12:
                ctx.enqueue_function[gateup[12]](P32(unsafe_from_address=p.o["xn"]), Int64(w.addr(b + "ffn_gate.weight")), Int64(w.addr(b + "ffn_up.weight")), P32(unsafe_from_address=p.o["act"]), grid_dim=FFN // 8, block_dim=256)
            else:
                ctx.enqueue_function[gateup[14]](P32(unsafe_from_address=p.o["xn"]), Int64(w.addr(b + "ffn_gate.weight")), Int64(w.addr(b + "ffn_up.weight")), P32(unsafe_from_address=p.o["act"]), grid_dim=FFN // 8, block_dim=256)
        else:
            linear[0, FFN, H](ctx, w, p, b + "ffn_gate.weight", T, "xn", "g", "g")
            linear[3, FFN, H](ctx, w, p, b + "ffn_up.weight", T, "xn", "g", "act")
        linear[1, H, FFN](ctx, w, p, b + "ffn_down.weight", T, "act", "x", "x")
