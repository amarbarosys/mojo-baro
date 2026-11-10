# ncm.mojo MODEL.gguf [N B DATADIR]: NCM sudoku (HazyResearch next-concept model) generated in Mojo on gfx1100, greedy, f32.
# MODEL.gguf from tools/ncm_pack.py (arch ncm, Linear weights stored [in, out]). With DATADIR (puzzles/solutions.i32 from
# ncm_pack) it runs N boards and writes DATADIR/codes.i32, tokens.i32; without, it checks itself: the 64 puzzles and the
# oracle's tokens stored in the gguf (ncm.run.*) must match, exit 1 otherwise.
# Op list: ~/Projects/research/ncm/docs/mojo-port-spec.md. The prefix never sees target rows, so its K/V per layer are
# computed once per batch and every level runs only its own rows. Writes WDIR/codes.i32 [N, sum l_k] and
# WDIR/tokens.i32 [N, 89]; prints solved boards and boards/s.
from std.collections import Dict
from std.math import ceildiv, erf, exp, sqrt
from std.os import getenv, stat
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu import block_idx, thread_idx, lane_id, WARP_SIZE
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from max.gpu.memory import AddressSpace
from max.gpu.primitives import warp
from max.gpu.sync import barrier
from layout import row_major, stack_allocation
from gguf_reader import GGUFModel

comptime f32 = DType.float32
comptime i32 = DType.int32
comptime FP = MutPointer[Scalar[f32], MutAnyOrigin]
comptime IP = MutPointer[Scalar[i32], MutAnyOrigin]
comptime T = 16  # GEMM k step and thread grid side
comptime TM = 64  # GEMM output tile side
comptime HD = 64  # head dim (generator and decoder)
comptime P = 91  # prefix tokens
comptime L = 89  # full length
comptime D = 256  # VQ width
comptime NL = 12  # levels
comptime EW = 256  # elementwise threads


# C[m, n] = act(A[m, :] . B[:, n] + bias[n]) (+ R[m, n]); B is [K, N]; R may be C (in-place residual).
# 64x64 tile per block, 16x16 threads, 4x4 outputs per thread (rows ty + 16 i, cols tx + 16 j); every output still
# sums k in order, so results equal a plain dot loop.
# CONV: Phi, C += R * 0.5 + (A conv B + bias) * 0.5 with A = U [rows * L, D] read as shifted rows (k = j * D + ci,
# zero past the board edge), B = Conv1d weight [co, ci, 3], R = U, n = D, k = 3 * D.
def gemm[BIAS: Bool, GELU: Bool, RES: Bool, CONV: Bool = False](A: FP, B: FP, bias: FP, R: FP, C: FP, m: Int32, n: Int32, k: Int32):
    var M = Int(m)
    var N = Int(n)
    var K = Int(k)
    var tx = Int(thread_idx.x)
    var ty = Int(thread_idx.y)
    var t = ty * T + tx
    var r0 = Int(block_idx.y) * TM
    var c0 = Int(block_idx.x) * TM
    var sa = stack_allocation[f32, address_space=AddressSpace.SHARED](row_major[T, TM]())  # [k, row]
    var sb = stack_allocation[f32, address_space=AddressSpace.SHARED](row_major[T, TM]())  # [k, col]
    var acc = SIMD[f32, 16](0)
    var k0 = 0
    while k0 < K:
        comptime for e in range(4):
            var x = t + e * T * T
            var ar = x // T
            var ak = x % T
            var bk = x // TM
            var bc = x % TM
            comptime if CONV:
                var g = r0 + ar
                var p = g % L + (k0 + ak) // D - 1
                sa[ak, ar] = A[unsafe_offset=(g - 1) * D + k0 + ak] if g < M and p >= 0 and p < L else 0
                var kb = k0 + bk
                sb[bk, bc] = B[unsafe_offset=((c0 + bc) * D + kb % D) * 3 + kb // D] if c0 + bc < N else 0
            else:
                sa[ak, ar] = A[unsafe_offset=(r0 + ar) * K + k0 + ak] if r0 + ar < M and k0 + ak < K else 0
                sb[bk, bc] = B[unsafe_offset=(k0 + bk) * N + c0 + bc] if c0 + bc < N and k0 + bk < K else 0
        barrier()
        for kk in range(T):
            var a = SIMD[f32, 4](0)
            var b = SIMD[f32, 4](0)
            comptime for i in range(4):
                a[i] = rebind[Float32](sa[kk, ty + T * i])
                b[i] = rebind[Float32](sb[kk, tx + T * i])
            comptime for i in range(4):
                comptime for j in range(4):
                    acc[i * 4 + j] += a[i] * b[j]
        barrier()
        k0 += T
    comptime for i in range(4):
        comptime for j in range(4):
            var row = r0 + ty + T * i
            var col = c0 + tx + T * j
            if row < M and col < N:
                var v = acc[i * 4 + j]
                comptime if BIAS:
                    v += bias[unsafe_offset=col]
                comptime if GELU:
                    v = 0.5 * v * (1 + erf(v * 0.7071067811865476))
                comptime if RES:
                    v += R[unsafe_offset=row * N + col]
                comptime if CONV:
                    C[unsafe_offset=row * N + col] += R[unsafe_offset=row * N + col] * 0.5 + v * 0.5
                else:
                    C[unsafe_offset=row * N + col] = v


# LayerNorm, eps 1e-6, one wave per row (N <= 512)
def layernorm[BIAS: Bool](X: FP, g: FP, b: FP, Y: FP, n: Int32):
    var N = Int(n)
    var r = Int(block_idx.x) * N
    var l = Int(lane_id())
    var s: Float32 = 0
    for i in range(l, N, WARP_SIZE):
        s += X[unsafe_offset=r + i]
    var mean = warp.sum(s) / Float32(N)
    var v: Float32 = 0
    for i in range(l, N, WARP_SIZE):
        var d = X[unsafe_offset=r + i] - mean
        v += d * d
    var inv = 1 / sqrt(warp.sum(v) / Float32(N) + 1e-6)
    for i in range(l, N, WARP_SIZE):
        var y = (X[unsafe_offset=r + i] - mean) * inv * g[unsafe_offset=i]
        comptime if BIAS:
            y += b[unsafe_offset=i]
        Y[unsafe_offset=r + i] = y


# per (row, head, q|k): RMSNorm over the head (eps 1e-6) then RoPE at position row % S, in place in QKV [M, 3C]
def qknorm_rope[ROPE: Bool](QKV: FP, qn: FP, kn: FP, cs: FP, sn: FP, m: Int32, s: Int32, c: Int32, h: Int32):
    var t = Int(block_idx.x) * EW + Int(thread_idx.x)
    var H = Int(h)
    if t >= Int(m) * H * 2:
        return
    var row = t // (H * 2)
    var head = (t // 2) % H
    var isk = t % 2
    var base = row * 3 * Int(c) + isk * Int(c) + head * HD
    var w = kn if isk == 1 else qn
    var ss: Float32 = 0
    for i in range(HD):
        var x = QKV[unsafe_offset=base + i]
        ss += x * x
    var inv = 1 / sqrt(ss / HD + 1e-6)
    var p = (row % Int(s)) * (HD // 2)
    for i in range(HD // 2):
        var x1 = QKV[unsafe_offset=base + i] * inv * w[unsafe_offset=i]
        var x2 = QKV[unsafe_offset=base + i + HD // 2] * inv * w[unsafe_offset=i + HD // 2]
        comptime if ROPE:
            var co = cs[unsafe_offset=p + i]
            var si = sn[unsafe_offset=p + i]
            QKV[unsafe_offset=base + i] = x1 * co - x2 * si
            QKV[unsafe_offset=base + i + HD // 2] = x2 * co + x1 * si
        else:
            QKV[unsafe_offset=base + i] = x1
            QKV[unsafe_offset=base + i + HD // 2] = x2


# one wave per (board, row, head): softmax(q k^T / 8) v over the prefix keys (n1 rows of PQ) then the own keys
# (S rows of QKV, up to the query row when CAUSAL); online softmax, lane holds dims lane and lane + 32
def attn[CAUSAL: Bool](QKV: FP, PQ: FP, O: FP, n1: Int32, s: Int32, c: Int32, h: Int32):
    var H = Int(h)
    var C = Int(c)
    var S = Int(s)
    var g = Int(block_idx.x)
    var head = g % H
    var row = g // H  # b * S + query row
    var b = row // S
    var l = Int(lane_id())
    var qb = row * 3 * C + head * HD + l
    var q0 = QKV[unsafe_offset=qb]
    var q1 = QKV[unsafe_offset=qb + 32]
    var mx: Float32 = -1e30
    var den: Float32 = 0
    var a0: Float32 = 0
    var a1: Float32 = 0
    var n_own = (row % S) + 1 if CAUSAL else S
    for j in range(Int(n1) + n_own):
        var kb = ((b * Int(n1) + j) * 3 * C if j < Int(n1) else (b * S + j - Int(n1)) * 3 * C) + C + head * HD + l
        var src = PQ if j < Int(n1) else QKV
        var sc = warp.sum(q0 * src[unsafe_offset=kb] + q1 * src[unsafe_offset=kb + 32]) * 0.125
        var nm = max(mx, sc)
        var corr = exp(mx - nm)
        var e = exp(sc - nm)
        den = den * corr + e
        a0 = a0 * corr + e * src[unsafe_offset=kb + C]
        a1 = a1 * corr + e * src[unsafe_offset=kb + C + 32]
        mx = nm
    O[unsafe_offset=row * C + head * HD + l] = a0 / den
    O[unsafe_offset=row * C + head * HD + l + 32] = a1 / den


def gather(ids: IP, tab: FP, dst: FP, rows: Int32, w: Int32):
    var t = Int(block_idx.x) * EW + Int(thread_idx.x)
    if t < Int(rows) * Int(w):
        dst[unsafe_offset=t] = tab[unsafe_offset=Int(ids[unsafe_offset=t // Int(w)]) * Int(w) + t % Int(w)]


def argmax(X: FP, dst: IP, rows: Int32, n: Int32):
    var r = Int(block_idx.x) * EW + Int(thread_idx.x)
    if r >= Int(rows):
        return
    var best = 0
    for i in range(1, Int(n)):  # first maximum wins, as torch.argmax
        if X[unsafe_offset=r * Int(n) + i] > X[unsafe_offset=r * Int(n) + best]:
            best = i
    dst[unsafe_offset=r] = Int32(best)


# level k -> full length: level 0 is a ConvTranspose1d (l0 = 1, kernel L, W [ci, co, L]); else linear interp
def upsample[CONVT: Bool](E: FP, W: FP, bias: FP, U: FP, rows: Int32, lk: Int32):
    var t = Int(block_idx.x) * EW + Int(thread_idx.x)
    if t >= Int(rows) * L * D:
        return
    var co = t % D
    var pos = (t // D) % L
    var b = t // (D * L)
    var lv = Int(lk)
    comptime if CONVT:
        var acc = bias[unsafe_offset=co]
        for ci in range(D):
            acc += E[unsafe_offset=b * D + ci] * W[unsafe_offset=(ci * D + co) * L + pos]
        U[unsafe_offset=t] = acc
    else:
        var src = max((Float32(pos) + 0.5) * (Float32(lv) / Float32(L)) - 0.5, 0)  # align_corners=False
        var i0 = Int(src)
        var i1 = min(i0 + 1, lv - 1)
        var wt = src - Float32(i0)
        U[unsafe_offset=t] = E[unsafe_offset=(b * lv + i0) * D + co] * (1 - wt) + E[unsafe_offset=(b * lv + i1) * D + co] * wt


# adaptive average pool, full length L -> lk (torch "area")
def pool(Z: FP, X: FP, rows: Int32, lk: Int32):
    var t = Int(block_idx.x) * EW + Int(thread_idx.x)
    var lv = Int(lk)
    if t >= Int(rows) * lv * D:
        return
    var co = t % D
    var i = (t // D) % lv
    var b = t // (D * lv)
    var s0 = (i * L) // lv
    var s1 = ((i + 1) * L + lv - 1) // lv
    var acc: Float32 = 0
    for p in range(s0, s1):
        acc += Z[unsafe_offset=(b * L + p) * D + co]
    X[unsafe_offset=t] = acc / Float32(s1 - s0)


def fp(b: DeviceBuffer[f32], at: Int = 0) -> FP:
    return FP(unsafe_from_address=Int(b.unsafe_ptr()) + at * 4)


def ip(b: DeviceBuffer[i32], at: Int = 0) -> IP:
    return IP(unsafe_from_address=Int(b.unsafe_ptr()) + at * 4)


def save[dt: DType](path: String, h: HostBuffer[dt], n: Int) raises:
    var bytes = List[UInt8](length=n * 4, fill=0)
    var src = h.unsafe_ptr().bitcast[UInt8]()
    for i in range(n * 4):
        bytes[i] = src[i]
    with open(path, "w") as f:
        f.write_bytes(Span(bytes))


struct Net:
    var ctx: DeviceContext
    var w: DeviceBuffer[f32]
    var off: Dict[String, Int]

    def __init__(out self, var ctx: DeviceContext, path: String) raises:
        self.ctx = ctx^
        self.off = Dict[String, Int]()  # tensor -> offset in floats from the start of the file
        var g = GGUFModel(path)
        if g.arch != "ncm":
            raise Error("not an ncm gguf: " + g.arch)
        for name in g.tensor_names:
            self.off[name] = (g.data_start + g.tensor_off[name]) // 4
        var n = stat(path).st_size // 4
        var raw = open(path, "r").read_bytes(n * 4)
        var hb = self.ctx.enqueue_create_host_buffer[f32](n)
        var src = raw.unsafe_ptr().bitcast[Float32]()
        for i in range(n):
            hb[i] = src[i]
        self.w = self.ctx.enqueue_create_buffer[f32](n)
        self.ctx.enqueue_copy(dst_buf=self.w, src_buf=hb)
        self.ctx.synchronize()

    def p(self, name: String) raises -> FP:
        return fp(self.w, self.off[name])

    def mm[BIAS: Bool, GELU: Bool, RES: Bool](self, A: FP, Wn: String, bn: String, R: FP, C: FP, m: Int, n: Int, k: Int) raises:
        var bias = self.p(bn) if BIAS else A
        self.ctx.enqueue_function[gemm[BIAS, GELU, RES]](A, self.p(Wn), bias, R, C, Int32(m), Int32(n), Int32(k),
            grid_dim=(ceildiv(n, TM), ceildiv(m, TM)), block_dim=(T, T))

    def ln[BIAS: Bool](self, X: FP, name: String, Y: FP, m: Int, n: Int) raises:
        var b = self.p(name + ".bias") if BIAS else X
        self.ctx.enqueue_function[layernorm[BIAS]](X, self.p(name + ".weight"), b, Y, Int32(n), grid_dim=m, block_dim=WARP_SIZE)

    # one pre-LN transformer block on rows H [m, c]; prefix keys from PQ (n1 rows per board); qkv kept in QKV
    def block[ROPE: Bool, CAUSAL: Bool](self, pre: String, H: FP, N_: FP, QKV: FP, PQ: FP, A: FP, M_: FP,
                                        cs: FP, sn: FP, m: Int, s: Int, n1: Int, c: Int, h: Int, ff: Int) raises:
        self.ln[True](H, pre + ".ln_1", N_, m, c)
        self.mm[True, False, False](N_, pre + ".attn.c_attn.weight", pre + ".attn.c_attn.bias", N_, QKV, m, 3 * c, c)
        self.ctx.enqueue_function[qknorm_rope[ROPE]](QKV, self.p(pre + ".attn.q_norm.weight"), self.p(pre + ".attn.k_norm.weight"),
            cs, sn, Int32(m), Int32(s), Int32(c), Int32(h), grid_dim=ceildiv(m * h * 2, EW), block_dim=EW)
        self.ctx.enqueue_function[attn[CAUSAL]](QKV, PQ, A, Int32(n1), Int32(s), Int32(c), Int32(h), grid_dim=m * h, block_dim=WARP_SIZE)
        self.mm[True, False, True](A, pre + ".attn.c_proj.weight", pre + ".attn.c_proj.bias", H, H, m, c, c)
        self.ln[True](H, pre + ".ln_2", N_, m, c)
        self.mm[True, True, False](N_, pre + ".mlp.c_fc.weight", pre + ".mlp.c_fc.bias", N_, M_, m, ff, c)
        self.mm[True, False, True](M_, pre + ".mlp.c_proj.weight", pre + ".mlp.c_proj.bias", H, H, m, c, ff)


def main() raises:
    var a = argv()
    # Two legal shapes: MODEL (self-check) or MODEL NB B DIR (batch run). Anything between is a forgotten argument that used
    # to turn a batch run into a self-check ignoring both numbers; no argument at all indexed a[1] out of range.
    if len(a) != 2 and len(a) != 5:
        raise Error("FAIL usage: ncm MODEL [NB B DIR]")
    var model = String(a[1])
    var selfcheck = len(a) < 5
    var NB = 64 if selfcheck else Int(String(a[2]))
    var B = 64 if selfcheck else Int(String(a[3]))
    var dir = String(a[4]) if not selfcheck else String("")
    comptime LV: List[Int] = [1, 2, 4, 6, 9, 13, 18, 26, 37, 50, 67, 89]
    var lv = materialize[LV]()
    var LSUM = 0
    for x in lv:
        LSUM += x
    var net = Net(DeviceContext(), model)
    ref ctx = net.ctx
    var C = 512
    var R = P  # max rows per board in one pass
    # host tables: RoPE cos/sin (base 10000) for positions 0..P-1; per-level input bias = wte.bias + wle[k]
    var hcs = ctx.enqueue_create_host_buffer[f32](P * HD // 2)
    var hsn = ctx.enqueue_create_host_buffer[f32](P * HD // 2)
    for p in range(P):
        for i in range(HD // 2):
            var ang = Float64(p) / (10000.0 ** (Float64(i) / Float64(HD // 2)))
            hcs[p * HD // 2 + i] = Float32(std.math.cos(ang))
            hsn[p * HD // 2 + i] = Float32(std.math.sin(ang))
    var cs = ctx.enqueue_create_buffer[f32](P * HD // 2)
    var sn = ctx.enqueue_create_buffer[f32](P * HD // 2)
    ctx.enqueue_copy(dst_buf=cs, src_buf=hcs)
    ctx.enqueue_copy(dst_buf=sn, src_buf=hsn)
    var wb = ctx.enqueue_create_host_buffer[f32](C)
    var hbias = ctx.enqueue_create_host_buffer[f32](NL * C)
    ctx.enqueue_copy(dst_buf=wb, src_buf=net.w.create_sub_buffer[f32](net.off["g.transformer.wte.bias"], C))
    var hwle = ctx.enqueue_create_host_buffer[f32](NL * C)
    ctx.enqueue_copy(dst_buf=hwle, src_buf=net.w.create_sub_buffer[f32](net.off["g.transformer.wle.weight"], NL * C))
    ctx.synchronize()
    for k in range(NL):
        for i in range(C):
            hbias[k * C + i] = wb[i] + hwle[k * C + i]
    var lbias = ctx.enqueue_create_buffer[f32](NL * C)
    ctx.enqueue_copy(dst_buf=lbias, src_buf=hbias)
    # boards
    var raw: List[UInt8]
    var sol: List[UInt8]
    if selfcheck:  # boards and reference tokens from the gguf's own i32 tensors
        var all = open(model, "r").read_bytes(stat(model).st_size)
        raw = List[UInt8](length=NB * P * 4, fill=0)
        sol = List[UInt8](length=NB * L * 4, fill=0)
        for i in range(NB * P * 4):
            raw[i] = all[net.off["ncm.run.puzzles"] * 4 + i]
        for i in range(NB * L * 4):
            sol[i] = all[net.off["ncm.run.ref.tokens"] * 4 + i]
    else:
        raw = open(dir + "/puzzles.i32", "r").read_bytes(NB * P * 4)
        sol = open(dir + "/solutions.i32", "r").read_bytes(NB * L * 4)
    var hids = ctx.enqueue_create_host_buffer[i32](NB * P)
    var src = raw.unsafe_ptr().bitcast[Int32]()
    for i in range(NB * P):
        hids[i] = src[i]
    # arena (one batch of B boards), allocated once
    var ids = ctx.enqueue_create_buffer[i32](NB * P)
    ctx.enqueue_copy(dst_buf=ids, src_buf=hids)
    var zero_ids = ctx.enqueue_create_buffer[i32](B * L)
    zero_ids.enqueue_fill(0)
    var x256 = ctx.enqueue_create_buffer[f32](B * R * D)
    var hbuf = ctx.enqueue_create_buffer[f32](B * R * C)
    var nbuf = ctx.enqueue_create_buffer[f32](B * R * C)
    var qkv = ctx.enqueue_create_buffer[f32](B * R * 3 * C)
    var abuf = ctx.enqueue_create_buffer[f32](B * R * C)
    var mbuf = ctx.enqueue_create_buffer[f32](B * R * 4 * C)
    var pq = ctx.enqueue_create_buffer[f32](8 * B * P * 3 * C)  # prefix QKV per layer
    var z = ctx.enqueue_create_buffer[f32](B * L * D)
    var u = ctx.enqueue_create_buffer[f32](B * L * D)
    var e = ctx.enqueue_create_buffer[f32](B * L * D)
    var logits = ctx.enqueue_create_buffer[f32](B * L * 16)
    var codes = ctx.enqueue_create_buffer[i32](NB * LSUM)
    var toks = ctx.enqueue_create_buffer[i32](NB * L)
    var arena = (B * R * (D + 3 * C + 3 * C + 4 * C) + 8 * B * P * 3 * C + 3 * B * L * D + B * L * 16) * 4
    print("arena", arena // (1 << 20), "MB, weights", len(net.w) * 4 // (1 << 20), "MB, batch", B)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    var b0 = 0
    while b0 < NB:
        var nb = min(B, NB - b0)
        var mp = nb * P
        # prefix: token embedding (VQ-VAE encoder table) -> wte, 8 blocks, keep each layer's QKV
        ctx.enqueue_function[gather](ip(ids, b0 * P), net.p("t.encoder.wte.weight"), fp(x256),
            Int32(mp), Int32(D), grid_dim=ceildiv(mp * D, EW), block_dim=EW)
        net.mm[True, False, False](fp(x256), "g.transformer.wte.weight", "g.transformer.wte.bias", fp(x256),
            fp(hbuf), mp, C, D)
        for li in range(8):
            var pqi = fp(pq, li * B * P * 3 * C)
            net.block[True, False]("g.transformer.blocks." + String(li), fp(hbuf), fp(nbuf), pqi, pqi,
                fp(abuf), fp(mbuf), fp(cs), fp(sn), mp, P, 0, C, 8, 4 * C)
        z.enqueue_fill(0)
        var coff = 0
        for k in range(NL):
            var lk = lv[k]
            var m = nb * lk
            if k == 0:  # level-0 query (l0_queries, the same row for every board)
                ctx.enqueue_function[gather](ip(zero_ids), net.p("g.l0_queries"), fp(x256), Int32(m), Int32(D),
                    grid_dim=ceildiv(m * D, EW), block_dim=EW)
            ctx.enqueue_function[gemm[True, False, False]](fp(x256), net.p("g.transformer.wte.weight"),
                fp(lbias, k * C), fp(x256), fp(hbuf), Int32(m), Int32(C), Int32(D),
                grid_dim=(ceildiv(C, TM), ceildiv(m, TM)), block_dim=(T, T))
            for li in range(8):
                var pqi = fp(pq, li * B * P * 3 * C)
                net.block[True, False]("g.transformer.blocks." + String(li), fp(hbuf), fp(nbuf), fp(qkv),
                    pqi, fp(abuf), fp(mbuf), fp(cs), fp(sn), m, lk, P, C, 8, 4 * C)
            net.ln[True](fp(hbuf), "g.transformer.ln_f", fp(nbuf), m, C)
            net.mm[True, False, False](fp(nbuf), "g.lm_heads." + String(k) + ".weight", "g.lm_heads." + String(k) + ".bias",
                fp(nbuf), fp(logits), m, 16, C)
            var ck = ip(codes, b0 * LSUM + nb * coff)  # level-major inside the batch
            ctx.enqueue_function[argmax](fp(logits), ck, Int32(m), Int32(16), grid_dim=ceildiv(m, EW), block_dim=EW)
            coff += lk
            # VQ: codebook rows -> full length -> Phi -> z_hat; then the next level's input by area pooling
            ctx.enqueue_function[gather](ck, net.p("t.vq.embeddings." + String(k) + ".weight"), fp(e), Int32(m), Int32(D),
                grid_dim=ceildiv(m * D, EW), block_dim=EW)
            var up = fp(e) if k == NL - 1 else fp(u)
            if k == 0:
                ctx.enqueue_function[upsample[True]](fp(e), net.p("t.vq.learned_up_l1.0.weight"), net.p("t.vq.learned_up_l1.0.bias"),
                    up, Int32(nb), Int32(lk), grid_dim=ceildiv(nb * L * D, EW), block_dim=EW)
            elif k < NL - 1:
                ctx.enqueue_function[upsample[False]](fp(e), fp(e), fp(e), up, Int32(nb), Int32(lk),
                    grid_dim=ceildiv(nb * L * D, EW), block_dim=EW)
            ctx.enqueue_function[gemm[True, False, False, True]](up, net.p("t.vq.quant_resi." + String(k) + ".weight"),
                net.p("t.vq.quant_resi." + String(k) + ".bias"), up, fp(z), Int32(nb * L), Int32(D), Int32(3 * D),
                grid_dim=(ceildiv(D, TM), ceildiv(nb * L, TM)), block_dim=(T, T))
            if k < NL - 1:
                ctx.enqueue_function[pool](fp(z), fp(x256), Int32(nb), Int32(lv[k + 1]),
                    grid_dim=ceildiv(nb * lv[k + 1] * D, EW), block_dim=EW)
        # VQ-VAE decoder on z_hat: one bidirectional block (no RoPE, 4 heads; matches the oracle to 2e-4), ln_f (no bias), output, argmax
        var md = nb * L
        if b0 == 0 and getenv("NCM_DUMP") and not selfcheck:  # decoder input of the first batch, for the parity hunt
            var hz = ctx.enqueue_create_host_buffer[f32](md * D)
            ctx.enqueue_copy(dst_buf=hz, src_buf=z.create_sub_buffer[f32](0, md * D))
            ctx.synchronize()
            save(dir + "/z.f32", hz, md * D)
        net.block[False, False]("t.decoder.middle_block", fp(z), fp(nbuf), fp(qkv), fp(qkv),
            fp(abuf), fp(mbuf), fp(cs), fp(sn), md, L, 0, D, 4, 4 * D)
        net.ln[False](fp(z), "t.decoder.ln_f", fp(nbuf), md, D)
        net.mm[False, False, False](fp(nbuf), "t.decoder.output_layer.weight", "", fp(nbuf), fp(logits), md, 12, D)
        if b0 == 0 and getenv("NCM_DUMP") and not selfcheck:
            var hl = ctx.enqueue_create_host_buffer[f32](md * 12)
            ctx.enqueue_copy(dst_buf=hl, src_buf=logits.create_sub_buffer[f32](0, md * 12))
            ctx.synchronize()
            save(dir + "/declogits.f32", hl, md * 12)
        ctx.enqueue_function[argmax](fp(logits), ip(toks, b0 * L), Int32(md), Int32(12),
            grid_dim=ceildiv(md, EW), block_dim=EW)
        b0 += nb
    ctx.synchronize()
    var dt = Float64(perf_counter_ns() - t0) / 1e9
    var ht = ctx.enqueue_create_host_buffer[i32](NB * L)
    var hc = ctx.enqueue_create_host_buffer[i32](NB * LSUM)
    ctx.enqueue_copy(dst_buf=ht, src_buf=toks)
    ctx.enqueue_copy(dst_buf=hc, src_buf=codes)
    ctx.synchronize()
    var sp = sol.unsafe_ptr().bitcast[Int32]()
    var solved = 0
    for i in range(NB):
        var ok = True
        for j in range(L):
            ok = ok and ht[i * L + j] == sp[i * L + j]
        solved += 1 if ok else 0
    if not selfcheck:
        save(dir + "/tokens.i32", ht, NB * L)
        save(dir + "/codes.i32", hc, NB * LSUM)
    _ = (ids^, zero_ids^, x256^, hbuf^, nbuf^, qkv^, abuf^, mbuf^, pq^, z^, u^, e^, logits^, codes^, toks^, cs^, sn^, lbias^)
    if selfcheck:  # here sol holds the oracle's tokens, so "solved" counts boards identical to the oracle
        print("selfcheck", solved, "/", NB, "boards identical to the oracle tokens stored in the gguf")
        if solved != NB:
            raise Error("FAIL selfcheck")
        return
    print("solved", solved, "/", NB, " ", dt, "s ", Float64(NB) / dt, "boards/s")
