from std.math import ceildiv, sqrt
from std.sys import get_defined_int, align_of
from std.utils import IndexList
from max.gpu.host import DeviceContext, DeviceBuffer
from ops import F, I, THREADS, grid, upload, gemm, rope_tables, swiglu_fwd_k, swiglu_bwd_k, add_k
from lops import *
from layout import TileTensor, row_major, Coord
from linalg.matmul import matmul as lmatmul
from nlook import Look, EB
from lora import Experts, EP, XRCH, xgate_k, xgbwd_k, xrfwd_k, xrbwd_k, xrdx_k, xrdw_k

comptime V = get_defined_int["LAT_V", 32]()  # 256: byte-level text
comptime D = DM
comptime H = get_defined_int["LAT_H", 4]()
comptime HD = D // H
comptime FF = get_defined_int["LAT_FF", D * 43 // 16]()  # 63M: -D LAT_FF=2816 (2752 pads to 22 tiles of 128 anyway)
comptime L = get_defined_int["LAT_L", 2]()
comptime NTH = 7
comptime F16 = (OPT & 2) != 0
comptime EPI = (OPT & 8) != 0
comptime FQKV = FQ and F16 and DD >= (1 << 20)  # fused only on the linalg dW path
comptime BT = (OPT & 256) != 0 and SW16 and DD >= (1 << 20)  # swiglu keeps bf16 only (in the fp32 buffers' bytes); dW transposes read bf16
comptime RT = (OPT & 2048) != 0 and BT and FQKV  # rms16 keeps n1/n2 as bf16 in their buffers' bytes; dW transposes read bf16
comptime DWG = (OPT & 4096) != 0 and RT  # dW by dwg_k straight from [tokens, features] operands: no transposes
comptime RDG = (OPT & 32768) != 0  # rms_dxg_k: dx and the gain-gradient partials in one pass
comptime WOS = (OPT & 65536) != 0 and DWG  # wo dW by split-K dwg_k
comptime DW4 = get_defined_int["LAT_DW4", 0]() == 1  # dwg_k 4-wave 64 x 64 warp tiles (A/B)
comptime R3 = (OPT & 131072) != 0 and DWG and RF  # rope3_k: inverse rope + interleaved bf16 q/k/v grads, read by dX and dwg_k
comptime AB16 = (OPT & 262144) != 0 and BT  # w1/w3 outputs and dff stored bf16 by the GEMM epilogue
comptime BF2 = OB16  # (needs WOS + DWG, both in every build that sets 1048576) bf16 o and dx for the wo / w2 dW launches (dwg_k bf16 x bf16)
comptime NOMS = (OPT & 16) != 0  # first backward segment writes dk/dv instead of memset + accumulate
comptime DD = D * D
comptime DF = D * FF
comptime LAYER = 2 * D + 4 * DD + 3 * DF
comptime GF = V * D + L * LAYER
comptime PAUSE = GF + D
comptime BH0 = PAUSE + NTH * D  # block heads: W_k [D, D] for k = 2 .. BHK at BH0 + (k - 2) DD
comptime NBH = BHK - 1 if BHK > 1 else 0
comptime NP = BH0 + NBH * DD
comptime EPS: Float32 = 1e-6
comptime TOK = 0
comptime LAT = 1
comptime PSE = 2
comptime GEN = 3
comptime SOFT = 4
comptime ENT = 5
comptime NEU = 6


@fieldwise_init
struct W(ImplicitlyCopyable):
    var g1: F
    var wq: F
    var wk: F
    var wv: F
    var wo: F
    var g2: F
    var w1: F
    var w3: F
    var w2: F


def layer(p: F, l: Int) -> W:
    var b = p.unsafe_offset(V * D + l * LAYER)
    return W(b, b.unsafe_offset(D), b.unsafe_offset(D + DD), b.unsafe_offset(D + 2 * DD), b.unsafe_offset(D + 3 * DD),
             b.unsafe_offset(D + 4 * DD), b.unsafe_offset(2 * D + 4 * DD), b.unsafe_offset(2 * D + 4 * DD + DF),
             b.unsafe_offset(2 * D + 4 * DD + 2 * DF))


def grow_dl(ctx: DeviceContext, src: F, n: Int) raises -> List[Float32]:
    """n floats from the device to a host list (grow=: the one-time host-side seeding)."""
    var h = List[Float32](length=n, fill=0)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=DeviceBuffer[DType.float32](ctx, src, n, owning=False))
    ctx.synchronize()
    return h^


def grow_ul(ctx: DeviceContext, dst: F, h: List[Float32]) raises:
    ctx.enqueue_copy(dst_buf=DeviceBuffer[DType.float32](ctx, dst, len(h), owning=False), src_ptr=h.unsafe_ptr())
    ctx.synchronize()


@fieldwise_init
struct Acts(ImplicitlyCopyable):
    var x: F
    var n1: F
    var r1: F
    var q: F
    var k: F
    var v: F
    var p: F
    var o: F
    var h: F
    var n2: F
    var r2: F
    var a: F
    var b: F
    var s: F
    var dk: F
    var dv: F


struct Model[exact: Bool = False](Movable):
    """Position-major batch: row r*b + i is position r of sequence i. Segments are row ranges [r0, r1)."""
    var b: Int
    var SR: Int  # SMAX * b rows
    var sg: Bool
    var itemp: Float32
    var mem: DeviceBuffer[DType.float32]
    var top: Int
    var g: Gemm
    var acts: List[Acts]
    var xl: F
    var nf: F
    var rf: F
    var z: F
    var dz: F
    var lrow: F
    var dnf: F
    var dx: F
    var dn: F
    var dt: F
    var dq: F
    var da: F
    var db: F
    var dff: F
    var part: F
    var w16: DeviceBuffer[DType.bfloat16]
    var w16src: Int  # address of the p that adamw16_k last wrote w16 from (0: stale); refresh skips the cast on a match
    var t16: DeviceBuffer[DType.bfloat16]
    var a16: DeviceBuffer[DType.bfloat16]
    var esort: Int  # > 0: rows sorted by token id are in eord / estart (set_sort), embed grad by embed_srt_k
    var eord: DeviceBuffer[DType.int32]
    var bhl: Float32  # block-head loss weight
    var ffm: Int  # Matryoshka: active FFN width (prefix of every w1 / w3 output and w2 input), FF = full
    var nl: Int  # dgrow: active layers (the first nl run, the rest are skipped: no kernels), L = all
    var bhd: Bool  # detached drafter: no head gradient into the trunk (dnf) or the shared embedding
    var dist: Float32  # mdist: > 0 makes the head loss (1 - dist) CE + dist KL(teacher || student) against zt (V > 256 only)
    var zt: DeviceBuffer[DType.bfloat16]  # mdist teacher: bf16 copy of the full-width update's logits z [R, V] (1 element until teacher_alloc)
    var b16x: DeviceBuffer[DType.bfloat16]  # block heads: gathered x | h | du, bf16 [3][NC, D]
    var bf32: DeviceBuffer[DType.float32]  # u | dh | dx [3][NC, D], weights [NC], losses [NBH][NC]
    var btk: DeviceBuffer[DType.int32]
    var estart: DeviceBuffer[DType.int32]
    var c16: DeviceBuffer[DType.bfloat16]  # V > 256: bf16 logit grads [R, V]
    var f16: DeviceBuffer[DType.bfloat16]  # FLASH: per layer roped bf16 q, k and bf16 v ([3][R, D]), then bf16 dO [R, D]
    var b16: DeviceBuffer[DType.bfloat16]  # dW: bf16 transpose of the gradient operand
    var i16: DeviceBuffer[DType.bfloat16]  # per layer [D, 3D]: wq|wk|wv rows side by side (fused q/k/v dX)
    var ctmp: F
    var sp: F
    var cs: DeviceBuffer[DType.float32]
    var qr: DeviceBuffer[DType.float32]  # QKN: per layer the q and k head RMS [2][R, H]
    var sn: DeviceBuffer[DType.float32]
    var lk: Look
    var xe: Experts  # low-rank experts (lora.mojo); xe.on False: the base path only, kernel for kernel
    var frz: Bool  # frozen base: no base weight gradient kernels (dX only), the experts train
    var t16p: Int  # frz: address of the p whose bf16 copy and transposes are current (refresh skips all of it)

    def __init__(out self, ctx: DeviceContext, b: Int) raises:
        comptime assert D % 256 == 0 and D % H == 0 and D // H <= 64 and V % 32 == 0
        self.b = b
        self.SR = SMAX * b
        self.lk = Look(ctx, b, 0, 0, 0, 0, 0)
        self.xe = Experts(ctx)
        self.frz = False
        self.t16p = 0
        self.sg = False
        self.w16src = 0
        self.itemp = 1
        self.bhl = 0.3
        self.ffm = FF
        self.nl = L
        self.bhd = False
        self.dist = 0
        self.zt = ctx.enqueue_create_buffer[DType.bfloat16](1)
        var R = SMAX * b
        comptime assert not FLASH or R3, "flash attention writes the R3 interleaved q/k/v-grad rows"
        var PS = b * H * (SMAX if FLASH else SMAX * SMAX)  # FLASH: per-row lse instead of P
        var per = 12 * R * D + 2 * R + 3 * R * FF + PS
        var total = L * per + (L + 1) * R * D + R + 2 * R * V + R + 7 * R * D + 3 * R * FF
        self.mem = ctx.enqueue_create_buffer[DType.float32](total)
        ctx.enqueue_memset(self.mem, 0)
        self.top = 0
        self.g = Gemm(ctx, 8 * 1024 * 1024)
        self.acts = List[Acts]()
        var base = F(unsafe_from_address=Int(self.mem.unsafe_ptr()))
        self.xl = base
        self.nf = base
        self.rf = base
        self.z = base
        self.dz = base
        self.lrow = base
        self.dnf = base
        self.dx = base
        self.dn = base
        self.dt = base
        self.dq = base
        self.da = base
        self.db = base
        self.dff = base
        self.part = base
        self.ctmp = base
        self.w16 = ctx.enqueue_create_buffer[DType.bfloat16](NP)
        self.t16 = ctx.enqueue_create_buffer[DType.bfloat16](NP)
        self.a16 = ctx.enqueue_create_buffer[DType.bfloat16](R * max(FF, 3 * D))
        self.esort = 0
        self.eord = ctx.enqueue_create_buffer[DType.int32](R)
        var nc = ceildiv(SMAX, BHS) * b if NBH > 0 else 1
        self.b16x = ctx.enqueue_create_buffer[DType.bfloat16](3 * nc * D)
        self.bf32 = ctx.enqueue_create_buffer[DType.float32](3 * nc * D + nc + max(NBH, 1) * nc)
        self.btk = ctx.enqueue_create_buffer[DType.int32](nc)
        self.estart = ctx.enqueue_create_buffer[DType.int32](V + 1)
        self.c16 = ctx.enqueue_create_buffer[DType.bfloat16](R * V if V > 256 else 1)
        self.f16 = ctx.enqueue_create_buffer[DType.bfloat16]((3 * L + 1) * R * D if FLASH else 1)
        self.b16 = ctx.enqueue_create_buffer[DType.bfloat16](R * max(FF, 3 * D))
        self.i16 = ctx.enqueue_create_buffer[DType.bfloat16](L * 3 * DD if FQKV else 1)
        self.sp = base
        self.qr = ctx.enqueue_create_buffer[DType.float32](L * 2 * R * H if QKN else 1)
        var tab = rope_tables(SMAX, HD, 10000)
        self.cs = ctx.enqueue_create_buffer[DType.float32](len(tab[0]))
        self.sn = ctx.enqueue_create_buffer[DType.float32](len(tab[1]))
        upload(ctx, self.cs, tab[0])
        upload(ctx, self.sn, tab[1])
        for _ in range(L):
            var x = self.take(R * D)
            self.acts.append(Acts(x, self.take(R * D), self.take(R), self.take(R * D), self.take(R * D), self.take(R * D),
                self.take(PS), self.take(R * D), self.take(R * D), self.take(R * D), self.take(R),
                self.take(R * FF), self.take(R * FF), self.take(R * FF), self.take(R * D), self.take(R * D)))
        self.xl = self.take(R * D)
        self.nf = self.take(R * D)
        self.rf = self.take(R)
        self.z = self.take(R * V)
        self.dz = self.take(R * V)
        self.lrow = self.take(R)
        self.dnf = self.take(R * D)
        self.dx = self.take(R * D)
        self.dn = self.take(R * D)
        self.dt = self.take(R * D)
        self.dq = self.take(R * D)
        self.da = self.take(R * FF)
        self.db = self.take(R * FF)
        self.dff = self.take(R * FF)
        self.part = self.take(max(ceildiv(R, RC) * D, ceildiv(R, EC) * V * D))
        self.sp = self.take(b * V)
        self.ctmp = self.take(R * D)

    def qkv16(self, l: Int) -> H16:
        return H16(unsafe_from_address=Int(self.f16.unsafe_ptr())).unsafe_offset(3 * l * self.SR * D)

    def set_sort(mut self, ctx: DeviceContext, ids: List[Int32], n: Int) raises:
        """Stable counting sort of the n row ids (one segment) for embed_srt_k; ids < 0 are skipped."""
        var cnt = List[Int32](length=V + 1, fill=0)
        for i in range(n):
            if ids[i] >= 0:
                cnt[Int(ids[i]) + 1] += 1
        for v in range(V):
            cnt[v + 1] += cnt[v]
        var pos = cnt.copy()
        var order = List[Int32](length=n, fill=0)
        for i in range(n):
            if ids[i] >= 0:
                order[Int(pos[Int(ids[i])])] = Int32(i)
                pos[Int(ids[i])] += 1
        upload(ctx, self.eord, order)
        upload(ctx, self.estart, cnt)
        self.esort = 1

    def teacher_alloc(mut self, ctx: DeviceContext) raises:
        """mdist: room for the teacher logits, [SMAX * b, V] bf16 (256 MB at S512 b32 V8192)."""
        self.zt = ctx.enqueue_create_buffer[DType.bfloat16](self.SR * V)

    def save_teacher(self, ctx: DeviceContext, n: Int) raises:
        """mdist: keep the logits z of the first n rows of the step just enqueued (bf16); the next step distils from them."""
        ctx.enqueue_function[cast16_k](self.z, H16(unsafe_from_address=Int(self.zt.unsafe_ptr())), Int32(n * V), grid_dim=grid(n * V), block_dim=THREADS)

    def grow_ffn(mut self, ctx: DeviceContext, p: F, mp: F, vp: F, f0: Int, f1: Int, e: Float32, rows: List[Int], tgt: List[Int32], nr: Int, plant: Int) raises:
        """grow=STEP:FROM:TO: ready the FFN neurons [f0, f1) of every layer for use. w2 rows [f0, f1) go to 0 in p and in both Adam
        moments (function-preserving). e > 0 also seeds them from the mistakes of the forward at width f0 that just ran on this
        batch (rows[k] = the k-th highest-CE row of [0, nr)): w1 and w3 column f0 + k = that row's FFN input n2 of the layer, scaled
        to the mean column norm of w1[:, :f0]; w2 row f0 + k = e * the tied-embedding row of that row's target; their Adam moments 0.
        plant >= 0 leaves w2 row `plant` of layer 0 at 0.5 (the gate's planted fault). The bf16 copy is marked stale."""
        var nn = f1 - f0
        var d0 = nn * D
        for l in range(L):
            var w = layer(p, l)
            ctx.enqueue_memset(DeviceBuffer[DType.float32](ctx, w.w2.unsafe_offset(f0 * D), d0, owning=False), 0)
            ctx.enqueue_memset(DeviceBuffer[DType.float32](ctx, layer(mp, l).w2.unsafe_offset(f0 * D), d0, owning=False), 0)
            ctx.enqueue_memset(DeviceBuffer[DType.float32](ctx, layer(vp, l).w2.unsafe_offset(f0 * D), d0, owning=False), 0)
        ctx.synchronize()
        self.w16src = 0
        if e > 0:
            comptime if not (RT and not Self.exact):
                raise Error("FAIL grow: gseed needs the bf16 RT build (n2 kept as bf16 in its buffer)")
            if len(rows) != nn or nr > self.SR:
                raise Error("FAIL grow: " + String(len(rows)) + " seed rows for " + String(nn) + " new neurons")
            var emb = grow_dl(ctx, p, V * D)
            var w2s = List[Float32](length=d0, fill=0)
            for k in range(nn):
                var t = Int(tgt[rows[k]])
                for d in range(D):
                    w2s[k * D + d] = e * emb[t * D + d]
            for l in range(self.nl):  # dgrow: only active layers ran this forward (inactive ones keep zero growth: w2 rows 0)
                var w = layer(p, l)
                var n2h = List[BFloat16](length=nr * D, fill=BFloat16(0))
                ctx.enqueue_copy(dst_ptr=n2h.unsafe_ptr(), src_buf=DeviceBuffer[DType.bfloat16](ctx, H16(unsafe_from_address=Int(self.acts[l].n2)), nr * D, owning=False))
                ctx.synchronize()
                var w1 = grow_dl(ctx, w.w1, DF)
                var w3 = grow_dl(ctx, w.w3, DF)
                var cs = List[Float64](length=f0, fill=0)
                for d in range(D):
                    for j in range(f0):
                        var x = Float64(w1[d * FF + j])
                        cs[j] += x * x
                var cn = Float64(0)
                for j in range(f0):
                    cn += sqrt(cs[j])
                cn /= Float64(f0)
                for k in range(nn):
                    var r = rows[k]
                    var s = Float64(0)
                    for d in range(D):
                        var x = Float64(n2h[r * D + d].cast[DType.float32]())
                        s += x * x
                    var sc = cn / sqrt(s) if s > 0 else Float64(0)
                    for d in range(D):
                        var v = Float32(Float64(n2h[r * D + d].cast[DType.float32]()) * sc)
                        w1[d * FF + f0 + k] = v
                        w3[d * FF + f0 + k] = v
                grow_ul(ctx, w.w1, w1)
                grow_ul(ctx, w.w3, w3)
                grow_ul(ctx, w.w2.unsafe_offset(f0 * D), w2s)
                for q in range(2):  # Adam moments m, v of the new w1 / w3 columns
                    var mm = layer(mp, l) if q == 0 else layer(vp, l)
                    var a1 = grow_dl(ctx, mm.w1, DF)
                    var a3 = grow_dl(ctx, mm.w3, DF)
                    for d in range(D):
                        for j in range(f0, f1):
                            a1[d * FF + j] = 0
                            a3[d * FF + j] = 0
                    grow_ul(ctx, mm.w1, a1)
                    grow_ul(ctx, mm.w3, a3)
            var emb2 = grow_dl(ctx, p, V * D)  # readback from the device: the seeded w2 rows are nonzero and equal e * E[target]
            var worst = Float64(0)
            var nz = 0
            for l in range(L):
                var back = grow_dl(ctx, layer(p, l).w2.unsafe_offset(f0 * D), d0)
                for k in range(nn):
                    var any = False
                    for d in range(D):
                        var want = e * emb2[Int(tgt[rows[k]]) * D + d]
                        var df = Float64(back[k * D + d] - want)
                        if df < 0:
                            df = -df
                        worst = max(worst, df)
                        if back[k * D + d] != 0:
                            any = True
                    if any and l == 0:
                        nz += 1
            print("GROW w2 check layers", L, "rows nonzero", nz, "of", nn, "maxerr", worst)
        if plant >= f0 and plant < f1:
            grow_ul(ctx, layer(p, 0).w2.unsafe_offset(plant * D), List[Float32](length=D, fill=0.5))
            print("GROW planted: w2 row", plant, "of layer 0 left at 0.5")
        self.w16src = 0

    def dgrow_zero(mut self, ctx: DeviceContext, p: F, mp: F, vp: F, l0: Int, plant: Int) raises:
        """dgrow=STEP:L0: make the layers [l0, L) the identity before they join: wo and w2 go to 0 in p and in both Adam moments
        (a zero residual branch; no gradient reached them while inactive). plant in [l0, L) leaves row 0 of that layer's wo at 0.5
        (the gate's planted fault). The bf16 copy is marked stale."""
        if l0 >= L:
            return
        for l in range(l0, L):
            for q in range(3):
                var w = layer(p if q == 0 else (mp if q == 1 else vp), l)
                ctx.enqueue_memset(DeviceBuffer[DType.float32](ctx, w.wo, DD, owning=False), 0)
                ctx.enqueue_memset(DeviceBuffer[DType.float32](ctx, w.w2, DF, owning=False), 0)
        ctx.synchronize()
        if plant >= l0 and plant < L:
            grow_ul(ctx, layer(p, plant).wo, List[Float32](length=D, fill=0.5))
            print("DGROW planted: row 0 of wo of layer", plant, "left at 0.5")
        self.w16src = 0

    def take(mut self, n: Int) -> F:
        var p = F(unsafe_from_address=Int(self.mem.unsafe_ptr())).unsafe_offset(self.top)
        self.top += n
        return p

    def mm[ta: Bool, tb: Bool](self, ctx: DeviceContext, p: F, a: F, b: F, c: F, m: Int, n: Int, k: Int, acc: Bool = False, cast: Bool = True, asrc: Int = 0) raises:
        """b is a parameter matrix when not ta: its bf16 copy (tb) or bf16 transpose (not tb) feeds linalg.
        cast=False: a16 already holds this call's A operand (same a as the previous call, or written by rms16)."""
        var fresh = cast or (OPT & 1) == 0
        comptime if Self.exact:
            gemm[ta, tb](ctx, a, b, c, m, n, k, acc)
        elif ta:
            # small dW (1.59M default: 256 x 688): split-K tgemm wins (10.0 vs 16.9 ms/step with linalg + transposes)
            if get_defined_int["LAT_TGEMM_DW", 0]() == 1 or m * n < (1 << 20):
                self.g[ta, tb](ctx, a, b, c, m, n, k, acc)
                return
            # dW = a^T b, a [k, m], b [k, n] (activations): both to bf16 [m, k] / [n, k], then linalg transpose_b.
            # The split-K tgemm was right at 1.59M (tiny m x n); at 63M it ran at ~15 TFLOPS, 45% of the step.
            comptime assert not tb, "dW path expects b untransposed"
            var a16p = H16(unsafe_from_address=Int(self.a16.unsafe_ptr()))
            var b16p = H16(unsafe_from_address=Int(self.b16.unsafe_ptr()))
            if fresh:
                ctx.enqueue_function[tr16t_k](a, a16p, Int32(k), Int32(m), grid_dim=(ceildiv(m, TT), ceildiv(k, TT)), block_dim=(TT, 8))
            ctx.enqueue_function[tr16t_k](b, b16p, Int32(k), Int32(n), grid_dim=(ceildiv(n, TT), ceildiv(k, TT)), block_dim=(TT, 8))
            self.lin(ctx, c, a16p, b16p, m, n, k, acc)
        else:
            var off = (Int(b) - Int(p)) // 4
            var a16p = H16(unsafe_from_address=Int(self.b16.unsafe_ptr() if asrc == 1 else self.a16.unsafe_ptr()))
            if fresh:
                ctx.enqueue_function[cast16_k](a, a16p, Int32(m * k), grid_dim=grid(m * k), block_dim=THREADS)
            var bsrc = self.w16 if tb else self.t16
            var b16p = H16(unsafe_from_address=Int(bsrc.unsafe_ptr())).unsafe_offset(off)
            self.lin(ctx, c, a16p, b16p, m, n, k, acc)

    def dwg[ta: DType, tb: DType](self, ctx: DeviceContext, a: MutPointer[Scalar[ta], MutUntrackedOrigin], b0: MutPointer[Scalar[tb], MutUntrackedOrigin],
            b1: MutPointer[Scalar[tb], MutUntrackedOrigin], b2: MutPointer[Scalar[tb], MutUntrackedOrigin], c0: F, c1: F, c2: F, m: Int, n: Int, k: Int, nz: Int, ks: Int = 1, ldb: Int = 0) raises:
        """c_z += a^T b_z for z < nz (a [k, m], b_z [k, n] row-major): dW with no transposes."""
        ctx.enqueue_function[dwg_k[ta, tb, DW4]](a, b0, b1, b2, c0, c1, c2, Int32(m), Int32(n), Int32(k), Int32(1), Int32(ks), Int32(ldb),
            grid_dim=(ceildiv(n, WB), ceildiv(m, WB), nz * ks), block_dim=128 if DW4 else 256)

    def wsrc(self, p: F, w: F, tb: Bool) -> H16:
        """bf16 copy (tb) or bf16 transpose of parameter matrix w, as mm's B operand."""
        var bsrc = self.w16 if tb else self.t16
        return H16(unsafe_from_address=Int(bsrc.unsafe_ptr())).unsafe_offset((Int(w) - Int(p)) // 4)

    def lin(self, ctx: DeviceContext, c: F, a16p: H16, b16p: H16, m: Int, n: Int, k: Int, acc: Bool) raises:
        """c (+)= A B^T, A [m, k], B [n, k] bf16. EPI: the add happens in the GEMM epilogue (no temp, no add_k)."""
        var abuf = DeviceBuffer[DType.bfloat16](ctx, a16p, m * k, owning=False)
        var bbuf = DeviceBuffer[DType.bfloat16](ctx, b16p, n * k, owning=False)
        var tA = TileTensor(abuf, row_major(m, k))
        var tB = TileTensor(bbuf, row_major(n, k))
        comptime if EPI:
            if acc:
                var cbuf = DeviceBuffer[DType.float32](ctx, c, m * n, owning=False)
                var tc = TileTensor(cbuf, row_major(m, n))
                var cp = c
                var nn = n

                @__parameter
                @always_inline
                @__copy_capture(cp, nn)
                def addc[_dtype: DType, width: SIMDLength, *, alignment: Int = 1](
                    idx: Coord, val: SIMD[_dtype, width]
                ) capturing -> SIMD[_dtype, width]:
                    return val + cp.unsafe_offset(Int(idx[0]) * nn + Int(idx[1])).load[width=width]().cast[_dtype]()

                lmatmul[transpose_b=True, elementwise_compute_lambda_fn=addc, target="gpu"](tc, tA, tB, ctx)
                return
        var out = self.ctmp if acc else c
        var cbuf = DeviceBuffer[DType.float32](ctx, out, m * n, owning=False)
        var tc = TileTensor(cbuf, row_major(m, n))
        lmatmul[transpose_b=True, target="gpu"](tc, tA, tB, ctx)
        if acc:
            self.add(ctx, self.ctmp, c, m * n)

    def lin16(self, ctx: DeviceContext, dst: H16, a16p: H16, b16p: H16, m: Int, n: Int, k: Int) raises:
        """dst [m, n] bf16 = A B^T (A [m, k], B [n, k] bf16): the epilogue stores bf16 (half the output bytes)."""
        var abuf = DeviceBuffer[DType.bfloat16](ctx, a16p, m * k, owning=False)
        var bbuf = DeviceBuffer[DType.bfloat16](ctx, b16p, n * k, owning=False)
        # C view: dst's own region (bf16 dst lives in the bytes of an fp32 buffer of >= m * n floats), so the
        # TileTensor's extent is real memory even though the store lambda replaces every write to it
        var cbuf = DeviceBuffer[DType.float32](ctx, F(unsafe_from_address=Int(dst)), m * n, owning=False)
        var tA = TileTensor(abuf, row_major(m, k))
        var tB = TileTensor(bbuf, row_major(n, k))
        var tc = TileTensor(cbuf, row_major(m, n))
        var dp = dst
        var nn = n

        @__parameter
        @always_inline
        @__copy_capture(dp, nn)
        def st[_dtype: DType, width: SIMDLength, *, alignment: Int = 1](idx: Coord, val: SIMD[_dtype, width]) capturing -> None:
            dp.unsafe_offset(Int(idx[0]) * nn + Int(idx[1])).store(val.cast[DType.bfloat16]())

        lmatmul[transpose_b=True, elementwise_lambda_fn=st, target="gpu"](tc, tA, tB, ctx)

    def lin16a(self, ctx: DeviceContext, dst: H16, a16p: H16, b16p: H16, m: Int, n: Int, k: Int) raises:
        """lin16 that adds: dst [m, n] bf16 += A B^T (an expert delta into a bf16 activation; B = 0 leaves dst's bytes)."""
        var abuf = DeviceBuffer[DType.bfloat16](ctx, a16p, m * k, owning=False)
        var bbuf = DeviceBuffer[DType.bfloat16](ctx, b16p, n * k, owning=False)
        var cbuf = DeviceBuffer[DType.float32](ctx, F(unsafe_from_address=Int(dst)), m * n, owning=False)
        var tA = TileTensor(abuf, row_major(m, k))
        var tB = TileTensor(bbuf, row_major(n, k))
        var tc = TileTensor(cbuf, row_major(m, n))
        var dp = dst
        var nn = n

        @__parameter
        @always_inline
        @__copy_capture(dp, nn)
        def st[_dtype: DType, width: SIMDLength, *, alignment: Int = 1](idx: Coord, val: SIMD[_dtype, width]) capturing -> None:
            var q = dp.unsafe_offset(Int(idx[0]) * nn + Int(idx[1]))
            q.store((q.load[width=width]().cast[DType.float32]() + val.cast[DType.float32]()).cast[DType.bfloat16]())

        lmatmul[transpose_b=True, elementwise_lambda_fn=st, target="gpu"](tc, tA, tB, ctx)

    def xf(self, ctx: DeviceContext, l: Int, t: Int, x: Int, y: Int, ybf: Bool, n: Int, ro: Int) raises:
        """Expert forward of target t in layer l on n rows from row ro: u = x A, u' = s g u, y (+)= u' B. x and y are the
        addresses of the base path's own input and output rows (bf16 where it keeps bf16, fp32 in exact; ybf: y is bf16)."""
        var er = self.xe.er()
        var ni = self.xe.tin(t)
        var no = self.xe.tout(t)
        var u = self.xe.uraw(l, t).unsafe_offset(ro * er)
        var uq = self.xe.ugat(l, t).unsafe_offset(ro * er)
        var gt = self.xe.gates(l).unsafe_offset(ro * EP)
        var rt = Int32(Int(self.xe.routed()))
        comptime if Self.exact:
            var xp = self.xe.p()
            gemm[False, True](ctx, F(unsafe_from_address=x), xp.unsafe_offset(self.xe.aofs(l, t)), u, n, er, ni)
            ctx.enqueue_function[xgate_k[DType.float32]](u, gt, uq, Int32(n), Int32(er), Int32(self.xe.r), Int32(self.xe.null), self.xe.s, rt,
                grid_dim=grid(n * er), block_dim=THREADS)
            gemm[False, False](ctx, uq, xp.unsafe_offset(self.xe.bofs(l, t)), F(unsafe_from_address=y), n, no, er, True)
        else:
            var w16 = H16(unsafe_from_address=Int(self.xe.x16.unsafe_ptr()))
            var t16 = H16(unsafe_from_address=Int(self.xe.xt16.unsafe_ptr()))
            var uh = H16(unsafe_from_address=Int(uq))
            self.lin(ctx, u, H16(unsafe_from_address=x), w16.unsafe_offset(self.xe.aofs(l, t)), n, er, ni, False)
            ctx.enqueue_function[xgate_k[bf]](u, gt, uh, Int32(n), Int32(er), Int32(self.xe.r), Int32(self.xe.null), self.xe.s, rt,
                grid_dim=grid(n * er), block_dim=THREADS)
            if ybf:
                self.lin16a(ctx, H16(unsafe_from_address=y), uh, t16.unsafe_offset(self.xe.bofs(l, t)), n, no, er)
            else:
                self.lin(ctx, F(unsafe_from_address=y), uh, t16.unsafe_offset(self.xe.bofs(l, t)), n, no, er, True)

    def xbw(self, ctx: DeviceContext, l: Int, t: Int, x: Int, dy: Int, dxi: Int, dxbf: Bool, n: Int, ro: Int, first: Bool) raises:
        """Expert backward of target t: t = dy B^T, t' = s g t, dB += u'^T dy, dAT += t'^T x, dx_in (+)= t' AT, and (routed) the
        gate gradient into dg (first: this target writes it, later ones add). dy / dxi as the base path keeps them (dxbf: bf16)."""
        var er = self.xe.er()
        var ni = self.xe.tin(t)
        var no = self.xe.tout(t)
        var u = self.xe.uraw(l, t).unsafe_offset(ro * er)
        var uq = self.xe.ugat(l, t).unsafe_offset(ro * er)
        var gt = self.xe.gates(l).unsafe_offset(ro * EP)
        var rt = Int32(Int(self.xe.routed()))
        var tt = self.xe.tbuf()
        var tq = self.xe.tqbuf()
        var xg = self.xe.gp()
        var gb = xg.unsafe_offset(self.xe.bofs(l, t))
        var ga = xg.unsafe_offset(self.xe.aofs(l, t))
        comptime if Self.exact:
            var xp = self.xe.p()
            gemm[False, True](ctx, F(unsafe_from_address=dy), xp.unsafe_offset(self.xe.bofs(l, t)), tt, n, er, no)
            ctx.enqueue_function[xgbwd_k[DType.float32]](tt, u, gt, tq, self.xe.dgbuf(), Int32(n), Int32(self.xe.E), Int32(self.xe.r), Int32(self.xe.null),
                self.xe.s, rt, Int32(Int(not first)), grid_dim=grid(n * self.xe.E), block_dim=THREADS)
            gemm[True, False](ctx, uq, F(unsafe_from_address=dy), gb, er, no, n, True)
            gemm[True, False](ctx, tq, F(unsafe_from_address=x), ga, er, ni, n, True)
            gemm[False, False](ctx, tq, xp.unsafe_offset(self.xe.aofs(l, t)), F(unsafe_from_address=dxi), n, ni, er, True)
        else:
            var w16 = H16(unsafe_from_address=Int(self.xe.x16.unsafe_ptr()))
            var t16 = H16(unsafe_from_address=Int(self.xe.xt16.unsafe_ptr()))
            var tqh = H16(unsafe_from_address=Int(tq))
            var uh = H16(unsafe_from_address=Int(uq))
            var dyh = H16(unsafe_from_address=dy)
            var xh = H16(unsafe_from_address=x)
            self.lin(ctx, tt, dyh, w16.unsafe_offset(self.xe.bofs(l, t)), n, er, no, False)
            ctx.enqueue_function[xgbwd_k[bf]](tt, u, gt, tqh, self.xe.dgbuf(), Int32(n), Int32(self.xe.E), Int32(self.xe.r), Int32(self.xe.null),
                self.xe.s, rt, Int32(Int(not first)), grid_dim=grid(n * self.xe.E), block_dim=THREADS)
            # M = E r is one 128-row block: split K in two (twice the workgroups), the second half into a zeroed scratch, then one add
            var sp = self.xe.wpart()
            ctx.enqueue_memset(DeviceBuffer[DType.float32](ctx, sp, er * no, owning=False), 0)
            self.dwg[bf, bf](ctx, uh, dyh, dyh, dyh, gb, sp, gb, er, no, n, 1, 2)
            self.add(ctx, sp, gb, er * no)
            ctx.enqueue_memset(DeviceBuffer[DType.float32](ctx, sp, er * ni, owning=False), 0)
            self.dwg[bf, bf](ctx, tqh, xh, xh, xh, ga, sp, ga, er, ni, n, 1, 2)
            self.add(ctx, sp, ga, er * ni)
            if dxbf:
                self.lin16a(ctx, H16(unsafe_from_address=dxi), tqh, t16.unsafe_offset(self.xe.aofs(l, t)), n, ni, er)
            else:
                self.lin(ctx, F(unsafe_from_address=dxi), tqh, t16.unsafe_offset(self.xe.aofs(l, t)), n, ni, er, True)

    def xrf(self, ctx: DeviceContext, l: Int, x: Int, n: Int, ro: Int) raises:
        """Router forward on the FFN input rows: the gates of layer l."""
        var gt = self.xe.gates(l).unsafe_offset(ro * EP)
        var rw = self.xe.p().unsafe_offset(self.xe.rofs(l))
        comptime if Self.exact:
            ctx.enqueue_function[xrfwd_k[DType.float32]](F(unsafe_from_address=x), rw, gt, Int32(n), Int32(self.xe.nopt), Int32(self.xe.topk),
                grid_dim=grid(n * 32), block_dim=THREADS)
        else:
            ctx.enqueue_function[xrfwd_k[bf]](H16(unsafe_from_address=x), rw, gt, Int32(n), Int32(self.xe.nopt), Int32(self.xe.topk),
                grid_dim=grid(n * 32), block_dim=THREADS)

    def xrb(self, ctx: DeviceContext, l: Int, x: Int, dxi: F, n: Int, ro: Int) raises:
        """Router backward: softmax backward of dg, dR += x^T dl (row-chunk partials, summed in order), dx_in += dl R^T."""
        var gt = self.xe.gates(l).unsafe_offset(ro * EP)
        var rw = self.xe.p().unsafe_offset(self.xe.rofs(l))
        var dl = self.xe.dlbuf()
        var pt = self.xe.rpart()
        ctx.enqueue_function[xrbwd_k](gt, self.xe.dgbuf(), dl, Int32(n), grid_dim=grid(n), block_dim=THREADS)
        var nch = ceildiv(n, XRCH)
        comptime if Self.exact:
            ctx.enqueue_function[xrdw_k[DType.float32]](F(unsafe_from_address=x), dl, pt, Int32(n), grid_dim=grid(nch * D), block_dim=THREADS)
        else:
            ctx.enqueue_function[xrdw_k[bf]](H16(unsafe_from_address=x), dl, pt, Int32(n), grid_dim=grid(nch * D), block_dim=THREADS)
        ctx.enqueue_function[colsum_k](pt, self.xe.gp().unsafe_offset(self.xe.rofs(l)), Int32(nch), Int32(D * EP), grid_dim=grid(D * EP), block_dim=THREADS)
        ctx.enqueue_function[xrdx_k](dl, rw, dxi, Int32(n), grid_dim=grid(n * D), block_dim=THREADS)

    def lin_nt(self, ctx: DeviceContext, c: F, a16p: H16, b16p: H16, m: Int, n: Int, k: Int, acc: Bool) raises:
        """c (+)= A B, A [m, k], B [k, n] bf16 row-major (B not transposed): prefix rows of a [FF, D] weight as B."""
        var abuf = DeviceBuffer[DType.bfloat16](ctx, a16p, m * k, owning=False)
        var bbuf = DeviceBuffer[DType.bfloat16](ctx, b16p, k * n, owning=False)
        var tA = TileTensor(abuf, row_major(m, k))
        var tB = TileTensor(bbuf, row_major(k, n))
        var cbuf = DeviceBuffer[DType.float32](ctx, c, m * n, owning=False)
        var tc = TileTensor(cbuf, row_major(m, n))
        if acc:
            var cp = c
            var nn = n

            @__parameter
            @always_inline
            @__copy_capture(cp, nn)
            def addc[_dtype: DType, width: SIMDLength, *, alignment: Int = 1](
                idx: Coord, val: SIMD[_dtype, width]
            ) capturing -> SIMD[_dtype, width]:
                return val + cp.unsafe_offset(Int(idx[0]) * nn + Int(idx[1])).load[width=width]().cast[_dtype]()

            lmatmul[transpose_b=False, elementwise_compute_lambda_fn=addc, target="gpu"](tc, tA, tB, ctx)
            return
        lmatmul[transpose_b=False, target="gpu"](tc, tA, tB, ctx)

    def lin3(self, ctx: DeviceContext, d0: F, d1: F, d2: F, a16p: H16, b16p: H16, m: Int, k: Int, acc: Bool) raises:
        """C [m, 3D] = A B^T (A [m, k], B [3D, k] bf16); the epilogue scatters column block j into dj [m, D] (+= if acc)."""
        var abuf = DeviceBuffer[DType.bfloat16](ctx, a16p, m * k, owning=False)
        var bbuf = DeviceBuffer[DType.bfloat16](ctx, b16p, 3 * D * k, owning=False)
        var cbuf = DeviceBuffer[DType.float32](ctx, d0, m * D, owning=False)
        var tA = TileTensor(abuf, row_major(m, k))
        var tB = TileTensor(bbuf, row_major(3 * D, k))
        var tc = TileTensor(cbuf, row_major(m, 3 * D))
        var p0 = d0
        var p1 = d1
        var p2 = d2
        var ac = acc

        @__parameter
        @always_inline
        @__copy_capture(p0, p1, p2, ac)
        def sc[_dtype: DType, width: SIMDLength, *, alignment: Int = 1](idx: Coord, val: SIMD[_dtype, width]) capturing -> None:
            var c = Int(idx[1])
            var j = c // D
            var dst = (p0 if j == 0 else (p1 if j == 1 else p2)).unsafe_offset(Int(idx[0]) * D + c - j * D)
            var v = val.cast[DType.float32]()
            if ac:
                v += dst.load[width=width]()
            dst.store(v)

        lmatmul[transpose_b=True, elementwise_lambda_fn=sc, target="gpu"](tc, tA, tB, ctx)

    def refresh(self, ctx: DeviceContext, p: F) raises:
        """bf16 copy of all params and bf16 transposes of every matrix, from the current p."""
        comptime if not Self.exact:
            if self.frz and self.t16p == Int(p) and self.w16src == Int(p):  # frozen base: the copies of the first step stay current
                return
            var w = H16(unsafe_from_address=Int(self.w16.unsafe_ptr()))
            var t = H16(unsafe_from_address=Int(self.t16.unsafe_ptr()))
            if self.w16src != Int(p):
                ctx.enqueue_function[cast16_k](p, w, Int32(NP), grid_dim=grid(NP), block_dim=THREADS)
            comptime if BF2:  # transpose the bf16 copy (half the read bytes, same values)
                ctx.enqueue_function[tr16h_k](w, t, Int32(V), Int32(D), grid_dim=(ceildiv(D, TT), ceildiv(V, TT)), block_dim=(TT, 8))
            else:
                ctx.enqueue_function[tr16t_k](p, t, Int32(V), Int32(D), grid_dim=(ceildiv(D, TT), ceildiv(V, TT)), block_dim=(TT, 8))
            for l in range(L):
                var o = V * D + l * LAYER
                comptime if FQKV:
                    var iv = H16(unsafe_from_address=Int(self.i16.unsafe_ptr())).unsafe_offset(l * 3 * DD)
                    ctx.enqueue_function[ileave_k](w.unsafe_offset(o + D), iv, Int32(3 * DD), grid_dim=grid(3 * DD), block_dim=THREADS)
                for j in range(4):
                    var q = o + D + j * DD
                    comptime if BF2:  # transpose the bf16 copy (half the read bytes, same values)
                        ctx.enqueue_function[tr16h_k](w.unsafe_offset(q), t.unsafe_offset(q), Int32(D), Int32(D), grid_dim=(ceildiv(D, TT), ceildiv(D, TT)), block_dim=(TT, 8))
                    else:
                        ctx.enqueue_function[tr16t_k](p.unsafe_offset(q), t.unsafe_offset(q), Int32(D), Int32(D), grid_dim=(ceildiv(D, TT), ceildiv(D, TT)), block_dim=(TT, 8))
                for j in range(2):
                    var q = o + 2 * D + 4 * DD + j * DF
                    comptime if BF2:  # transpose the bf16 copy (half the read bytes, same values)
                        ctx.enqueue_function[tr16h_k](w.unsafe_offset(q), t.unsafe_offset(q), Int32(D), Int32(FF), grid_dim=(ceildiv(FF, TT), ceildiv(D, TT)), block_dim=(TT, 8))
                    else:
                        ctx.enqueue_function[tr16t_k](p.unsafe_offset(q), t.unsafe_offset(q), Int32(D), Int32(FF), grid_dim=(ceildiv(FF, TT), ceildiv(D, TT)), block_dim=(TT, 8))
                var q = o + 2 * D + 4 * DD + 2 * DF
                comptime if BF2:  # transpose the bf16 copy (half the read bytes, same values)
                    ctx.enqueue_function[tr16h_k](w.unsafe_offset(q), t.unsafe_offset(q), Int32(FF), Int32(D), grid_dim=(ceildiv(D, TT), ceildiv(FF, TT)), block_dim=(TT, 8))
                else:
                    ctx.enqueue_function[tr16t_k](p.unsafe_offset(q), t.unsafe_offset(q), Int32(FF), Int32(D), grid_dim=(ceildiv(D, TT), ceildiv(FF, TT)), block_dim=(TT, 8))
            for k in range(NBH):
                var q = BH0 + k * DD
                ctx.enqueue_function[tr16h_k](w.unsafe_offset(q), t.unsafe_offset(q), Int32(D), Int32(D), grid_dim=(ceildiv(D, TT), ceildiv(D, TT)), block_dim=(TT, 8))

    def bheads(self, ctx: DeviceContext, p: F, g: F, tgt: I, wr: F, r0: Int, r1: Int) raises:
        """Block heads, forward and backward in one go (after the main head consumed c16): head k predicts the token
        k - 1 positions after the main head's target from h_k = x + silu(x W_k^T) through the tied embedding, on every
        BHS-th position; its dnf joins the main one. Losses stay in bf32 (bh_losses), apart from the main loss."""
        var b = self.b
        var sn = r1 - r0
        var nc = ceildiv(sn, BHS) * b
        var o = r0 * b * D
        var x16 = H16(unsafe_from_address=Int(self.b16x.unsafe_ptr()))
        var h16 = x16.unsafe_offset(nc * D)
        var du16 = x16.unsafe_offset(2 * nc * D)
        var u = F(unsafe_from_address=Int(self.bf32.unsafe_ptr()))
        var dh = u.unsafe_offset(nc * D)
        var dxu = u.unsafe_offset(2 * nc * D)
        var wk = u.unsafe_offset(3 * nc * D)
        var tk = I(unsafe_from_address=Int(self.btk.unsafe_ptr()))
        var c16 = H16(unsafe_from_address=Int(self.c16.unsafe_ptr()))
        var nf = self.nf.unsafe_offset(o)
        for k in range(NBH):
            var wp = p.unsafe_offset(BH0 + k * DD)
            ctx.enqueue_function[bh_gather_k](nf, tgt.unsafe_offset(r0 * b), wr.unsafe_offset(r0 * b), x16, tk, wk, Int32(nc), Int32(b), Int32(sn), Int32(k + 2), self.bhl,
                grid_dim=grid(nc * D), block_dim=THREADS)
            self.lin(ctx, u, x16, self.wsrc(p, wp, True), nc, D, D, False)
            ctx.enqueue_function[bh_act_k](nf, u, h16, Int32(nc), Int32(b), grid_dim=grid(nc * D), block_dim=THREADS)
            self.lin(ctx, self.z, h16, self.wsrc(p, p, True), nc, V, D, False)
            ctx.enqueue_function[ce_w_k[False]](self.z, tk, wk, u.unsafe_offset(3 * nc * D + nc + k * nc), self.dz, Int32(nc), Int32(V), c16,
                grid_dim=grid(nc * 32), block_dim=THREADS)
            self.lin(ctx, dh, c16, self.wsrc(p, p, False), nc, D, V, False)
            if not self.bhd:
                self.dwg[bf, bf](ctx, c16, h16, h16, h16, g, g, g, V, D, nc, 1)
            ctx.enqueue_function[bh_du_k](u, dh, du16, Int32(nc * D), grid_dim=grid(nc * D), block_dim=THREADS)
            self.lin(ctx, dxu, du16, self.wsrc(p, wp, False), nc, D, D, False)
            self.dwg[bf, bf](ctx, du16, x16, x16, x16, g.unsafe_offset(BH0 + k * DD), g.unsafe_offset(BH0 + k * DD), g.unsafe_offset(BH0 + k * DD), D, D, nc, 1)
            if not self.bhd:
                ctx.enqueue_function[bh_scat_k](dh, dxu, self.dnf.unsafe_offset(o), Int32(nc), Int32(b), grid_dim=grid(nc * D), block_dim=THREADS)

    def bh_losses(self, ctx: DeviceContext, sn: Int) raises -> List[Float64]:
        """Per block head: summed weighted CE of the last bheads call (divide by bhl for the per-row mean)."""
        var out = List[Float64]()
        if NBH == 0:
            return out^
        var nc = ceildiv(sn, BHS) * self.b
        var h = List[Float32](length=NBH * nc, fill=0)
        ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=DeviceBuffer[DType.float32](ctx, F(unsafe_from_address=Int(self.bf32.unsafe_ptr())).unsafe_offset(3 * nc * D + nc), NBH * nc, owning=False))
        ctx.synchronize()
        for k in range(NBH):
            var s: Float64 = 0
            for c in range(nc):
                s += Float64(h[k * nc + c])
            out.append(s)
        return out^

    def rms(self, ctx: DeviceContext, x: F, g: F, y: F, r: F, n: Int) raises:
        ctx.enqueue_function[rms_fwd_k](x, g, y, r, Int32(n), EPS, grid_dim=ceildiv(n, NW), block_dim=AT)

    def rms16(self, ctx: DeviceContext, hres: F, x: F, g: F, y: F, r: F, n: Int, add: Bool, kb: Bool = False) raises:
        """add: x += hres first. Also writes y as bf16 into a16, so the next mm runs with cast=False."""
        var y16 = H16(unsafe_from_address=Int(self.a16.unsafe_ptr()))
        comptime if RT and not Self.exact:
            if kb:
                ctx.enqueue_function[rms_fwdh_k[not (FQKV and AB16)]](x, hres, g, H16(unsafe_from_address=Int(y)), r, y16, Int32(n), EPS, Int32(add), grid_dim=ceildiv(n, NW), block_dim=AT)
                return
        ctx.enqueue_function[rms_fwd16_k](x, hres, g, y, r, y16, Int32(n), EPS, Int32(add), grid_dim=ceildiv(n, NW), block_dim=AT)

    def rms_bwd(self, ctx: DeviceContext, x: F, g: F, r: F, dy: F, dx: F, dg: F, n: Int, acc: Bool, dgon: Bool = True) raises:
        """dgon False (frozen base): dx only, no gain gradient lands in dg."""
        comptime if RDG and not Self.exact:
            ctx.enqueue_function[rms_dxg_k](x, g, r, dy, dx, self.part, Int32(n), Int32(acc), grid_dim=ceildiv(n, RC), block_dim=AT)
        else:
            ctx.enqueue_function[rms_dx_k](x, g, r, dy, dx, Int32(n), Int32(acc), grid_dim=ceildiv(n, NW), block_dim=AT)
            if dgon:
                ctx.enqueue_function[rms_dwp_k](x, r, dy, self.part, Int32(n), grid_dim=ceildiv(n, RC), block_dim=D)
        if dgon:
            ctx.enqueue_function[colsum_k](self.part, dg, Int32(ceildiv(n, RC)), Int32(D), grid_dim=grid(D), block_dim=THREADS)

    def add(self, ctx: DeviceContext, x: F, y: F, n: Int) raises:
        ctx.enqueue_function[add_k](x, y, Int32(n), grid_dim=grid(n), block_dim=THREADS)

    def copy(self, ctx: DeviceContext, src: F, dst: F, n: Int, acc: Bool) raises:
        ctx.enqueue_function[copy_k](src, dst, Int32(n), Int32(acc), grid_dim=grid(n), block_dim=THREADS)

    def rope[inv: Bool](self, ctx: DeviceContext, x: F, n: Int, r0: Int) raises:
        var c = F(unsafe_from_address=Int(self.cs.unsafe_ptr()))
        var s = F(unsafe_from_address=Int(self.sn.unsafe_ptr()))
        ctx.enqueue_function[rope_k[inv]](x, c, s, Int32(n), Int32(self.b), Int32(r0), Int32(H), Int32(HD), grid_dim=grid(n * D // 2), block_dim=THREADS)

    def embed(self, ctx: DeviceContext, p: F, ids: I, kinds: List[Int], S: Int) raises:
        """Token rows from the table; pause rows from the pause block; latent rows are filled by forward."""
        var b = self.b
        var x = self.acts[0].x
        ctx.enqueue_function[embed_k](p, ids, x, Int32(S * b), Int32(D), grid_dim=grid(S * b * D), block_dim=THREADS)
        var j = 0
        for r in range(S):
            if kinds[r] == PSE:
                ctx.enqueue_function[bcast_k](p.unsafe_offset(PAUSE + j * D), x.unsafe_offset(r * b * D), Int32(b), Int32(D), grid_dim=grid(b * D), block_dim=THREADS)
                j += 1

    def fwd_seg(self, ctx: DeviceContext, p: F, tgt: I, wr: F, r0: Int, r1: Int, head: Bool) raises:
        if self.ffm != FF:
            comptime if Self.exact or not (AB16 and BT and DWG and BF2):
                raise Error("FAIL ffm: Matryoshka widths need the bf16 AB16/BT/DWG/BF2 path (got ffm " + String(self.ffm) + ")")
        var b = self.b
        var n = (r1 - r0) * b
        var o = r0 * b * D
        for l in range(self.nl):
            var a = self.acts[l]
            var w = layer(p, l)
            var out = self.acts[l + 1].x if l + 1 < self.nl else self.xl
            comptime if F16:
                # l > 0: the previous layer's residual add (x += h) is fused here
                var hp = self.acts[l - 1].h.unsafe_offset(o) if l > 0 else a.x.unsafe_offset(o)
                self.rms16(ctx, hp, a.x.unsafe_offset(o), w.g1, a.n1.unsafe_offset(o), a.r1.unsafe_offset(r0 * b), n, l > 0, True)
            else:
                self.rms(ctx, a.x.unsafe_offset(o), w.g1, a.n1.unsafe_offset(o), a.r1.unsafe_offset(r0 * b), n)
            comptime if FQKV and not Self.exact:
                # t16 holds wq^T, wk^T, wv^T back to back: one [3D, D] B operand
                var b16p = H16(unsafe_from_address=Int(self.t16.unsafe_ptr())).unsafe_offset((Int(w.wq) - Int(p)) // 4)
                var n1h = H16(unsafe_from_address=Int(a.n1.unsafe_offset(o))) if AB16 else H16(unsafe_from_address=Int(self.a16.unsafe_ptr()))
                self.lin3(ctx, a.q.unsafe_offset(o), a.k.unsafe_offset(o), a.v.unsafe_offset(o), n1h, b16p, n, D, False)
            else:
                self.mm[False, False](ctx, p, a.n1.unsafe_offset(o), w.wq, a.q.unsafe_offset(o), n, D, D, False, not F16)
                self.mm[False, False](ctx, p, a.n1.unsafe_offset(o), w.wk, a.k.unsafe_offset(o), n, D, D, False, False)
                self.mm[False, False](ctx, p, a.n1.unsafe_offset(o), w.wv, a.v.unsafe_offset(o), n, D, D, False, False)
            comptime if QKN:
                var qrp = F(unsafe_from_address=Int(self.qr.unsafe_ptr())).unsafe_offset(l * 2 * self.SR * H + r0 * b * H)
                ctx.enqueue_function[qkn_k](a.q.unsafe_offset(o), qrp, Int32(n), grid_dim=grid(n * H * 32), block_dim=THREADS)
                ctx.enqueue_function[qkn_k](a.k.unsafe_offset(o), qrp.unsafe_offset(self.SR * H), Int32(n), grid_dim=grid(n * H * 32), block_dim=THREADS)
            comptime if not (RF and AW and not Self.exact) or (FLASH and Self.exact):  # RF: attention ropes q/k on load
                self.rope[False](ctx, a.q.unsafe_offset(o), n, r0)
                self.rope[False](ctx, a.k.unsafe_offset(o), n, r0)
            var csp = F(unsafe_from_address=Int(self.cs.unsafe_ptr()))
            var snp = F(unsafe_from_address=Int(self.sn.unsafe_ptr()))
            comptime if FLASH:
                if r0 != 0:
                    raise Error("FAIL flash: one segment only (r0 " + String(r0) + ")")
                comptime if Self.exact:
                    ctx.enqueue_function[attn_ref_fwd_k](a.q, a.k, a.v, a.o, a.p, Int32(b), Int32(H), Int32(r1), Float32(HD) ** -0.5,
                        grid_dim=grid(r1 * b * H), block_dim=THREADS)
                else:
                    var qkv = self.qkv16(l)
                    comptime for t in range(3):
                        ctx.enqueue_function[rope16_k]((a.q, a.k, a.v)[t], qkv.unsafe_offset(t * self.SR * D), csp, snp, Int32(n), Int32(b),
                            Int32(Int(t < 2)), grid_dim=grid(n * D // 16), block_dim=THREADS)
                    ctx.enqueue_function[flash_fwd8t_k if (FY & 1) != 0 else flash_fwd8_k](qkv, qkv.unsafe_offset(self.SR * D), qkv.unsafe_offset(2 * self.SR * D), a.o, a.p, Int32(b), Int32(H), Int32(r1), Float32(HD) ** -0.5, csp, snp,
                        grid_dim=(b * H, ceildiv(r1, FQR)), block_dim=F8)
            elif AW and not Self.exact:
                ctx.enqueue_function[attn_fwd_w_k](a.q, a.k, a.v, a.p, a.o, Int32(b), Int32(H), Int32(r0), Int32(r1),
                    Float32(HD) ** -0.5, csp, snp, grid_dim=b * H, block_dim=AT)
            else:
                ctx.enqueue_function[attn_fwd_k](a.q, a.k, a.v, a.p, a.o, Int32(b), Int32(H), Int32(HD), Int32(r0), Int32(r1),
                    Float32(HD) ** -0.5, grid_dim=b * H, block_dim=AT)
            comptime if BF2 and not Self.exact:
                self.lin(ctx, a.h.unsafe_offset(o), H16(unsafe_from_address=Int(a.o)).unsafe_offset(o), self.wsrc(p, w.wo, False), n, D, D, False)
            else:
                self.mm[False, False](ctx, p, a.o.unsafe_offset(o), w.wo, a.h.unsafe_offset(o), n, D, D)
            if self.xe.on and self.xe.has(3):
                comptime if Self.exact:
                    self.xf(ctx, l, 3, Int(a.o.unsafe_offset(o)), Int(a.h.unsafe_offset(o)), False, n, r0 * b)
                else:
                    self.xf(ctx, l, 3, Int(H16(unsafe_from_address=Int(a.o)).unsafe_offset(o)), Int(a.h.unsafe_offset(o)), False, n, r0 * b)
            comptime if F16:
                self.rms16(ctx, a.x.unsafe_offset(o), a.h.unsafe_offset(o), w.g2, a.n2.unsafe_offset(o), a.r2.unsafe_offset(r0 * b), n, True, True)
            else:
                self.add(ctx, a.x.unsafe_offset(o), a.h.unsafe_offset(o), n * D)
                self.rms(ctx, a.h.unsafe_offset(o), w.g2, a.n2.unsafe_offset(o), a.r2.unsafe_offset(r0 * b), n)
            if self.xe.on and self.xe.routed():
                self.xrf(ctx, l, Int(a.n2.unsafe_offset(o)), n, r0 * b)
            comptime if AB16 and not Self.exact:
                var a16p = H16(unsafe_from_address=Int(a.n2.unsafe_offset(o))) if FQKV else H16(unsafe_from_address=Int(self.a16.unsafe_ptr()))
                self.lin16(ctx, H16(unsafe_from_address=Int(a.a.unsafe_offset(r0 * b * FF))), a16p, self.wsrc(p, w.w1, False), n, self.ffm, D)
                self.lin16(ctx, H16(unsafe_from_address=Int(a.b.unsafe_offset(r0 * b * FF))), a16p, self.wsrc(p, w.w3, False), n, self.ffm, D)
            else:
                self.mm[False, False](ctx, p, a.n2.unsafe_offset(o), w.w1, a.a.unsafe_offset(r0 * b * FF), n, FF, D, False, not F16)
                self.mm[False, False](ctx, p, a.n2.unsafe_offset(o), w.w3, a.b.unsafe_offset(r0 * b * FF), n, FF, D, False, False)
            if self.xe.on:
                if self.xe.has(0):
                    self.xf(ctx, l, 0, Int(a.n2.unsafe_offset(o)), Int(a.a.unsafe_offset(r0 * b * FF)), not Self.exact, n, r0 * b)
                if self.xe.has(1):
                    self.xf(ctx, l, 1, Int(a.n2.unsafe_offset(o)), Int(a.b.unsafe_offset(r0 * b * FF)), not Self.exact, n, r0 * b)
            comptime if BT and not Self.exact:
                var s16 = H16(unsafe_from_address=Int(a.s.unsafe_offset(r0 * b * FF)))
                comptime if AB16:
                    ctx.enqueue_function[swiglu_fwdb_k](H16(unsafe_from_address=Int(a.a.unsafe_offset(r0 * b * FF))),
                        H16(unsafe_from_address=Int(a.b.unsafe_offset(r0 * b * FF))), s16, Int32(n * self.ffm), grid_dim=grid(n * self.ffm), block_dim=THREADS)
                else:
                    ctx.enqueue_function[swiglu_fwdh_k](a.a.unsafe_offset(r0 * b * FF), a.b.unsafe_offset(r0 * b * FF), s16,
                        Int32(n * FF), grid_dim=grid(n * FF), block_dim=THREADS)
                if self.ffm == FF:
                    self.lin(ctx, out.unsafe_offset(o), s16, self.wsrc(p, w.w2, False), n, D, FF, False)
                else:  # Matryoshka: s [n, m] times the first m rows of w2 [FF, D] (its bf16 copy, not transposed)
                    self.lin_nt(ctx, out.unsafe_offset(o), s16, self.wsrc(p, w.w2, True), n, D, self.ffm, False)
            elif SW16:
                ctx.enqueue_function[swiglu_fwd16_k](a.a.unsafe_offset(r0 * b * FF), a.b.unsafe_offset(r0 * b * FF), a.s.unsafe_offset(r0 * b * FF),
                    H16(unsafe_from_address=Int(self.a16.unsafe_ptr())), Int32(n * FF), grid_dim=grid(n * FF), block_dim=THREADS)
            else:
                ctx.enqueue_function[swiglu_fwd_k](a.a.unsafe_offset(r0 * b * FF), a.b.unsafe_offset(r0 * b * FF), a.s.unsafe_offset(r0 * b * FF),
                    Int32(n * FF), grid_dim=grid(n * FF), block_dim=THREADS)
            comptime if not (BT and not Self.exact):
                self.mm[False, False](ctx, p, a.s.unsafe_offset(r0 * b * FF), w.w2, out.unsafe_offset(o), n, D, FF, False, not SW16)
            if self.xe.on and self.xe.has(2):
                self.xf(ctx, l, 2, Int(a.s.unsafe_offset(r0 * b * FF)), Int(out.unsafe_offset(o)), False, n, r0 * b)
            comptime if not F16:
                self.add(ctx, a.h.unsafe_offset(o), out.unsafe_offset(o), n * D)
        comptime if F16:
            self.rms16(ctx, self.acts[self.nl - 1].h.unsafe_offset(o), self.xl.unsafe_offset(o), p.unsafe_offset(GF), self.nf.unsafe_offset(o),
                self.rf.unsafe_offset(r0 * b), n, True)
        else:
            self.rms(ctx, self.xl.unsafe_offset(o), p.unsafe_offset(GF), self.nf.unsafe_offset(o), self.rf.unsafe_offset(r0 * b), n)
        if head:
            self.mm[False, True](ctx, p, self.nf.unsafe_offset(o), p, self.z.unsafe_offset(r0 * b * V), n, V, D, False, not F16)
            comptime if V > 256:  # one wave per row; small V keeps the thread-per-row kernel (bit-identical builds)
                if self.dist > 0:  # mdist: the same row loss plus in-place distillation from the saved full-width logits
                    ctx.enqueue_function[ce_kd_k[Self.exact]](self.z.unsafe_offset(r0 * b * V), H16(unsafe_from_address=Int(self.zt.unsafe_ptr())).unsafe_offset(r0 * b * V),
                        tgt.unsafe_offset(r0 * b), wr.unsafe_offset(r0 * b), self.lrow.unsafe_offset(r0 * b), self.dz.unsafe_offset(r0 * b * V), Int32(n), Int32(V),
                        H16(unsafe_from_address=Int(self.c16.unsafe_ptr())), self.dist, grid_dim=grid(n * 32), block_dim=THREADS)
                else:
                    ctx.enqueue_function[ce_w_k[Self.exact]](self.z.unsafe_offset(r0 * b * V), tgt.unsafe_offset(r0 * b), wr.unsafe_offset(r0 * b), self.lrow.unsafe_offset(r0 * b),
                        self.dz.unsafe_offset(r0 * b * V), Int32(n), Int32(V), H16(unsafe_from_address=Int(self.c16.unsafe_ptr())), grid_dim=grid(n * 32), block_dim=THREADS)
            else:
                ctx.enqueue_function[ce_k](self.z.unsafe_offset(r0 * b * V), tgt.unsafe_offset(r0 * b), wr.unsafe_offset(r0 * b), self.lrow.unsafe_offset(r0 * b),
                    self.dz.unsafe_offset(r0 * b * V), Int32(n), Int32(V), grid_dim=grid(n), block_dim=THREADS)

    def bwd_seg(self, ctx: DeviceContext, p: F, g: F, r0: Int, r1: Int, tgt: I, wr: F, first: Bool = False) raises:
        var b = self.b
        var n = (r1 - r0) * b
        var o = r0 * b * D
        var of = r0 * b * FF
        comptime if V > 256 and not Self.exact:  # ce_w_k left bf16 dz in c16: dX by linalg, dE by dwg_k, no [rows, V] cast/transpose
            var dz16 = H16(unsafe_from_address=Int(self.c16.unsafe_ptr()))
            self.lin(ctx, self.dnf.unsafe_offset(o), dz16, self.wsrc(p, p, False), n, D, V, True)
            if not self.frz:
                self.dwg[bf, DType.float32](ctx, dz16, self.nf.unsafe_offset(o), self.nf.unsafe_offset(o), self.nf.unsafe_offset(o), g, g, g, V, D, n, 1)
            comptime if NBH > 0:
                if not self.frz:  # (run.mojo refuses frz with block heads; this keeps the guard local too)
                    self.bheads(ctx, p, g, tgt, wr, r0, r1)
        else:
            self.mm[False, False](ctx, p, self.dz.unsafe_offset(r0 * b * V), p, self.dnf.unsafe_offset(o), n, D, V, True)
            if not self.frz:
                self.mm[True, False](ctx, p, self.dz.unsafe_offset(r0 * b * V), self.nf.unsafe_offset(o), g, V, D, n, True)
        self.rms_bwd(ctx, self.xl.unsafe_offset(o), p.unsafe_offset(GF), self.rf.unsafe_offset(r0 * b), self.dnf.unsafe_offset(o),
            self.dx.unsafe_offset(o), g.unsafe_offset(GF), n, False, not self.frz)
        var dx = self.dx.unsafe_offset(o)
        var dn = self.dn.unsafe_offset(o)
        var dt = self.dt.unsafe_offset(o)
        var dq = self.dq.unsafe_offset(o)
        for l in reversed(range(self.nl)):
            var a = self.acts[l]
            var w = layer(p, l)
            var gw = layer(g, l)
            comptime if AB16 and not Self.exact:
                var a16p = H16(unsafe_from_address=Int(self.a16.unsafe_ptr()))
                ctx.enqueue_function[cast16_k](dx, a16p, Int32(n * D), grid_dim=grid(n * D), block_dim=THREADS)
                self.lin16(ctx, H16(unsafe_from_address=Int(self.dff.unsafe_offset(of))), a16p, self.wsrc(p, w.w2, True), n, self.ffm, D)
            else:
                self.mm[False, True](ctx, p, dx, w.w2, self.dff.unsafe_offset(of), n, FF, D)
            if self.xe.on and self.xe.has(2):  # w2 expert: dy = dx (a16 holds bf16(dx) on the bf16 path), dx_in = dff
                comptime if Self.exact:
                    self.xbw(ctx, l, 2, Int(a.s.unsafe_offset(of)), Int(dx), Int(self.dff.unsafe_offset(of)), False, n, r0 * b, True)
                else:
                    self.xbw(ctx, l, 2, Int(a.s.unsafe_offset(of)), Int(self.a16.unsafe_ptr()), Int(self.dff.unsafe_offset(of)), True, n, r0 * b, True)
            comptime if BT and not Self.exact:
                var a16p = H16(unsafe_from_address=Int(self.a16.unsafe_ptr()))
                var b16p = H16(unsafe_from_address=Int(self.b16.unsafe_ptr()))
                var s16 = H16(unsafe_from_address=Int(a.s.unsafe_offset(of)))
                var da16 = H16(unsafe_from_address=Int(self.da.unsafe_offset(of)))
                var db16 = H16(unsafe_from_address=Int(self.db.unsafe_offset(of)))
                var tf = (ceildiv(FF, TT), ceildiv(n, TT))
                if not self.frz:
                    comptime if DWG:
                        comptime if BF2:  # a16 holds bf16(dx): cast for the dff dX GEMM just above
                            self.dwg[bf, bf](ctx, s16, a16p, a16p, a16p, gw.w2, gw.w2, gw.w2, self.ffm, D, n, 1)
                        else:
                            self.dwg[bf, DType.float32](ctx, s16, dx, dx, dx, gw.w2, gw.w2, gw.w2, FF, D, n, 1)
                    else:
                        ctx.enqueue_function[tr16h_k](s16, a16p, Int32(n), Int32(FF), grid_dim=tf, block_dim=(TT, 8))
                        ctx.enqueue_function[tr16t_k](dx, b16p, Int32(n), Int32(D), grid_dim=(ceildiv(D, TT), ceildiv(n, TT)), block_dim=(TT, 8))
                        self.lin(ctx, gw.w2, a16p, b16p, FF, D, n, True)
                comptime if AB16:
                    ctx.enqueue_function[swiglu_bwdb_k](H16(unsafe_from_address=Int(a.a.unsafe_offset(of))), H16(unsafe_from_address=Int(a.b.unsafe_offset(of))),
                        H16(unsafe_from_address=Int(self.dff.unsafe_offset(of))), da16, db16, Int32(n * self.ffm), grid_dim=grid(n * self.ffm), block_dim=THREADS)
                else:
                    ctx.enqueue_function[swiglu_bwdh_k](a.a.unsafe_offset(of), a.b.unsafe_offset(of), self.dff.unsafe_offset(of), da16, db16,
                        Int32(n * FF), grid_dim=grid(n * FF), block_dim=THREADS)
                if self.ffm == FF:
                    self.lin(ctx, dn, da16, self.wsrc(p, w.w1, True), n, D, FF, False)
                    self.lin(ctx, dn, db16, self.wsrc(p, w.w3, True), n, D, FF, True)
                else:  # dn = da w1[:, :m]^T = da (first m rows of the transposed copy)
                    self.lin_nt(ctx, dn, da16, self.wsrc(p, w.w1, False), n, D, self.ffm, False)
                    self.lin_nt(ctx, dn, db16, self.wsrc(p, w.w3, False), n, D, self.ffm, True)
                if not self.frz:
                    comptime if DWG:
                        if self.ffm == FF:
                            self.dwg[bf, bf](ctx, H16(unsafe_from_address=Int(a.n2.unsafe_offset(o))), da16, db16, db16, gw.w1, gw.w3, gw.w3, D, FF, n, 2)
                        else:  # dW1^T, dW3^T [m, D] = da^T n2 into ctmp, then added into the first m columns of [D, FF]
                            var t0 = self.ctmp
                            var t1 = self.ctmp.unsafe_offset(self.ffm * D)
                            ctx.enqueue_memset(DeviceBuffer[DType.float32](ctx, t0, 2 * self.ffm * D, owning=False), 0)
                            var n2h = H16(unsafe_from_address=Int(a.n2.unsafe_offset(o)))
                            self.dwg[bf, bf](ctx, da16, n2h, n2h, n2h, t0, t0, t0, self.ffm, D, n, 1)
                            self.dwg[bf, bf](ctx, db16, n2h, n2h, n2h, t1, t1, t1, self.ffm, D, n, 1)
                            ctx.enqueue_function[tradd_k](t0, gw.w1, Int32(self.ffm), grid_dim=grid(self.ffm * D), block_dim=THREADS)
                            ctx.enqueue_function[tradd_k](t1, gw.w3, Int32(self.ffm), grid_dim=grid(self.ffm * D), block_dim=THREADS)
                    elif RT:
                        ctx.enqueue_function[tr16h_k](H16(unsafe_from_address=Int(a.n2.unsafe_offset(o))), a16p, Int32(n), Int32(D), grid_dim=(ceildiv(D, TT), ceildiv(n, TT)), block_dim=(TT, 8))
                    else:
                        ctx.enqueue_function[tr16t_k](a.n2.unsafe_offset(o), a16p, Int32(n), Int32(D), grid_dim=(ceildiv(D, TT), ceildiv(n, TT)), block_dim=(TT, 8))
                    comptime if not DWG:
                        ctx.enqueue_function[tr16h_k](da16, b16p, Int32(n), Int32(FF), grid_dim=tf, block_dim=(TT, 8))
                        self.lin(ctx, gw.w1, a16p, b16p, D, FF, n, True)
                        ctx.enqueue_function[tr16h_k](db16, b16p, Int32(n), Int32(FF), grid_dim=tf, block_dim=(TT, 8))
                        self.lin(ctx, gw.w3, a16p, b16p, D, FF, n, True)
            else:
                if not self.frz:
                    self.mm[True, False](ctx, p, a.s.unsafe_offset(of), dx, gw.w2, FF, D, n, True)
            comptime if BT and not Self.exact:
                pass
            elif SW16:
                ctx.enqueue_function[swiglu_bwd16_k](a.a.unsafe_offset(of), a.b.unsafe_offset(of), self.dff.unsafe_offset(of), self.da.unsafe_offset(of),
                    self.db.unsafe_offset(of), H16(unsafe_from_address=Int(self.a16.unsafe_ptr())), H16(unsafe_from_address=Int(self.b16.unsafe_ptr())),
                    Int32(n * FF), grid_dim=grid(n * FF), block_dim=THREADS)
                # dX first: a16/b16 hold da/db in bf16 until the dW transposes overwrite them
                self.mm[False, True](ctx, p, self.da.unsafe_offset(of), w.w1, dn, n, D, FF, False, False)
                self.mm[False, True](ctx, p, self.db.unsafe_offset(of), w.w3, dn, n, D, FF, True, False, 1)
                if not self.frz:
                    self.mm[True, False](ctx, p, a.n2.unsafe_offset(o), self.da.unsafe_offset(of), gw.w1, D, FF, n, True)
                    self.mm[True, False](ctx, p, a.n2.unsafe_offset(o), self.db.unsafe_offset(of), gw.w3, D, FF, n, True, False)
            else:
                ctx.enqueue_function[swiglu_bwd_k](a.a.unsafe_offset(of), a.b.unsafe_offset(of), self.dff.unsafe_offset(of), self.da.unsafe_offset(of),
                    self.db.unsafe_offset(of), Int32(n * FF), grid_dim=grid(n * FF), block_dim=THREADS)
                if not self.frz:
                    self.mm[True, False](ctx, p, a.n2.unsafe_offset(o), self.da.unsafe_offset(of), gw.w1, D, FF, n, True)
                    self.mm[True, False](ctx, p, a.n2.unsafe_offset(o), self.db.unsafe_offset(of), gw.w3, D, FF, n, True, False)
                self.mm[False, True](ctx, p, self.da.unsafe_offset(of), w.w1, dn, n, D, FF)
                self.mm[False, True](ctx, p, self.db.unsafe_offset(of), w.w3, dn, n, D, FF, True)
            if self.xe.on:  # w1 / w3 experts (dy = da / db as the base path keeps them, dx_in = dn), then the router
                var n2a = Int(a.n2.unsafe_offset(o))
                var f1 = not self.xe.has(2)
                if self.xe.has(0):
                    self.xbw(ctx, l, 0, n2a, Int(self.da.unsafe_offset(of)), Int(dn), False, n, r0 * b, f1)
                    f1 = False
                if self.xe.has(1):
                    self.xbw(ctx, l, 1, n2a, Int(self.db.unsafe_offset(of)), Int(dn), False, n, r0 * b, f1)
                if self.xe.routed():
                    self.xrb(ctx, l, n2a, dn, n, r0 * b)
            self.rms_bwd(ctx, a.h.unsafe_offset(o), w.g2, a.r2.unsafe_offset(r0 * b), dn, dx, gw.g2, n, True, not self.frz)
            self.mm[False, True](ctx, p, dx, w.wo, dt, n, D, D)
            if self.xe.on and self.xe.has(3):  # wo expert: dy = dx (a16 = bf16(dx) after the dX mm), dx_in = dt
                comptime if Self.exact:
                    self.xbw(ctx, l, 3, Int(a.o.unsafe_offset(o)), Int(dx), Int(dt), False, n, r0 * b, True)
                else:
                    self.xbw(ctx, l, 3, Int(H16(unsafe_from_address=Int(a.o)).unsafe_offset(o)), Int(self.a16.unsafe_ptr()), Int(dt), False, n, r0 * b, True)
            if not self.frz:
                comptime if WOS and not Self.exact:  # wo dW: 64 output blocks, so split K in two (128 blocks) + one add
                    ctx.enqueue_memset(DeviceBuffer[DType.float32](ctx, self.ctmp, DD, owning=False), 0)
                    comptime if BF2:  # o is bf16; a16 holds bf16(dx) from the wo dX mm just above
                        var a16w = H16(unsafe_from_address=Int(self.a16.unsafe_ptr()))
                        self.dwg[bf, bf](ctx, H16(unsafe_from_address=Int(a.o)).unsafe_offset(o), a16w, a16w, a16w, gw.wo, self.ctmp, gw.wo, D, D, n, 1, 2)
                    else:
                        self.dwg[DType.float32, DType.float32](ctx, a.o.unsafe_offset(o), dx, dx, dx, gw.wo, self.ctmp, gw.wo, D, D, n, 1, 2)
                    self.add(ctx, self.ctmp, gw.wo, DD)
                else:
                    self.mm[True, False](ctx, p, a.o.unsafe_offset(o), dx, gw.wo, D, D, n, True)
            if self.frz and l == 0:  # frozen base: layer 0's attention backward only feeds the embedding gradient (none)
                continue
            comptime if FLASH:
                var sc = Float32(HD) ** -0.5
                comptime if Self.exact:
                    comptime for part in range(3):
                        ctx.enqueue_function[attn_ref_bwd_k](a.q, a.k, a.v, a.o, self.dt, a.p, self.da, self.dq, a.dk, a.dv, Int32(b), Int32(H), Int32(r1),
                            sc, Int32(part), grid_dim=grid(r1 * b * H), block_dim=THREADS)
                else:
                    var csf = F(unsafe_from_address=Int(self.cs.unsafe_ptr()))
                    var snf = F(unsafe_from_address=Int(self.sn.unsafe_ptr()))
                    var y16 = H16(unsafe_from_address=Int(self.a16.unsafe_ptr()))
                    var qkv = self.qkv16(l)
                    var do16 = qkv.unsafe_offset(3 * (L - l) * self.SR * D)  # the slot after the last layer's
                    ctx.enqueue_function[rope16_k](self.dt, do16, csf, snf, Int32(n), Int32(b), Int32(0), grid_dim=grid(n * D // 16), block_dim=THREADS)
                    ctx.enqueue_function[dlt_k](self.dt, a.o, self.da, Int32(b), Int32(H), Int32(r1), grid_dim=grid(r1 * b * H * 32), block_dim=THREADS)
                    ctx.enqueue_function[flash_dkv8t_k if (FY & 4) != 0 else flash_dkv8_k](qkv, qkv.unsafe_offset(self.SR * D), qkv.unsafe_offset(2 * self.SR * D), do16, a.p, self.da, y16, Int32(b), Int32(H), Int32(r1), sc, csf, snf,
                        grid_dim=(b * H, ceildiv(r1, FQR)), block_dim=F8)
                    ctx.enqueue_function[flash_dq8t_k if (FY & 2) != 0 else flash_dq8_k](qkv, qkv.unsafe_offset(self.SR * D), qkv.unsafe_offset(2 * self.SR * D), do16, a.p, self.da, y16, Int32(b), Int32(H), Int32(r1), sc, csf, snf,
                        grid_dim=(b * H, ceildiv(r1, FQR)), block_dim=F8)
            elif AW and not Self.exact:
                ctx.enqueue_function[attn_bwd_w_k[AR3 and R3]](a.q, a.k, a.v, a.p, self.dt, self.dq, a.dk, a.dv, Int32(b), Int32(H), Int32(r0), Int32(r1),
                    Float32(HD) ** -0.5, Int32(first), F(unsafe_from_address=Int(self.cs.unsafe_ptr())),
                    F(unsafe_from_address=Int(self.sn.unsafe_ptr())), H16(unsafe_from_address=Int(self.a16.unsafe_ptr())), grid_dim=b * H, block_dim=AT)
            else:
                ctx.enqueue_function[attn_bwd_k](a.q, a.k, a.v, a.p, self.dt, self.dq, a.dk, a.dv, Int32(b), Int32(H), Int32(HD), Int32(r0), Int32(r1),
                    Float32(HD) ** -0.5, Int32(first), grid_dim=b * H, block_dim=AT)
            comptime if R3 and not Self.exact:
                # rope3_k: inverse rope + bf16 [n, 3D] in one pass; q/k/v dW and dX both read it
                var a16p = H16(unsafe_from_address=Int(self.a16.unsafe_ptr()))
                comptime if not (AR3 or FLASH):  # AR3 / FLASH: the attention backward already wrote these rows
                    ctx.enqueue_function[rope3_k](dq, a.dk.unsafe_offset(o), a.dv.unsafe_offset(o), a16p, F(unsafe_from_address=Int(self.cs.unsafe_ptr())),
                        F(unsafe_from_address=Int(self.sn.unsafe_ptr())), Int32(n), Int32(b), Int32(r0), grid_dim=grid(n * 3 * D), block_dim=THREADS)
                comptime if QKN:  # a16p rows hold inverse-roped dq | dk | dv; a.q / a.k hold the unroped normalised q / k
                    var qrp = F(unsafe_from_address=Int(self.qr.unsafe_ptr())).unsafe_offset(l * 2 * self.SR * H + r0 * b * H)
                    ctx.enqueue_function[qknb_k[True]](dq, a16p, Int32(3 * D), a.q.unsafe_offset(o), qrp, Int32(n), grid_dim=grid(n * H * 32), block_dim=THREADS)
                    ctx.enqueue_function[qknb_k[True]](dq, a16p.unsafe_offset(D), Int32(3 * D), a.k.unsafe_offset(o), qrp.unsafe_offset(self.SR * H), Int32(n), grid_dim=grid(n * H * 32), block_dim=THREADS)
                if not self.frz:
                    self.dwg[bf, bf](ctx, H16(unsafe_from_address=Int(a.n1.unsafe_offset(o))), a16p, a16p.unsafe_offset(D), a16p.unsafe_offset(2 * D),
                        gw.wq, gw.wk, gw.wv, D, D, n, 3, 1, 3 * D)
                var iv = H16(unsafe_from_address=Int(self.i16.unsafe_ptr())).unsafe_offset(l * 3 * DD)
                self.lin(ctx, dn, a16p, iv, n, D, 3 * D, False)
            else:
                comptime if QKN:  # dq / dk and a.q / a.k are both in the roped frame here (rope ran in place)
                    var qrp = F(unsafe_from_address=Int(self.qr.unsafe_ptr())).unsafe_offset(l * 2 * self.SR * H + r0 * b * H)
                    var nh = H16(unsafe_from_address=Int(self.a16.unsafe_ptr()))
                    ctx.enqueue_function[qknb_k[False]](dq, nh, Int32(D), a.q.unsafe_offset(o), qrp, Int32(n), grid_dim=grid(n * H * 32), block_dim=THREADS)
                    ctx.enqueue_function[qknb_k[False]](a.dk.unsafe_offset(o), nh, Int32(D), a.k.unsafe_offset(o), qrp.unsafe_offset(self.SR * H), Int32(n), grid_dim=grid(n * H * 32), block_dim=THREADS)
                self.rope[True](ctx, dq, n, r0)
                self.rope[True](ctx, a.dk.unsafe_offset(o), n, r0)
            comptime if R3 and not Self.exact:
                pass
            elif FQKV and not Self.exact:
                var a16p = H16(unsafe_from_address=Int(self.a16.unsafe_ptr()))
                var b16p = H16(unsafe_from_address=Int(self.b16.unsafe_ptr()))
                var tg = (ceildiv(D, TT), ceildiv(n, TT))
                comptime if DWG:
                    if not self.frz:
                        self.dwg[bf, DType.float32](ctx, H16(unsafe_from_address=Int(a.n1.unsafe_offset(o))), dq, a.dk.unsafe_offset(o), a.dv.unsafe_offset(o),
                            gw.wq, gw.wk, gw.wv, D, D, n, 3)
                else:
                    if not self.frz:  # the dW transposes and lin3 into gw; cast3_k and the dX lin below still run
                        comptime if RT:
                            ctx.enqueue_function[tr16h_k](H16(unsafe_from_address=Int(a.n1.unsafe_offset(o))), a16p, Int32(n), Int32(D), grid_dim=tg, block_dim=(TT, 8))
                        else:
                            ctx.enqueue_function[tr16t_k](a.n1.unsafe_offset(o), a16p, Int32(n), Int32(D), grid_dim=tg, block_dim=(TT, 8))
                        ctx.enqueue_function[tr16t_k](dq, b16p, Int32(n), Int32(D), grid_dim=tg, block_dim=(TT, 8))
                        ctx.enqueue_function[tr16t_k](a.dk.unsafe_offset(o), b16p.unsafe_offset(D * n), Int32(n), Int32(D), grid_dim=tg, block_dim=(TT, 8))
                        ctx.enqueue_function[tr16t_k](a.dv.unsafe_offset(o), b16p.unsafe_offset(2 * D * n), Int32(n), Int32(D), grid_dim=tg, block_dim=(TT, 8))
                        self.lin3(ctx, gw.wq, gw.wk, gw.wv, a16p, b16p, D, n, True)
                ctx.enqueue_function[cast3_k](dq, a.dk.unsafe_offset(o), a.dv.unsafe_offset(o), a16p, Int32(n), grid_dim=grid(n * 3 * D), block_dim=THREADS)
                var iv = H16(unsafe_from_address=Int(self.i16.unsafe_ptr())).unsafe_offset(l * 3 * DD)
                self.lin(ctx, dn, a16p, iv, n, D, 3 * D, False)
            else:
                if not self.frz:
                    self.mm[True, False](ctx, p, a.n1.unsafe_offset(o), dq, gw.wq, D, D, n, True)
                    self.mm[True, False](ctx, p, a.n1.unsafe_offset(o), a.dk.unsafe_offset(o), gw.wk, D, D, n, True, False)
                    self.mm[True, False](ctx, p, a.n1.unsafe_offset(o), a.dv.unsafe_offset(o), gw.wv, D, D, n, True, False)
                self.mm[False, True](ctx, p, dq, w.wq, dn, n, D, D)
                self.mm[False, True](ctx, p, a.dk.unsafe_offset(o), w.wk, dn, n, D, D, True)
                self.mm[False, True](ctx, p, a.dv.unsafe_offset(o), w.wv, dn, n, D, D, True)
            self.rms_bwd(ctx, a.x.unsafe_offset(o), w.g1, a.r1.unsafe_offset(r0 * b), dn, dx, gw.g1, n, True, not self.frz)

    def step(self, ctx: DeviceContext, p: F, g: F, ids: I, tgt: I, wr: F, kinds: List[Int], S: Int, train: Bool) raises:
        """Forward (and backward when train) of one batch; S rows per sequence, kinds[r] in TOK/LAT/PSE/GEN; row loss weight wr."""
        var b = self.b
        var segs = List[Int]()
        segs.append(0)
        for r in range(1, S):
            if kinds[r] == LAT or kinds[r] == GEN or kinds[r] == SOFT or kinds[r] == NEU:
                segs.append(r)
        segs.append(S)
        self.refresh(ctx, p)
        comptime if not Self.exact:
            if self.xe.on:
                self.xe.refresh(ctx)
        self.embed(ctx, p, ids, kinds, S)
        var hop = List[Int](length=S, fill=-1)
        var nneu = 0
        for r in range(S):
            if kinds[r] == NEU:
                hop[r] = nneu
                nneu += 1
        if kinds[0] == ENT:
            self.lk.ent_fwd(ctx, p, self.acts[0].x)
        for i in range(len(segs) - 1):
            var r0 = segs[i]
            if kinds[r0] == NEU:
                self.lk.fwd[Self.exact](ctx, p, self.nf.unsafe_offset((r0 - 1) * b * D), self.acts[0].x.unsafe_offset(r0 * b * D), hop[r0])
            if kinds[r0] == LAT:
                self.copy(ctx, self.nf.unsafe_offset((r0 - 1) * b * D), self.acts[0].x.unsafe_offset(r0 * b * D), b * D, False)
            if kinds[r0] == SOFT:
                ctx.enqueue_function[soft_k](self.z.unsafe_offset((r0 - 1) * b * V), p, self.acts[0].x.unsafe_offset(r0 * b * D), Int32(b), Int32(V), Int32(D),
                    self.itemp, grid_dim=grid(b * D), block_dim=THREADS)
            if kinds[r0] == GEN:
                ctx.enqueue_function[argmax_k](self.z.unsafe_offset((r0 - 1) * b * V), ids.unsafe_offset(r0 * b), Int32(b), Int32(V), grid_dim=grid(b), block_dim=THREADS)
                ctx.enqueue_function[embed_k](p, ids.unsafe_offset(r0 * b), self.acts[0].x.unsafe_offset(r0 * b * D), Int32(b), Int32(D), grid_dim=grid(b * D), block_dim=THREADS)
            self.fwd_seg(ctx, p, tgt, wr, r0, segs[i + 1], True)
        if not train:
            return
        ctx.enqueue_memset(DeviceBuffer[DType.float32](ctx, self.dnf, S * b * D, owning=False), 0)
        comptime if not NOMS:
            for l in range(self.nl):
                ctx.enqueue_memset(DeviceBuffer[DType.float32](ctx, self.acts[l].dk, S * b * D, owning=False), 0)
                ctx.enqueue_memset(DeviceBuffer[DType.float32](ctx, self.acts[l].dv, S * b * D, owning=False), 0)
        for i in reversed(range(len(segs) - 1)):
            var r0 = segs[i]
            var r1 = segs[i + 1]
            self.bwd_seg(ctx, p, g, r0, r1, tgt, wr, NOMS and i == len(segs) - 2)
            var nr = (r1 - r0) * b
            if self.frz:  # frozen base: no embedding gradient (toks arm only: no pause, latent or soft rows)
                continue
            if self.esort > 0 and len(segs) == 2:
                ctx.enqueue_function[embed_srt_k](I(unsafe_from_address=Int(self.eord.unsafe_ptr())), I(unsafe_from_address=Int(self.estart.unsafe_ptr())),
                    self.dx, g, grid_dim=(V, D // 256), block_dim=256)
            else:
                ctx.enqueue_function[embed_bwd_k](ids.unsafe_offset(r0 * b), self.dx.unsafe_offset(r0 * b * D), self.part, Int32(nr), Int32(V),
                    grid_dim=(ceildiv(nr, EC), D // 256, V // 32), block_dim=256)
                ctx.enqueue_function[colsum_k](self.part, g, Int32(ceildiv(nr, EC)), Int32(V * D), grid_dim=grid(V * D), block_dim=THREADS)
            var j = 0
            for r in range(S):
                if kinds[r] == PSE:
                    if r >= r0 and r < r1:
                        ctx.enqueue_function[colsum_k](self.dx.unsafe_offset(r * b * D), g.unsafe_offset(PAUSE + j * D), Int32(b), Int32(D),
                            grid_dim=grid(D), block_dim=THREADS)
                    j += 1
            if kinds[r0] == LAT:
                self.copy(ctx, self.dx.unsafe_offset(r0 * b * D), self.dnf.unsafe_offset((r0 - 1) * b * D), b * D, True)
            if kinds[r0] == NEU:
                self.lk.bwd[Self.exact](ctx, p, g, self.dx.unsafe_offset(r0 * b * D), self.nf.unsafe_offset((r0 - 1) * b * D),
                    self.dnf.unsafe_offset((r0 - 1) * b * D), hop[r0])
            if kinds[0] == ENT and i == 0:
                self.lk.ent_bwd(ctx, g, self.dx)
            if kinds[r0] == SOFT and not self.sg:
                ctx.enqueue_function[soft_dz_k](self.z.unsafe_offset((r0 - 1) * b * V), p, self.dx.unsafe_offset(r0 * b * D),
                    self.dz.unsafe_offset((r0 - 1) * b * V), self.sp, Int32(b), self.itemp, grid_dim=grid(b), block_dim=THREADS)
                ctx.enqueue_function[soft_de_k](self.sp, self.dx.unsafe_offset(r0 * b * D), g, Int32(b), grid_dim=grid(V * D), block_dim=THREADS)


def row_weights(kinds: List[Int], tgt: List[Int32], b: Int, aux: Float64, rw: Bool = False, base: Int = 0) -> List[Float32]:
    """Per-row loss weights, rows r * b + i: rows feeding a LAT/SOFT row weigh aux / n_aux when aux > 0, other targeted rows
    1 / n_main (rw: row j = r - base weighs (1 + j/7) / (12 b)). Shared by run.weigh and test_latent, which checks it against the oracle."""
    var S = len(kinds)
    var w = List[Float32](length=S * b, fill=0)
    var nm = 0
    var na = 0
    for r in range(S):
        var feeds = aux > 0 and r + 1 < S and (kinds[r + 1] == LAT or kinds[r + 1] == SOFT)
        for i in range(b):
            if tgt[r * b + i] >= 0:
                if feeds:
                    na += 1
                else:
                    nm += 1
    for r in range(S):
        var feeds = aux > 0 and r + 1 < S and (kinds[r + 1] == LAT or kinds[r + 1] == SOFT)
        for i in range(b):
            if tgt[r * b + i] < 0:
                continue
            if feeds:
                w[r * b + i] = Float32(aux / Float64(na))
            elif rw:
                w[r * b + i] = Float32((1 + Float64(r - base) / 7) / (Float64(b) * 12))
            else:
                w[r * b + i] = Float32(1 / Float64(nm))
    return w^
