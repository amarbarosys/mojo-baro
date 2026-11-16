# DFlash2 block-diffusion drafter (lane-dflash2 D2c). Semantics: llama.cpp src/models/dflash.cpp graph<false>
# (injection + draft block), build_dflash2_conv, build_dflash2_selector; common/speculative.cpp (lattice walk).
#   inject(feat rows, positions): g = rmsnorm(fc @ feat, enc.output_norm); per layer K = rope(rmsnorm_head(Wk g)),
#                                 V = Wv g, written to ring cell pos % RING (latest write per position wins)
#   draft(anchor, pos0):          tokens [anchor, MASK x 7] at pos0 .. pos0 + 7, target token_embd; 5 non-causal SWA
#                                 layers (two-tap convs around attention and FFN); output_norm; target LM head;
#                                 top-16 per row; gate = selector_hidden @ h; lattice -> lat_h [NB, COLS]
# Activations enter every q8 GEMM as bf16 (gemm_q8 A); the ring and the residual stay f32. eps 1e-6.
from std.ffi import c_ssize_t, external_call
from std.math import ceildiv, sqrt
from std.collections import Dict
from std.time import perf_counter_ns
from std.os import getenv

from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import TileTensor, TensorLayout, row_major

from model import VOCAB
from window import gemm_w, tens_f32, tens_bf16, tens_q8q, tens_q8s
from registry import quant_rows, gemm_q8dot
from matmul_skinny import amar_skinny_reduce, amar_skinny_reduce_swiglu_bf16, SM
from elementwise import amar_rmsnorm, amar_rmsnorm_cast, amar_embed_lookup_pos, EW_THREADS
from spark_kernels import amar_rope_plain
from dflash_kernels import amar_df_attn, amar_df_conv, amar_df_put, amar_df_lattice, amar_df_gather, amar_df_topk

comptime f32 = DType.float32
comptime bf16 = DType.bfloat16
comptime DH = 5120  # draft hidden = target hidden (the draft embeds with the target token_embd)
comptime DNL = 5
comptime DNQH = 32
comptime DNKVH = 8
comptime DHD = 128
comptime DQ = DNQH * DHD
comptime DKV = DNKVH * DHD
comptime DFFN = 17408
comptime DFEAT = 5 * DH  # target taps 5/19/33/47/61, concatenated
comptime DGS = 16
comptime DDYN = 4 * DH // DGS  # [2 side, 2 tap, groups]
comptime RING = 4096
comptime SWA = 2048
comptime NB = 8  # block_size: anchor + 7 masks
comptime TOPK = 16
comptime RANK = 256
comptime COLS = TOPK + TOPK * TOPK
comptime MASK = 248070
comptime FREQ = Float32(1e7)
comptime EPS = Float32(1e-6)
# per-layer tensor order in DFlash.lo
comptime LNAMES: Array[StaticString, 15] = [
    "attn_norm.weight", "attn_conv_proj.weight", "attn_conv_base", "attn_q.weight", "attn_k.weight",
    "attn_v.weight", "attn_q_norm.weight", "attn_k_norm.weight", "attn_output.weight", "ffn_norm.weight",
    "ffn_conv_proj.weight", "ffn_conv_base", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
]
comptime L_ANORM = 0
comptime L_ACP = 1
comptime L_ACB = 2
comptime L_Q = 3
comptime L_K = 4
comptime L_V = 5
comptime L_QN = 6
comptime L_KN = 7
comptime L_O = 8
comptime L_FNORM = 9
comptime L_FCP = 10
comptime L_FCB = 11
comptime L_G = 12
comptime L_U = 13
comptime L_D = 14


def tt[dt: DType, LT: TensorLayout](
    ctx: DeviceContext, b: DeviceBuffer[dt], o: Int, lt: LT
) -> TileTensor[dt, LT, MutAnyOrigin]:
    """TileTensor view of b at element offset o, shaped lt."""
    var v = DeviceBuffer[dt](ctx, b.unsafe_ptr().unsafe_offset(o), len(b) - o, owning=False)
    return rebind[TileTensor[dt, LT, MutAnyOrigin]](TileTensor(v, lt))


def ptr[dt: DType](b: DeviceBuffer[dt], o: Int = 0) -> Pointer[Scalar[dt], MutAnyOrigin]:
    return Pointer[Scalar[dt], MutAnyOrigin](unsafe_from_address=Int(b.unsafe_ptr().unsafe_offset(o)))


def wptr[dt: DType](w: DeviceBuffer[DType.uint8], byte_off: Int) -> Pointer[Scalar[dt], MutAnyOrigin]:
    return Pointer[Scalar[dt], MutAnyOrigin](unsafe_from_address=Int(w.unsafe_ptr()) + byte_off)


def read_index(dir: String) raises -> Dict[String, Tuple[String, Int, Int]]:
    """index.txt -> name: (dtype, byte offset, numel)."""
    var d = Dict[String, Tuple[String, Int, Int]]()
    with open(dir + "/index.txt", "r") as f:
        for line in f.read().splitlines():
            var p = line.split(" ")
            if len(p) >= 4:
                d[String(p[0])] = (String(p[1]), Int(p[2]), Int(p[3]))
    return d^


def nbytes(dt: String, n: Int) raises -> Int:
    if dt == "q8":
        return n + n // 16
    if dt == "q4":
        return n // 2 + n // 16
    if dt == "bf16":
        return 2 * n
    if dt == "f32":
        return 4 * n
    raise Error("dflash: no size rule for dtype " + dt)


def dev_load(ctx: DeviceContext, path: String, offs: List[Int], sizes: List[Int]) raises -> DeviceBuffer[DType.uint8]:
    """Byte ranges [offs[t], offs[t] + sizes[t]) of path, back to back, in one device buffer."""
    var total = 0
    for s in sizes:
        total += s
    var d = ctx.enqueue_create_buffer[DType.uint8](total)
    comptime CH = 1 << 26
    var st = ctx.enqueue_create_host_buffer[DType.uint8](CH)
    var dst = 0
    with open(path, "r") as f:
        var fd = f._get_raw_fd()
        for t in range(len(offs)):
            var done = 0
            while done < sizes[t]:
                var want = min(CH, sizes[t] - done)
                ctx.synchronize()
                var got = 0
                while got < want:
                    var n = external_call["pread", c_ssize_t](
                        fd, st.unsafe_ptr().unsafe_offset(got), want - got, Int64(offs[t] + done + got)
                    )
                    if n <= 0:
                        raise Error("dflash: short read " + path)
                    got += Int(n)
                ctx.enqueue_copy(
                    dst_buf=DeviceBuffer[DType.uint8](ctx, d.unsafe_ptr().unsafe_offset(dst), want, owning=False),
                    src_buf=st.create_sub_buffer[DType.uint8](0, want),
                )
                done += want
                dst += want
    ctx.synchronize()
    return d^


comptime x_l = row_major[SM, DH]()
comptime h_l = row_major[DH]()
comptime hd_l = row_major[DHD]()
comptime qh_l = row_major[SM * DNQH, DHD]()
comptime kh_l = row_major[SM * DNKVH, DHD]()
comptime k1_l = row_major[DNKVH, DHD]()
comptime ffn_l = row_major[SM, DFFN]()
comptime pffn_l = row_major[1, SM, DFFN]()
comptime v_l = row_major[SM, VOCAB]()
comptime tok_l = row_major[SM]()
comptime emb_l = row_major[VOCAB, DH]()
comptime rms_x = amar_rmsnorm[type_of(x_l), type_of(h_l), type_of(x_l)]
comptime rmsc_x = amar_rmsnorm_cast[type_of(x_l), type_of(h_l), type_of(x_l)]
comptime rms_q = amar_rmsnorm[type_of(qh_l), type_of(hd_l), type_of(qh_l)]
comptime rms_k = amar_rmsnorm[type_of(kh_l), type_of(hd_l), type_of(kh_l)]
comptime rope_q = amar_rope_plain[DHD, type_of(qh_l), NEOX=True]
comptime rope_k = amar_rope_plain[DHD, type_of(kh_l), NEOX=True]
comptime rope_k1 = amar_rope_plain[DHD, type_of(k1_l), NEOX=True]
comptime swiglu_b = amar_skinny_reduce_swiglu_bf16[type_of(pffn_l), type_of(ffn_l), 1]
comptime embed_d = amar_embed_lookup_pos[type_of(emb_l), type_of(x_l), type_of(tok_l)]
comptime attn_d = amar_df_attn[DHD, DNQH, DNKVH, RING, SWA]
comptime put_d = amar_df_put[DKV]
comptime gather_d = amar_df_gather[DH, DFEAT // DH, SM]
comptime lat_d = amar_df_lattice[TOPK, RANK]
comptime topk_d = amar_df_topk[TOPK, 256]
comptime conv_b = amar_df_conv[DH, DGS, bf16, False]
comptime conv_acc = amar_df_conv[DH, DGS, f32, True]


struct DFlash(Movable):
    var dw: DeviceBuffer[DType.uint8]  # draft pack (all q8 matrices + f32 norms / conv bases)
    var lo: List[Int]  # byte offsets, layer l tensor j at lo[l * 15 + j] (LNAMES order)
    var fc: Int
    var enc_norm: Int
    var out_norm: Int
    var sel_h: Int
    var sel_s: Int
    var sel_p: Int
    var tw: DeviceBuffer[DType.uint8]  # target weights holding token_embd (bf16) and the LM head
    var emb_off: Int
    var head_off: Int
    var head_kind: Int  # gemm_w kind: 1 q4, 0 q8
    var kr: DeviceBuffer[f32]  # [DNL, RING, DKV]
    var vr: DeviceBuffer[f32]
    var kpos: List[Int32]  # position held by each ring cell, -1 empty (host truth, uploaded per call)
    var kpos_d: DeviceBuffer[DType.int32]
    var featb: DeviceBuffer[bf16]  # [SM, DFEAT]
    var h: DeviceBuffer[f32]  # [SM, DH] residual (draft) / fc output (inject)
    var x0: DeviceBuffer[f32]  # [SM, DH] normed input of a conv
    var xb: DeviceBuffer[bf16]  # [SM, DH] bf16 GEMM input
    var dyn: DeviceBuffer[f32]  # [SM, DDYN]
    var q: DeviceBuffer[f32]  # [SM, DQ]
    var k: DeviceBuffer[f32]  # [SM, DKV]
    var v: DeviceBuffer[f32]
    var ao: DeviceBuffer[bf16]  # [SM, DQ]
    var o: DeviceBuffer[f32]  # [SM, DH]
    var p: DeviceBuffer[f32]  # GEMM partials [1, SM, n <= VOCAB]
    var p2: DeviceBuffer[f32]  # [1, SM, DFFN] (up)
    var fb: DeviceBuffer[bf16]  # [SM, DFFN]
    var logits: DeviceBuffer[f32]  # [SM, VOCAB]
    var gate: DeviceBuffer[f32]  # [SM, RANK]
    var cand: DeviceBuffer[DType.int32]  # [NB, TOPK]
    var unary: DeviceBuffer[f32]  # [NB, TOPK]
    var lat: DeviceBuffer[f32]  # [NB, COLS]
    var toks: DeviceBuffer[DType.int32]  # [SM]
    var cells: DeviceBuffer[DType.int32]  # [SM]
    var featb_h: HostBuffer[bf16]
    var lat_h: HostBuffer[f32]
    var toks_h: HostBuffer[DType.int32]
    var cells_h: HostBuffer[DType.int32]
    var kpos_h: HostBuffer[DType.int32]
    var dot: Bool  # BARO_DF_DOT (default 1): q8 draft GEMMs on the int8-dot kernel, activations quantized per 32
    var aq: DeviceBuffer[DType.int8]  # [SM, DFEAT] quantized A
    var asc: DeviceBuffer[DType.float16]  # [SM, DFEAT / 32]
    var prof: Bool  # BARO_DF_PROF=1: sync timers per draft stage (serialises the draft)
    var pt: List[Int]  # ns: embed, attn, ffn, head, select, inject; [6] = drafts
    var t0: Int

    def __init__(
        out self, ctx: DeviceContext, draft_dir: String, var tw: DeviceBuffer[DType.uint8], emb_off: Int,
        head_off: Int, head_kind: Int,
    ) raises:
        var ix = read_index(draft_dir)
        var size = 0
        for e in ix.items():
            size = max(size, e.value[1] + nbytes(e.value[0], e.value[2]))
        self.dw = dev_load(ctx, draft_dir + "/pack.bin", [0], [size])
        self.lo = List[Int]()
        for l in range(DNL):
            comptime for j in range(15):
                comptime nm = LNAMES[j]
                self.lo.append(ix["blk." + String(l) + "." + nm][1])
        self.fc = ix["fc.weight"][1]
        self.enc_norm = ix["enc.output_norm.weight"][1]
        self.out_norm = ix["output_norm.weight"][1]
        self.sel_h = ix["selector_hidden.weight"][1]
        self.sel_s = ix["selector_successor.weight"][1]
        self.sel_p = ix["selector_predecessor.weight"][1]
        self.tw = tw^
        self.emb_off = emb_off
        self.head_off = head_off
        self.head_kind = head_kind
        self.kr = ctx.enqueue_create_buffer[f32](DNL * RING * DKV)
        self.vr = ctx.enqueue_create_buffer[f32](DNL * RING * DKV)
        self.kpos = List[Int32](length=RING, fill=-1)
        self.kpos_d = ctx.enqueue_create_buffer[DType.int32](RING)
        self.featb = ctx.enqueue_create_buffer[bf16](SM * DFEAT)
        self.h = ctx.enqueue_create_buffer[f32](SM * DH)
        self.x0 = ctx.enqueue_create_buffer[f32](SM * DH)
        self.xb = ctx.enqueue_create_buffer[bf16](SM * DH)
        self.dyn = ctx.enqueue_create_buffer[f32](SM * DDYN)
        self.q = ctx.enqueue_create_buffer[f32](SM * DQ)
        self.k = ctx.enqueue_create_buffer[f32](SM * DKV)
        self.v = ctx.enqueue_create_buffer[f32](SM * DKV)
        self.ao = ctx.enqueue_create_buffer[bf16](SM * DQ)
        self.o = ctx.enqueue_create_buffer[f32](SM * DH)
        self.p = ctx.enqueue_create_buffer[f32](SM * VOCAB)
        self.p2 = ctx.enqueue_create_buffer[f32](SM * DFFN)
        self.fb = ctx.enqueue_create_buffer[bf16](SM * DFFN)
        self.logits = ctx.enqueue_create_buffer[f32](SM * VOCAB)
        self.gate = ctx.enqueue_create_buffer[f32](SM * RANK)
        self.cand = ctx.enqueue_create_buffer[DType.int32](NB * TOPK)
        self.unary = ctx.enqueue_create_buffer[f32](NB * TOPK)
        self.lat = ctx.enqueue_create_buffer[f32](NB * COLS)
        self.toks = ctx.enqueue_create_buffer[DType.int32](SM)
        self.cells = ctx.enqueue_create_buffer[DType.int32](SM)
        self.featb_h = ctx.enqueue_create_host_buffer[bf16](SM * DFEAT)
        self.lat_h = ctx.enqueue_create_host_buffer[f32](NB * COLS)
        self.toks_h = ctx.enqueue_create_host_buffer[DType.int32](SM)
        self.cells_h = ctx.enqueue_create_host_buffer[DType.int32](SM)
        self.kpos_h = ctx.enqueue_create_host_buffer[DType.int32](RING)
        self.prof = getenv("BARO_DF_PROF", "0") == "1"
        self.dot = getenv("BARO_DF_DOT", "1") == "1"
        self.aq = ctx.enqueue_create_buffer[DType.int8](SM * DFEAT)
        self.asc = ctx.enqueue_create_buffer[DType.float16](SM * DFEAT // 32)
        self.pt = List[Int](length=7, fill=0)
        self.t0 = 0
        ctx.synchronize()
        for i in range(NB * COLS):
            self.lat_h[i] = 0
        ctx.enqueue_copy(dst_buf=self.lat, src_buf=self.lat_h)  # row 0 stays zero: the lattice kernel writes rows >= 1
        ctx.synchronize()

    def wl(self, l: Int, j: Int) -> Int:
        return self.lo[l * 15 + j]

    def proj[N: Int, K: Int](
        self, ctx: DeviceContext, a: DeviceBuffer[bf16], w: DeviceBuffer[DType.uint8], o: Int, kind: Int,
        c: DeviceBuffer[f32], m: Int,
    ) raises:
        """c[m, N] (f32, row stride N) = a[m, K] (bf16) @ W[N, K]^T."""
        comptime p_l = row_major[1, SM, N]()
        comptime c_l = row_major[SM, N]()
        self.gemm[N, K](ctx, a, w, o, kind, self.p, m)
        ctx.enqueue_function[amar_skinny_reduce[type_of(p_l), type_of(c_l), 1]](
            tt(ctx, self.p, 0, p_l), tt(ctx, c, 0, c_l), Int32(m), Int32(N),
            grid_dim=ceildiv(m * N, 256), block_dim=256,
        )

    def gemm[N: Int, K: Int](
        self, ctx: DeviceContext, a: DeviceBuffer[bf16], w: DeviceBuffer[DType.uint8], o: Int, kind: Int,
        p: DeviceBuffer[f32], m: Int,
    ) raises:
        """Partials p[0, m, N] of a[m, K] @ W[N, K]^T: q8 weights on the int8-dot kernel when dot is on, else gemm_w."""
        comptime a_l = row_major[SM, K]()
        comptime p_l = row_major[1, SM, N]()
        if kind == 0 and self.dot:
            comptime aq_l = row_major[SM, K]()
            comptime as_l = row_major[SM, K // 32]()
            var Aq = tt(ctx, self.aq, 0, aq_l)
            var As = tt(ctx, self.asc, 0, as_l)
            quant_rows(ctx, tt(ctx, a, 0, a_l), Aq, As, m, K)
            gemm_q8dot(
                ctx, Aq, As, tens_q8q(ctx, w, o, N * K, row_major[N, K]()), tens_q8s(ctx, w, o, N * K, row_major[N, K // 32]()),
                tt(ctx, p, 0, p_l), m, N, K,
            )
        else:
            gemm_w[N, K](ctx, tt(ctx, a, 0, a_l), w, o, kind, tt(ctx, p, 0, p_l), m)

    def put_kpos(mut self, pos: Int):
        self.kpos[pos % RING] = Int32(pos)

    def upload_kpos(mut self, ctx: DeviceContext) raises:
        for i in range(RING):
            self.kpos_h[i] = self.kpos[i]
        ctx.enqueue_copy(dst_buf=self.kpos_d, src_buf=self.kpos_h)

    def inject(
        mut self, ctx: DeviceContext, feat: Pointer[Float32, MutAnyOrigin], pos: List[Int],
        tap: Optional[DeviceBuffer[f32]] = None,
    ) raises:
        """Feature rows at positions pos[n] into every layer's ring: host f32 feat[n, DFEAT], or (tap given) window
        rows 0 .. n - 1 of the target taps, f32 [5, SM, DH] as window.mojo's tap_d holds them."""
        var ti = self.start(ctx)
        var n = len(pos)
        for c0 in range(0, n, SM):
            var m = min(SM, n - c0)
            ctx.synchronize()  # featb_h / cells_h free again
            if tap:
                ctx.enqueue_function[gather_d](
                    ptr(tap.value()), ptr(self.featb), Int32(c0), Int32(m), grid_dim=ceildiv(m * DFEAT, 256),
                    block_dim=256,
                )
            else:
                for i in range(m * DFEAT):
                    self.featb_h[i] = feat[c0 * DFEAT + i].cast[bf16]()
                ctx.enqueue_copy(dst_buf=self.featb, src_buf=self.featb_h)
            for r in range(m):
                self.cells_h[r] = Int32(pos[c0 + r] % RING)
                self.put_kpos(pos[c0 + r])
            ctx.enqueue_copy(dst_buf=self.cells, src_buf=self.cells_h)
            self.proj[DH, DFEAT](ctx, self.featb, self.dw, self.fc, 0, self.h, m)
            ctx.enqueue_function[rmsc_x](
                tt(ctx, self.h, 0, x_l), tens_f32(ctx, self.dw, self.enc_norm, DH, h_l), tt(ctx, self.xb, 0, x_l),
                Int32(DH), EPS, grid_dim=m, block_dim=EW_THREADS,
            )
            for l in range(DNL):
                self.proj[DKV, DH](ctx, self.xb, self.dw, self.wl(l, L_K), 0, self.k, m)
                self.proj[DKV, DH](ctx, self.xb, self.dw, self.wl(l, L_V), 0, self.v, m)
                ctx.enqueue_function[rms_k](
                    tt(ctx, self.k, 0, kh_l), tens_f32(ctx, self.dw, self.wl(l, L_KN), DHD, hd_l), tt(ctx, self.k, 0, kh_l),
                    Int32(DHD), EPS, grid_dim=m * DNKVH, block_dim=EW_THREADS,
                )
                for r in range(m):
                    ctx.enqueue_function[rope_k1](
                        tt(ctx, self.k, r * DKV, k1_l), Int32(pos[c0 + r]), Int32(DNKVH), FREQ, ptr(self.q),
                        grid_dim=(DNKVH, 1), block_dim=DHD // 2,
                    )
                ctx.enqueue_function[put_d](
                    ptr(self.k), ptr(self.v), ptr(self.kr, l * RING * DKV), ptr(self.vr, l * RING * DKV),
                    ptr(self.cells), grid_dim=m, block_dim=256,
                )
        self.upload_kpos(ctx)
        ctx.synchronize()
        if self.prof:
            self.pt[5] += Int(perf_counter_ns()) - ti

    def norm2(self, ctx: DeviceContext, src: DeviceBuffer[f32], g_off: Int, m: Int) raises:
        """x0 = rmsnorm(src, g) in f32 (the conv input) and xb = the same in bf16 (the conv-proj GEMM input)."""
        var G = tens_f32(ctx, self.dw, g_off, DH, h_l)
        ctx.enqueue_function[rms_x](
            tt(ctx, src, 0, x_l), G, tt(ctx, self.x0, 0, x_l), Int32(DH), EPS, grid_dim=m, block_dim=EW_THREADS
        )
        ctx.enqueue_function[rmsc_x](
            tt(ctx, src, 0, x_l), G, tt(ctx, self.xb, 0, x_l), Int32(DH), EPS, grid_dim=m, block_dim=EW_THREADS
        )

    def conv[ODT: DType, ACC: Bool](
        self, ctx: DeviceContext, x: DeviceBuffer[f32], base_off: Int, dst: DeviceBuffer[ODT], side: Int
    ) raises:
        ctx.enqueue_function[amar_df_conv[DH, DGS, ODT, ACC]](
            ptr(x), ptr(self.dyn), wptr[f32](self.dw, base_off), ptr(dst), Int32(NB), Int32(side),
            grid_dim=ceildiv(NB * DH, 256), block_dim=256,
        )

    def draft(mut self, ctx: DeviceContext, anchor: Int, pos0: Int) raises:
        """One block [anchor, MASK x 7] at pos0 .. pos0 + NB - 1; the lattice lands in lat_h [NB, COLS], row 0 zero.
        The block's own K/V go to the ring first (llama's build_attn stores them before attending)."""
        ctx.synchronize()
        self.t0 = Int(perf_counter_ns())
        self.pt[6] += 1
        for r in range(NB):
            self.toks_h[r] = Int32(anchor) if r == 0 else Int32(MASK)
            self.cells_h[r] = Int32((pos0 + r) % RING)
            self.put_kpos(pos0 + r)
        ctx.enqueue_copy(dst_buf=self.toks, src_buf=self.toks_h)
        ctx.enqueue_copy(dst_buf=self.cells, src_buf=self.cells_h)
        self.upload_kpos(ctx)
        ctx.enqueue_function[embed_d](
            tens_bf16(ctx, self.tw, self.emb_off, VOCAB * DH, emb_l), tt(ctx, self.h, 0, x_l),
            tt(ctx, self.toks, 0, tok_l), Int32(0), Int32(DH), grid_dim=(ceildiv(DH, 256), NB), block_dim=256,
        )
        var scale = Float32(1.0) / sqrt(Float32(DHD))
        self.tick(ctx, 0)
        for l in range(DNL):
            self.norm2(ctx, self.h, self.wl(l, L_ANORM), NB)
            self.proj[DDYN, DH](ctx, self.xb, self.dw, self.wl(l, L_ACP), 0, self.dyn, NB)
            self.conv[bf16, False](ctx, self.x0, self.wl(l, L_ACB), self.xb, 0)
            self.proj[DQ, DH](ctx, self.xb, self.dw, self.wl(l, L_Q), 0, self.q, NB)
            self.proj[DKV, DH](ctx, self.xb, self.dw, self.wl(l, L_K), 0, self.k, NB)
            self.proj[DKV, DH](ctx, self.xb, self.dw, self.wl(l, L_V), 0, self.v, NB)
            ctx.enqueue_function[rms_q](
                tt(ctx, self.q, 0, qh_l), tens_f32(ctx, self.dw, self.wl(l, L_QN), DHD, hd_l), tt(ctx, self.q, 0, qh_l),
                Int32(DHD), EPS, grid_dim=NB * DNQH, block_dim=EW_THREADS,
            )
            ctx.enqueue_function[rms_k](
                tt(ctx, self.k, 0, kh_l), tens_f32(ctx, self.dw, self.wl(l, L_KN), DHD, hd_l), tt(ctx, self.k, 0, kh_l),
                Int32(DHD), EPS, grid_dim=NB * DNKVH, block_dim=EW_THREADS,
            )
            ctx.enqueue_function[rope_q](
                tt(ctx, self.q, 0, qh_l), Int32(pos0), Int32(DNQH), FREQ, ptr(self.q),
                grid_dim=(DNQH, NB), block_dim=DHD // 2,
            )
            ctx.enqueue_function[rope_k](
                tt(ctx, self.k, 0, kh_l), Int32(pos0), Int32(DNKVH), FREQ, ptr(self.q),
                grid_dim=(DNKVH, NB), block_dim=DHD // 2,
            )
            ctx.enqueue_function[put_d](
                ptr(self.k), ptr(self.v), ptr(self.kr, l * RING * DKV), ptr(self.vr, l * RING * DKV),
                ptr(self.cells), grid_dim=NB, block_dim=256,
            )
            ctx.enqueue_function[attn_d](
                ptr(self.q), ptr(self.kr, l * RING * DKV), ptr(self.vr, l * RING * DKV), ptr(self.kpos_d),
                ptr(self.ao), Int32(pos0), Int32(NB), scale, grid_dim=(DNQH, NB), block_dim=DHD,
            )
            self.proj[DH, DQ](ctx, self.ao, self.dw, self.wl(l, L_O), 0, self.o, NB)
            self.conv[f32, True](ctx, self.o, self.wl(l, L_ACB), self.h, 1)
            self.tick(ctx, 1)
            self.norm2(ctx, self.h, self.wl(l, L_FNORM), NB)
            self.proj[DDYN, DH](ctx, self.xb, self.dw, self.wl(l, L_FCP), 0, self.dyn, NB)
            self.conv[bf16, False](ctx, self.x0, self.wl(l, L_FCB), self.xb, 0)
            self.gemm[DFFN, DH](ctx, self.xb, self.dw, self.wl(l, L_G), 0, self.p, NB)
            self.gemm[DFFN, DH](ctx, self.xb, self.dw, self.wl(l, L_U), 0, self.p2, NB)
            ctx.enqueue_function[swiglu_b](
                tt(ctx, self.p, 0, pffn_l), tt(ctx, self.p2, 0, pffn_l), tt(ctx, self.fb, 0, ffn_l), Int32(NB),
                Int32(DFFN), grid_dim=ceildiv(NB * DFFN, 256), block_dim=256,
            )
            self.proj[DH, DFFN](ctx, self.fb, self.dw, self.wl(l, L_D), 0, self.o, NB)
            self.conv[f32, True](ctx, self.o, self.wl(l, L_FCB), self.h, 1)
            self.tick(ctx, 2)
        ctx.enqueue_function[rmsc_x](
            tt(ctx, self.h, 0, x_l), tens_f32(ctx, self.dw, self.out_norm, DH, h_l), tt(ctx, self.xb, 0, x_l),
            Int32(DH), EPS, grid_dim=NB, block_dim=EW_THREADS,
        )
        self.proj[VOCAB, DH](ctx, self.xb, self.tw, self.head_off, self.head_kind, self.logits, NB)
        self.tick(ctx, 3)
        self.proj[RANK, DH](ctx, self.xb, self.dw, self.sel_h, 0, self.gate, NB)
        ctx.enqueue_function[topk_d](
            ptr(self.logits), ptr(self.cand), ptr(self.unary), Int32(VOCAB), grid_dim=NB - 1, block_dim=256
        )
        comptime NSEL = VOCAB * RANK
        ctx.enqueue_function[lat_d](
            ptr(self.cand), ptr(self.unary), ptr(self.gate), wptr[DType.int8](self.dw, self.sel_s),
            wptr[DType.float16](self.dw, self.sel_s + NSEL), wptr[DType.int8](self.dw, self.sel_p),
            wptr[DType.float16](self.dw, self.sel_p + NSEL), Int32(anchor), ptr(self.lat), Int32(COLS),
            grid_dim=(NB - 1, TOPK), block_dim=TOPK,
        )
        ctx.enqueue_copy(dst_buf=self.lat_h, src_buf=self.lat)
        ctx.synchronize()
        self.tick(ctx, 4)

    def start(self, ctx: DeviceContext) raises -> Int:
        if self.prof:
            ctx.synchronize()
        return Int(perf_counter_ns())

    def tick(mut self, ctx: DeviceContext, i: Int) raises:
        if self.prof:
            ctx.synchronize()
            var now = Int(perf_counter_ns())
            self.pt[i] += now - self.t0
            self.t0 = now

    def prof_line(mut self) -> String:
        """Mean ms per draft of each stage since the last call; resets the counters."""
        var names: List[String] = ["embed", "attn", "ffn", "head", "select", "inject"]
        var s = String("dflash2 prof ms/draft (") + String(self.pt[6]) + " drafts):"
        for i in range(6):
            s += " " + names[i] + " " + String(Float64(self.pt[i]) / 1e6 / Float64(max(self.pt[6], 1)))
        for i in range(7):
            self.pt[i] = 0
        return s

    def walk(self) -> List[Int]:
        """Greedy walk of lat_h (speculative.cpp:1419-1466, p_min 0): the NB - 1 draft ids."""
        var ids = List[Int]()
        var pred = 0
        for i in range(1, NB):
            var row = i * COLS + TOPK + pred * TOPK
            var best = 0
            for j in range(1, TOPK):
                if self.lat_h[row + j] > self.lat_h[row + best]:
                    best = j
            pred = best
            ids.append(Int(self.lat_h[i * COLS + pred]))
        return ids^

    def reset(mut self):
        for i in range(RING):
            self.kpos[i] = -1
