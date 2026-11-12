from max.gpu import MAX_THREADS_PER_BLOCK_METADATA
from max.gpu import block_idx, global_idx, lane_id, thread_idx, WARP_SIZE
from max.gpu.primitives import warp
from std.math import log1p, sqrt
from bexp import exp, rsqrt
from std.sys import get_defined_int
from std.utils import StaticTuple
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from layout import TileTensor, TensorLayout, row_major, stack_allocation

from matmul_skinny import SPLITK

comptime f32 = DType.float32
comptime NH_K = get_defined_int["BARO_SSM_NH_K", 16]()
comptime NH_V = get_defined_int["BARO_SSM_NH_V", 32]()
comptime SSTATE = get_defined_int["BARO_SSM_STATE", 128]()
comptime KDIM = NH_K * SSTATE
comptime CONV = 2 * KDIM + NH_V * SSTATE
# amar_ssm_conv_l2g grid: one block per 256 conv channels, also one per v-head for the gates.
comptime CONV_GATE_BLOCKS = max(CONV // 256, NH_V)
comptime SSM_EPS = Float32(1e-6)


def amar_cast_bf16[
    XLayout: TensorLayout, OLayout: TensorLayout
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    O: TileTensor[DType.bfloat16, OLayout, MutAnyOrigin],
    n: Int32,
):
    comptime assert X.flat_rank == 1 and O.flat_rank == 1
    var i = global_idx.x
    if i < Int(n):
        O[i] = rebind[O.ElementType](
            rebind[Scalar[f32]](X[i]).cast[DType.bfloat16]()
        )


def amar_widen_bf16[
    XLayout: TensorLayout, OLayout: TensorLayout
](
    X: TileTensor[DType.bfloat16, XLayout, MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    n: Int32,
):
    comptime assert X.flat_rank == 1 and O.flat_rank == 1
    var i = global_idx.x
    if i < Int(n):
        O[i] = rebind[Scalar[f32]](X[i].cast[f32]())


def amar_residual_add[
    XLayout: TensorLayout, YLayout: TensorLayout
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    Y: TileTensor[f32, YLayout, MutAnyOrigin],
    n: Int32,
):
    comptime assert X.flat_rank == 1 and Y.flat_rank == 1
    var i = global_idx.x
    if i < Int(n):
        Y[i] = rebind[Y.ElementType](
            rebind[Scalar[f32]](Y[i]) + rebind[Scalar[f32]](X[i])
        )


def amar_ssm_gates[
    ALayout: TensorLayout, BLayout: TensorLayout, GLayout: TensorLayout,
    WLayout: TensorLayout, DLayout: TensorLayout
](
    AlphaRaw: TileTensor[f32, ALayout, MutAnyOrigin],
    BetaRaw: TileTensor[f32, BLayout, MutAnyOrigin],
    EgOut: TileTensor[f32, GLayout, MutAnyOrigin],
    BetaOut: TileTensor[f32, WLayout, MutAnyOrigin],
    SsmA: TileTensor[f32, DLayout, MutAnyOrigin],
    DtBias: TileTensor[f32, DLayout, MutAnyOrigin],
):
    comptime assert AlphaRaw.flat_rank == 1 and BetaRaw.flat_rank == 1
    comptime assert EgOut.flat_rank == 1 and BetaOut.flat_rank == 1
    comptime assert SsmA.flat_rank == 1 and DtBias.flat_rank == 1
    var h = global_idx.x
    if h >= NH_V:
        return
    var braw = rebind[Scalar[f32]](BetaRaw[h])
    BetaOut[h] = rebind[BetaOut.ElementType](1 / (1 + exp(-braw)))
    var araw = rebind[Scalar[f32]](AlphaRaw[h]) + rebind[Scalar[f32]](DtBias[h])
    var sp = log1p(exp(araw))
    EgOut[h] = rebind[EgOut.ElementType](exp(sp * rebind[Scalar[f32]](SsmA[h])))


def amar_ssm_reduce_gates[
    PLayout: TensorLayout, GLayout: TensorLayout, DLayout: TensorLayout
](
    Ap: TileTensor[f32, PLayout, MutAnyOrigin],
    Bp: TileTensor[f32, PLayout, MutAnyOrigin],
    EgOut: TileTensor[f32, GLayout, MutAnyOrigin],
    BetaOut: TileTensor[f32, GLayout, MutAnyOrigin],
    SsmA: TileTensor[f32, DLayout, MutAnyOrigin],
    DtBias: TileTensor[f32, DLayout, MutAnyOrigin],
    m: Int32,
):
    comptime assert Ap.flat_rank == 3 and Bp.flat_rank == 3
    comptime assert EgOut.flat_rank == 2 and BetaOut.flat_rank == 2
    comptime assert SsmA.flat_rank == 1 and DtBias.flat_rank == 1
    var h = Int(thread_idx.x)
    var r = Int(block_idx.x)
    if h >= NH_V or r >= Int(m):
        return
    var sa = rebind[Scalar[f32]](SsmA[h])
    var db = rebind[Scalar[f32]](DtBias[h])
    var araw: Scalar[f32] = 0
    var braw: Scalar[f32] = 0
    comptime for s in range(SPLITK):
        araw += rebind[Scalar[f32]](Ap[s, r, h])
        braw += rebind[Scalar[f32]](Bp[s, r, h])
    BetaOut[r, h] = rebind[BetaOut.ElementType](1 / (1 + exp(-braw)))
    var asum = araw + db
    var sp = log1p(exp(asum))
    EgOut[r, h] = rebind[EgOut.ElementType](exp(sp * sa))


def amar_ssm_conv[
    QLayout: TensorLayout, SLayout: TensorLayout, WLayout: TensorLayout,
    OLayout: TensorLayout
](
    Qkv: TileTensor[f32, QLayout, MutAnyOrigin],
    ConvState: TileTensor[f32, SLayout, MutAnyOrigin],
    ConvW: TileTensor[f32, WLayout, MutAnyOrigin],
    Out: TileTensor[f32, OLayout, MutAnyOrigin],
    ring: Int32, ssm_i: Int32, slots: Int32, m: Int32,
):
    comptime assert Qkv.flat_rank == 2 and ConvState.flat_rank == 4
    comptime assert ConvW.flat_rank == 2 and Out.flat_rank == 2
    var c = global_idx.x
    if c >= CONV:
        return
    var si = Int(ssm_i)
    var sl = Int(slots)
    var rg = Int(ring)
    var cw0 = rebind[Scalar[f32]](ConvW[c, 0])
    var cw1 = rebind[Scalar[f32]](ConvW[c, 1])
    var cw2 = rebind[Scalar[f32]](ConvW[c, 2])
    var cw3 = rebind[Scalar[f32]](ConvW[c, 3])
    for r in range(Int(m)):
        var rs = (rg + r) % sl
        var ws = (rg + r + 1) % sl
        var w0 = rebind[Scalar[f32]](ConvState[rs, si, 0, c])
        var w1 = rebind[Scalar[f32]](ConvState[rs, si, 1, c])
        var w2 = rebind[Scalar[f32]](ConvState[rs, si, 2, c])
        var w3 = rebind[Scalar[f32]](Qkv[r, c])
        var acc = w0 * cw0 + w1 * cw1 + w2 * cw2 + w3 * cw3
        Out[r, c] = rebind[Out.ElementType](acc / (1 + exp(-acc)))
        ConvState[ws, si, 0, c] = rebind[ConvState.ElementType](w1)
        ConvState[ws, si, 1, c] = rebind[ConvState.ElementType](w2)
        ConvState[ws, si, 2, c] = rebind[ConvState.ElementType](w3)


def amar_ssm_conv_l2g[
    QLayout: TensorLayout, SLayout: TensorLayout, WLayout: TensorLayout, OLayout: TensorLayout,
    PLayout: TensorLayout, GLayout: TensorLayout, DLayout: TensorLayout
](
    Qkv: TileTensor[f32, QLayout, MutAnyOrigin],
    ConvState: TileTensor[f32, SLayout, MutAnyOrigin],
    ConvW: TileTensor[f32, WLayout, MutAnyOrigin],
    Out: TileTensor[f32, OLayout, MutAnyOrigin],
    Ap: TileTensor[f32, PLayout, MutAnyOrigin],
    Bp: TileTensor[f32, PLayout, MutAnyOrigin],
    EgOut: TileTensor[f32, GLayout, MutAnyOrigin],
    BetaOut: TileTensor[f32, GLayout, MutAnyOrigin],
    SsmA: TileTensor[f32, DLayout, MutAnyOrigin],
    DtBias: TileTensor[f32, DLayout, MutAnyOrigin],
    ring: Int32, ssm_i: Int32, slots: Int32, m: Int32,
):
    comptime assert Qkv.flat_rank == 2 and ConvState.flat_rank == 4
    comptime assert ConvW.flat_rank == 2 and Out.flat_rank == 2
    comptime assert Ap.flat_rank == 3 and Bp.flat_rank == 3
    comptime assert EgOut.flat_rank == 2 and BetaOut.flat_rank == 2
    comptime assert SsmA.flat_rank == 1 and DtBias.flat_rank == 1
    comptime assert CONV % 256 == 0 and (2 * KDIM) % 256 == 0 and SSTATE == 128
    var blk = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var c = blk * 256 + tid
    if tid == 0 and blk < NH_V:
        var sa = rebind[Scalar[f32]](SsmA[blk])
        var db = rebind[Scalar[f32]](DtBias[blk])
        for r in range(Int(m)):
            var araw: Scalar[f32] = 0
            var braw: Scalar[f32] = 0
            comptime for s in range(SPLITK):
                araw += rebind[Scalar[f32]](Ap[s, r, blk])
                braw += rebind[Scalar[f32]](Bp[s, r, blk])
            BetaOut[r, blk] = rebind[BetaOut.ElementType](1 / (1 + exp(-braw)))
            var asum = araw + db
            var sp = log1p(exp(asum))
            EgOut[r, blk] = rebind[EgOut.ElementType](exp(sp * sa))
    # Gate-only blocks (NH_V > CONV // 256, OrcaSAQ-27B: 48 > 40) end here. Block-uniform: CONV % 256 == 0.
    if c >= CONV:
        return
    var si = Int(ssm_i)
    var sl = Int(slots)
    var rg = Int(ring)
    var cw0 = rebind[Scalar[f32]](ConvW[c, 0])
    var cw1 = rebind[Scalar[f32]](ConvW[c, 1])
    var cw2 = rebind[Scalar[f32]](ConvW[c, 2])
    var cw3 = rebind[Scalar[f32]](ConvW[c, 3])
    var sums = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[8]())
    var norm = c < 2 * KDIM
    var head = c // SSTATE
    var w4 = (tid // WARP_SIZE) // 4 * 4
    for r in range(Int(m)):
        var rs = (rg + r) % sl
        var ws = (rg + r + 1) % sl
        var w0 = rebind[Scalar[f32]](ConvState[rs, si, 0, c])
        var w1 = rebind[Scalar[f32]](ConvState[rs, si, 1, c])
        var w2 = rebind[Scalar[f32]](ConvState[rs, si, 2, c])
        var w3 = rebind[Scalar[f32]](Qkv[r, c])
        var acc = w0 * cw0 + w1 * cw1 + w2 * cw2 + w3 * cw3
        var v = acc / (1 + exp(-acc))
        ConvState[ws, si, 0, c] = rebind[ConvState.ElementType](w1)
        ConvState[ws, si, 1, c] = rebind[ConvState.ElementType](w2)
        ConvState[ws, si, 2, c] = rebind[ConvState.ElementType](w3)
        if norm:
            var ssq = warp.sum(v * v)
            if lane_id() == 0:
                sums[tid // WARP_SIZE] = rebind[sums.ElementType](ssq)
            barrier()
            var total: Float32 = 0
            comptime for w in range(SSTATE // WARP_SIZE):
                total += rebind[Scalar[f32]](sums[w4 + w])
            var inv = rsqrt(total + SSM_EPS)
            if head < NH_K:
                inv = inv / sqrt(Float32(SSTATE))
            v = v * inv
            barrier()
        Out[r, c] = rebind[Out.ElementType](v)


def amar_ssm_qk_l2norm[
    XLayout: TensorLayout
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    m: Int32,
):
    comptime assert X.flat_rank == 2
    var head = block_idx.x
    var tid = thread_idx.x
    var base = head * SSTATE
    var sums = stack_allocation[f32, address_space = AddressSpace.SHARED](
        row_major[SSTATE // WARP_SIZE]()
    )
    for r in range(Int(m)):
        var v = rebind[Scalar[f32]](X[r, base + tid])
        var ssq = warp.sum(v * v)
        if lane_id() == 0:
            sums[tid // WARP_SIZE] = rebind[sums.ElementType](ssq)
        barrier()
        var total: Float32 = 0
        comptime for w in range(SSTATE // WARP_SIZE):
            total += rebind[Scalar[f32]](sums[w])
        var inv = rsqrt(total + SSM_EPS)
        if head < NH_K:
            inv = inv / sqrt(Float32(SSTATE))
        X[r, base + tid] = rebind[X.ElementType](v * inv)
        barrier()


def amar_ssm_qk_l2norm_gates[
    XLayout: TensorLayout, PLayout: TensorLayout, GLayout: TensorLayout, DLayout: TensorLayout
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    Ap: TileTensor[f32, PLayout, MutAnyOrigin],
    Bp: TileTensor[f32, PLayout, MutAnyOrigin],
    EgOut: TileTensor[f32, GLayout, MutAnyOrigin],
    BetaOut: TileTensor[f32, GLayout, MutAnyOrigin],
    SsmA: TileTensor[f32, DLayout, MutAnyOrigin],
    DtBias: TileTensor[f32, DLayout, MutAnyOrigin],
    m: Int32,
):
    comptime assert X.flat_rank == 2 and Ap.flat_rank == 3 and Bp.flat_rank == 3
    comptime assert EgOut.flat_rank == 2 and BetaOut.flat_rank == 2
    comptime assert SsmA.flat_rank == 1 and DtBias.flat_rank == 1
    var head = block_idx.x
    var tid = thread_idx.x
    if tid == 0:
        var sa = rebind[Scalar[f32]](SsmA[head])
        var db = rebind[Scalar[f32]](DtBias[head])
        for r in range(Int(m)):
            var araw: Scalar[f32] = 0
            var braw: Scalar[f32] = 0
            comptime for s in range(SPLITK):
                araw += rebind[Scalar[f32]](Ap[s, r, head])
                braw += rebind[Scalar[f32]](Bp[s, r, head])
            BetaOut[r, head] = rebind[BetaOut.ElementType](1 / (1 + exp(-braw)))
            var asum = araw + db
            var sp = log1p(exp(asum))
            EgOut[r, head] = rebind[EgOut.ElementType](exp(sp * sa))
    var base = head * SSTATE
    var sums = stack_allocation[f32, address_space = AddressSpace.SHARED](
        row_major[SSTATE // WARP_SIZE]()
    )
    for r in range(Int(m)):
        var v = rebind[Scalar[f32]](X[r, base + tid])
        var ssq = warp.sum(v * v)
        if lane_id() == 0:
            sums[tid // WARP_SIZE] = rebind[sums.ElementType](ssq)
        barrier()
        var total: Float32 = 0
        comptime for w in range(SSTATE // WARP_SIZE):
            total += rebind[Scalar[f32]](sums[w])
        var inv = rsqrt(total + SSM_EPS)
        if head < NH_K:
            inv = inv / sqrt(Float32(SSTATE))
        X[r, base + tid] = rebind[X.ElementType](v * inv)
        barrier()


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(SSTATE)))
def amar_ssm_delta_step[
    MR: Int,
    S0Layout: TensorLayout, CLayout: TensorLayout, GLayout: TensorLayout,
    OLayout: TensorLayout
](
    SAll: TileTensor[f32, S0Layout, MutAnyOrigin],
    ConvOut: TileTensor[f32, CLayout, MutAnyOrigin],
    Eg: TileTensor[f32, GLayout, MutAnyOrigin],
    Beta: TileTensor[f32, GLayout, MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    ring: Int32, ssm_i: Int32, slots: Int32,
):
    comptime assert SAll.flat_rank == 5 and ConvOut.flat_rank == 2
    comptime assert Eg.flat_rank == 2 and O.flat_rank == 3
    var h = block_idx.x
    var j = thread_idx.x
    var kh = h % NH_K
    var si = Int(ssm_i)
    var sl = Int(slots)
    var rg = Int(ring)

    var kq = stack_allocation[f32, address_space = AddressSpace.SHARED](
        row_major[2, SSTATE]()
    )
    comptime for r in range(MR):
        var rs = (rg + r) % sl
        var ws = (rg + r + 1) % sl
        kq[0, j] = rebind[kq.ElementType](ConvOut[r, kh * SSTATE + j])
        kq[1, j] = rebind[kq.ElementType](ConvOut[r, KDIM + kh * SSTATE + j])
        barrier()

        var eg = rebind[Scalar[f32]](Eg[r, h])
        var beta = rebind[Scalar[f32]](Beta[r, h])
        var vj = rebind[Scalar[f32]](ConvOut[r, 2 * KDIM + h * SSTATE + j])

        var col = SIMD[f32, SSTATE]()
        comptime for i in range(SSTATE):
            col[i] = rebind[Scalar[f32]](SAll[rs, si, h, i, j])

        var sk: Float32 = 0
        comptime for i in range(SSTATE):
            sk += col[i] * eg * rebind[Scalar[f32]](kq[1, i])
        var d = (vj - sk) * beta

        var o: Float32 = 0
        comptime for i in range(SSTATE):
            var s = col[i] * eg + rebind[Scalar[f32]](kq[1, i]) * d
            SAll[ws, si, h, i, j] = rebind[SAll.ElementType](s)
            o += s * rebind[Scalar[f32]](kq[0, i])
        O[r, h, j] = rebind[O.ElementType](o)
        barrier()


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(SSTATE)))
def amar_ssm_delta_rows[
    S0Layout: TensorLayout, CLayout: TensorLayout, GLayout: TensorLayout,
    OLayout: TensorLayout
](
    SAll: TileTensor[f32, S0Layout, MutAnyOrigin],
    ConvOut: TileTensor[f32, CLayout, MutAnyOrigin],
    Eg: TileTensor[f32, GLayout, MutAnyOrigin],
    Beta: TileTensor[f32, GLayout, MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    ring: Int32, ssm_i: Int32, slots: Int32, m: Int32,
):
    comptime assert SAll.flat_rank == 5 and ConvOut.flat_rank == 2
    comptime assert Eg.flat_rank == 2 and O.flat_rank == 3
    var h = block_idx.x
    var j = thread_idx.x
    var kh = h % NH_K
    var si = Int(ssm_i)
    var rs = Int(ring) % Int(slots)
    var ws = (Int(ring) + Int(m)) % Int(slots)
    var M = Int(m)
    var kq = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[2, 2, SSTATE]())
    var col = SIMD[f32, SSTATE]()
    comptime for i in range(SSTATE):
        col[i] = rebind[Scalar[f32]](SAll[rs, si, h, i, j])
    kq[0, 0, j] = ConvOut[0, KDIM + kh * SSTATE + j]
    kq[0, 1, j] = ConvOut[0, kh * SSTATE + j]
    var eg = rebind[Scalar[f32]](Eg[0, h])
    var beta = rebind[Scalar[f32]](Beta[0, h])
    var vj = rebind[Scalar[f32]](ConvOut[0, 2 * KDIM + h * SSTATE + j])
    barrier()
    for r in range(M):
        var cur = r % 2
        var kn: Float32 = 0
        var qn: Float32 = 0
        var egn: Float32 = 0
        var betan: Float32 = 0
        var vjn: Float32 = 0
        if r + 1 < M:
            kn = rebind[Scalar[f32]](ConvOut[r + 1, KDIM + kh * SSTATE + j])
            qn = rebind[Scalar[f32]](ConvOut[r + 1, kh * SSTATE + j])
            egn = rebind[Scalar[f32]](Eg[r + 1, h])
            betan = rebind[Scalar[f32]](Beta[r + 1, h])
            vjn = rebind[Scalar[f32]](ConvOut[r + 1, 2 * KDIM + h * SSTATE + j])
        var sk: Float32 = 0
        comptime for i in range(SSTATE):
            sk += col[i] * eg * rebind[Scalar[f32]](kq[cur, 0, i])
        var d = (vj - sk) * beta
        var o: Float32 = 0
        comptime for i in range(SSTATE):
            var s = col[i] * eg + rebind[Scalar[f32]](kq[cur, 0, i]) * d
            col[i] = s
            o += s * rebind[Scalar[f32]](kq[cur, 1, i])
        O[r, h, j] = rebind[O.ElementType](o)
        if r + 1 < M:
            kq[1 - cur, 0, j] = rebind[kq.ElementType](kn)
            kq[1 - cur, 1, j] = rebind[kq.ElementType](qn)
            eg = egn
            beta = betan
            vj = vjn
        barrier()
    comptime for i in range(SSTATE):
        SAll[ws, si, h, i, j] = rebind[SAll.ElementType](col[i])


def amar_ssm_gated_out[
    OLayout: TensorLayout, ZLayout: TensorLayout, NLayout: TensorLayout,
    RLayout: TensorLayout
](
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    Z: TileTensor[f32, ZLayout, MutAnyOrigin],
    NormW: TileTensor[f32, NLayout, MutAnyOrigin],
    Res: TileTensor[f32, RLayout, MutAnyOrigin],
):
    comptime assert O.flat_rank == 2 and Z.flat_rank == 1
    comptime assert NormW.flat_rank == 1 and Res.flat_rank == 1
    var h = block_idx.x
    var j = thread_idx.x
    var v = rebind[Scalar[f32]](O[h, j])
    var ssq = warp.sum(v * v)
    var sums = stack_allocation[f32, address_space = AddressSpace.SHARED](
        row_major[SSTATE // WARP_SIZE]()
    )
    if lane_id() == 0:
        sums[j // WARP_SIZE] = rebind[sums.ElementType](ssq)
    barrier()
    var total: Float32 = 0
    comptime for w in range(SSTATE // WARP_SIZE):
        total += rebind[Scalar[f32]](sums[w])
    var scale = rsqrt(total / Float32(SSTATE) + SSM_EPS)
    var z = rebind[Scalar[f32]](Z[h * SSTATE + j])
    Res[h * SSTATE + j] = rebind[Res.ElementType](
        v * scale * rebind[Scalar[f32]](NormW[j]) * (z / (1 + exp(-z)))
    )


def amar_ssm_gated_out_bf16[
    OLayout: TensorLayout, ZLayout: TensorLayout, NLayout: TensorLayout,
    RLayout: TensorLayout
](
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    Z: TileTensor[f32, ZLayout, MutAnyOrigin],
    NormW: TileTensor[f32, NLayout, MutAnyOrigin],
    Res: TileTensor[DType.bfloat16, RLayout, MutAnyOrigin],
    m: Int32,
):
    comptime assert O.flat_rank == 3 and Z.flat_rank == 2
    comptime assert NormW.flat_rank == 1 and Res.flat_rank == 2
    var h = block_idx.x
    var j = thread_idx.x
    var sums = stack_allocation[f32, address_space = AddressSpace.SHARED](
        row_major[SSTATE // WARP_SIZE]()
    )
    var nwj = rebind[Scalar[f32]](NormW[j])
    for r in range(Int(m)):
        var v = rebind[Scalar[f32]](O[r, h, j])
        var ssq = warp.sum(v * v)
        if lane_id() == 0:
            sums[j // WARP_SIZE] = rebind[sums.ElementType](ssq)
        barrier()
        var total: Float32 = 0
        comptime for w in range(SSTATE // WARP_SIZE):
            total += rebind[Scalar[f32]](sums[w])
        var scale = rsqrt(total / Float32(SSTATE) + SSM_EPS)
        var z = rebind[Scalar[f32]](Z[r, h * SSTATE + j])
        Res[r, h * SSTATE + j] = rebind[Res.ElementType](
            (
                v * scale * nwj * (z / (1 + exp(-z)))
            ).cast[DType.bfloat16]()
        )
        barrier()


def amar_ssm_gates_rows[
    RLayout: TensorLayout, GLayout: TensorLayout, DLayout: TensorLayout
](
    AlphaRaw: TileTensor[f32, RLayout, MutAnyOrigin],
    BetaRaw: TileTensor[f32, RLayout, MutAnyOrigin],
    EgOut: TileTensor[f32, GLayout, MutAnyOrigin],
    BetaOut: TileTensor[f32, GLayout, MutAnyOrigin],
    SsmA: TileTensor[f32, DLayout, MutAnyOrigin],
    DtBias: TileTensor[f32, DLayout, MutAnyOrigin],
    m: Int32,
):
    comptime assert AlphaRaw.flat_rank == 2 and BetaRaw.flat_rank == 2
    comptime assert EgOut.flat_rank == 2 and BetaOut.flat_rank == 2
    comptime assert SsmA.flat_rank == 1 and DtBias.flat_rank == 1
    var gid = Int(global_idx.x)
    if gid >= Int(m) * NH_V:
        return
    var r = gid // NH_V
    var h = gid % NH_V
    var braw = rebind[Scalar[f32]](BetaRaw[r, h])
    BetaOut[r, h] = rebind[BetaOut.ElementType](1 / (1 + exp(-braw)))
    var asum = rebind[Scalar[f32]](AlphaRaw[r, h]) + rebind[Scalar[f32]](DtBias[h])
    var sp = log1p(exp(asum))
    EgOut[r, h] = rebind[EgOut.ElementType](exp(sp * rebind[Scalar[f32]](SsmA[h])))


def amar_ssm_conv_chunk[
    QLayout: TensorLayout, SLayout: TensorLayout, WLayout: TensorLayout,
    OLayout: TensorLayout
](
    Qkv: TileTensor[f32, QLayout, MutAnyOrigin],
    ConvState: TileTensor[f32, SLayout, MutAnyOrigin],
    ConvW: TileTensor[f32, WLayout, MutAnyOrigin],
    Out: TileTensor[f32, OLayout, MutAnyOrigin],
    ring: Int32, ssm_i: Int32, slots: Int32, m: Int32,
):
    comptime assert Qkv.flat_rank == 2 and ConvState.flat_rank == 4
    comptime assert ConvW.flat_rank == 2 and Out.flat_rank == 2
    var c = global_idx.x
    if c >= CONV:
        return
    var si = Int(ssm_i)
    var rs = Int(ring) % Int(slots)
    var ws = (Int(ring) + Int(m)) % Int(slots)
    var cw0 = rebind[Scalar[f32]](ConvW[c, 0])
    var cw1 = rebind[Scalar[f32]](ConvW[c, 1])
    var cw2 = rebind[Scalar[f32]](ConvW[c, 2])
    var cw3 = rebind[Scalar[f32]](ConvW[c, 3])
    var w0 = rebind[Scalar[f32]](ConvState[rs, si, 0, c])
    var w1 = rebind[Scalar[f32]](ConvState[rs, si, 1, c])
    var w2 = rebind[Scalar[f32]](ConvState[rs, si, 2, c])
    for r in range(Int(m)):
        var w3 = rebind[Scalar[f32]](Qkv[r, c])
        var acc = w0 * cw0 + w1 * cw1 + w2 * cw2 + w3 * cw3
        Out[r, c] = rebind[Out.ElementType](acc / (1 + exp(-acc)))
        w0 = w1
        w1 = w2
        w2 = w3
    ConvState[ws, si, 0, c] = rebind[ConvState.ElementType](w0)
    ConvState[ws, si, 1, c] = rebind[ConvState.ElementType](w1)
    ConvState[ws, si, 2, c] = rebind[ConvState.ElementType](w2)


def amar_ssm_qk_l2norm_rows[
    XLayout: TensorLayout
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    m: Int32,
):
    comptime assert X.flat_rank == 2
    var head = block_idx.x
    var r = Int(block_idx.y)
    var tid = thread_idx.x
    var base = head * SSTATE
    var sums = stack_allocation[f32, address_space = AddressSpace.SHARED](
        row_major[SSTATE // WARP_SIZE]()
    )
    var v = rebind[Scalar[f32]](X[r, base + tid])
    var ssq = warp.sum(v * v)
    if lane_id() == 0:
        sums[tid // WARP_SIZE] = rebind[sums.ElementType](ssq)
    barrier()
    var total: Float32 = 0
    comptime for w in range(SSTATE // WARP_SIZE):
        total += rebind[Scalar[f32]](sums[w])
    var inv = rsqrt(total + SSM_EPS)
    if head < NH_K:
        inv = inv / sqrt(Float32(SSTATE))
    X[r, base + tid] = rebind[X.ElementType](v * inv)


def amar_ssm_delta_chunk[
    S0Layout: TensorLayout, CLayout: TensorLayout, GLayout: TensorLayout,
    OLayout: TensorLayout
](
    SAll: TileTensor[f32, S0Layout, MutAnyOrigin],
    ConvOut: TileTensor[f32, CLayout, MutAnyOrigin],
    Eg: TileTensor[f32, GLayout, MutAnyOrigin],
    Beta: TileTensor[f32, GLayout, MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    ring: Int32, ssm_i: Int32, slots: Int32, m: Int32,
):
    comptime assert SAll.flat_rank == 5 and ConvOut.flat_rank == 2
    comptime assert Eg.flat_rank == 2 and O.flat_rank == 3
    var h = block_idx.x
    var j = thread_idx.x
    var kh = h % NH_K
    var si = Int(ssm_i)
    var rs = Int(ring) % Int(slots)
    var ws = (Int(ring) + Int(m)) % Int(slots)

    var kq = stack_allocation[f32, address_space = AddressSpace.SHARED](
        row_major[2, SSTATE]()
    )
    var col = SIMD[f32, SSTATE]()
    comptime for i in range(SSTATE):
        col[i] = rebind[Scalar[f32]](SAll[rs, si, h, i, j])
    for r in range(Int(m)):
        kq[0, j] = rebind[kq.ElementType](ConvOut[r, kh * SSTATE + j])
        kq[1, j] = rebind[kq.ElementType](ConvOut[r, KDIM + kh * SSTATE + j])
        barrier()

        var eg = rebind[Scalar[f32]](Eg[r, h])
        var beta = rebind[Scalar[f32]](Beta[r, h])
        var vj = rebind[Scalar[f32]](ConvOut[r, 2 * KDIM + h * SSTATE + j])

        var sk: Float32 = 0
        comptime for i in range(SSTATE):
            sk += col[i] * eg * rebind[Scalar[f32]](kq[1, i])
        var d = (vj - sk) * beta

        var o: Float32 = 0
        comptime for i in range(SSTATE):
            var s = col[i] * eg + rebind[Scalar[f32]](kq[1, i]) * d
            col[i] = s
            o += s * rebind[Scalar[f32]](kq[0, i])
        O[r, h, j] = rebind[O.ElementType](o)
        barrier()
    comptime for i in range(SSTATE):
        SAll[ws, si, h, i, j] = rebind[SAll.ElementType](col[i])


def amar_ssm_gated_out_rows_bf16[
    OLayout: TensorLayout, ZLayout: TensorLayout, NLayout: TensorLayout,
    RLayout: TensorLayout
](
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    Z: TileTensor[f32, ZLayout, MutAnyOrigin],
    NormW: TileTensor[f32, NLayout, MutAnyOrigin],
    Res: TileTensor[DType.bfloat16, RLayout, MutAnyOrigin],
):
    comptime assert O.flat_rank == 3 and Z.flat_rank == 2
    comptime assert NormW.flat_rank == 1 and Res.flat_rank == 2
    var h = block_idx.x
    var r = Int(block_idx.y)
    var j = thread_idx.x
    var sums = stack_allocation[f32, address_space = AddressSpace.SHARED](
        row_major[SSTATE // WARP_SIZE]()
    )
    var v = rebind[Scalar[f32]](O[r, h, j])
    var ssq = warp.sum(v * v)
    if lane_id() == 0:
        sums[j // WARP_SIZE] = rebind[sums.ElementType](ssq)
    barrier()
    var total: Float32 = 0
    comptime for w in range(SSTATE // WARP_SIZE):
        total += rebind[Scalar[f32]](sums[w])
    var scale = rsqrt(total / Float32(SSTATE) + SSM_EPS)
    var z = rebind[Scalar[f32]](Z[r, h * SSTATE + j])
    Res[r, h * SSTATE + j] = rebind[Res.ElementType](
        (v * scale * rebind[Scalar[f32]](NormW[j]) * (z / (1 + exp(-z)))).cast[DType.bfloat16]()
    )


comptime DC_COLS = 8
comptime DC_IQ = WARP_SIZE // DC_COLS
comptime DC_IL = SSTATE // DC_IQ
comptime DC_GROUPS = SSTATE // DC_COLS
comptime DC_BLOCKS = NH_V * DC_GROUPS


@always_inline
def dc_load[CLayout: TensorLayout](
    ConvOut: TileTensor[f32, CLayout, MutAnyOrigin], r: Int, base: Int
) -> SIMD[f32, DC_IL]:
    comptime assert ConvOut.flat_rank == 2
    var Cv = ConvOut.vectorize[1, 8]()
    var b8 = base // 8
    var a = rebind[SIMD[f32, 8]](Cv[r, b8]).join(rebind[SIMD[f32, 8]](Cv[r, b8 + 1]))
    var b = rebind[SIMD[f32, 8]](Cv[r, b8 + 2]).join(rebind[SIMD[f32, 8]](Cv[r, b8 + 3]))
    return rebind[SIMD[f32, DC_IL]](a.join(b))


def amar_ssm_delta_chunk_w[
    S0Layout: TensorLayout, CLayout: TensorLayout, GLayout: TensorLayout,
    OLayout: TensorLayout
](
    SAll: TileTensor[f32, S0Layout, MutAnyOrigin],
    ConvOut: TileTensor[f32, CLayout, MutAnyOrigin],
    Eg: TileTensor[f32, GLayout, MutAnyOrigin],
    Beta: TileTensor[f32, GLayout, MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    ring: Int32, ssm_i: Int32, slots: Int32, m: Int32,
):
    comptime assert SAll.flat_rank == 5 and ConvOut.flat_rank == 2
    comptime assert Eg.flat_rank == 2 and O.flat_rank == 3
    comptime assert DC_IL == 32
    var h = Int(block_idx.x) // DC_GROUPS
    var lane = Int(thread_idx.x)
    var jj = lane // DC_IQ
    var iq = lane % DC_IQ
    var j = (Int(block_idx.x) % DC_GROUPS) * DC_COLS + jj
    var i0 = iq * DC_IL
    var kh = h % NH_K
    var si = Int(ssm_i)
    var rs = Int(ring) % Int(slots)
    var ws = (Int(ring) + Int(m)) % Int(slots)
    var M = Int(m)
    var qb = kh * SSTATE + i0
    var kb = KDIM + kh * SSTATE + i0
    var vb = 2 * KDIM + h * SSTATE + j

    var col = SIMD[f32, DC_IL]()
    comptime for i in range(DC_IL):
        col[i] = rebind[Scalar[f32]](SAll[rs, si, h, i0 + i, j])
    var kc = dc_load(ConvOut, 0, kb)
    var qc = dc_load(ConvOut, 0, qb)
    var vc = rebind[Scalar[f32]](ConvOut[0, vb])
    var ec = rebind[Scalar[f32]](Eg[0, h])
    var bc = rebind[Scalar[f32]](Beta[0, h])
    for r in range(M):
        var rn = min(r + 1, M - 1)
        var kn = dc_load(ConvOut, rn, kb)
        var qn = dc_load(ConvOut, rn, qb)
        var vn = rebind[Scalar[f32]](ConvOut[rn, vb])
        var en = rebind[Scalar[f32]](Eg[rn, h])
        var bn = rebind[Scalar[f32]](Beta[rn, h])

        var ce = col * ec
        var sk = (ce * kc).reduce_add()
        sk += warp.shuffle_xor(sk, UInt32(1))
        sk += warp.shuffle_xor(sk, UInt32(2))
        var d = (vc - sk) * bc
        col = ce + kc * d
        var o = (col * qc).reduce_add()
        o += warp.shuffle_xor(o, UInt32(1))
        o += warp.shuffle_xor(o, UInt32(2))
        if iq == 0:
            O[r, h, j] = rebind[O.ElementType](o)

        kc = kn
        qc = qn
        vc = vn
        ec = en
        bc = bn
    comptime for i in range(DC_IL):
        SAll[ws, si, h, i0 + i, j] = rebind[SAll.ElementType](col[i])
