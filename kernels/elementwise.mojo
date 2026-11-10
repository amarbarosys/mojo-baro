
from max.gpu import block_dim, block_idx, global_idx, lane_id, thread_idx, WARP_SIZE
from max.gpu.primitives import warp
from bexp import cos, exp, log, rsqrt, sin
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from layout import TileTensor, TensorLayout, row_major, stack_allocation

comptime EW_THREADS = 256
comptime f32 = DType.float32


def amar_rmsnorm[
    XLayout: TensorLayout, GLayout: TensorLayout, OLayout: TensorLayout
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    G: TileTensor[f32, GLayout, MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    n: Int32,
    eps: Float32,
):
    comptime assert X.flat_rank == 2 and G.flat_rank == 1 and O.flat_rank == 2

    var N = Int(n)
    var row = block_idx.x
    var tid = thread_idx.x

    var partial: Float32 = 0
    var i = tid
    while i < N:
        var v = rebind[Scalar[f32]](X[row, i])
        partial += v * v
        i += EW_THREADS

    var sums = stack_allocation[
        f32, address_space = AddressSpace.SHARED
    ](row_major[EW_THREADS // WARP_SIZE]())
    var wsum = warp.sum(partial)
    if lane_id() == 0:
        sums[tid // WARP_SIZE] = rebind[sums.ElementType](wsum)
    barrier()
    var total: Float32 = 0
    comptime for w in range(EW_THREADS // WARP_SIZE):
        total += rebind[Scalar[f32]](sums[w])

    var scale = rsqrt(total / Float32(N) + eps)
    i = tid
    while i < N:
        O[row, i] = rebind[O.ElementType](
            rebind[Scalar[f32]](X[row, i])
            * scale
            * rebind[Scalar[f32]](G[i])
        )
        i += EW_THREADS


def amar_rmsnorm_cast[
    XLayout: TensorLayout, GLayout: TensorLayout, OLayout: TensorLayout, ODT: DType = DType.bfloat16
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    G: TileTensor[f32, GLayout, MutAnyOrigin],
    O: TileTensor[ODT, OLayout, MutAnyOrigin],
    n: Int32,
    eps: Float32,
):
    comptime assert X.flat_rank == 2 and G.flat_rank == 1 and O.flat_rank == 2

    var N = Int(n)
    var row = block_idx.x
    var tid = thread_idx.x

    var partial: Float32 = 0
    var i = tid
    while i + 3 * EW_THREADS < N:
        var vs = Array[Float32, 4](uninitialized=True)
        comptime for k in range(4):
            vs[k] = rebind[Scalar[f32]](X[row, i + k * EW_THREADS])
        comptime for k in range(4):
            partial += vs[k] * vs[k]
        i += 4 * EW_THREADS
    while i < N:
        var v = rebind[Scalar[f32]](X[row, i])
        partial += v * v
        i += EW_THREADS

    var sums = stack_allocation[
        f32, address_space = AddressSpace.SHARED
    ](row_major[EW_THREADS // WARP_SIZE]())
    var wsum = warp.sum(partial)
    if lane_id() == 0:
        sums[tid // WARP_SIZE] = rebind[sums.ElementType](wsum)
    barrier()
    var total: Float32 = 0
    comptime for w in range(EW_THREADS // WARP_SIZE):
        total += rebind[Scalar[f32]](sums[w])

    var scale = rsqrt(total / Float32(N) + eps)
    i = tid
    while i + 3 * EW_THREADS < N:
        var xs = Array[Float32, 4](uninitialized=True)
        var gs = Array[Float32, 4](uninitialized=True)
        comptime for k in range(4):
            xs[k] = rebind[Scalar[f32]](X[row, i + k * EW_THREADS])
            gs[k] = rebind[Scalar[f32]](G[i + k * EW_THREADS])
        comptime for k in range(4):
            O[row, i + k * EW_THREADS] = rebind[O.ElementType]((xs[k] * scale * gs[k]).cast[ODT]())
        i += 4 * EW_THREADS
    while i < N:
        O[row, i] = rebind[O.ElementType](
            (
                rebind[Scalar[f32]](X[row, i])
                * scale
                * rebind[Scalar[f32]](G[i])
            ).cast[ODT]()
        )
        i += EW_THREADS


def amar_rmsnorm_cast_steer[
    XLayout: TensorLayout, GLayout: TensorLayout, OLayout: TensorLayout, SLayout: TensorLayout, ODT: DType = DType.bfloat16
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    G: TileTensor[f32, GLayout, MutAnyOrigin],
    O: TileTensor[ODT, OLayout, MutAnyOrigin],
    n: Int32,
    eps: Float32,
    S: TileTensor[f32, SLayout, MutAnyOrigin],
    soff: Int32,
):
    """amar_rmsnorm_cast after X += S[soff : soff + n] (a steering vector, written back to X)."""
    comptime assert X.flat_rank == 2 and G.flat_rank == 1 and O.flat_rank == 2 and S.flat_rank == 1

    var N = Int(n)
    var row = block_idx.x
    var tid = thread_idx.x

    var partial: Float32 = 0
    var i = tid
    while i + 3 * EW_THREADS < N:
        var vs = Array[Float32, 4](uninitialized=True)
        comptime for k in range(4):
            vs[k] = rebind[Scalar[f32]](X[row, i + k * EW_THREADS]) + rebind[Scalar[f32]](S[Int(soff) + i + k * EW_THREADS])
            X[row, i + k * EW_THREADS] = rebind[X.ElementType](vs[k])
        comptime for k in range(4):
            partial += vs[k] * vs[k]
        i += 4 * EW_THREADS
    while i < N:
        var v = rebind[Scalar[f32]](X[row, i]) + rebind[Scalar[f32]](S[Int(soff) + i])
        X[row, i] = rebind[X.ElementType](v)
        partial += v * v
        i += EW_THREADS

    var sums = stack_allocation[
        f32, address_space = AddressSpace.SHARED
    ](row_major[EW_THREADS // WARP_SIZE]())
    var wsum = warp.sum(partial)
    if lane_id() == 0:
        sums[tid // WARP_SIZE] = rebind[sums.ElementType](wsum)
    barrier()
    var total: Float32 = 0
    comptime for w in range(EW_THREADS // WARP_SIZE):
        total += rebind[Scalar[f32]](sums[w])

    var scale = rsqrt(total / Float32(N) + eps)
    i = tid
    while i + 3 * EW_THREADS < N:
        var xs = Array[Float32, 4](uninitialized=True)
        var gs = Array[Float32, 4](uninitialized=True)
        comptime for k in range(4):
            xs[k] = rebind[Scalar[f32]](X[row, i + k * EW_THREADS])
            gs[k] = rebind[Scalar[f32]](G[i + k * EW_THREADS])
        comptime for k in range(4):
            O[row, i + k * EW_THREADS] = rebind[O.ElementType]((xs[k] * scale * gs[k]).cast[ODT]())
        i += 4 * EW_THREADS
    while i < N:
        O[row, i] = rebind[O.ElementType](
            (
                rebind[Scalar[f32]](X[row, i])
                * scale
                * rebind[Scalar[f32]](G[i])
            ).cast[ODT]()
        )
        i += EW_THREADS


def amar_rmsnorm_cast2[
    XLayout: TensorLayout, GLayout: TensorLayout, OLayout: TensorLayout,
    FLayout: TensorLayout
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    G: TileTensor[f32, GLayout, MutAnyOrigin],
    O: TileTensor[DType.bfloat16, OLayout, MutAnyOrigin],
    F: TileTensor[f32, FLayout, MutAnyOrigin],
    n: Int32,
    eps: Float32,
):
    comptime assert X.flat_rank == 2 and G.flat_rank == 1 and O.flat_rank == 2 and F.flat_rank == 2

    var N = Int(n)
    var row = block_idx.x
    var tid = thread_idx.x

    var partial: Float32 = 0
    var i = tid
    while i < N:
        var v = rebind[Scalar[f32]](X[row, i])
        partial += v * v
        i += EW_THREADS

    var sums = stack_allocation[
        f32, address_space = AddressSpace.SHARED
    ](row_major[EW_THREADS // WARP_SIZE]())
    var wsum = warp.sum(partial)
    if lane_id() == 0:
        sums[tid // WARP_SIZE] = rebind[sums.ElementType](wsum)
    barrier()
    var total: Float32 = 0
    comptime for w in range(EW_THREADS // WARP_SIZE):
        total += rebind[Scalar[f32]](sums[w])

    var scale = rsqrt(total / Float32(N) + eps)
    i = tid
    while i < N:
        var h = (
            rebind[Scalar[f32]](X[row, i])
            * scale
            * rebind[Scalar[f32]](G[i])
        ).cast[DType.bfloat16]()
        O[row, i] = rebind[O.ElementType](h)
        if row == 0:
            F[0, i] = rebind[F.ElementType](h.cast[f32]())
        i += EW_THREADS


def amar_rmsnorm_cast2_q8[
    NMAX: Int, XLayout: TensorLayout, GLayout: TensorLayout, OLayout: TensorLayout,
    FLayout: TensorLayout
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    G: TileTensor[f32, GLayout, MutAnyOrigin],
    O: TileTensor[DType.bfloat16, OLayout, MutAnyOrigin],
    F: TileTensor[f32, FLayout, MutAnyOrigin],
    Aq: MutPointer[Scalar[DType.int8], MutAnyOrigin],
    As: MutPointer[Scalar[DType.float16], MutAnyOrigin],
    n: Int32,
    eps: Float32,
):
    comptime assert X.flat_rank == 2 and G.flat_rank == 1 and O.flat_rank == 2 and F.flat_rank == 2

    var N = Int(n)
    var row = block_idx.x
    var tid = thread_idx.x

    var partial: Float32 = 0
    var i = tid
    while i < N:
        var v = rebind[Scalar[f32]](X[row, i])
        partial += v * v
        i += EW_THREADS

    var sums = stack_allocation[
        f32, address_space = AddressSpace.SHARED
    ](row_major[EW_THREADS // WARP_SIZE]())
    var hv = stack_allocation[
        f32, address_space = AddressSpace.SHARED
    ](row_major[NMAX]())
    var wsum = warp.sum(partial)
    if lane_id() == 0:
        sums[tid // WARP_SIZE] = rebind[sums.ElementType](wsum)
    barrier()
    var total: Float32 = 0
    comptime for w in range(EW_THREADS // WARP_SIZE):
        total += rebind[Scalar[f32]](sums[w])

    var scale = rsqrt(total / Float32(N) + eps)
    i = tid
    while i < N:
        var h = (
            rebind[Scalar[f32]](X[row, i])
            * scale
            * rebind[Scalar[f32]](G[i])
        ).cast[DType.bfloat16]()
        O[row, i] = rebind[O.ElementType](h)
        hv[i] = rebind[hv.ElementType](h.cast[f32]())
        if row == 0:
            F[0, i] = rebind[F.ElementType](h.cast[f32]())
        i += EW_THREADS
    barrier()
    if row == 0:
        var lane = Int(lane_id())
        var g = tid // WARP_SIZE
        while g < N // 32:
            var x = rebind[Scalar[f32]](hv[g * 32 + lane])
            var wmax = warp.max(abs(x))
            var qs = wmax / Float32(127) if wmax > 0 else Float32(1)
            var v = x / qs
            var q = Int32(v + Float32(0.5)) if v >= 0 else -Int32(-v + Float32(0.5))
            if q > 127:
                q = 127
            if q < -127:
                q = -127
            Aq[unsafe_offset=g * 32 + lane] = Int8(q)
            if lane == 0:
                As[unsafe_offset=g] = qs.cast[DType.float16]()
            g += EW_THREADS // WARP_SIZE


def amar_rmsnorm_cast2_rows[
    XLayout: TensorLayout, GLayout: TensorLayout, OLayout: TensorLayout,
    FLayout: TensorLayout
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    G: TileTensor[f32, GLayout, MutAnyOrigin],
    O: TileTensor[DType.bfloat16, OLayout, MutAnyOrigin],
    F: TileTensor[f32, FLayout, MutAnyOrigin],
    n: Int32,
    eps: Float32,
):
    comptime assert X.flat_rank == 2 and G.flat_rank == 1 and O.flat_rank == 2 and F.flat_rank == 2

    var N = Int(n)
    var row = block_idx.x
    var tid = thread_idx.x

    var partial: Float32 = 0
    var i = tid
    while i < N:
        var v = rebind[Scalar[f32]](X[row, i])
        partial += v * v
        i += EW_THREADS

    var sums = stack_allocation[
        f32, address_space = AddressSpace.SHARED
    ](row_major[EW_THREADS // WARP_SIZE]())
    var wsum = warp.sum(partial)
    if lane_id() == 0:
        sums[tid // WARP_SIZE] = rebind[sums.ElementType](wsum)
    barrier()
    var total: Float32 = 0
    comptime for w in range(EW_THREADS // WARP_SIZE):
        total += rebind[Scalar[f32]](sums[w])

    var scale = rsqrt(total / Float32(N) + eps)
    i = tid
    while i < N:
        var h = (
            rebind[Scalar[f32]](X[row, i])
            * scale
            * rebind[Scalar[f32]](G[i])
        ).cast[DType.bfloat16]()
        O[row, i] = rebind[O.ElementType](h)
        F[row, i] = rebind[F.ElementType](h.cast[f32]())
        i += EW_THREADS


def amar_swiglu[
    GLayout: TensorLayout, ULayout: TensorLayout, OLayout: TensorLayout
](
    Gate: TileTensor[f32, GLayout, MutAnyOrigin],
    Up: TileTensor[f32, ULayout, MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    count: Int32,
):
    comptime assert Gate.flat_rank == 1 and Up.flat_rank == 1 and O.flat_rank == 1

    var idx = global_idx.x
    if idx >= Int(count):
        return
    var g = rebind[Scalar[f32]](Gate[idx])
    var u = rebind[Scalar[f32]](Up[idx])
    var silu = g / (1 + exp(-g))
    O[idx] = rebind[O.ElementType](silu * u)


def amar_rope_rows[
    XLayout: TensorLayout
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    n_heads: Int32,
    head_dim: Int32,
    pos: Int32,
    theta_base: Float32,
):
    comptime assert X.flat_rank == 2

    var row = block_idx.x
    var half = Int(head_dim) // 2
    var pairs = Int(n_heads) * half
    var idx = thread_idx.x
    while idx < pairs:
        var h = idx // half
        var i = idx % half
        var base = h * Int(head_dim)
        var freq = exp(
            Float32(-2 * i) / Float32(head_dim) * log(theta_base)
        )
        var angle = Float32(Int(pos) + row) * freq
        var c = cos(angle)
        var s = sin(angle)
        var x0 = rebind[Scalar[f32]](X[row, base + i])
        var x1 = rebind[Scalar[f32]](X[row, base + half + i])
        X[row, base + i] = rebind[X.ElementType](x0 * c - x1 * s)
        X[row, base + half + i] = rebind[X.ElementType](x0 * s + x1 * c)
        idx += block_dim.x


def amar_softmax_rows[
    XLayout: TensorLayout
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    n: Int32,
):
    comptime assert X.flat_rank == 2

    var N = Int(n)
    var row = block_idx.x
    var tid = thread_idx.x

    var scratch = stack_allocation[
        f32, address_space = AddressSpace.SHARED
    ](row_major[EW_THREADS // WARP_SIZE]())

    var local_max = Float32(-3.4e38)
    var i = tid
    while i < N:
        var v = rebind[Scalar[f32]](X[row, i])
        if v > local_max:
            local_max = v
        i += EW_THREADS
    var wmax = warp.max(local_max)
    if lane_id() == 0:
        scratch[tid // WARP_SIZE] = rebind[scratch.ElementType](wmax)
    barrier()
    var row_max = Float32(-3.4e38)
    comptime for w in range(EW_THREADS // WARP_SIZE):
        var v = rebind[Scalar[f32]](scratch[w])
        if v > row_max:
            row_max = v
    barrier()

    var partial: Float32 = 0
    i = tid
    while i < N:
        var e = exp(rebind[Scalar[f32]](X[row, i]) - row_max)
        X[row, i] = rebind[X.ElementType](e)
        partial += e
        i += EW_THREADS
    var wsum = warp.sum(partial)
    if lane_id() == 0:
        scratch[tid // WARP_SIZE] = rebind[scratch.ElementType](wsum)
    barrier()
    var total: Float32 = 0
    comptime for w in range(EW_THREADS // WARP_SIZE):
        total += rebind[Scalar[f32]](scratch[w])

    var inv = 1 / total
    i = tid
    while i < N:
        X[row, i] = rebind[X.ElementType](rebind[Scalar[f32]](X[row, i]) * inv)
        i += EW_THREADS


def amar_embed_lookup[
    TLayout: TensorLayout, OLayout: TensorLayout
](
    Table: TileTensor[DType.bfloat16, TLayout, MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    token: Int32,
    n: Int32,
):
    comptime assert Table.flat_rank == 2 and O.flat_rank == 2

    var idx = global_idx.x
    if idx >= Int(n):
        return
    O[block_idx.y, idx] = rebind[O.ElementType](
        rebind[Scalar[DType.bfloat16]](Table[Int(token), idx]).cast[f32]()
    )


def amar_embed_lookup_pos[
    TLayout: TensorLayout, OLayout: TensorLayout, KLayout: TensorLayout
](
    Table: TileTensor[DType.bfloat16, TLayout, MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    Toks: TileTensor[DType.int32, KLayout, MutAnyOrigin],
    pos: Int32,
    n: Int32,
):
    comptime assert Table.flat_rank == 2 and O.flat_rank == 2 and Toks.flat_rank == 1

    var idx = global_idx.x
    if idx >= Int(n):
        return
    var token = Int(rebind[Scalar[DType.int32]](Toks[Int(pos) + Int(block_idx.y)]))
    O[block_idx.y, idx] = rebind[O.ElementType](
        rebind[Scalar[DType.bfloat16]](Table[token, idx]).cast[f32]()
    )


def amar_argmax_pos[
    XLayout: TensorLayout, OLayout: TensorLayout
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    Out: TileTensor[DType.int32, OLayout, MutAnyOrigin],
    n: Int32,
    wpos: Int32,
):
    comptime assert X.flat_rank == 2 and Out.flat_rank == 1

    var N = Int(n)
    var row = block_idx.x
    var tid = thread_idx.x

    var best_v = Float32(-3.4e38)
    var best_i: Int32 = 0
    var i = tid
    while i + 7 * EW_THREADS < N:
        var vs = Array[Float32, 8](uninitialized=True)
        comptime for k in range(8):
            vs[k] = rebind[Scalar[f32]](X[row, i + k * EW_THREADS])
        comptime for k in range(8):
            if vs[k] > best_v:
                best_v = vs[k]
                best_i = Int32(i + k * EW_THREADS)
        i += 8 * EW_THREADS
    while i < N:
        var v = rebind[Scalar[f32]](X[row, i])
        if v > best_v:
            best_v = v
            best_i = Int32(i)
        i += EW_THREADS

    var vals = stack_allocation[
        f32, address_space = AddressSpace.SHARED
    ](row_major[EW_THREADS]())
    var idxs = stack_allocation[
        DType.int32, address_space = AddressSpace.SHARED
    ](row_major[EW_THREADS]())
    vals[tid] = rebind[vals.ElementType](best_v)
    idxs[tid] = rebind[idxs.ElementType](best_i)
    barrier()

    if tid == 0:
        var bv = Float32(-3.4e38)
        var bi: Int32 = 0
        comptime for t in range(EW_THREADS):
            var v = rebind[Scalar[f32]](vals[t])
            var ix = rebind[Scalar[DType.int32]](idxs[t])
            if v > bv or (v == bv and ix < bi):
                bv = v
                bi = ix
        Out[Int(wpos) + Int(row)] = rebind[Out.ElementType](bi)


def amar_argmax_row[
    XLayout: TensorLayout, OLayout: TensorLayout
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    Out: TileTensor[DType.int32, OLayout, MutAnyOrigin],
    n: Int32,
):
    comptime assert X.flat_rank == 2 and Out.flat_rank == 1

    var N = Int(n)
    var row = block_idx.x
    var tid = thread_idx.x

    var best_v = Float32(-3.4e38)
    var best_i: Int32 = 0
    var i = tid
    while i < N:
        var v = rebind[Scalar[f32]](X[row, i])
        if v > best_v:
            best_v = v
            best_i = Int32(i)
        i += EW_THREADS

    var vals = stack_allocation[
        f32, address_space = AddressSpace.SHARED
    ](row_major[EW_THREADS]())
    var idxs = stack_allocation[
        DType.int32, address_space = AddressSpace.SHARED
    ](row_major[EW_THREADS]())
    vals[tid] = rebind[vals.ElementType](best_v)
    idxs[tid] = rebind[idxs.ElementType](best_i)
    barrier()

    if tid == 0:
        var bv = Float32(-3.4e38)
        var bi: Int32 = 0
        comptime for t in range(EW_THREADS):
            var v = rebind[Scalar[f32]](vals[t])
            var ix = rebind[Scalar[DType.int32]](idxs[t])
            if v > bv or (v == bv and ix < bi):
                bv = v
                bi = ix
        Out[row] = rebind[Out.ElementType](bi)


def amar_tok_copy[
    SLayout: TensorLayout, DLayout: TensorLayout
](
    Src: TileTensor[DType.int32, SLayout, MutAnyOrigin],
    Dst: TileTensor[DType.int32, DLayout, MutAnyOrigin],
    si: Int32,
    di: Int32,
    n: Int32,
):
    comptime assert Src.flat_rank == 1 and Dst.flat_rank == 1
    var i = global_idx.x
    if i >= Int(n):
        return
    Dst[Int(di) + i] = rebind[Dst.ElementType](Src[Int(si) + i])


def amar_tok_remap[
    MLayout: TensorLayout, DLayout: TensorLayout
](
    Map: TileTensor[DType.int32, MLayout, MutAnyOrigin],
    Dtok: TileTensor[DType.int32, DLayout, MutAnyOrigin],
    n: Int32,
):
    comptime assert Map.flat_rank == 1 and Dtok.flat_rank == 1
    var i = global_idx.x
    if i >= Int(n):
        return
    var j = Int(rebind[Scalar[DType.int32]](Dtok[i]))
    Dtok[i] = rebind[Dtok.ElementType](Map[j])


def amar_quantize_q8_rows[
    ALayout: TensorLayout, QLayout: TensorLayout, SLayout: TensorLayout
](
    A: TileTensor[DType.bfloat16, ALayout, MutAnyOrigin],
    Aq: TileTensor[DType.int8, QLayout, MutAnyOrigin],
    As: TileTensor[DType.float16, SLayout, MutAnyOrigin],
    m: Int32,
    k_dim: Int32,
):
    comptime assert A.flat_rank == 2 and Aq.flat_rank == 2 and As.flat_rank == 2

    var M = Int(m)
    var K = Int(k_dim)
    var row = Int(block_idx.x)
    var blk = Int(block_idx.y)
    var lane = Int(lane_id())
    var k = blk * 32 + lane
    if row >= M or k >= K:
        return

    var x = rebind[Scalar[DType.bfloat16]](A[row, k]).cast[f32]()
    var wmax = warp.max(abs(x))
    var scale = wmax / Float32(127) if wmax > 0 else Float32(1)
    var v = x / scale
    var q = Int32(v + Float32(0.5)) if v >= 0 else -Int32(-v + Float32(0.5))
    if q > 127:
        q = 127
    if q < -127:
        q = -127
    Aq[row, k] = rebind[Aq.ElementType](Int8(q))
    if lane == 0:
        As[row, blk] = rebind[As.ElementType](scale.cast[DType.float16]())


def amar_graft_mean[
    Layout: TensorLayout
](
    X: TileTensor[f32, Layout, MutAnyOrigin],
    V: TileTensor[f32, Layout, MutAnyOrigin],
    n: Int32,
):
    comptime assert X.flat_rank == 1
    var i = global_idx.x
    if i >= Int(n):
        return
    X[i] = (X[i] + V[i]) * 0.5


def amar_scale_one(p: MutPointer[Scalar[f32], MutAnyOrigin], idx: Int32, k: Float32):
    """p[idx] *= k: a neuron's gain applied to its FFN activation before the down projection."""
    if global_idx.x == 0:
        p[Int(idx)] = p[Int(idx)] * k
