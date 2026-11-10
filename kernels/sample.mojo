from std.atomic import Atomic
from max.gpu import block_idx, lane_id, thread_idx, WARP_SIZE
from max.gpu.primitives import warp
from std.math import ceil
from bexp import exp, log
from std.memory import bitcast
from std.utils.numerics import inf, nan
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from layout import TileTensor, TensorLayout, row_major, stack_allocation

comptime SAMP_THREADS = 1024
comptime SAMP_CAP = 2048
comptime NBAND = 257
comptime f32 = DType.float32
comptime u32 = DType.uint32
comptime u64 = DType.uint64
comptime i32 = DType.int32
comptime FMAX = Float32(3.4028234663852886e38)
comptime MASS_ONE = Float32(1099511627776.0)
comptime NO_IDX = Int32(2147483647)


@always_inline
def philox4x32(c: SIMD[u32, 4], k0: UInt32, k1: UInt32) -> SIMD[u32, 4]:
    var x = c
    var a = k0
    var b = k1
    comptime for _ in range(10):
        var p0 = UInt64(0xD2511F53) * x[0].cast[u64]()
        var p1 = UInt64(0xCD9E8D57) * x[2].cast[u64]()
        x = SIMD[u32, 4](
            (p1 >> 32).cast[u32]() ^ x[1] ^ a,
            p1.cast[u32](),
            (p0 >> 32).cast[u32]() ^ x[3] ^ b,
            p0.cast[u32](),
        )
        a += 0x9E3779B9
        b += 0xBB67AE85
    return x


@always_inline
def rng4(seed: UInt64, counter: UInt64, row: Int, stream: Int, g: Int) -> SIMD[u32, 4]:
    var c = SIMD[u32, 4](
        counter.cast[u32](),
        (counter >> 32).cast[u32](),
        UInt32(row),
        (UInt32(stream) << 28) | UInt32(g >> 2),
    )
    return philox4x32(c, seed.cast[u32](), (seed >> 32).cast[u32]())


@always_inline
def rng_word(seed: UInt64, counter: UInt64, row: Int, stream: Int, i: Int) -> UInt32:
    return rng4(seed, counter, row, stream, i)[i & 3]


@always_inline
def unif(w: UInt32) -> Float32:
    return ((w >> 8).cast[f32]() + 0.5) * Float32(5.9604644775390625e-08)


@always_inline
def gumbel(w: UInt32) -> Float32:
    return -log(-log(unif(w)))


@always_inline
def gumbel2(w1: UInt32, w2: UInt32) -> Float32:
    var u = ((w1 >> 5).cast[DType.float64]() * 67108864.0 + (w2 >> 6).cast[DType.float64]() + 0.5) * 1.1102230246251565e-16
    return (-log(-log(u))).cast[f32]()


@always_inline
def okey(v: Float32) -> UInt32:
    var b = bitcast[u32, 1](v)
    if (b >> 31) != 0:
        return ~b
    return b | 0x80000000


@always_inline
def okey_inv(k: UInt32) -> Float32:
    if (k >> 31) != 0:
        return bitcast[f32, 1](k & 0x7FFFFFFF)
    return bitcast[f32, 1](~k)


@always_inline
def is_valid(v: Float32) -> Bool:
    return v >= -FMAX and v <= FMAX


@always_inline
def band_of(v: Float32, lmax: Float32) -> Int:
    var t = (lmax - v) * 8.0
    if t < 256.0:
        return Int(t)
    return NBAND - 1


@always_inline
def fixed_mass(v: Float32, lmax: Float32) -> UInt64:
    return (exp(v - lmax) * MASS_ONE).cast[u64]()


@always_inline
def in_cut(k: UInt32, i: Int, ck: UInt32, ci: Int) -> Bool:
    return k > ck or (k == ck and i <= ci)


@always_inline
def load4[
    XL: TensorLayout
](X: TileTensor[f32, XL, MutAnyOrigin], base: Int, i: Int, N: Int) -> SIMD[f32, 4]:
    if i + 4 <= N:
        return X.ptr.unsafe_load[width=4, alignment=4](base + i)
    var r = SIMD[f32, 4](nan[f32]())
    comptime for e in range(4):
        if i + e < N:
            r[e] = X.ptr.unsafe_load[width=1](base + i + e)
    return r


@always_inline
def mload4[
    MASK: Bool, XL: TensorLayout
](X: TileTensor[f32, XL, MutAnyOrigin], base: Int, i: Int, N: Int, mask: MutPointer[Scalar[u64], MutAnyOrigin], mrow: Int) -> SIMD[f32, 4]:
    var a = load4(X, base, i, N)
    comptime if MASK:
        var mb = (mask[unsafe_offset=mrow + (i >> 6)] >> UInt64(i & 63)) & UInt64(15)
        comptime for e in range(4):
            if ((mb >> UInt64(e)) & 1) == 0:
                a[e] = nan[f32]()
    return a


@always_inline
def mload1[
    MASK: Bool, XL: TensorLayout
](X: TileTensor[f32, XL, MutAnyOrigin], row: Int, i: Int, mask: MutPointer[Scalar[u64], MutAnyOrigin], mrow: Int) -> Float32:
    var v = rebind[Scalar[f32]](X[row, i])
    comptime if MASK:
        if ((mask[unsafe_offset=mrow + (i >> 6)] >> UInt64(i & 63)) & UInt64(1)) == 0:
            return nan[f32]()
    return v


comptime NWAVE = SAMP_THREADS // WARP_SIZE


@always_inline
def bsum_u64[
    RL: TensorLayout
](
    mut red: TileTensor[u64, RL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    tid: Int,
    v: UInt64,
) -> UInt64:
    comptime assert red.flat_rank == 1
    var s = warp.sum(v)
    if lane_id() == 0:
        red[tid // WARP_SIZE] = rebind[red.ElementType](s)
    barrier()
    if tid < WARP_SIZE:
        var x: UInt64 = 0
        if tid < NWAVE:
            x = rebind[Scalar[u64]](red[tid])
        var t = warp.sum(x)
        if tid == 0:
            red[NWAVE] = rebind[red.ElementType](t)
    barrier()
    var r = rebind[Scalar[u64]](red[NWAVE])
    barrier()
    return r


@always_inline
def bsum_f32[
    RL: TensorLayout
](
    mut red: TileTensor[f32, RL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    tid: Int,
    v: Float32,
) -> Float32:
    comptime assert red.flat_rank == 1
    var s = warp.sum(v)
    if lane_id() == 0:
        red[tid // WARP_SIZE] = rebind[red.ElementType](s)
    barrier()
    if tid < WARP_SIZE:
        var x: Float32 = 0
        if tid < NWAVE:
            x = rebind[Scalar[f32]](red[tid])
        var t = warp.sum(x)
        if tid == 0:
            red[NWAVE] = rebind[red.ElementType](t)
    barrier()
    var r = rebind[Scalar[f32]](red[NWAVE])
    barrier()
    return r


@always_inline
def bmax_f32[
    RL: TensorLayout
](
    mut red: TileTensor[f32, RL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    tid: Int,
    v: Float32,
) -> Float32:
    comptime assert red.flat_rank == 1
    var s = warp.max(v)
    if lane_id() == 0:
        red[tid // WARP_SIZE] = rebind[red.ElementType](s)
    barrier()
    if tid < WARP_SIZE:
        var x = -FMAX
        if tid < NWAVE:
            x = rebind[Scalar[f32]](red[tid])
        var t = warp.max(x)
        if tid == 0:
            red[NWAVE] = rebind[red.ElementType](t)
    barrier()
    var r = rebind[Scalar[f32]](red[NWAVE])
    barrier()
    return r


@always_inline
def bargmax[
    FL: TensorLayout, IL: TensorLayout
](
    mut redf: TileTensor[f32, FL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    mut redi: TileTensor[i32, IL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    tid: Int,
    v: Float32,
    ix: Int32,
) -> Int32:
    comptime assert redf.flat_rank == 1 and redi.flat_rank == 1
    var m = warp.max(v)
    var c = warp.min(ix if v == m else NO_IDX)
    if lane_id() == 0:
        redf[tid // WARP_SIZE] = rebind[redf.ElementType](m)
        redi[tid // WARP_SIZE] = rebind[redi.ElementType](c)
    barrier()
    if tid < WARP_SIZE:
        var v2 = Float32(-3.4028234663852886e38)
        var i2 = NO_IDX
        if tid < NWAVE:
            v2 = rebind[Scalar[f32]](redf[tid])
            i2 = rebind[Scalar[i32]](redi[tid])
        var m2 = warp.max(v2)
        var c2 = warp.min(i2 if v2 == m2 else NO_IDX)
        if tid == 0:
            redi[NWAVE] = rebind[redi.ElementType](c2)
    barrier()
    var r = rebind[Scalar[i32]](redi[NWAVE])
    barrier()
    return r


@always_inline
def greedy_tok[
    MASK: Bool, XL: TensorLayout, FL: TensorLayout, IL: TensorLayout
](
    X: TileTensor[f32, XL, MutAnyOrigin],
    mut redf: TileTensor[f32, FL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    mut redi: TileTensor[i32, IL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    base: Int,
    N: Int,
    tid: Int,
    mask: MutPointer[Scalar[u64], MutAnyOrigin],
    mrow: Int,
) -> Int32:
    var best_v = Float32(-3.4e38)
    var best_i: Int32 = NO_IDX if MASK else 0
    var g = tid * 4
    while g < N:
        var g2 = g + 4 * SAMP_THREADS
        var a = mload4[MASK](X, base, g, N, mask, mrow)
        var b = mload4[MASK](X, base, g2, N, mask, mrow)
        comptime for e in range(4):
            if a[e] > best_v:
                best_v = a[e]
                best_i = Int32(g + e)
        comptime for e in range(4):
            if b[e] > best_v:
                best_v = b[e]
                best_i = Int32(g2 + e)
        g += 8 * SAMP_THREADS
    return bargmax(redf, redi, tid, best_v, best_i)


def refine[
    MASK: Bool, XL: TensorLayout, HL: TensorLayout, SL: TensorLayout
](
    X: TileTensor[f32, XL, MutAnyOrigin],
    mut hist: TileTensor[u64, HL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    mut st: TileTensor[u64, SL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    row: Int,
    N: Int,
    tid: Int,
    lmax: Float32,
    b: Int,
    ck: UInt32,
    ci: Int,
    use_mass: Bool,
    W: UInt64,
    mask: MutPointer[Scalar[u64], MutAnyOrigin],
    mrow: Int,
) -> Tuple[UInt32, Int]:
    comptime assert X.flat_rank == 2 and hist.flat_rank == 1 and st.flat_rank == 1
    var prefix: UInt32 = 0
    var pmask: UInt32 = 0
    var need = W
    var tiew: UInt64 = 0
    for d in range(4):
        var sh = UInt32(24 - 8 * d)
        if tid < 256:
            hist[tid] = 0
        barrier()
        var i = tid
        while i < N:
            var v = mload1[MASK](X, row, i, mask, mrow)
            if is_valid(v) and band_of(v, lmax) == b:
                var k = okey(v)
                if (k & pmask) == prefix and in_cut(k, i, ck, ci):
                    var w: UInt64 = 1
                    if use_mass:
                        w = fixed_mass(v, lmax)
                    _ = Atomic.fetch_add(hist.ptr.unsafe_offset(Int((k >> sh) & 255)), w)
            i += SAMP_THREADS
        barrier()
        if tid == 0:
            var acc: UInt64 = 0
            var sel = 0
            for bb in range(255, -1, -1):
                var h = rebind[Scalar[u64]](hist[bb])
                if acc + h >= need:
                    sel = bb
                    break
                acc += h
            st[0] = rebind[st.ElementType]((prefix | (UInt32(sel) << sh)).cast[u64]())
            st[1] = rebind[st.ElementType](need - acc)
            st[2] = hist[sel]
        barrier()
        prefix = rebind[Scalar[u64]](st[0]).cast[u32]()
        need = rebind[Scalar[u64]](st[1])
        tiew = rebind[Scalar[u64]](st[2])
        pmask |= UInt32(255) << sh
    barrier()

    var w0: UInt64 = 1
    if use_mass:
        w0 = fixed_mass(okey_inv(prefix), lmax)
        if w0 == 0:
            w0 = 1
    var ties = tiew // w0
    var r = (need + w0 - 1) // w0
    if r >= ties:
        if prefix == ck:
            return (prefix, ci)
        return (prefix, N)

    var nbits = 0
    while (1 << nbits) < N:
        nbits += 1
    var ipre = 0
    var imask = 0
    var rn = r
    for d in range((nbits + 7) // 8 - 1, -1, -1):
        var sh = 8 * d
        if tid < 256:
            hist[tid] = 0
        barrier()
        var i = tid
        while i < N:
            var v = mload1[MASK](X, row, i, mask, mrow)
            if is_valid(v) and band_of(v, lmax) == b:
                var k = okey(v)
                if k == prefix and in_cut(k, i, ck, ci) and (i & imask) == ipre:
                    _ = Atomic.fetch_add(hist.ptr.unsafe_offset((i >> sh) & 255), UInt64(1))
            i += SAMP_THREADS
        barrier()
        if tid == 0:
            var acc: UInt64 = 0
            var sel = 255
            for bb in range(256):
                var h = rebind[Scalar[u64]](hist[bb])
                if acc + h >= rn:
                    sel = bb
                    break
                acc += h
            st[0] = rebind[st.ElementType](UInt64(ipre | (sel << sh)))
            st[1] = rebind[st.ElementType](rn - acc)
        barrier()
        ipre = Int(rebind[Scalar[u64]](st[0]))
        rn = rebind[Scalar[u64]](st[1])
        imask |= 255 << sh
    barrier()
    return (prefix, ipre)


@always_inline
def pmass_target(top_p: Float32, z: UInt64) -> UInt64:
    var W = ceil(Float64(top_p) * z.cast[DType.float64]()).cast[u64]()
    if W < 1:
        W = 1
    if W > z:
        W = z
    return W


@always_inline
def samp_update(
    v: Float32, lmax: Float32, dth: SIMD[f32, 16], mut cnt: SIMD[f32, 16], mut ms: SIMD[f32, 16], z_on: Bool,
):
    if not is_valid(v):
        return
    var inb = dth.ge(SIMD[f32, 16](lmax - v))
    cnt += inb.select(SIMD[f32, 16](1), SIMD[f32, 16](0))
    if z_on:
        ms += inb.select(SIMD[f32, 16](exp(v - lmax)), SIMD[f32, 16](0))


def fast_cut[
    MASK: Bool, XL: TensorLayout, SL: TensorLayout, CL: TensorLayout, UL: TensorLayout,
    FL: TensorLayout, IL: TensorLayout, CAP: Int
](
    X: TileTensor[f32, XL, MutAnyOrigin],
    mut st: TileTensor[u64, SL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    mut srt: TileTensor[u64, CL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    mut mas: TileTensor[u64, CL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    mut redu: TileTensor[u64, UL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    mut redf: TileTensor[f32, FL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    mut redi: TileTensor[i32, IL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    base: Int,
    N: Int,
    tid: Int,
    lmax: Float32,
    k_on: Bool,
    top_k: Int32,
    p_on: Bool,
    top_p: Float32,
    min_p: Float32,
    mask: MutPointer[Scalar[u64], MutAnyOrigin],
    mrow: Int,
) -> Tuple[Bool, UInt32, Int]:
    comptime assert st.flat_rank == 1 and srt.flat_rank == 1 and mas.flat_rank == 1
    comptime assert redf.flat_rank == 1 and redi.flat_rank == 1
    var dth = SIMD[f32, 16](
        0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 32.0, inf[f32]()
    )
    var z_on = p_on and not k_on
    var cnt = SIMD[f32, 16](0)
    var ms = SIMD[f32, 16](0)
    var nsamp = N
    if N <= 16384:
        var g = tid * 4
        while g < N:
            var a = mload4[MASK](X, base, g, N, mask, mrow)
            comptime for e in range(4):
                samp_update(a[e], lmax, dth, cnt, ms, z_on)
            g += 4 * SAMP_THREADS
    else:
        nsamp = 16384
        var g0 = ((((tid // 64) * N) // 16) & ~3) + (tid % 64) * 16
        comptime for q in range(4):
            var a = mload4[MASK](X, base, g0 + 4 * q, N, mask, mrow)
            comptime for e in range(4):
                samp_update(a[e], lmax, dth, cnt, ms, z_on)
    var wv = tid // WARP_SIZE
    comptime for j in range(16):
        var cj = warp.sum(cnt[j])
        var mj = warp.sum(ms[j])
        if lane_id() == 0:
            redf[wv * 32 + j] = rebind[redf.ElementType](cj)
            redf[wv * 32 + 16 + j] = rebind[redf.ElementType](mj)
    barrier()
    var tot: Float32 = 0
    if tid < 32:
        for w in range(NWAVE):
            tot += rebind[Scalar[f32]](redf[w * 32 + tid])
    barrier()
    if tid < 32:
        redf[tid] = rebind[redf.ElementType](tot)
    barrier()
    var kk = Int(top_k)
    var scale = Float32(N) / Float32(nsamp)
    var need = Float32(2 * kk + 16)
    var r = top_p + 0.5 * (1.0 - top_p)
    var mall = rebind[Scalar[f32]](redf[31])
    var j = -1
    for jj in range(15):
        if j < 0:
            var c = rebind[Scalar[f32]](redf[jj]) * scale
            var m = rebind[Scalar[f32]](redf[16 + jj])
            if (k_on and c >= need) or (not k_on and m >= r * mall):
                j = jj
    if j < 0:
        j = 14
    barrier()

    var ncand = 0
    var zt: UInt64 = 0
    var ok = False
    for _ in range(4):
        var dsel = dth[j]
        if tid == 0:
            st[3] = 0
        barrier()
        var zp: UInt64 = 0
        var cm: UInt64 = 0
        var g = tid * 4
        while g < N:
            var a = mload4[MASK](X, base, g, N, mask, mrow)
            comptime for e in range(4):
                var v = a[e]
                if is_valid(v):
                    var w: UInt64 = 0
                    if z_on:
                        w = fixed_mass(v, lmax)
                        zp += w
                    if lmax - v <= dsel:
                        cm += w
                        var slot = Int(Atomic.fetch_add(st.ptr.unsafe_offset(3), UInt64(1)))
                        if slot < CAP:
                            srt[slot] = rebind[srt.ElementType](
                                (okey(v).cast[u64]() << 32) | (UInt32(0xFFFFFFFF) - UInt32(g + e)).cast[u64]()
                            )
            g += 4 * SAMP_THREADS
        barrier()
        ncand = Int(rebind[Scalar[u64]](st[3]))
        var cmass: UInt64 = 0
        if z_on:
            zt = bsum_u64(redu, tid, zp)
            cmass = bsum_u64(redu, tid, cm)
        barrier()
        if ncand > CAP:
            if j == 0:
                break
            j -= 1
            continue
        if k_on and ncand < kk:
            if j == 14:
                break
            j += 1
            continue
        if z_on and cmass < pmass_target(top_p, zt):
            if j == 14:
                break
            j += 1
            continue
        ok = True
        break
    if not ok:
        return (False, UInt32(0), 0)

    var P = 2
    while P < ncand:
        P <<= 1
    var t = ncand + tid
    while t < P:
        srt[t] = 0
        t += SAMP_THREADS
    barrier()
    var size = 2
    while size <= P:
        var stride = size >> 1
        while stride > 0:
            t = tid
            while t < (P >> 1):
                var lo = ((t & ~(stride - 1)) << 1) | (t & (stride - 1))
                var hi = lo + stride
                var x0 = rebind[Scalar[u64]](srt[lo])
                var x1 = rebind[Scalar[u64]](srt[hi])
                if (x0 < x1) == ((lo & size) == 0):
                    srt[lo] = rebind[srt.ElementType](x1)
                    srt[hi] = rebind[srt.ElementType](x0)
                t += SAMP_THREADS
            barrier()
            stride >>= 1
        size <<= 1

    var mk = kk if k_on else ncand
    var zp: UInt64 = 0
    t = tid
    while t < mk:
        var w = fixed_mass(okey_inv((rebind[Scalar[u64]](srt[t]) >> 32).cast[u32]()), lmax)
        mas[t] = rebind[mas.ElementType](w)
        zp += w
        t += SAMP_THREADS
    var zc = bsum_u64(redu, tid, zp)
    if not k_on:
        zc = zt
    if tid == 0:
        var m = mk
        var ok: UInt64 = 1
        if p_on:
            var W = ceil(Float64(top_p) * zc.cast[DType.float64]()).cast[u64]()
            if W < 1:
                W = 1
            if W > zc:
                W = zc
            var cum: UInt64 = 0
            var mp = -1
            for j in range(mk):
                cum += rebind[Scalar[u64]](mas[j])
                if cum >= W:
                    mp = j + 1
                    break
            if mp < 0:
                ok = 0
            else:
                m = mp
        if min_p > 0:
            var mpe = min(min_p, Float32(1))
            var mm = 0
            while mm < m:
                var v = okey_inv((rebind[Scalar[u64]](srt[mm]) >> 32).cast[u32]())
                if not (exp(v - lmax) >= mpe):
                    break
                mm += 1
            m = max(mm, 1)
        var last = rebind[Scalar[u64]](srt[m - 1])
        st[0] = rebind[st.ElementType](ok)
        st[1] = rebind[st.ElementType](last >> 32)
        st[2] = rebind[st.ElementType](UInt64(0xFFFFFFFF) - (last & 0xFFFFFFFF))
    barrier()
    var okf = rebind[Scalar[u64]](st[0]) != 0
    var ck = rebind[Scalar[u64]](st[1]).cast[u32]()
    var ci = Int(rebind[Scalar[u64]](st[2]))
    barrier()
    return (okf, ck, ci)


def sample_cut[
    MASK: Bool, XL: TensorLayout, HL: TensorLayout, BL: TensorLayout, SL: TensorLayout,
    CL: TensorLayout, UL: TensorLayout, FL: TensorLayout, IL: TensorLayout, CAP: Int
](
    X: TileTensor[f32, XL, MutAnyOrigin],
    mut hist: TileTensor[u64, HL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    mut hc: TileTensor[u64, BL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    mut hm: TileTensor[u64, BL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    mut st: TileTensor[u64, SL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    mut srt: TileTensor[u64, CL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    mut mas: TileTensor[u64, CL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    mut redu: TileTensor[u64, UL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    mut redf: TileTensor[f32, FL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    mut redi: TileTensor[i32, IL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    row: Int,
    base: Int,
    N: Int,
    tid: Int,
    top_k: Int32,
    top_p: Float32,
    min_p: Float32,
    mask: MutPointer[Scalar[u64], MutAnyOrigin],
    mrow: Int,
) -> Tuple[Float32, UInt64, UInt32, Int]:
    comptime assert X.flat_rank == 2 and hc.flat_rank == 1 and hm.flat_rank == 1 and st.flat_rank == 1
    var lm = -FMAX
    var cnt: UInt64 = 0
    var g = tid * 4
    while g < N:
        var g2 = g + 4 * SAMP_THREADS
        var a = mload4[MASK](X, base, g, N, mask, mrow)
        var b = mload4[MASK](X, base, g2, N, mask, mrow)
        comptime for e in range(4):
            if is_valid(a[e]):
                if a[e] > lm:
                    lm = a[e]
                cnt += 1
        comptime for e in range(4):
            if is_valid(b[e]):
                if b[e] > lm:
                    lm = b[e]
                cnt += 1
        g += 8 * SAMP_THREADS
    var lmax = bmax_f32(redf, tid, lm)
    var nvalid = bsum_u64(redu, tid, cnt)
    var ck: UInt32 = 0
    var ci = N
    if nvalid == 0:
        return (lmax, nvalid, ck, ci)

    var k_on = top_k > 0 and UInt64(Int(top_k)) < nvalid
    var p_on = top_p < 1.0
    if not k_on and not p_on:
        return (lmax, nvalid, ck, ci)

    var fc = fast_cut[MASK, CAP=CAP](
        X, st, srt, mas, redu, redf, redi, base, N, tid, lmax, k_on, top_k, p_on, top_p, min_p, mask, mrow
    )
    if fc[0]:
        return (lmax, nvalid, fc[1], fc[2])

    if tid < NBAND:
        hc[tid] = 0
        hm[tid] = 0
    barrier()
    var i = tid
    while i < N:
        var v = mload1[MASK](X, row, i, mask, mrow)
        if is_valid(v):
            var bi = band_of(v, lmax)
            _ = Atomic.fetch_add(hc.ptr.unsafe_offset(bi), UInt64(1))
            _ = Atomic.fetch_add(hm.ptr.unsafe_offset(bi), fixed_mass(v, lmax))
        i += SAMP_THREADS
    barrier()

    var bk = NBAND
    if k_on:
        if tid == 0:
            var kk = UInt64(Int(top_k))
            var cum: UInt64 = 0
            var sel = NBAND - 1
            for bb in range(NBAND):
                var h = rebind[Scalar[u64]](hc[bb])
                if cum + h >= kk:
                    sel = bb
                    break
                cum += h
            st[0] = rebind[st.ElementType](UInt64(sel))
            st[1] = rebind[st.ElementType](kk - cum)
        barrier()
        bk = Int(rebind[Scalar[u64]](st[0]))
        var needk = rebind[Scalar[u64]](st[1])
        barrier()
        var cut = refine[MASK](X, hist, st, row, N, tid, lmax, bk, ck, ci, False, needk, mask, mrow)
        ck = cut[0]
        ci = cut[1]

    if p_on:
        var z: UInt64 = 0
        if k_on:
            i = tid
            while i < N:
                var v = mload1[MASK](X, row, i, mask, mrow)
                if is_valid(v) and in_cut(okey(v), i, ck, ci):
                    z += fixed_mass(v, lmax)
                i += SAMP_THREADS
            z = bsum_u64(redu, tid, z)
        else:
            var hv: UInt64 = 0
            if tid < NBAND:
                hv = rebind[Scalar[u64]](hm[tid])
            z = bsum_u64(redu, tid, hv)
        var W = ceil(Float64(top_p) * z.cast[DType.float64]()).cast[u64]()
        if W < 1:
            W = 1
        if W > z:
            W = z
        if tid == 0:
            var cum: UInt64 = 0
            var sel = bk
            var lim = bk if bk < NBAND else NBAND
            for bb in range(lim):
                var h = rebind[Scalar[u64]](hm[bb])
                if cum + h >= W:
                    sel = bb
                    break
                cum += h
            st[0] = rebind[st.ElementType](UInt64(sel))
            st[1] = rebind[st.ElementType](W - cum)
        barrier()
        var bp = Int(rebind[Scalar[u64]](st[0]))
        var needp = rebind[Scalar[u64]](st[1])
        barrier()
        var cut = refine[MASK](X, hist, st, row, N, tid, lmax, bp, ck, ci, True, needp, mask, mrow)
        ck = cut[0]
        ci = cut[1]
    return (lmax, nvalid, ck, ci)


@always_inline
def cut_val(ck: UInt32) -> Float32:
    if ck == 0:
        return -FMAX
    return okey_inv(ck)


@always_inline
def member(v: Float32, i: Int, lmax: Float32, vcut: Float32, ck: UInt32, ci: Int, mpe: Float32) -> Bool:
    if not (v <= FMAX):
        return False
    if not (v > vcut or (v == vcut and in_cut(okey(v), i, ck, ci))):
        return False
    return mpe <= 0 or exp(v - lmax) >= mpe


@always_inline
def sample_row_body[
    MASK: Bool, XLayout: TensorLayout, OLayout: TensorLayout, PLayout: TensorLayout, CAP: Int
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    Out: TileTensor[i32, OLayout, MutAnyOrigin],
    Prob: TileTensor[f32, PLayout, MutAnyOrigin],
    n: Int32,
    temperature: Float32,
    top_k: Int32,
    top_p: Float32,
    min_p: Float32,
    seed: UInt64,
    counter: UInt64,
    mask: MutPointer[Scalar[u64], MutAnyOrigin],
    mask_stride: Int,
):
    comptime assert X.flat_rank == 2 and Out.flat_rank == 1 and Prob.flat_rank == 1
    var N = Int(n)
    var row = block_idx.x
    var tid = thread_idx.x
    var base = row * Int(X.dim[1]())
    var mrow = Int(row) * mask_stride
    var redf = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[SAMP_THREADS]())
    var redi = stack_allocation[i32, address_space = AddressSpace.SHARED](row_major[SAMP_THREADS]())

    if temperature <= 0:
        var g = greedy_tok[MASK](X, redf, redi, base, N, tid, mask, mrow)
        if tid == 0:
            if g == NO_IDX:
                Out[row] = rebind[Out.ElementType](Int32(-1))
                Prob[row] = rebind[Prob.ElementType](Float32(0))
            else:
                Out[row] = rebind[Out.ElementType](g)
                Prob[row] = rebind[Prob.ElementType](Float32(1))
        return

    var hist = stack_allocation[u64, address_space = AddressSpace.SHARED](row_major[256]())
    var hc = stack_allocation[u64, address_space = AddressSpace.SHARED](row_major[NBAND]())
    var hm = stack_allocation[u64, address_space = AddressSpace.SHARED](row_major[NBAND]())
    var st = stack_allocation[u64, address_space = AddressSpace.SHARED](row_major[4]())
    var srt = stack_allocation[u64, address_space = AddressSpace.SHARED](row_major[CAP]())
    var mas = stack_allocation[u64, address_space = AddressSpace.SHARED](row_major[CAP]())
    var redu = stack_allocation[u64, address_space = AddressSpace.SHARED](row_major[SAMP_THREADS]())
    var cut = sample_cut[MASK, CAP=CAP](
        X, hist, hc, hm, st, srt, mas, redu, redf, redi, row, base, N, tid, top_k, top_p, min_p, mask, mrow
    )
    var lmax = cut[0]
    var ck = cut[2]
    var ci = cut[3]
    if cut[1] == 0:
        if tid == 0:
            Out[row] = rebind[Out.ElementType](Int32(-1))
            Prob[row] = rebind[Prob.ElementType](Float32(0))
        return

    var mpe = min(min_p, Float32(1))
    var vcut = cut_val(ck)
    var bs = -FMAX
    var bi = NO_IDX
    var zt: Float32 = 0
    var g = tid * 4
    while g < N:
        var a = load4(X, base, g, N)
        var mb = UInt64(15)
        comptime if MASK:
            mb = (mask[unsafe_offset=mrow + (g >> 6)] >> UInt64(g & 63)) & UInt64(15)
        var hit = False
        comptime for e in range(4):
            if ((mb >> UInt64(e)) & 1) != 0 and member(a[e], g + e, lmax, vcut, ck, ci, mpe):
                hit = True
        if hit:
            var w = rng4(seed, counter, row, 0, g)
            var w2 = rng4(seed, counter, row, 4, g)
            comptime for e in range(4):
                var v = a[e]
                if ((mb >> UInt64(e)) & 1) != 0 and member(v, g + e, lmax, vcut, ck, ci, mpe):
                    var ev = (v - lmax) / temperature
                    var s = ev + gumbel2(w[e], w2[e])
                    if s > bs:
                        bs = s
                        bi = Int32(g + e)
                    zt += exp(ev)
        g += 4 * SAMP_THREADS
    var tok = bargmax(redf, redi, tid, bs, bi)
    var z = bsum_f32(redf, tid, zt)
    if tid == 0:
        var lt = rebind[Scalar[f32]](X[row, Int(tok)])
        Out[row] = rebind[Out.ElementType](tok)
        Prob[row] = rebind[Prob.ElementType](exp((lt - lmax) / temperature) / z)


def amar_sample_row[
    XLayout: TensorLayout, OLayout: TensorLayout, PLayout: TensorLayout, CAP: Int = SAMP_CAP
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    Out: TileTensor[i32, OLayout, MutAnyOrigin],
    Prob: TileTensor[f32, PLayout, MutAnyOrigin],
    n: Int32,
    temperature: Float32,
    top_k: Int32,
    top_p: Float32,
    min_p: Float32,
    seed: UInt64,
    counter: UInt64,
):
    sample_row_body[False, XLayout, OLayout, PLayout, CAP](X, Out, Prob, n, temperature, top_k, top_p, min_p, seed, counter, X.ptr.unsafe_bitcast[Scalar[u64]](), 0)


def amar_sample_row_masked[
    XLayout: TensorLayout, OLayout: TensorLayout, PLayout: TensorLayout, CAP: Int = SAMP_CAP
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    Out: TileTensor[i32, OLayout, MutAnyOrigin],
    Prob: TileTensor[f32, PLayout, MutAnyOrigin],
    n: Int32,
    temperature: Float32,
    top_k: Int32,
    top_p: Float32,
    min_p: Float32,
    seed: UInt64,
    counter: UInt64,
    mask: MutPointer[Scalar[u64], MutAnyOrigin],
    mask_stride: Int32,
):
    sample_row_body[True, XLayout, OLayout, PLayout, CAP](X, Out, Prob, n, temperature, top_k, top_p, min_p, seed, counter, mask, Int(mask_stride))


@always_inline
def sample_probs_body[
    MASK: Bool, XLayout: TensorLayout, PLayout: TensorLayout, CAP: Int
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    P: TileTensor[f32, PLayout, MutAnyOrigin],
    n: Int32,
    temperature: Float32,
    top_k: Int32,
    top_p: Float32,
    min_p: Float32,
    mask: MutPointer[Scalar[u64], MutAnyOrigin],
    mask_stride: Int,
):
    comptime assert X.flat_rank == 2 and P.flat_rank == 2
    var N = Int(n)
    var row = block_idx.x
    var tid = thread_idx.x
    var base = row * Int(X.dim[1]())
    var mrow = Int(row) * mask_stride
    var redf = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[SAMP_THREADS]())
    var redi = stack_allocation[i32, address_space = AddressSpace.SHARED](row_major[SAMP_THREADS]())

    if temperature <= 0:
        var gt = Int(greedy_tok[MASK](X, redf, redi, base, N, tid, mask, mrow))
        var i = tid
        while i < N:
            P[row, i] = rebind[P.ElementType](Float32(1) if i == gt else Float32(0))
            i += SAMP_THREADS
        return

    var hist = stack_allocation[u64, address_space = AddressSpace.SHARED](row_major[256]())
    var hc = stack_allocation[u64, address_space = AddressSpace.SHARED](row_major[NBAND]())
    var hm = stack_allocation[u64, address_space = AddressSpace.SHARED](row_major[NBAND]())
    var st = stack_allocation[u64, address_space = AddressSpace.SHARED](row_major[4]())
    var srt = stack_allocation[u64, address_space = AddressSpace.SHARED](row_major[CAP]())
    var mas = stack_allocation[u64, address_space = AddressSpace.SHARED](row_major[CAP]())
    var redu = stack_allocation[u64, address_space = AddressSpace.SHARED](row_major[SAMP_THREADS]())
    var cut = sample_cut[MASK, CAP=CAP](
        X, hist, hc, hm, st, srt, mas, redu, redf, redi, row, base, N, tid, top_k, top_p, min_p, mask, mrow
    )
    var lmax = cut[0]
    var ck = cut[2]
    var ci = cut[3]
    var none = cut[1] == 0
    var mpe = min(min_p, Float32(1))
    var vcut = cut_val(ck)
    var zt: Float32 = 0
    var g = tid * 4
    while g < N and not none:
        var a = mload4[MASK](X, base, g, N, mask, mrow)
        comptime for e in range(4):
            if member(a[e], g + e, lmax, vcut, ck, ci, mpe):
                zt += exp((a[e] - lmax) / temperature)
        g += 4 * SAMP_THREADS
    var z = bsum_f32(redf, tid, zt)
    g = tid * 4
    while g < N:
        var a = mload4[MASK](X, base, g, N, mask, mrow)
        comptime for e in range(4):
            if g + e < N:
                var p: Float32 = 0
                if not none and member(a[e], g + e, lmax, vcut, ck, ci, mpe):
                    p = exp((a[e] - lmax) / temperature) / z
                P[row, g + e] = rebind[P.ElementType](p)
        g += 4 * SAMP_THREADS



def amar_sample_probs[
    XLayout: TensorLayout, PLayout: TensorLayout, CAP: Int = SAMP_CAP
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    P: TileTensor[f32, PLayout, MutAnyOrigin],
    n: Int32,
    temperature: Float32,
    top_k: Int32,
    top_p: Float32,
    min_p: Float32,
):
    sample_probs_body[False, XLayout, PLayout, CAP](X, P, n, temperature, top_k, top_p, min_p, X.ptr.unsafe_bitcast[Scalar[u64]](), 0)


def amar_sample_probs_masked[
    XLayout: TensorLayout, PLayout: TensorLayout, CAP: Int = SAMP_CAP
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    P: TileTensor[f32, PLayout, MutAnyOrigin],
    n: Int32,
    temperature: Float32,
    top_k: Int32,
    top_p: Float32,
    min_p: Float32,
    mask: MutPointer[Scalar[u64], MutAnyOrigin],
    mask_stride: Int32,
):
    sample_probs_body[True, XLayout, PLayout, CAP](X, P, n, temperature, top_k, top_p, min_p, mask, Int(mask_stride))


def amar_spec_accept[
    PLayout: TensorLayout, TLayout: TensorLayout
](
    Pt: TileTensor[f32, PLayout, MutAnyOrigin],
    Pd: TileTensor[f32, PLayout, MutAnyOrigin],
    Dtok: TileTensor[i32, TLayout, MutAnyOrigin],
    Out: TileTensor[i32, TLayout, MutAnyOrigin],
    Acc: TileTensor[i32, TLayout, MutAnyOrigin],
    n: Int32,
    seed: UInt64,
    counter: UInt64,
):
    comptime assert Pt.flat_rank == 2 and Pd.flat_rank == 2 and Dtok.flat_rank == 1
    comptime assert Out.flat_rank == 1 and Acc.flat_rank == 1
    var N = Int(n)
    var row = block_idx.x
    var tid = thread_idx.x
    var base = row * Int(Pt.dim[1]())
    var redf = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[SAMP_THREADS]())
    var redi = stack_allocation[i32, address_space = AddressSpace.SHARED](row_major[SAMP_THREADS]())

    var x = Int(rebind[Scalar[i32]](Dtok[row]))
    var accept = False
    if x >= 0 and x < N:
        var pt = rebind[Scalar[f32]](Pt[row, x])
        var pd = rebind[Scalar[f32]](Pd[row, x])
        accept = unif(rng_word(seed, counter, row, 1, 0)) * pd < pt
    if accept:
        if tid == 0:
            Out[row] = rebind[Out.ElementType](Int32(x))
            Acc[row] = rebind[Acc.ElementType](Int32(1))
        return

    var bs = -FMAX
    var bi = NO_IDX
    var g = tid * 4
    while g < N:
        var at = load4(Pt, base, g, N)
        var ad = load4(Pd, base, g, N)
        var r = at - ad
        if r.gt(SIMD[f32, 4](0)).reduce_or():
            var w = rng4(seed, counter, row, 2, g)
            var w2 = rng4(seed, counter, row, 6, g)
            comptime for e in range(4):
                if r[e] > 0:
                    var s = log(r[e]) + gumbel2(w[e], w2[e])
                    if s > bs:
                        bs = s
                        bi = Int32(g + e)
        g += 4 * SAMP_THREADS
    var tok = bargmax(redf, redi, tid, bs, bi)
    if tok == NO_IDX:
        bs = -FMAX
        bi = NO_IDX
        g = tid * 4
        while g < N:
            var at = load4(Pt, base, g, N)
            if at.gt(SIMD[f32, 4](0)).reduce_or():
                var w = rng4(seed, counter, row, 3, g)
                var w2 = rng4(seed, counter, row, 7, g)
                comptime for e in range(4):
                    if at[e] > 0:
                        var s = log(at[e]) + gumbel2(w[e], w2[e])
                        if s > bs:
                            bs = s
                            bi = Int32(g + e)
            g += 4 * SAMP_THREADS
        tok = bargmax(redf, redi, tid, bs, bi)
    if tid == 0:
        Out[row] = rebind[Out.ElementType](Int32(-1) if tok == NO_IDX else tok)
        Acc[row] = rebind[Acc.ElementType](Int32(0))


comptime PEN_THREADS = 256


def amar_apply_penalties[
    XLayout: TensorLayout, ILayout: TensorLayout, CLayout: TensorLayout, NLayout: TensorLayout
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    Ids: TileTensor[i32, ILayout, MutAnyOrigin],
    Cnt: TileTensor[i32, CLayout, MutAnyOrigin],
    Npen: TileTensor[i32, NLayout, MutAnyOrigin],
    n: Int32,
    presence: Float32,
    frequency: Float32,
):
    comptime assert X.flat_rank == 2 and Ids.flat_rank == 2 and Cnt.flat_rank == 2 and Npen.flat_rank == 1
    var row = Int(block_idx.x)
    var N = Int(n)
    var m = Int(rebind[Scalar[i32]](Npen[row]))
    var cap = Int(Ids.dim[1]())
    if m > cap:
        m = cap
    var i = Int(thread_idx.x)
    while i < m:
        var t = Int(rebind[Scalar[i32]](Ids[row, i]))
        var c = Int(rebind[Scalar[i32]](Cnt[row, i]))
        if t >= 0 and t < N and c > 0:
            X[row, t] = rebind[X.ElementType](
                rebind[Scalar[f32]](X[row, t]) - (presence + frequency * Float32(c))
            )
        i += PEN_THREADS


def amar_topn_probs[
    XLayout: TensorLayout, ILayout: TensorLayout, PLayout: TensorLayout, CAP: Int = SAMP_CAP
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    TopIds: TileTensor[i32, ILayout, MutAnyOrigin],
    TopProbs: TileTensor[f32, PLayout, MutAnyOrigin],
    n: Int32,
    nsel: Int32,
    temperature: Float32,
    top_k: Int32,
    top_p: Float32,
    min_p: Float32,
):
    comptime assert X.flat_rank == 2 and TopIds.flat_rank == 2 and TopProbs.flat_rank == 2
    var N = Int(n)
    var row = block_idx.x
    var tid = thread_idx.x
    var base = row * Int(X.dim[1]())
    var NMAX = Int(TopIds.dim[1]())
    var K = Int(nsel)
    if K > NMAX:
        K = NMAX
    if K > CAP:
        K = CAP
    var redf = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[SAMP_THREADS]())
    var redi = stack_allocation[i32, address_space = AddressSpace.SHARED](row_major[SAMP_THREADS]())
    var hist = stack_allocation[u64, address_space = AddressSpace.SHARED](row_major[256]())
    var hc = stack_allocation[u64, address_space = AddressSpace.SHARED](row_major[NBAND]())
    var hm = stack_allocation[u64, address_space = AddressSpace.SHARED](row_major[NBAND]())
    var st = stack_allocation[u64, address_space = AddressSpace.SHARED](row_major[4]())
    var srt = stack_allocation[u64, address_space = AddressSpace.SHARED](row_major[CAP]())
    var mas = stack_allocation[u64, address_space = AddressSpace.SHARED](row_major[CAP]())
    var redu = stack_allocation[u64, address_space = AddressSpace.SHARED](row_major[SAMP_THREADS]())

    var temp = temperature
    var tk = top_k
    var tp = top_p
    var mp = min_p
    if temperature <= 0:
        temp = Float32(1)
        tk = Int32(0)
        tp = Float32(1)
        mp = Float32(0)
    var nomask = X.ptr.unsafe_bitcast[Scalar[u64]]()
    var cut = sample_cut[False, CAP=CAP](
        X, hist, hc, hm, st, srt, mas, redu, redf, redi, row, base, N, tid, tk, tp, mp, nomask, 0
    )
    var lmax = cut[0]
    var none = cut[1] == 0
    var ck = cut[2]
    var ci = cut[3]
    var mpe = min(mp, Float32(1))
    var vcut = cut_val(ck)
    var zt: Float32 = 0
    var g = tid * 4
    while g < N and not none:
        var a = load4(X, base, g, N)
        comptime for e in range(4):
            if member(a[e], g + e, lmax, vcut, ck, ci, mpe):
                zt += exp((a[e] - lmax) / temp)
        g += 4 * SAMP_THREADS
    var z = bsum_f32(redf, tid, zt)

    var sel = sample_cut[False, CAP=CAP](
        X, hist, hc, hm, st, srt, mas, redu, redf, redi, row, base, N, tid, Int32(K), Float32(1), Float32(0), nomask, 0
    )
    var ck2 = sel[2]
    var ci2 = sel[3]
    var vcut2 = cut_val(ck2)
    if tid == 0:
        st[3] = 0
    barrier()
    var i = tid
    while i < N and not none:
        var v = rebind[Scalar[f32]](X[row, i])
        if member(v, i, lmax, vcut2, ck2, ci2, Float32(0)):
            var slot = Int(Atomic.fetch_add(st.ptr.unsafe_offset(3), UInt64(1)))
            if slot < K:
                srt[slot] = rebind[srt.ElementType](
                    (okey(v).cast[u64]() << 32) | (UInt32(0xFFFFFFFF) - UInt32(i)).cast[u64]()
                )
        i += SAMP_THREADS
    barrier()
    var cnt = Int(rebind[Scalar[u64]](st[3]))
    if cnt > K:
        cnt = K
    if tid == 0:
        for a in range(1, cnt):
            var key = rebind[Scalar[u64]](srt[a])
            var b = a
            while b > 0 and rebind[Scalar[u64]](srt[b - 1]) < key:
                srt[b] = rebind[srt.ElementType](rebind[Scalar[u64]](srt[b - 1]))
                b -= 1
            srt[b] = rebind[srt.ElementType](key)
    barrier()
    if tid < NMAX:
        if tid < cnt:
            var key = rebind[Scalar[u64]](srt[tid])
            var v = okey_inv((key >> 32).cast[u32]())
            var idx = Int(UInt32(0xFFFFFFFF) - (key & 0xFFFFFFFF).cast[u32]())
            var p: Float32 = 0
            if member(v, idx, lmax, vcut, ck, ci, mpe):
                p = exp((v - lmax) / temp) / z
            TopIds[row, tid] = rebind[TopIds.ElementType](Int32(idx))
            TopProbs[row, tid] = rebind[TopProbs.ElementType](p)
        else:
            TopIds[row, tid] = rebind[TopIds.ElementType](Int32(-1))
            TopProbs[row, tid] = rebind[TopProbs.ElementType](Float32(0))
