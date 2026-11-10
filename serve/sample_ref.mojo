"""Host reference sampler (bench/chat-protocol.md C3, plan brief M3 section 2).

Sequential CPU equivalent of `kernels/sample.mojo`'s `amar_sample_row` /
`amar_spec_accept` (lane-KSAMP, unmerged -- read via `git show
lane-KSAMP:kernels/sample.mojo`): same Philox4x32-10, same rng4/rng_word/
unif/gumbel transform, same stream numbering (0 = primary draw, 1 = accept
decision, 2 = residual resample, 3 = fallback-to-target resample), same cut
order (top-k, then top-p mass at T=1, then min-p), same tie-break (lower
index). A full sort stands in for the kernel's parallel radix/histogram
selection -- a reference is not required to be sub-linear per cut, only
correct. Not the device kernel: this is the flag-gated reference path and
the oracle the kernel is checked against once it lands.
"""
from bexp import exp, log
from std.utils.numerics import nan

comptime FMAX = Float32(3.4028234663852886e38)


def is_valid(v: Float32) -> Bool:
    return v >= -FMAX and v <= FMAX


# ---- Philox4x32-10, matching kernels/sample.mojo::philox4x32 exactly ------

def philox4x32(c: SIMD[DType.uint32, 4], k0: UInt32, k1: UInt32) -> SIMD[DType.uint32, 4]:
    var x = c
    var a = k0
    var b = k1
    for _ in range(10):
        var p0 = UInt64(0xD2511F53) * UInt64(x[0])
        var p1 = UInt64(0xCD9E8D57) * UInt64(x[2])
        x = SIMD[DType.uint32, 4](
            UInt32((p1 >> 32) & 0xFFFFFFFF) ^ x[1] ^ a,
            UInt32(p1 & 0xFFFFFFFF),
            UInt32((p0 >> 32) & 0xFFFFFFFF) ^ x[3] ^ b,
            UInt32(p0 & 0xFFFFFFFF),
        )
        a += UInt32(0x9E3779B9)
        b += UInt32(0xBB67AE85)
    return x


def rng4(seed: UInt64, counter: UInt64, row: Int, stream: Int, g: Int) -> SIMD[DType.uint32, 4]:
    var c = SIMD[DType.uint32, 4](
        UInt32(counter & 0xFFFFFFFF),
        UInt32((counter >> 32) & 0xFFFFFFFF),
        UInt32(row),
        (UInt32(stream) << 28) | UInt32(g >> 2),
    )
    return philox4x32(c, UInt32(seed & 0xFFFFFFFF), UInt32((seed >> 32) & 0xFFFFFFFF))


def rng_word(seed: UInt64, counter: UInt64, row: Int, stream: Int, i: Int) -> UInt32:
    var g = i - (i % 4)
    return rng4(seed, counter, row, stream, g)[i % 4]


def unif(w: UInt32) -> Float32:
    return (Float32(w >> 8) + 0.5) * Float32(5.9604644775390625e-08)


def gumbel(w: UInt32) -> Float32:
    return -log(-log(unif(w)))


def gumbel2(w1: UInt32, w2: UInt32) -> Float32:
    # 53-bit uniform in float64 from two Philox words (C3 tail round): the
    # 24-bit float32 uniform floors every token at 2^-24 per draw.
    var u = (Float64(w1 >> 5) * 67108864.0 + Float64(w2 >> 6) + 0.5) * 1.1102230246251565e-16
    return Float32(-log(-log(u)))


# ---- stable sort (value desc, ties by ascending original index) ----------

def sort_desc_stable(mut order: List[Int], values: List[Float32]):
    # Bottom-up stable merge sort over `order`, comparing values[order[i]].
    # Stability preserves the incoming (ascending index) order among ties,
    # which is exactly the required tie-break.
    var n = len(order)
    if n < 2:
        return
    var buf = List[Int](unsafe_uninit_length=n)
    var width = 1
    while width < n:
        var i = 0
        while i < n:
            var mid = min(i + width, n)
            var end = min(i + 2 * width, n)
            var a = i
            var b = mid
            var k = i
            while a < mid and b < end:
                if values[order[a]] >= values[order[b]]:
                    buf[k] = order[a]
                    a += 1
                else:
                    buf[k] = order[b]
                    b += 1
                k += 1
            while a < mid:
                buf[k] = order[a]
                a += 1
                k += 1
            while b < end:
                buf[k] = order[b]
                b += 1
                k += 1
            i += 2 * width
        for j in range(n):
            order[j] = buf[j]
        width *= 2


# ---- amar_sample_row reference --------------------------------------------

def sample_row_ref(
    logits: List[Float32], temperature: Float32, top_k: Int, top_p: Float32, min_p: Float32,
    seed: UInt64, counter: UInt64, row: Int,
) -> Tuple[Int, Float32]:
    var n = len(logits)
    if temperature <= 0:
        var best_i = -1
        var best_v = -FMAX
        for i in range(n):
            var v = logits[i]
            if is_valid(v) and v > best_v:
                best_v = v
                best_i = i
        return (best_i, Float32(1) if best_i >= 0 else Float32(0))

    var lmax = -FMAX
    var nvalid = 0
    for i in range(n):
        if is_valid(logits[i]):
            nvalid += 1
            if logits[i] > lmax:
                lmax = logits[i]
    if nvalid == 0:
        return (-1, Float32(0))

    var k_on = top_k > 0 and top_k < nvalid
    var p_on = top_p < Float32(1.0)

    var order = List[Int]()
    for i in range(n):
        if is_valid(logits[i]):
            order.append(i)
    sort_desc_stable(order, logits)

    var mk = nvalid
    if k_on:
        mk = top_k

    if p_on:
        var zc: Float64 = 0
        for j in range(mk):
            zc += Float64(exp(logits[order[j]] - lmax))
        # C3 fix round (bench/chat-protocol.md, exchange/2026-09-15-m5-sampler-diagnosis.md):
        # the device's pmass_target ceils a FIXED-POINT mass (2^-40 units,
        # where ceil is exact); this host port had ceil'd a float64 mass
        # whose unit is exp(lmax) = 1, rounding the target up to most or all
        # of the top-k set on a peaked row. No ceil here: w is the raw mass
        # target, clamped to [the top-1 token's own mass (1.0 in these
        # units), the full retained mass].
        var w = Float64(top_p) * zc
        if w < 1:
            w = 1
        if w > zc:
            w = zc
        var cum: Float64 = 0
        var m = mk
        for j in range(mk):
            cum += Float64(exp(logits[order[j]] - lmax))
            if cum >= w:
                m = j + 1
                break
        mk = m

    if min_p > 0:
        var mpe = min(min_p, Float32(1))
        var m = 0
        while m < mk:
            if not (exp(logits[order[m]] - lmax) >= mpe):
                break
            m += 1
        mk = max(m, 1)

    var bs = -FMAX
    var bi = -1
    var zt: Float64 = 0
    for j in range(mk):
        var i = order[j]
        var ev = (logits[i] - lmax) / temperature
        var wd = rng_word(seed, counter, row, 0, i)
        var s = ev + gumbel2(wd, rng_word(seed, counter, row, 4, i))
        if s > bs:
            bs = s
            bi = i
        zt += Float64(exp(ev))
    var prob = Float32(Float64(exp((logits[bi] - lmax) / temperature)) / zt)
    return (bi, prob)


# ---- amar_sample_probs reference: full retained-set distribution ---------

def sample_probs_ref(
    logits: List[Float32], temperature: Float32, top_k: Int, top_p: Float32, min_p: Float32,
) -> List[Float32]:
    var n = len(logits)
    var probs = List[Float32](unsafe_uninit_length=n)
    for i in range(n):
        probs[i] = 0

    if temperature <= 0:
        var best_i = -1
        var best_v = -FMAX
        for i in range(n):
            var v = logits[i]
            if is_valid(v) and v > best_v:
                best_v = v
                best_i = i
        if best_i >= 0:
            probs[best_i] = 1
        return probs^

    var lmax = -FMAX
    var nvalid = 0
    for i in range(n):
        if is_valid(logits[i]):
            nvalid += 1
            if logits[i] > lmax:
                lmax = logits[i]
    if nvalid == 0:
        return probs^

    var k_on = top_k > 0 and top_k < nvalid
    var p_on = top_p < Float32(1.0)

    var order = List[Int]()
    for i in range(n):
        if is_valid(logits[i]):
            order.append(i)
    sort_desc_stable(order, logits)

    var mk = nvalid
    if k_on:
        mk = top_k

    if p_on:
        var zc: Float64 = 0
        for j in range(mk):
            zc += Float64(exp(logits[order[j]] - lmax))
        # C3 fix round (bench/chat-protocol.md, exchange/2026-09-15-m5-sampler-diagnosis.md):
        # the device's pmass_target ceils a FIXED-POINT mass (2^-40 units,
        # where ceil is exact); this host port had ceil'd a float64 mass
        # whose unit is exp(lmax) = 1, rounding the target up to most or all
        # of the top-k set on a peaked row. No ceil here: w is the raw mass
        # target, clamped to [the top-1 token's own mass (1.0 in these
        # units), the full retained mass].
        var w = Float64(top_p) * zc
        if w < 1:
            w = 1
        if w > zc:
            w = zc
        var cum: Float64 = 0
        var m = mk
        for j in range(mk):
            cum += Float64(exp(logits[order[j]] - lmax))
            if cum >= w:
                m = j + 1
                break
        mk = m

    if min_p > 0:
        var mpe = min(min_p, Float32(1))
        var m = 0
        while m < mk:
            if not (exp(logits[order[m]] - lmax) >= mpe):
                break
            m += 1
        mk = max(m, 1)

    var zt: Float64 = 0
    for j in range(mk):
        zt += Float64(exp((logits[order[j]] - lmax) / temperature))
    for j in range(mk):
        var i = order[j]
        probs[i] = Float32(Float64(exp((logits[i] - lmax) / temperature)) / zt)
    return probs^


# ---- amar_spec_accept reference -------------------------------------------

def spec_accept_ref(
    pt: List[Float32], pd: List[Float32], x: Int, seed: UInt64, counter: UInt64, row: Int,
) -> Tuple[Int, Bool]:
    var n = len(pt)
    if x >= 0 and x < n:
        var u = unif(rng_word(seed, counter, row, 1, 0))
        if u * pd[x] < pt[x]:
            return (x, True)

    var bs = -FMAX
    var bi = -1
    for i in range(n):
        var r = pt[i] - pd[i]
        if r > 0:
            var wd = rng_word(seed, counter, row, 2, i)
            var s = log(r) + gumbel2(wd, rng_word(seed, counter, row, 6, i))
            if s > bs:
                bs = s
                bi = i
    if bi < 0:
        for i in range(n):
            if pt[i] > 0:
                var wd = rng_word(seed, counter, row, 3, i)
                var s = log(pt[i]) + gumbel2(wd, rng_word(seed, counter, row, 7, i))
                if s > bs:
                    bs = s
                    bi = i
    return (bi, False)


# ---- presence / frequency penalties (host-only, not in the kernel) -------

def apply_penalties(
    mut logits: List[Float32], generated: List[Int], presence_penalty: Float32, frequency_penalty: Float32,
):
    if presence_penalty == 0 and frequency_penalty == 0:
        return
    var counts = Dict[Int, Int]()
    for i in range(len(generated)):
        var t = generated[i]
        counts[t] = counts.get(t, 0) + 1
    for entry in counts.items():
        var tok = entry.key
        var c = entry.value
        if tok >= 0 and tok < len(logits):
            logits[tok] -= presence_penalty + frequency_penalty * Float32(c)


# ---- grammar mask reference ----------------------------------------------

def mask_logits(logits: List[Float32], words: List[UInt64]) -> List[Float32]:
    # Masked tokens (bit clear) become NaN, which is_valid rejects, so the
    # existing sample_row_ref / sample_probs_ref on this copy are the masked
    # reference: the mask is applied before every truncating sampler and a
    # draw is impossible only when no allowed token exists.
    var out = List[Float32](unsafe_uninit_length=len(logits))
    for i in range(len(logits)):
        var w = i // 64
        var allowed = w < len(words) and ((words[w] >> UInt64(i % 64)) & 1) == 1
        out[i] = logits[i] if allowed else nan[DType.float32]()
    return out^
