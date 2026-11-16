# DFlash2 draft kernels (lane-dflash2 D2c). Semantics: llama.cpp src/models/dflash.cpp (graph<false>,
# build_dflash2_conv, build_dflash2_selector); layouts follow its ggml shapes, ne0 fastest.
#   draft attention: non-causal block of NB rows at positions pos0 .. pos0+NB-1 over a position-keyed KV ring,
#                    key k_pos visible to query q_pos iff q_pos - k_pos < SWA (block keys past q_pos included)
#   two-tap conv:    out[t, c] = sum_tap (dyn[t, side, tap, c / GS] + base[side, tap, c]) * x[t - tap, c], x[<0] = 0
#   lattice:         row 0 zero; row i >= 1 = [cand ids (K floats) | score[p * K + j]],
#                    score = unary[i, j] + sum_r succ[cand_ij, r] * pred[prev_ip, r] * gate[i, r]
from max.gpu import block_idx, global_idx, lane_id, thread_idx, WARP_SIZE
from max.gpu.primitives import warp
from max.gpu.sync import barrier
from max.gpu.memory import AddressSpace
from layout import row_major, stack_allocation
from bexp import exp

comptime f32 = DType.float32
comptime P32 = Pointer[Float32, MutAnyOrigin]
comptime PI32 = Pointer[Int32, MutAnyOrigin]
comptime PI8 = Pointer[Int8, MutAnyOrigin]
comptime PF16 = Pointer[Float16, MutAnyOrigin]
comptime PBF16 = Pointer[BFloat16, MutAnyOrigin]


def amar_df_attn[HD: Int, NQH: Int, NKVH: Int, RING: Int, SWA: Int](
    Q: P32, Kr: P32, Vr: P32, kpos: PI32, O: PBF16, pos0: Int32, nb: Int32, scale: Float32,
):
    """grid (NQH, nb), block HD. Q [nb, NQH, HD]; Kr/Vr [RING, NKVH, HD] of one layer; kpos[RING] = position held
    by each cell (-1 empty). O [nb, NQH * HD] bf16 (its only reader is the Wo GEMM)."""
    var h = Int(block_idx.x)
    var r = Int(block_idx.y)
    var d = Int(thread_idx.x)
    var kvh = h // (NQH // NKVH)
    var qp = Int(pos0) + r
    var q = Q[(r * NQH + h) * HD + d]
    var sums = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[HD // WARP_SIZE]())
    var m: Float32 = -1e30
    var den: Float32 = 0
    var acc: Float32 = 0
    for kp in range(max(0, qp - SWA + 1), Int(pos0) + Int(nb)):
        var cell = kp % RING
        if Int(kpos[cell]) != kp:
            continue
        var kv = cell * NKVH + kvh
        var w = warp.sum(q * Kr[kv * HD + d])
        if lane_id() == 0:
            sums[d // WARP_SIZE] = rebind[sums.ElementType](w)
        barrier()
        var tot: Float32 = 0
        comptime for i in range(HD // WARP_SIZE):
            tot += rebind[Scalar[f32]](sums[i])
        barrier()
        var s = tot * scale
        var mn = max(m, s)
        var a = exp(m - mn)
        var p = exp(s - mn)
        acc = acc * a + p * Vr[kv * HD + d]
        den = den * a + p
        m = mn
    O[r * (NQH * HD) + h * HD + d] = (acc / den).cast[DType.bfloat16]()


def amar_df_conv[H: Int, GS: Int, ODT: DType, ACC: Bool](
    X: P32, dyn: P32, base: P32, O: Pointer[Scalar[ODT], MutAnyOrigin], nb: Int32, side: Int32
):
    """One thread per (t, c) of an nb x H block. dyn [nb, 2 side, 2 tap, H / GS]; base [2 side, 2 tap, H].
    ACC adds into O (side 1: the residual add); else O = out (side 0: bf16, the next GEMM's A)."""
    comptime NG = H // GS
    var i = Int(global_idx.x)
    if i >= Int(nb) * H:
        return
    var t = i // H
    var c = i % H
    var g = c // GS
    var sd = Int(side)
    var dt = dyn.unsafe_offset(t * 4 * NG + sd * 2 * NG)
    var w0 = dt[g] + base[(sd * 2 + 0) * H + c]
    var out = w0 * X[i]
    if t > 0:
        var w1 = dt[NG + g] + base[(sd * 2 + 1) * H + c]
        out += w1 * X[i - H]
    comptime if ACC:
        O[i] = (O[i].cast[f32]() + out).cast[ODT]()
    else:
        O[i] = out.cast[ODT]()


def amar_df_topk[K: Int, T: Int](X: P32, ids: PI32, vals: P32, n: Int32):
    """grid nb - 1, block T: top-K of row r = block + 1 of X [nb, n] into ids / vals [nb, K], descending (row 0 is
    never read). Each thread keeps a register top-K of its stride (comptime indices only), then K block-argmax rounds
    over the T x K shared candidates; the winning thread emits and advances its head."""
    var r = Int(block_idx.x) + 1
    var t = Int(thread_idx.x)
    var row = X.unsafe_offset(r * Int(n))
    var v = Array[Float32, K](fill=-Float32.MAX)
    var ix = Array[Int32, K](fill=0)
    var c = t
    while c < Int(n):
        var x = row[c]
        if x > v[K - 1]:
            var xi = Int32(c)
            comptime for j in range(K):
                if x > v[j]:
                    var tv = v[j]
                    var ti = ix[j]
                    v[j] = x
                    ix[j] = xi
                    x = tv
                    xi = ti
        c += T
    var sv = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[T * K]())
    var si = stack_allocation[DType.int32, address_space = AddressSpace.SHARED](row_major[T * K]())
    var rv = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[T]())
    var rt = stack_allocation[DType.int32, address_space = AddressSpace.SHARED](row_major[T]())
    comptime for j in range(K):
        sv[t * K + j] = rebind[sv.ElementType](v[j])
        si[t * K + j] = rebind[si.ElementType](ix[j])
    var h = 0
    for k in range(K):
        rv[t] = rebind[rv.ElementType](rebind[Scalar[f32]](sv[t * K + h]) if h < K else -Float32.MAX)
        rt[t] = rebind[rt.ElementType](Int32(t))
        barrier()
        var s = T // 2
        while s > 0:
            if t < s and rebind[Scalar[f32]](rv[t + s]) > rebind[Scalar[f32]](rv[t]):
                rv[t] = rv[t + s]
                rt[t] = rt[t + s]
            barrier()
            s //= 2
        var w = Int(rebind[Scalar[DType.int32]](rt[0]))
        barrier()
        if t == w:
            ids[r * K + k] = rebind[Scalar[DType.int32]](si[t * K + h])
            vals[r * K + k] = rebind[Scalar[f32]](sv[t * K + h])
            h += 1


def amar_df_gather[H: Int, NT: Int, MR: Int](T: P32, O: PBF16, r0: Int32, m: Int32):
    """Window taps T [NT, MR, H] (f32) rows r0 .. r0 + m - 1 into feature rows O [m, NT * H] (bf16, tap-major per row)."""
    var i = Int(global_idx.x)
    if i >= Int(m) * NT * H:
        return
    var r = i // (NT * H)
    var s = (i % (NT * H)) // H
    var c = i % H
    O[i] = T[(s * MR + Int(r0) + r) * H + c].cast[DType.bfloat16]()


def amar_df_put[W: Int](Ks: P32, Vs: P32, Kr: P32, Vr: P32, cells: PI32):
    """grid nb, block 256: row r of Ks/Vs [nb, W] into ring cell cells[r] of one layer's Kr/Vr [RING, W]."""
    var r = Int(block_idx.x)
    var c = Int(cells[r])
    var i = Int(thread_idx.x)
    while i < W:
        Kr[c * W + i] = Ks[r * W + i]
        Vr[c * W + i] = Vs[r * W + i]
        i += 256


@always_inline
def q8_row_val(qs: PI8, sc: PF16, row: Int, r: Int, R: Int) -> Float32:
    """Element r of row `row` of a q8 matrix with rows of R (engine q8 layout: qs [rows * R], f16 scales [rows * R / 32])."""
    return Float32(qs[row * R + r]) * sc[(row * R + r) // 32].cast[f32]()


def amar_df_lattice[K: Int, R: Int](
    cand: PI32, unary: P32, gate: P32, succ_q: PI8, succ_s: PF16, pred_q: PI8, pred_s: PF16,
    anchor: Int32, L: P32, ncols: Int32,
):
    """grid (nb - 1, K), block K: one block per (row i >= 1, predecessor p), one thread per successor j, each a rank-R
    dot. Row 1 has one predecessor (the anchor); its K blocks write the same values. cand/unary [nb, K], gate [nb, R];
    L [nb, ncols], row 0 is zeroed by the caller."""
    var i = Int(block_idx.x) + 1
    var p = Int(block_idx.y)
    var j = Int(thread_idx.x)
    var prev = Int(anchor) if i == 1 else Int(cand[(i - 1) * K + p])
    var nxt = Int(cand[i * K + j])
    var s: Float32 = 0
    for r in range(R):
        s += q8_row_val(succ_q, succ_s, nxt, r, R) * q8_row_val(pred_q, pred_s, prev, r, R) * gate[i * R + r]
    var row = L.unsafe_offset(i * Int(ncols))
    row[K + p * K + j] = s + unary[i * K + j]
    if p == 0:
        row[j] = Float32(nxt)
