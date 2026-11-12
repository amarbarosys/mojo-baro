from std.math import exp, log, sqrt, rsqrt
from max.gpu import block_idx, thread_idx, global_idx
from max.gpu.primitives import warp
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from layout import row_major, stack_allocation
from ops import F, I

comptime f32 = DType.float32
comptime AT = 256
comptime NW = 8
comptime NQ = 32
comptime HDM = 256
comptime KT = 128
comptime RC = 32


def rmsn_fwd_k(x: F, w: F, y: F, r: F, n32: Int32, d32: Int32, eps: Float32):
    var row = Int(block_idx.x) * NW + Int(thread_idx.x) // 32
    var lane = Int(thread_idx.x) % 32
    var d = Int(d32)
    if row >= Int(n32):
        return
    var o = row * d
    var ss: Float32 = 0
    var c = lane
    while c < d:
        var v = x[unsafe_offset=o + c]
        ss += v * v
        c += 32
    var rr = rsqrt(warp.sum(ss) / Float32(d) + eps)
    if lane == 0:
        r[unsafe_offset=row] = rr
    c = lane
    while c < d:
        y[unsafe_offset=o + c] = x[unsafe_offset=o + c] * rr * w[unsafe_offset=c]
        c += 32


def rmsn_dx_k(x: F, w: F, r: F, dy: F, dx: F, n32: Int32, d32: Int32, acc: Int32):
    var row = Int(block_idx.x) * NW + Int(thread_idx.x) // 32
    var lane = Int(thread_idx.x) % 32
    var d = Int(d32)
    if row >= Int(n32):
        return
    var o = row * d
    var rr = r[unsafe_offset=row]
    var s: Float32 = 0
    var c = lane
    while c < d:
        s += dy[unsafe_offset=o + c] * w[unsafe_offset=c] * x[unsafe_offset=o + c]
        c += 32
    var cc = rr * rr * rr * warp.sum(s) / Float32(d)
    c = lane
    while c < d:
        var val = rr * w[unsafe_offset=c] * dy[unsafe_offset=o + c] - x[unsafe_offset=o + c] * cc
        dx[unsafe_offset=o + c] = dx[unsafe_offset=o + c] + val if acc != 0 else val
        c += 32


def coldot_k(x: F, y: F, r: F, part: F, n32: Int32, d32: Int32, user: Int32):
    var j = Int(block_idx.y) * AT + Int(thread_idx.x)
    var c = Int(block_idx.x)
    var d = Int(d32)
    if j >= d:
        return
    var s: Float32 = 0
    for t in range(c * RC, min(Int(n32), (c + 1) * RC)):
        var v = x[unsafe_offset=t * d + j] * y[unsafe_offset=t * d + j]
        if user != 0:
            v *= r[unsafe_offset=t]
        s += v
    part[unsafe_offset=c * d + j] = s


def groupsum_k(src: F, dst: F, ng32: Int32, hs32: Int32):
    var i = global_idx.x
    var hs = Int(hs32)
    if i >= Int(ng32):
        return
    var s: Float32 = 0
    for k in range(hs):
        s += src[unsafe_offset=i * hs + k]
    dst[unsafe_offset=i] = dst[unsafe_offset=i] + s


def scale_cols_k(x: F, g: F, y: F, n32: Int32, d32: Int32, hs32: Int32):
    var i = global_idx.x
    var d = Int(d32)
    if i < Int(n32) * d:
        y[unsafe_offset=i] = x[unsafe_offset=i] * g[unsafe_offset=(i % d) // Int(hs32)]


def diff_k(x: F, t: F, r: F, n32: Int32):
    var i = global_idx.x
    if i < Int(n32):
        r[unsafe_offset=i] = x[unsafe_offset=i] - t[unsafe_offset=i]


def saxpy_k(x: F, y: F, a: Float32, s: F, n32: Int32, acc: Int32):
    var i = global_idx.x
    if i < Int(n32):
        var v = a * x[unsafe_offset=i] / s[unsafe_offset=0]
        y[unsafe_offset=i] = y[unsafe_offset=i] + v if acc != 0 else v


def rowmap_k(src: F, dst: F, map: I, srcid: I, e: F, n32: Int32, d32: Int32, inv: Int32):
    var i = global_idx.x
    var d = Int(d32)
    if i >= Int(n32) * d:
        return
    var r = i // d
    var c = i % d
    var m = Int(map[unsafe_offset=r])
    if inv != 0:
        dst[unsafe_offset=m * d + c] = src[unsafe_offset=i]
    else:
        dst[unsafe_offset=i] = src[unsafe_offset=m * d + c] + e[unsafe_offset=Int(srcid[unsafe_offset=r]) * d + c]


def reduce_k(x: F, n32: Int32, res: F, sq: Int32):
    var tid = Int(thread_idx.x)
    var red = stack_allocation[f32, address_space=AddressSpace.SHARED](row_major[NW]())
    var s: Float32 = 0
    var i = tid
    while i < Int(n32):
        var v = x[unsafe_offset=i]
        s += v * v if sq != 0 else v
        i += AT
    s = warp.sum(s)
    if tid % 32 == 0:
        red[tid // 32] = rebind[red.ElementType](s)
    barrier()
    if tid == 0:
        var t: Float32 = 0
        for w in range(NW):
            t += rebind[Float32](red[w])
        res[unsafe_offset=0] = t


def ce_big_k(z: F, tgt: I, loss: F, hit: F, n32: Int32, v32: Int32, w: Float32):
    var row = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var lane = tid % 32
    var wid = tid // 32
    var v = Int(v32)
    var o = row * v
    var red = stack_allocation[f32, address_space=AddressSpace.SHARED](row_major[2 * NW]())
    var y = Int(tgt[unsafe_offset=row])
    var zy = z[unsafe_offset=o + y]
    var m: Float32 = -3.0e38
    var mi = v
    var j = tid
    while j < v:
        var val = z[unsafe_offset=o + j]
        if val > m:
            m = val
            mi = j
        j += AT
    var wm = warp.max(m)
    var ci = warp.min(Int32(mi) if m == wm else Int32(v))
    if lane == 0:
        red[wid] = rebind[red.ElementType](wm)
        red[NW + wid] = rebind[red.ElementType](Float32(ci))
    barrier()
    var bm = rebind[Float32](red[0])
    var bi = rebind[Float32](red[NW])
    for k in range(1, NW):
        var km = rebind[Float32](red[k])
        var ki = rebind[Float32](red[NW + k])
        if km > bm or (km == bm and ki < bi):
            bm = km
            bi = ki
    barrier()
    var s: Float32 = 0
    j = tid
    while j < v:
        s += exp(z[unsafe_offset=o + j] - bm)
        j += AT
    s = warp.sum(s)
    if lane == 0:
        red[wid] = rebind[red.ElementType](s)
    barrier()
    var bs: Float32 = 0
    for k in range(NW):
        bs += rebind[Float32](red[k])
    var inv = 1 / bs
    j = tid
    while j < v:
        var p = exp(z[unsafe_offset=o + j] - bm) * inv
        z[unsafe_offset=o + j] = (p - Float32(Int(j == y))) * w
        j += AT
    if tid == 0:
        loss[unsafe_offset=row] = log(bs) + bm - zy
        hit[unsafe_offset=row] = Float32(Int(Int(bi) == y))


def cos_k(c: F, t: F, dc: F, lrow: F, n32: Int32, d32: Int32, wc: Float32):
    var row = Int(block_idx.x) * NW + Int(thread_idx.x) // 32
    var lane = Int(thread_idx.x) % 32
    var d = Int(d32)
    if row >= Int(n32):
        return
    var o = row * d
    var dot: Float32 = 0
    var cc: Float32 = 0
    var tt: Float32 = 0
    var i = lane
    while i < d:
        var a = c[unsafe_offset=o + i]
        var b = t[unsafe_offset=o + i]
        dot += a * b
        cc += a * a
        tt += b * b
        i += 32
    dot = warp.sum(dot)
    cc = warp.sum(cc)
    tt = warp.sum(tt)
    var den = sqrt(cc * tt) + 1e-20
    var cs = dot / den
    if lane == 0:
        lrow[unsafe_offset=row] = 1 - cs
    i = lane
    while i < d:
        dc[unsafe_offset=o + i] = -wc * (t[unsafe_offset=o + i] / den - dot * c[unsafe_offset=o + i] / (cc * den + 1e-20))
        i += 32


def attn_fwd_k(q: F, k: F, v: F, o: F, p: F, tab: I, qrs: Int32, qhs: Int32, krs: Int32, khs: Int32, vrs: Int32, vhs: Int32,
               ors: Int32, ohs: Int32, hdk32: Int32, hdv32: Int32, grp32: Int32, nh32: Int32, nkmax32: Int32, scale: Float32):
    var it = Int(block_idx.x)
    var kvh = Int(block_idx.y)
    var tid = Int(thread_idx.x)
    var lane = tid % 32
    var wid = tid // 32
    var qoff = Int(tab[unsafe_offset=it * 4])
    var nq = Int(tab[unsafe_offset=it * 4 + 1])
    var koff = Int(tab[unsafe_offset=it * 4 + 2])
    var nk = Int(tab[unsafe_offset=it * 4 + 3])
    var hdk = Int(hdk32)
    var hdv = Int(hdv32)
    var grp = Int(grp32)
    var nkmax = Int(nkmax32)
    var qa = stack_allocation[f32, address_space=AddressSpace.SHARED](row_major[NQ, HDM]())
    var pt = stack_allocation[f32, address_space=AddressSpace.SHARED](row_major[NQ, KT]())
    var ngrp = AT // hdv
    var dd = tid % hdv
    var g = tid // hdv
    for h in range(kvh * grp, kvh * grp + grp):
        var pb = p.unsafe_offset((it * Int(nh32) + h) * NQ * nkmax)
        var idx = tid
        while idx < NQ * hdk:
            var ii = idx // hdk
            qa[ii, idx % hdk] = rebind[qa.ElementType](q[unsafe_offset=(qoff + ii) * Int(qrs) + h * Int(qhs) + idx % hdk] if ii < nq else Float32(0))
            idx += AT
        barrier()
        var j = tid
        while j < nk:
            var kb = k.unsafe_offset((koff + j) * Int(krs) + kvh * Int(khs))
            var acc = SIMD[f32, NQ](0)
            for e in range(hdk):
                var kv = kb[unsafe_offset=e]
                comptime for i in range(NQ):
                    acc[i] += rebind[Float32](qa[i, e]) * kv
            comptime for i in range(NQ):
                if i < nq:
                    pb[unsafe_offset=i * nkmax + j] = acc[i] * scale
            j += AT
        barrier()
        var i = wid
        while i < nq:
            var row = pb.unsafe_offset(i * nkmax)
            var m: Float32 = -3.0e38
            j = lane
            while j < nk:
                m = max(m, row[unsafe_offset=j])
                j += 32
            m = warp.max(m)
            var s: Float32 = 0
            j = lane
            while j < nk:
                s += exp(row[unsafe_offset=j] - m)
                j += 32
            var inv = 1 / warp.sum(s)
            j = lane
            while j < nk:
                row[unsafe_offset=j] = exp(row[unsafe_offset=j] - m) * inv
                j += 32
            i += NW
        barrier()
        var acc = SIMD[f32, NQ](0)
        var j0 = 0
        while j0 < nk:
            var jn = min(KT, nk - j0)
            idx = tid
            while idx < NQ * jn:
                var ii = idx // jn
                pt[ii, idx % jn] = rebind[pt.ElementType](pb[unsafe_offset=ii * nkmax + j0 + idx % jn] if ii < nq else Float32(0))
                idx += AT
            barrier()
            if g < ngrp:
                for jj in range(jn):
                    var vv = v[unsafe_offset=(koff + j0 + jj) * Int(vrs) + kvh * Int(vhs) + dd]
                    comptime for i in range(NQ):
                        if i % ngrp == g:
                            acc[i] += rebind[Float32](pt[i, jj]) * vv
            barrier()
            j0 += KT
        if g < ngrp:
            comptime for i in range(NQ):
                if i < nq and i % ngrp == g:
                    o[unsafe_offset=(qoff + i) * Int(ors) + h * Int(ohs) + dd] = acc[i]
        barrier()


def attn_bwd_k(q: F, k: F, v: F, o: F, do: F, p: F, ds: F, dq: F, dk: F, dv: F, tab: I, qrs: Int32, qhs: Int32, krs: Int32, khs: Int32,
               vrs: Int32, vhs: Int32, ors: Int32, ohs: Int32, hdk32: Int32, hdv32: Int32, grp32: Int32, nh32: Int32, nkmax32: Int32, scale: Float32):
    var it = Int(block_idx.x)
    var kvh = Int(block_idx.y)
    var tid = Int(thread_idx.x)
    var lane = tid % 32
    var wid = tid // 32
    var qoff = Int(tab[unsafe_offset=it * 4])
    var nq = Int(tab[unsafe_offset=it * 4 + 1])
    var koff = Int(tab[unsafe_offset=it * 4 + 2])
    var nk = Int(tab[unsafe_offset=it * 4 + 3])
    var hdk = Int(hdk32)
    var hdv = Int(hdv32)
    var grp = Int(grp32)
    var nkmax = Int(nkmax32)
    var qa = stack_allocation[f32, address_space=AddressSpace.SHARED](row_major[NQ, HDM]())
    var pt = stack_allocation[f32, address_space=AddressSpace.SHARED](row_major[NQ, KT]())
    var dl = stack_allocation[f32, address_space=AddressSpace.SHARED](row_major[NQ]())
    var jj = tid % KT
    var half = tid // KT
    var ngrp = AT // hdk
    var dd = tid % hdk
    var g = tid // hdk
    for h in range(kvh * grp, kvh * grp + grp):
        var pb = p.unsafe_offset((it * Int(nh32) + h) * NQ * nkmax)
        var db = ds.unsafe_offset((it * Int(nh32) + h) * NQ * nkmax)
        var idx = tid
        while idx < NQ * hdv:
            var ii = idx // hdv
            qa[ii, idx % hdv] = rebind[qa.ElementType](do[unsafe_offset=(qoff + ii) * Int(ors) + h * Int(ohs) + idx % hdv] if ii < nq else Float32(0))
            idx += AT
        barrier()
        var i = wid
        while i < nq:
            var s: Float32 = 0
            var e = lane
            while e < hdv:
                s += rebind[Float32](qa[i, e]) * o[unsafe_offset=(qoff + i) * Int(ors) + h * Int(ohs) + e]
                e += 32
            s = warp.sum(s)
            if lane == 0:
                dl[i] = rebind[dl.ElementType](s)
            i += NW
        barrier()
        var j = tid
        while j < nk:
            var vb = v.unsafe_offset((koff + j) * Int(vrs) + kvh * Int(vhs))
            var acc = SIMD[f32, NQ](0)
            for e in range(hdv):
                var vv = vb[unsafe_offset=e]
                comptime for i in range(NQ):
                    acc[i] += rebind[Float32](qa[i, e]) * vv
            comptime for i in range(NQ):
                if i < nq:
                    db[unsafe_offset=i * nkmax + j] = pb[unsafe_offset=i * nkmax + j] * (acc[i] - rebind[Float32](dl[i])) * scale
            j += AT
        barrier()
        var j0 = 0
        while j0 < nk:
            var jn = min(KT, nk - j0)
            idx = tid
            while idx < NQ * jn:
                var ii = idx // jn
                pt[ii, idx % jn] = rebind[pt.ElementType](pb[unsafe_offset=ii * nkmax + j0 + idx % jn] if ii < nq else Float32(0))
                idx += AT
            barrier()
            if jj < jn:
                for e in range(half * (hdv // 2), (half + 1) * (hdv // 2)):
                    var acc: Float32 = 0
                    comptime for i in range(NQ):
                        acc += rebind[Float32](pt[i, jj]) * rebind[Float32](qa[i, e])
                    var oo = (koff + j0 + jj) * Int(vrs) + kvh * Int(vhs) + e
                    dv[unsafe_offset=oo] = dv[unsafe_offset=oo] + acc
            barrier()
            j0 += KT
        idx = tid
        while idx < NQ * hdk:
            var ii = idx // hdk
            qa[ii, idx % hdk] = rebind[qa.ElementType](q[unsafe_offset=(qoff + ii) * Int(qrs) + h * Int(qhs) + idx % hdk] if ii < nq else Float32(0))
            idx += AT
        barrier()
        var accq = SIMD[f32, NQ](0)
        j0 = 0
        while j0 < nk:
            var jn = min(KT, nk - j0)
            idx = tid
            while idx < NQ * jn:
                var ii = idx // jn
                pt[ii, idx % jn] = rebind[pt.ElementType](db[unsafe_offset=ii * nkmax + j0 + idx % jn] if ii < nq else Float32(0))
                idx += AT
            barrier()
            if jj < jn:
                for e in range(half * (hdk // 2), (half + 1) * (hdk // 2)):
                    var acc: Float32 = 0
                    comptime for i in range(NQ):
                        acc += rebind[Float32](pt[i, jj]) * rebind[Float32](qa[i, e])
                    var oo = (koff + j0 + jj) * Int(krs) + kvh * Int(khs) + e
                    dk[unsafe_offset=oo] = dk[unsafe_offset=oo] + acc
            if g < ngrp:
                for t in range(jn):
                    var kk = k[unsafe_offset=(koff + j0 + t) * Int(krs) + kvh * Int(khs) + dd]
                    comptime for i in range(NQ):
                        if i % ngrp == g:
                            accq[i] += rebind[Float32](pt[i, t]) * kk
            barrier()
            j0 += KT
        if g < ngrp:
            comptime for i in range(NQ):
                if i < nq and i % ngrp == g:
                    dq[unsafe_offset=(qoff + i) * Int(qrs) + h * Int(qhs) + dd] = accq[i]
        barrier()
