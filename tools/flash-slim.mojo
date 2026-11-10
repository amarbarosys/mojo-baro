"""Flash-slim round 1 analysis on the CPU (bench/flash-slim-protocol.md).

keep TRACE[,TRACE...] SEQLEN [v]COVER OUT CATS IN
    Expert hits per category from Q4X_EXPERTS traces of Q4X_SEQ batches (lines "il id0 .. id9", layer-major,
    token t of a batch in sequence t // SEQLEN, named by CATS, a comma list in sequence order that may repeat names
    and applies to every batch; IN = in-domain names). vCOVER: per layer, rank on the even-numbered sequences
    (numbered across batches), take the smallest k (at least TOPK) whose top-k covers COVER of the odd-numbered
    sequences' hits, keep the top k ranked on all hits.
    Writes OUT, lines "il id": per layer the smallest expert set covering COVER of the in-domain hits (count
    descending, lower id first on ties, at least TOPK). Prints the keep fraction of every category on its own,
    of IN, and the kill-line number (mean over layers of |keep| / NE for IN).
gkeep PARTS_PREFIX K TRACE[,TRACE...] SEQLEN COVER OUT CATS IN
    Group keep set: per layer, the parts of PARTS_PREFIX<layer>.k<K>.parts (graph-shard --parts) ranked by their
    in-domain hits in the traces (count descending, lower part id first on ties), kept whole until they cover COVER
    of the layer's in-domain hits. Same trace and category rules as keep. An expert hit in the traces but absent from
    the parts file (not in the co-activation graph) is its own one-expert group.
coact TRACE LAYER OUT
    Expert co-activation graph of one layer: edge list "u v w" for graph-shard, w = number of tokens whose top-10
    holds both experts.
groups TRACE SEQLEN CATS LAYER PARTS
    Scores a partition of that layer's experts (PARTS: one part id per line, graph-shard --parts) against 200 random
    relabelings with the same part sizes: share of co-activation weight inside parts, and mutual information (bits)
    between part and category over the layer's picks. Prints observed, baseline mean and std, and per part its hits
    and dominant category.
score GGUF REF.f32 ARM.f32 SEQLEN CATS IN [NROWS [IDS]]
    Final-norm hidden rows (Q4X_HIDDEN_F32) of two arms over the same batch; logits = output.weight x row, the head
    dequantized by ggml_quant.row_f32 (bit-equal to gguf-py) in vocab chunks. Per category: top-1 agreement of ARM
    with REF and mean KL(REF || ARM). NROWS > 0 also prints argmax and KL of the first NROWS rows (oracle check).
    IDS (the batch's token ids, one per line) adds the negative log-likelihood of the true next token in each
    sequence for both arms, and perplexity per category: a quality measure, where agreement measures fidelity.
"""
from std.sys import argv
from std.math import exp, log, sqrt
from std.ffi import external_call
from max.algorithm import parallelize
from std.utils import StaticTuple
from shard_index import ShardIndex
from ggml_quant import row_f32, block_bytes

comptime H = 2560
comptime NL = 48
comptime NE = 512
comptime TOPK = 10
comptime VC = 2048
comptime PG = 4
comptime F32P = Pointer[Float32, MutUntrackedOrigin]
comptime F64P = Pointer[Float64, MutUntrackedOrigin]
comptime I64P = Pointer[Int, MutUntrackedOrigin]


def f32p(mut l: List[Float32]) -> F32P:
    return F32P(unsafe_from_address=Int(l.unsafe_ptr()))


def f64p(mut l: List[Float64]) -> F64P:
    return F64P(unsafe_from_address=Int(l.unsafe_ptr()))


def i64p(mut l: List[Int]) -> I64P:
    return I64P(unsafe_from_address=Int(l.unsafe_ptr()))


def names(s: String) -> List[String]:
    var out = List[String]()
    for p in s.split(","):
        out.append(String(p))
    return out^


def keep_set(cnt: List[Int], cover: Float64) -> List[Int]:
    var tot = 0
    for e in range(NE):
        tot += cnt[e]
    var used = List[Bool](length=NE, fill=False)
    var ids = List[Int]()
    var acc = 0
    while len(ids) < NE and (len(ids) < TOPK or Float64(acc) < cover * Float64(tot)):
        var b = -1
        for e in range(NE):
            if not used[e] and (b < 0 or cnt[e] > cnt[b]):
                b = e
        used[b] = True
        ids.append(b)
        acc += cnt[b]
    return ids^


def rank(cnt: List[Int]) -> List[Int]:
    var used = List[Bool](length=NE, fill=False)
    var ids = List[Int]()
    for _ in range(NE):
        var b = -1
        for e in range(NE):
            if not used[e] and (b < 0 or cnt[e] > cnt[b]):
                b = e
        used[b] = True
        ids.append(b)
    return ids^


def calibrated(fit: List[Int], val: List[Int], all: List[Int], cover: Float64) -> List[Int]:
    var rf = rank(fit)
    var vt = 0
    for e in range(NE):
        vt += val[e]
    var k = 0
    var acc = 0
    while k < NE and (k < TOPK or Float64(acc) < cover * Float64(vt)):
        acc += val[rf[k]]
        k += 1
    var ra = rank(all)
    var ids = List[Int]()
    for i in range(k):
        ids.append(ra[i])
    return ids^


def hits(files: List[String], seqlen: Int, nc: Int) raises -> List[Int]:
    var G = len(files) * nc
    var cnt = List[Int](length=G * NL * NE, fill=0)
    for fi in range(len(files)):
        var raw: String
        with open(files[fi], "r") as f:
            raw = f.read()
        var lines = raw.split("\n")
        var nl = len(lines)
        if nl > 0 and lines[nl - 1].byte_length() == 0:
            nl -= 1
        if nl % NL != 0:
            raise Error("FAIL keep: " + String(nl) + " lines in " + files[fi] + " is not a multiple of " + String(NL) + " layers")
        var T = nl // NL
        if T != nc * seqlen:
            raise Error("FAIL keep: " + String(T) + " tokens per layer in " + files[fi] + ", want " + String(nc) + " x " + String(seqlen))
        for k in range(nl):
            var parts = lines[k].split(" ")
            if len(parts) != 1 + TOPK or Int(parts[0]) != k // T:
                raise Error("FAIL keep: " + files[fi] + " line " + String(k + 1) + " is not layer " + String(k // T) + " with " + String(TOPK) + " ids")
            var g = fi * nc + (k % T) // seqlen
            for j in range(TOPK):
                var e = Int(parts[1 + j])
                if e < 0 or e >= NE:
                    raise Error("FAIL keep: expert id " + String(e) + " out of range in " + files[fi] + " line " + String(k + 1))
                cnt[(g * NL + k // T) * NE + e] += 1
    return cnt^


def keep(args: List[String]) raises:
    var seqlen = Int(args[2])
    var cal = args[3].startswith("v")
    var cover = Float64(atof(String(args[3][byte=1:]) if cal else args[3]))
    var cats = names(args[5])
    var inc = names(args[6])
    var files = names(args[1])
    var nc = len(cats)
    var G = len(files) * nc
    var cnt = hits(files, seqlen, nc)
    var isin = List[Bool](length=nc, fill=False)
    for n in inc:
        var found = False
        for c in range(nc):
            if cats[c] == n:
                isin[c] = True
                found = True
        if not found:
            raise Error("FAIL keep: in-domain category " + n + " is not in " + args[5])
    var uniq = List[String]()
    for c in cats:
        var seen = False
        for u in uniq:
            if u == c:
                seen = True
        if not seen:
            uniq.append(c)
    for u in range(len(uniq) + 1):
        var fr = Float64(0)
        var lo = NE
        var hi = 0
        var zero = 0
        for il in range(NL):
            var lc = List[Int](length=NE, fill=0)
            for g in range(G):
                var c = g % nc
                if (u < len(uniq) and cats[c] == uniq[u]) or (u == len(uniq) and isin[c]):
                    for e in range(NE):
                        lc[e] += cnt[(g * NL + il) * NE + e]
            for e in range(NE):
                if lc[e] == 0:
                    zero += 1
            var n = len(keep_set(lc, cover))
            fr += Float64(n) / Float64(NE)
            lo = min(lo, n)
            hi = max(hi, n)
        var nm = uniq[u] if u < len(uniq) else "IN(" + args[6] + ")"
        print("keep", nm, "cover", cover, "mean_fraction", fr / Float64(NL), "min", lo, "max", hi, "unhit_mean", Float64(zero) / Float64(NL))
    var mode = ("calibrated on odd sequences " if cal else "") + "cover " + String(cover)
    var out = String("# flash-slim keep set: " + mode + " of in-domain hits (" + args[6] + "), traces " + args[1] + "\n")
    var total = 0
    var lo = NE
    var hi = 0
    for il in range(NL):
        var la = List[Int](length=NE, fill=0)
        var lf = List[Int](length=NE, fill=0)
        var lv = List[Int](length=NE, fill=0)
        for g in range(G):
            if isin[g % nc]:
                for e in range(NE):
                    var x = cnt[(g * NL + il) * NE + e]
                    la[e] += x
                    if g % 2 == 0:
                        lf[e] += x
                    else:
                        lv[e] += x
        var ids = calibrated(lf, lv, la, cover) if cal else keep_set(la, cover)
        total += len(ids)
        lo = min(lo, len(ids))
        hi = max(hi, len(ids))
        for e in ids:
            out += String(il) + " " + String(e) + "\n"
    with open(args[4], "w") as f:
        f.write(out)
    print("KILL-LINE mean_fraction", Float64(total) / Float64(NL * NE), "(dead at >= 0.60) keep", total, "of", NL * NE, "per-layer min", lo, "max", hi, "mode", mode, "written", args[4])


def gkeep(args: List[String]) raises:
    var K = Int(args[2])
    var seqlen = Int(args[4])
    var cover = Float64(atof(args[5]))
    var cats = names(args[7])
    var inc = names(args[8])
    var files = names(args[3])
    var nc = len(cats)
    var cnt = hits(files, seqlen, nc)
    var out = String("# flash-slim group keep set: k " + args[2] + " parts " + args[1] + "*, cover " + String(cover) + " of in-domain hits (" + args[8] + "), traces " + args[3] + "\n")
    var total = 0
    var lo = NE
    var hi = 0
    for il in range(NL):
        var raw: String
        with open(args[1] + String(il) + ".k" + args[2] + ".parts", "r") as f:
            raw = f.read()
        var part = List[Int](length=NE, fill=-1)
        var n = 0
        for l in raw.split("\n"):
            if l.byte_length() > 0 and n < NE:
                part[n] = Int(l)
                n += 1
        for e in range(NE):
            if part[e] < 0 or part[e] >= K:
                part[e] = K + e
        var ph = List[Int](length=K + NE, fill=0)
        var tot = 0
        for g in range(len(files) * nc):
            var isin = False
            for nm in inc:
                if nm == cats[g % nc]:
                    isin = True
            if not isin:
                continue
            for e in range(NE):
                var x = cnt[(g * NL + il) * NE + e]
                if x > 0:
                    ph[part[e]] += x
                    tot += x
        var used = List[Bool](length=K + NE, fill=False)
        var acc = 0
        var kept = 0
        while Float64(acc) < cover * Float64(tot):
            var b = -1
            for q in range(K + NE):
                if not used[q] and (b < 0 or ph[q] > ph[b]):
                    b = q
            used[b] = True
            acc += ph[b]
        for e in range(NE):
            if used[part[e]]:
                out += String(il) + " " + String(e) + "\n"
                kept += 1
        if kept < TOPK:
            raise Error("FAIL gkeep: layer " + String(il) + " keeps " + String(kept) + " experts, need " + String(TOPK))
        total += kept
        lo = min(lo, kept)
        hi = max(hi, kept)
    with open(args[6], "w") as f:
        f.write(out)
    print("KILL-LINE mean_fraction", Float64(total) / Float64(NL * NE), "(dead at >= 0.60) keep", total, "of", NL * NE, "per-layer min", lo, "max", hi, "mode groups k", args[2], "cover", cover, "written", args[6])


def layer_picks(path: String, layer: Int) raises -> List[Int]:
    var raw: String
    with open(path, "r") as f:
        raw = f.read()
    var out = List[Int]()
    for l in raw.split("\n"):
        var parts = l.split(" ")
        if len(parts) != 1 + TOPK or Int(parts[0]) != layer:
            continue
        for j in range(TOPK):
            out.append(Int(parts[1 + j]))
    if len(out) == 0:
        raise Error("FAIL: no picks of layer " + String(layer) + " in " + path)
    return out^


def coact(args: List[String]) raises:
    var pk = layer_picks(args[1], Int(args[2]))
    var w = List[Int](length=NE * NE, fill=0)
    for t in range(len(pk) // TOPK):
        for a in range(TOPK):
            for b in range(a + 1, TOPK):
                var u = min(pk[t * TOPK + a], pk[t * TOPK + b])
                var v = max(pk[t * TOPK + a], pk[t * TOPK + b])
                w[u * NE + v] += 1
    var out = String("# co-activation layer " + args[2] + " from " + args[1] + "\n")
    var m = 0
    for u in range(NE):
        for v in range(u + 1, NE):
            if w[u * NE + v] > 0:
                out += String(u) + " " + String(v) + " " + String(w[u * NE + v]) + "\n"
                m += 1
    with open(args[3], "w") as f:
        f.write(out)
    print("coact layer", args[2], "tokens", len(pk) // TOPK, "edges", m, "written", args[3])


def mi_bits(joint: List[Float64], np_: Int, nc: Int) -> Float64:
    var tot = Float64(0)
    for x in joint:
        tot += x
    var pp = List[Float64](length=np_, fill=0)
    var pc = List[Float64](length=nc, fill=0)
    for a in range(np_):
        for c in range(nc):
            pp[a] += joint[a * nc + c]
            pc[c] += joint[a * nc + c]
    var mi = Float64(0)
    for a in range(np_):
        for c in range(nc):
            var j = joint[a * nc + c]
            if j > 0:
                mi += j / tot * log(j * tot / (pp[a] * pc[c])) / log(2.0)
    return mi


def part_stats(pk: List[Int], seqlen: Int, nc: Int, part: List[Int], np_: Int) -> Tuple[Float64, Float64]:
    var intra = 0
    var total = 0
    var joint = List[Float64](length=np_ * nc, fill=0)
    for t in range(len(pk) // TOPK):
        var c = t // seqlen
        for a in range(TOPK):
            var pa = part[pk[t * TOPK + a]]
            joint[pa * nc + c] += 1
            for b in range(a + 1, TOPK):
                total += 1
                if part[pk[t * TOPK + b]] == pa:
                    intra += 1
    return (Float64(intra) / Float64(total), mi_bits(joint, np_, nc))


def groups(args: List[String]) raises:
    var seqlen = Int(args[2])
    var cats = names(args[3])
    var nc = len(cats)
    var pk = layer_picks(args[1], Int(args[4]))
    if len(pk) // TOPK != nc * seqlen:
        raise Error("FAIL groups: " + String(len(pk) // TOPK) + " tokens, want " + String(nc) + " x " + String(seqlen))
    var raw: String
    with open(args[5], "r") as f:
        raw = f.read()
    var part = List[Int](length=NE, fill=0)
    var n = 0
    var np_ = 0
    for l in raw.split("\n"):
        if l.byte_length() == 0:
            continue
        if n >= NE:
            raise Error("FAIL groups: more than " + String(NE) + " lines in " + args[5])
        part[n] = Int(l)
        np_ = max(np_, part[n] + 1)
        n += 1
    var obs = part_stats(pk, seqlen, nc, part, np_)
    var s = UInt64(12345)
    var bi = List[Float64]()
    var bm = List[Float64]()
    for _ in range(200):
        var perm = part.copy()
        for i in range(n - 1, 0, -1):
            s = s * 6364136223846793005 + 1442695040888963407
            var j = Int((s >> 33) % UInt64(i + 1))
            var x = perm[i]
            perm[i] = perm[j]
            perm[j] = x
        var r = part_stats(pk, seqlen, nc, perm, np_)
        bi.append(r[0])
        bm.append(r[1])
    def ms(v: List[Float64]) -> Tuple[Float64, Float64]:
        var m = Float64(0)
        for x in v:
            m += x
        m /= Float64(len(v))
        var q = Float64(0)
        for x in v:
            q += (x - m) * (x - m)
        return (m, sqrt(q / Float64(len(v))))
    var a = ms(bi)
    var b = ms(bm)
    print("groups layer", args[4], "parts", np_, "nodes_in_file", n, "intra", obs[0], "random", a[0], "sd", a[1], "z", (obs[0] - a[0]) / max(a[1], 1e-12),
          "mi_bits", obs[1], "random", b[0], "sd", b[1], "z", (obs[1] - b[0]) / max(b[1], 1e-12))
    var joint = List[Int](length=np_ * nc, fill=0)
    for t in range(len(pk) // TOPK):
        for j in range(TOPK):
            joint[part[pk[t * TOPK + j]] * nc + t // seqlen] += 1
    for p in range(np_):
        var tot = 0
        var best = 0
        var size = 0
        for e in range(NE):
            if part[e] == p:
                size += 1
        for c in range(nc):
            tot += joint[p * nc + c]
            if joint[p * nc + c] > joint[p * nc + best]:
                best = c
        print("  part", p, "experts", size, "hits", tot, "top", cats[best], "share", Float64(joint[p * nc + best]) / Float64(max(tot, 1)))


def read_rows(path: String) raises -> List[Float32]:
    var raw: List[UInt8]
    with open(path, "r") as f:
        raw = f.read_bytes()
    if len(raw) % (4 * H) != 0:
        raise Error("FAIL score: " + path + " holds " + String(len(raw)) + " bytes, not whole rows of " + String(H) + " f32")
    var n = len(raw) // 4
    var out = List[Float32](length=n, fill=0)
    var src = raw.unsafe_ptr().unsafe_bitcast[Float32]()
    for i in range(n):
        out[i] = src[unsafe_offset=i]
    return out^


def score(args: List[String]) raises:
    var seqlen = Int(args[4])
    var cats = names(args[5])
    var inc = names(args[6])
    var nshow = Int(args[7]) if len(args) > 7 else 0
    var tg = List[Int]()
    var rf = read_rows(args[2])
    var arm = read_rows(args[3])
    if len(rf) != len(arm):
        raise Error("FAIL score: REF and ARM differ in rows")
    var R = len(rf) // H
    if R != len(cats) * seqlen:
        raise Error("FAIL score: " + String(R) + " rows, want " + String(len(cats)) + " x " + String(seqlen))
    if len(args) > 8:
        var txt: String
        with open(args[8], "r") as f:
            txt = f.read()
        var ids = List[Int]()
        for l in txt.split("\n"):
            if l.byte_length() > 0:
                ids.append(Int(l))
        if len(ids) != R:
            raise Error("FAIL score: " + String(len(ids)) + " ids, want " + String(R))
        for p in range(R):
            tg.append(ids[p + 1] if p % seqlen != seqlen - 1 else -1)
    else:
        for p in range(R):
            tg.append(-1)
    var ltr = List[Float64](length=R, fill=0)
    var lta = List[Float64](length=R, fill=0)
    var ptg = i64p(tg)
    var pltr = f64p(ltr)
    var plta = f64p(lta)
    var ix = ShardIndex(args[1])
    var w = ix["output.weight"]
    if w.dims[0] != H:
        raise Error("FAIL score: output.weight row length " + String(w.dims[0]) + ", want " + String(H))
    var V = w.dims[1]
    var rb = (H // 256) * block_bytes(w.gtype)
    var waddr = w.addr
    var gt = w.gtype
    var mr = List[Float64](length=R, fill=-1e300)
    var sr = List[Float64](length=R, fill=0)
    var ma = List[Float64](length=R, fill=-1e300)
    var sa = List[Float64](length=R, fill=0)
    var cs = List[Float64](length=R, fill=0)
    var ar = List[Int](length=R, fill=-1)
    var aa = List[Int](length=R, fill=-1)
    var chunk = List[Float32](length=VC * H, fill=0)
    var cp = f32p(chunk)
    var rp = f32p(rf)
    var ap = f32p(arm)
    var pmr = f64p(mr)
    var psr = f64p(sr)
    var pma = f64p(ma)
    var psa = f64p(sa)
    var pcs = f64p(cs)
    var par = i64p(ar)
    var paa = i64p(aa)
    var ng = (R + PG - 1) // PG
    var err = List[Int](length=1, fill=0)
    var perr = i64p(err)
    var v0 = 0
    while v0 < V:
        var nv = min(VC, V - v0)

        def deq(r: Int) {var cp, var waddr, var rb, var gt, var v0, var perr}:
            var row = List[Float32](length=H, fill=0)
            try:
                row_f32(gt, waddr + (v0 + r) * rb, H, row)
            except:
                perr[unsafe_offset=0] = 1
            for i in range(H):
                cp[unsafe_offset=r * H + i] = row[i]

        parallelize(deq, nv)
        if err[0] != 0:
            raise Error("FAIL score: ggml type " + String(gt) + " of output.weight has no dequantizer")

        def grp(g: Int) {var cp, var rp, var ap, var pmr, var psr, var pma, var psa, var pcs, var par, var paa, var ptg, var pltr, var plta, var v0, var nv, var R}:
            var p0 = g * PG
            var np_ = min(PG, R - p0)
            for v in range(nv):
                var acc = StaticTuple[SIMD[DType.float32, 16], 2 * PG](fill=SIMD[DType.float32, 16](0))
                var k = 0
                while k < H:
                    var wv = cp.unsafe_load[width=16](v * H + k)
                    comptime for q in range(PG):
                        var pq = min(p0 + q, R - 1)
                        acc[2 * q] = wv.fma(rp.unsafe_load[width=16](pq * H + k), acc[2 * q])
                        acc[2 * q + 1] = wv.fma(ap.unsafe_load[width=16](pq * H + k), acc[2 * q + 1])
                    k += 16
                comptime for q in range(PG):
                    if q < np_:
                        var p = p0 + q
                        var lr = Float64(acc[2 * q].reduce_add())
                        var la = Float64(acc[2 * q + 1].reduce_add())
                        if v0 + v == ptg[unsafe_offset=p]:
                            pltr[unsafe_offset=p] = lr
                            plta[unsafe_offset=p] = la
                        if lr > pmr[unsafe_offset=p]:
                            var sc = exp(pmr[unsafe_offset=p] - lr)
                            psr[unsafe_offset=p] = psr[unsafe_offset=p] * sc
                            pcs[unsafe_offset=p] = pcs[unsafe_offset=p] * sc
                            pmr[unsafe_offset=p] = lr
                            par[unsafe_offset=p] = v0 + v
                        var er = exp(lr - pmr[unsafe_offset=p])
                        psr[unsafe_offset=p] = psr[unsafe_offset=p] + er
                        pcs[unsafe_offset=p] = pcs[unsafe_offset=p] + er * (lr - la)
                        if la > pma[unsafe_offset=p]:
                            psa[unsafe_offset=p] = psa[unsafe_offset=p] * exp(pma[unsafe_offset=p] - la)
                            pma[unsafe_offset=p] = la
                            paa[unsafe_offset=p] = v0 + v
                        psa[unsafe_offset=p] = psa[unsafe_offset=p] + exp(la - pma[unsafe_offset=p])

        parallelize(grp, ng)
        v0 += nv
    _ = len(chunk)
    _ = len(ix.paths)
    _ = len(rf)
    _ = len(arm)
    var kl = List[Float64](length=R, fill=0)
    var nr = List[Float64](length=R, fill=0)
    var na = List[Float64](length=R, fill=0)
    for p in range(R):
        kl[p] = cs[p] / sr[p] - (mr[p] + log(sr[p])) + (ma[p] + log(sa[p]))
        if tg[p] >= 0:
            nr[p] = mr[p] + log(sr[p]) - ltr[p]
            na[p] = ma[p] + log(sa[p]) - lta[p]
    for p in range(nshow):
        print("row", p, "argmax_ref", ar[p], "argmax_arm", aa[p], "kl", kl[p], "nll_ref", nr[p], "nll_arm", na[p])
    var nc = len(cats)
    var ia = 0
    var ik = Float64(0)
    var inn = 0
    var inr = Float64(0)
    var ina = Float64(0)
    var int_ = 0
    for c in range(nc):
        var ag = 0
        var ks = Float64(0)
        var sr_ = Float64(0)
        var sa_ = Float64(0)
        var nt = 0
        for p in range(c * seqlen, (c + 1) * seqlen):
            if ar[p] == aa[p]:
                ag += 1
            ks += kl[p]
            if tg[p] >= 0:
                sr_ += nr[p]
                sa_ += na[p]
                nt += 1
        var isin = False
        for n in inc:
            if n == cats[c]:
                isin = True
        if isin:
            ia += ag
            ik += ks
            inn += seqlen
            inr += sr_
            ina += sa_
            int_ += nt
        var pp = String("")
        if nt > 0:
            pp = "ppl_ref " + String(exp(sr_ / Float64(nt))) + " ppl_arm " + String(exp(sa_ / Float64(nt))) + " ppl_ratio " + String(exp((sa_ - sr_) / Float64(nt)))
        print("cat", cats[c], "in" if isin else "out", "agree", ag, "of", seqlen, "pct", 100.0 * Float64(ag) / Float64(seqlen), "kl_mean", ks / Float64(seqlen), pp)
    if inn > 0:
        var pp = String("")
        if int_ > 0:
            pp = "ppl_ref " + String(exp(inr / Float64(int_))) + " ppl_arm " + String(exp(ina / Float64(int_))) + " ppl_ratio " + String(exp((ina - inr) / Float64(int_)))
        print("IN agree", ia, "of", inn, "pct", 100.0 * Float64(ia) / Float64(inn), "kl_mean", ik / Float64(inn), pp)


def main() raises:
    var av = argv()
    var args = List[String]()
    for i in range(1, len(av)):
        args.append(String(av[i]))
    if len(args) == 7 and args[0] == "keep":
        keep(args)
    elif len(args) == 9 and args[0] == "gkeep":
        gkeep(args)
    elif len(args) == 4 and args[0] == "coact":
        coact(args)
    elif len(args) == 6 and args[0] == "groups":
        groups(args)
    elif len(args) >= 7 and len(args) <= 9 and args[0] == "score":
        score(args)
    else:
        print("usage: flash-slim keep TRACE[,TRACE...] SEQLEN [v]COVER OUT CATS IN | score GGUF REF.f32 ARM.f32 SEQLEN CATS IN [NROWS [IDS]]")
        raise Error("FAIL flash-slim: bad arguments")
