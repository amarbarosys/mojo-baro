"""Neuron-data: a small multi-hop subset of the wikinet store, for miniagi/latent/neuron.mojo.

usage: neuron-data STORE OUT [N=2048] [R=4] [K=16] [SEED=1]
STORE = wikinet store (read-only): net.bin (CSR, both-way links), vecs.f32 (n x 384 bge), titles.txt.
Takes the N articles with the most links, keeps the induced link graph, drops articles with fewer than R links inside the
subset until none is left, and writes into OUT:
  ent.f32    n x 384  bge vectors, centred on the subset mean, unit length
  nbr.i32    n x R    slot r of article i = its r-th in-subset link by store id
  clu.i32    n        k-means cluster of the vector (K clusters, 12 Lloyd rounds)
  split.i32  n        start-article family: 0 train, 1 val, 2 test (hash of the local id, 80/10/10)
  titles.txt n lines
  meta.txt   n R K E
Prints the no-lookup answer ceilings: end-cluster accuracy on test starts of predictors that never see the graph.
"""
from std.sys import argv
from std.math import sqrt
from std.collections import Dict

comptime E = 384


def rd(path: String, at: Int, n: Int) raises -> List[UInt8]:
    with open(path, "r") as f:
        _ = f.seek(at)
        return f.read_bytes(n)


def mix(x: Int) -> Int:
    var h = UInt64(x) * 0x9E3779B97F4A7C15 + 0x1234567
    h ^= h >> 29
    h *= 0xBF58476D1CE4E5B9
    h ^= h >> 32
    return Int(h & 0x7FFFFFFF)


def main() raises:
    var a = argv()
    var store = String(a[1])
    var out = String(a[2])
    var N = Int(String(a[3])) if len(a) > 3 else 2048
    var R = Int(String(a[4])) if len(a) > 4 else 4
    var K = Int(String(a[5])) if len(a) > 5 else 16
    var seed = Int(String(a[6])) if len(a) > 6 else 1
    var head = rd(store + "/net.bin", 0, 8)
    var hp = head.unsafe_ptr().unsafe_bitcast[UInt32]()
    var n = Int(hp[unsafe_offset=0])
    var raw = rd(store + "/net.bin", 8, 4 * (n + 1))
    var ptr = raw.unsafe_ptr().unsafe_bitcast[UInt32]()
    var keys = List[UInt64](capacity=n)
    for i in range(n):
        keys.append((UInt64(ptr[unsafe_offset=i + 1] - ptr[unsafe_offset=i]) << 32) | UInt64(i))
    sort(keys)
    var cand = List[Int]()
    for j in range(N):
        cand.append(Int(keys[n - 1 - j] & 0xFFFFFFFF))
    keys = List[UInt64]()
    var colbase = 8 + 4 * (n + 1)
    var nbrs = List[List[Int]]()
    for c in cand:
        var b = rd(store + "/net.bin", colbase + 4 * Int(ptr[unsafe_offset=c]), 4 * Int(ptr[unsafe_offset=c + 1] - ptr[unsafe_offset=c]))
        var cp = b.unsafe_ptr().unsafe_bitcast[UInt32]()
        var row = List[Int]()
        for e in range(len(b) // 4):
            row.append(Int(cp[unsafe_offset=e]))
        nbrs.append(row^)
    var alive = List[Bool](length=N, fill=True)
    while True:
        var at = Dict[Int, Int]()
        for i in range(N):
            if alive[i]:
                at[cand[i]] = i
        var changed = False
        for i in range(N):
            if not alive[i]:
                continue
            var cnt = 0
            var seen = Dict[Int, Bool]()
            for g in nbrs[i]:
                if g in at and g != cand[i] and g not in seen:
                    seen[g] = True
                    cnt += 1
            if cnt < R:
                alive[i] = False
                changed = True
        if not changed:
            break
    var gid = List[Int]()
    var at = Dict[Int, Int]()
    for i in range(N):
        if alive[i]:
            at[cand[i]] = len(gid)
            gid.append(cand[i])
    var m = len(gid)
    var nbr = List[Int32](length=m * R, fill=0)
    var j = 0
    for i in range(N):
        if not alive[i]:
            continue
        var loc = List[Int]()
        var seen = Dict[Int, Bool]()
        for g in nbrs[i]:
            if g in at and g != cand[i] and g not in seen:
                seen[g] = True
                loc.append(g)
        sort(loc)
        for r in range(R):
            nbr[j * R + r] = Int32(at[loc[r]])
        j += 1
    var ent = List[Float32](length=m * E, fill=0)
    for i in range(m):
        var b = rd(store + "/vecs.f32", 4 * E * gid[i], 4 * E)
        var vp = b.unsafe_ptr().unsafe_bitcast[Float32]()
        for d in range(E):
            ent[i * E + d] = vp[unsafe_offset=d]
    var mean = List[Float64](length=E, fill=0)
    for i in range(m):
        for d in range(E):
            mean[d] += Float64(ent[i * E + d]) / Float64(m)
    for i in range(m):
        var ss: Float64 = 0
        for d in range(E):
            var x = Float64(ent[i * E + d]) - mean[d]
            ent[i * E + d] = Float32(x)
            ss += x * x
        var inv = 1 / sqrt(ss)
        for d in range(E):
            ent[i * E + d] = Float32(Float64(ent[i * E + d]) * inv)
    var cen = List[Float32](length=K * E, fill=0)
    for k in range(K):
        var pick = mix(seed * 7919 + k) % m
        for d in range(E):
            cen[k * E + d] = ent[pick * E + d]
    var clu = List[Int32](length=m, fill=0)
    for _ in range(12):
        for i in range(m):
            var best = 0
            var bs: Float32 = -1e30
            for k in range(K):
                var s: Float32 = 0
                for d in range(E):
                    s += ent[i * E + d] * cen[k * E + d]
                if s > bs:
                    bs = s
                    best = k
            clu[i] = Int32(best)
        var acc = List[Float32](length=K * E, fill=0)
        var cnt = List[Int](length=K, fill=0)
        for i in range(m):
            var k = Int(clu[i])
            cnt[k] += 1
            for d in range(E):
                acc[k * E + d] += ent[i * E + d]
        for k in range(K):
            if cnt[k] == 0:
                continue
            var ss: Float32 = 0
            for d in range(E):
                ss += acc[k * E + d] * acc[k * E + d]
            var inv = 1 / sqrt(ss)
            for d in range(E):
                cen[k * E + d] = acc[k * E + d] * inv
    var split = List[Int32](length=m, fill=0)
    for i in range(m):
        var h = mix(i * 31 + seed) % 10
        split[i] = Int32(2 if h == 0 else (1 if h == 1 else 0))
    var tit = List[String](length=m, fill=String(""))
    var want = Dict[Int, Int]()
    for i in range(m):
        want[gid[i]] = i
    var tb = open(store + "/titles.txt", "r").read_bytes()
    var line = 0
    var s0 = 0
    for p in range(len(tb)):
        if tb[p] == 10:
            if line in want:
                tit[want[line]] = String(StringSlice(unsafe_from_utf8=Span(tb)[s0:p]))
            line += 1
            s0 = p + 1
    var cc = List[Int](length=K, fill=0)
    for i in range(m):
        cc[Int(clu[i])] += 1
    print("neuron-data: n", m, "of", N, "R", R, "K", K, "cluster sizes", cc)
    var H = 3
    var nq = 1
    for _ in range(H):
        nq *= R
    var cnt1 = List[Int](length=K * K, fill=0)
    var cnt3 = Dict[Int, List[Int]]()
    var cnt0 = List[Int](length=K, fill=0)
    var tests = List[List[Int]]()
    for i in range(m):
        for q in range(nq):
            var cur = i
            var qq = q
            var sl = List[Int]()
            for _ in range(H):
                sl.append(qq % R)
                qq //= R
            for h in range(H):
                cur = Int(nbr[cur * R + sl[h]])
            var ans = Int(clu[cur])
            var key3 = Int(clu[i]) * nq + q
            if split[i] == 0:
                cnt0[ans] += 1
                cnt1[Int(clu[i]) * K + ans] += 1
                if key3 not in cnt3:
                    cnt3[key3] = List[Int](length=K, fill=0)
                cnt3[key3][ans] += 1
            elif split[i] == 2:
                tests.append([Int(clu[i]), q, ans])
    var maj0 = 0
    for k in range(K):
        if cnt0[k] > cnt0[maj0]:
            maj0 = k
    var h0 = 0
    var h1 = 0
    var h3 = 0
    for t in tests:
        h0 += Int(t[2] == maj0)
        var b1 = 0
        for k in range(K):
            if cnt1[t[0] * K + k] > cnt1[t[0] * K + b1]:
                b1 = k
        h1 += Int(t[2] == b1)
        var key3 = t[0] * nq + t[1]
        if key3 in cnt3:
            var b3 = 0
            for k in range(K):
                if cnt3[key3][k] > cnt3[key3][b3]:
                    b3 = k
            h3 += Int(t[2] == b3)
    var nt = Float64(len(tests))
    print("no-lookup ceilings on", len(tests), "test (start, path) pairs: majority", Float64(h0) / nt, "| by start cluster", Float64(h1) / nt, "| by start cluster + slots", Float64(h3) / nt, "| chance", 1.0 / Float64(K))
    with open(out + "/ent.f32", "w") as f:
        f.write_bytes(Span[UInt8](unsafe_ptr=ent.unsafe_ptr().unsafe_bitcast[UInt8](), length=len(ent) * 4))
    with open(out + "/nbr.i32", "w") as f:
        f.write_bytes(Span[UInt8](unsafe_ptr=nbr.unsafe_ptr().unsafe_bitcast[UInt8](), length=len(nbr) * 4))
    with open(out + "/clu.i32", "w") as f:
        f.write_bytes(Span[UInt8](unsafe_ptr=clu.unsafe_ptr().unsafe_bitcast[UInt8](), length=len(clu) * 4))
    with open(out + "/split.i32", "w") as f:
        f.write_bytes(Span[UInt8](unsafe_ptr=split.unsafe_ptr().unsafe_bitcast[UInt8](), length=len(split) * 4))
    var ts = String()
    for i in range(m):
        ts += tit[i] + "\n"
    with open(out + "/titles.txt", "w") as f:
        f.write(ts)
    with open(out + "/meta.txt", "w") as f:
        f.write(String(m) + " " + String(R) + " " + String(K) + " " + String(E) + "\n")
