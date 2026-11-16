# DFlash2 parity harness (lane-dflash2 D2c). Oracle = a D2d dump written by llama.cpp (common/speculative.cpp,
# env LLAMA_DFLASH_DUMP; branch dflash2-dump ff30654a3). Records, int32 headers, native endian:
#   'F' seq n_chunk n_embd             | i32 pos[n_chunk] | f32 feat[n_chunk * n_embd]
#   'L' seq pos0 id_last n_block cols  | f32 lattice[n_block * cols] | i32 n_result | i32 ids[n_result]
# lattice row i: [K candidate ids as floats | K x K scores, row p = predecessor p]; row 0 unused.
# Usage:
#   dflash_parity DUMP...                         CPU: parse, counts, greedy walk of each dumped lattice vs its ids
#   dflash_parity --gpu DRAFT TARGET DUMP...      replay every dump through serve/dflash.mojo in file order
#       (F -> inject, L -> draft at its pos0/anchor, compare), arm fwd (dumped positions) and arm rev (every
#       feature at position + 1). Per position i = 1..7: candidate overlap /16, forced decision (our argmax at the
#       dumped predecessor == the dumped id), free greedy walk prefix, score |diff| over (prev id, cand id) pairs
#       present in both lattices. Bars: docs/lanes/2026-11-16-dflash2-report.md (frozen before the first run).
from std.sys import argv
from std.math import abs

from max.gpu.host import DeviceContext
from dflash import DFlash, read_index, nbytes, dev_load, NB, TOPK, COLS

comptime K = TOPK


struct Dump(Movable):
    var b: List[UInt8]
    var rec: List[Int]  # byte offset of every record header, file order
    var f_off: List[Int]
    var l_off: List[Int]
    var rows: Int

    def __init__(out self, path: String) raises:
        with open(path, "r") as f:
            self.b = f.read_bytes()
        self.rec = List[Int]()
        self.f_off = List[Int]()
        self.l_off = List[Int]()
        self.rows = 0
        var o = 0
        while o + 16 <= len(self.b):
            var kind = self.i32(o)
            self.rec.append(o)
            if kind == ord("F"):
                self.f_off.append(o)
                var n = self.i32(o + 8)
                self.rows += n
                o += 16 + 4 * n * (1 + self.i32(o + 12))
            elif kind == ord("L"):
                self.l_off.append(o)
                o += 24 + 4 * self.i32(o + 16) * self.i32(o + 20)
                o += 4 + 4 * self.i32(o)
            else:
                raise Error("dump: bad record kind " + String(kind) + " at byte " + String(o))
        if o != len(self.b):
            raise Error("dump: " + String(len(self.b) - o) + " trailing bytes")

    def i32(self, o: Int) -> Int:
        return Int(self.b.unsafe_ptr().unsafe_offset(o).unsafe_bitcast[Int32]()[])

    def f32(self, o: Int) -> Float32:
        return self.b.unsafe_ptr().unsafe_offset(o).unsafe_bitcast[Float32]()[]

    def lattice(self, o: Int) -> List[Float32]:
        """Lattice [n_block, cols] of the L record at byte o."""
        var n = self.i32(o + 16) * self.i32(o + 20)
        var out = List[Float32](capacity=n)
        for i in range(n):
            out.append(self.f32(o + 24 + 4 * i))
        return out^

    def ids(self, o: Int) -> List[Int]:
        var no = o + 24 + 4 * self.i32(o + 16) * self.i32(o + 20)
        var out = List[Int]()
        for i in range(self.i32(no)):
            out.append(self.i32(no + 4 + 4 * i))
        return out^


def walk(lat: List[Float32], nb: Int) -> List[Int]:
    """Greedy lattice walk (speculative.cpp:1419-1466, p_min 0): predecessor 0, row i picks argmax of its scores."""
    var ids = List[Int]()
    var pred = 0
    for i in range(1, nb):
        pred = argmax(lat, i, pred)
        ids.append(Int(lat[i * COLS + pred]))
    return ids^


def argmax(lat: List[Float32], i: Int, p: Int) -> Int:
    var best = 0
    for j in range(1, K):
        if lat[i * COLS + K + p * K + j] > lat[i * COLS + K + p * K + best]:
            best = j
    return best


def find(lat: List[Float32], i: Int, id: Int) -> Int:
    """Column of candidate id in row i, -1 if absent."""
    for j in range(K):
        if Int(lat[i * COLS + j]) == id:
            return j
    return -1


@fieldwise_init
struct Tally(Copyable, Movable):
    var rows: Int
    var overlap: Int
    var forced: Int
    var walk_full: Int
    var walk_pos: Int
    var cycles: Int
    var pairs: Int
    var dsum: Float64
    var dmax: Float32


def compare(ours: List[Float32], rl: List[Float32], anchor: Int, ref_ids: List[Int], mut t: Tally):
    t.cycles += 1
    for i in range(1, NB):
        t.rows += 1
        for j in range(K):
            t.overlap += Int(find(rl, i, Int(ours[i * COLS + j])) >= 0)
        var dprev = anchor if i == 1 else ref_ids[i - 2]
        var p = 0 if i == 1 else find(ours, i - 1, dprev)
        if p >= 0 and i - 1 < len(ref_ids):
            t.forced += Int(Int(ours[i * COLS + argmax(ours, i, p)]) == ref_ids[i - 1])
        for pp in range(1 if i == 1 else K):
            var a = anchor if i == 1 else Int(ours[(i - 1) * COLS + pp])
            var rp = 0 if i == 1 else find(rl, i - 1, a)
            if rp < 0:
                continue
            for j in range(K):
                var rj = find(rl, i, Int(ours[i * COLS + j]))
                if rj < 0:
                    continue
                var d = abs(ours[i * COLS + K + pp * K + j] - rl[i * COLS + K + rp * K + rj])
                t.pairs += 1
                t.dsum += Float64(d)
                t.dmax = max(t.dmax, d)
    var w = walk(ours, NB)
    var n = 0
    while n < min(len(w), len(ref_ids)) and w[n] == ref_ids[n]:
        n += 1
    t.walk_pos += n
    t.walk_full += Int(n == len(ref_ids))


def cpu(paths: List[String]) raises:
    var bad = 0
    for path in paths:
        var d = Dump(path)
        var seen = List[Bool](length=1 << 16, fill=False)
        var npos = 0
        for o in d.f_off:
            for i in range(d.i32(o + 8)):
                var p = d.i32(o + 16 + 4 * i)
                npos += Int(not seen[p])
                seen[p] = True
        var eq = 0
        for o in d.l_off:
            var w = walk(d.lattice(o), d.i32(o + 16))
            var ids = d.ids(o)
            var ok = d.i32(o + 20) == COLS and len(ids) <= len(w)
            for i in range(min(len(ids), len(w))):
                ok = ok and w[i] == ids[i]
            eq += Int(ok)
        bad += len(d.l_off) - eq
        print("dump", path, "| F", len(d.f_off), "rows", d.rows, "pos", npos, "| L", len(d.l_off), "| walk==ids", eq, "/", len(d.l_off))
    if bad:
        raise Error("FAIL walk: " + String(bad) + " L records differ from the dumped ids")


def gpu(draft_dir: String, target_dir: String, paths: List[String]) raises:
    var ctx = DeviceContext()
    var tix = read_index(target_dir)
    var emb = tix["token_embd.weight"]
    var head = tix["output.weight"]
    var eb = nbytes(emb[0], emb[2])
    var tw = dev_load(ctx, target_dir + "/pack.bin", [emb[1], head[1]], [eb, nbytes(head[0], head[2])])
    var df = DFlash(ctx, draft_dir, tw^, 0, eb, 1 if head[0] == "q4" else 0)
    print("loaded draft", draft_dir, "target emb", emb[0], "head", head[0])
    var tallies = List[Tally]()
    for arm in range(2):
        var t = Tally(0, 0, 0, 0, 0, 0, 0, 0.0, 0.0)
        for path in paths:
            var d = Dump(path)
            df.reset()
            var u = t.copy()
            for o in d.rec:
                if d.i32(o) == ord("F"):
                    var n = d.i32(o + 8)
                    var pos = List[Int]()
                    for i in range(n):
                        pos.append(d.i32(o + 16 + 4 * i) + arm)
                    var feat = Pointer[Float32, MutAnyOrigin](unsafe_from_address=Int(d.b.unsafe_ptr()) + o + 16 + 4 * n)
                    df.inject(ctx, feat, pos)
                else:
                    df.draft(ctx, d.i32(o + 12), d.i32(o + 8))
                    var ours = List[Float32](capacity=NB * COLS)
                    for i in range(NB * COLS):
                        ours.append(df.lat_h[i])
                    compare(ours, d.lattice(o), d.i32(o + 12), d.ids(o), t)
            print(
                "arm", "fwd" if arm == 0 else "rev", path, "| cycles", t.cycles - u.cycles, "forced",
                t.forced - u.forced, "/", t.rows - u.rows, "overlap", t.overlap - u.overlap, "/", K * (t.rows - u.rows),
                "walk full", t.walk_full - u.walk_full, "prefix", t.walk_pos - u.walk_pos,
            )
        tallies.append(t.copy())
        print(
            "arm", "fwd" if arm == 0 else "rev", "POOLED | forced", Float64(t.forced) / Float64(t.rows), "overlap",
            Float64(t.overlap) / Float64(t.rows), "/16 | walk full", t.walk_full, "/", t.cycles, "prefix mean",
            Float64(t.walk_pos) / Float64(t.cycles), "| score pairs", t.pairs, "mean |d|", t.dsum / Float64(max(t.pairs, 1)),
            "max |d|", t.dmax,
        )
    var f = Float64(tallies[0].forced) / Float64(tallies[0].rows)
    var r = Float64(tallies[1].forced) / Float64(tallies[1].rows)
    var ov = Float64(tallies[0].overlap) / Float64(tallies[0].rows)
    print("B1 forced fwd >= 0.90:", "PASS" if f >= 0.90 else "FAIL", f)
    print("B2 overlap fwd >= 14.0/16:", "PASS" if ov >= 14.0 else "FAIL", ov)
    print("R1 forced rev <= 0.5 x fwd:", "PASS" if r <= 0.5 * f else "FAIL", r)
    _ = df^


def main() raises:
    var args = argv()
    var paths = List[String]()
    if len(args) >= 5 and args[1] == "--gpu":
        for a in range(4, len(args)):
            paths.append(String(args[a]))
        gpu(String(args[2]), String(args[3]), paths)
        return
    if len(args) < 2:
        raise Error("usage: dflash_parity DUMP... | dflash_parity --gpu DRAFT_DIR TARGET_DIR DUMP...")
    for a in range(1, len(args)):
        paths.append(String(args[a]))
    cpu(paths)
