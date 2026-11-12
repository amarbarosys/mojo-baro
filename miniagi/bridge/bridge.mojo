"""Latent-bridge trainer: bridge RUN [k=v ...]  (README.md: data layout, checkpoint format, checks).

RUN: A | B | C | D | ALL | single | identical | noxattn  (single/identical/noxattn = D with that control on).
k=v: data=DIR (default ~/.cache/baro/latent-bridge), steps=2000, b=32 (two micro-batches of b/2), seed=1, lr=3e-4,
     warmup=100, wd=0.01, clip=1.0, lam=1.0 (reader A MSE weight), every=50 (log + eval), ckpt=500, out=DIR (checkpoints,
     default data/ckpt/RUN), items=N (train on the first N train items only), evalsplit=val|train, patience=500,
     ginit=0 (gate init), single=1, ident=1, own=1, check=CASE (dump params.f32 + grad.f32 of micro-batch 0, exit),
     gate0=1 (gate-0 identity check on micro-batch 0, exit), skip=N split=train|val (check/gate0 use that split's items N..N+b/2-1).
"""
from std.sys import argv
from std.ffi import c_ssize_t, external_call
from std.math import ceildiv, log, sqrt, cos
from std.time import perf_counter_ns
from std.collections import Dict
from std.os import makedirs, getenv
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from ops import F, I, THREADS, grid, add_k
from lops import Gemm, sumsq_k, clip_k, adamw_k, CH
from bblock import *

comptime TAIL = 12
comptime PI = 3.14159265358979


struct Rng(Movable):
    var s: UInt64

    def __init__(out self, seed: Int):
        self.s = UInt64(seed) * 0x9E3779B97F4A7C15 + 1

    def next(mut self) -> UInt64:
        self.s ^= self.s << 13
        self.s ^= self.s >> 7
        self.s ^= self.s << 17
        return self.s

    def below(mut self, n: Int) -> Int:
        return Int(self.next() % UInt64(n))

    def normal(mut self) -> Float64:
        var u = (Float64(self.next() >> 11) + 1) / 9007199254740993.0
        var v = Float64(self.next() >> 11) / 9007199254740992.0
        return sqrt(-2 * log(u)) * cos(6.283185307179586 * v)


def read_text(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def pread(fd: Int, dst: F, nfloat: Int, off_float: Int) raises:
    if fd < 0:
        return
    var want = nfloat * 4
    var got = 0
    while got < want:
        var p = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(dst) + got)
        var n = external_call["pread", c_ssize_t](Int32(fd), p, want - got, Int64(off_float * 4 + got))
        if n <= 0:
            raise Error("FAIL pread fd " + String(fd) + " at " + String(off_float) + " floats, want " + String(nfloat))
        got += Int(n)


struct Items(Movable):
    var lp: List[Int]
    var lq: List[Int]
    var la: List[Int]
    var ans: List[Int32]
    var ids: Dict[String, Int]

    def __init__(out self, path: String) raises:
        self.lp = List[Int]()
        self.lq = List[Int]()
        self.la = List[Int]()
        self.ans = List[Int32]()
        self.ids = Dict[String, Int]()
        var text = read_text(path)
        for line in text.split("\n"):
            var f = line.split("\t")
            if len(f) < 9:
                continue
            self.ids[String(f[0])] = len(self.lp)
            self.lp.append(len(f[5].split(" ")))
            self.lq.append(len(f[7].split(" ")))
            var a = f[8].split(" ")
            if len(a) > TAIL - 1:
                raise Error("FAIL item " + String(f[0]) + ": answer longer than " + String(TAIL - 1))
            self.la.append(len(a))
            for k in range(TAIL - 1):
                self.ans.append(Int32(Int(String(a[k]))) if k < len(a) else Int32(-1))


def idx_offsets(path: String, per_hid: Int, per_att: Int) raises -> Tuple[List[Int], List[Int], List[String], List[Int], List[Int]]:
    """Per item: hid float offset, att float offset, tag, hid rows, att rows (the idx counts are authoritative)."""
    var h = List[Int]()
    var a = List[Int]()
    var tags = List[String]()
    var hr = List[Int]()
    var ar = List[Int]()
    var oh = 0
    var oa = 0
    var text = String("")
    try:
        text = read_text(path)
    except:
        return (h^, a^, tags^, hr^, ar^)
    for line in text.split("\n"):
        var f = line.split(" ")
        if len(f) < 3:
            continue
        tags.append(String(f[0]))
        h.append(oh)
        a.append(oa)
        hr.append(Int(String(f[1])))
        ar.append(Int(String(f[2])))
        oh += Int(String(f[1])) * per_hid
        oa += Int(String(f[2])) * per_att
    return (h^, a^, tags^, hr^, ar^)


struct Split(Movable):
    """One dump split: item indices in dump order, float offsets per item in each file, open fds."""
    var n: Int
    var idx: List[Int]
    var hoff: List[Int]
    var toff: List[Int]
    var aoff: List[Int]
    var s1off: List[Int]
    var s2off: List[Int]
    var trows: List[Int]
    var arows: List[Int]
    var have: List[Bool]
    var files: List[FileHandle]
    var fds: List[Int]

    def __init__(out self, root: String, name: String, it: Items, c: Cfg, ident: Bool) raises:
        self.idx = List[Int]()
        self.files = List[FileHandle]()
        self.fds = List[Int]()
        self.have = List[Bool]()
        var tags = List[String]()
        var sel = read_text(root + "/data/sel-" + name + ".txt")
        for line in sel.split("\n"):
            var s = String(line).strip()
            if s.byte_length() > 0:
                tags.append(String(s))
                self.idx.append(it.ids[String(s)])
        self.n = len(self.idx)
        var dumps = root + "/dumps/"
        var st = idx_offsets(dumps + "flash-student-" + name + ".idx", c.d, c.qw() + c.d)
        var te = idx_offsets(dumps + "flash-teacher-" + name + ".idx", c.d, c.qw() + c.d)
        var s1n = "flash-student-" if ident else "qwythos9b-"
        var s2n = "flash-student-" if ident else "qwen35-4b-"
        var s1 = idx_offsets(dumps + s1n + name + ".idx", c.s1, 0)
        var s2 = idx_offsets(dumps + s2n + name + ".idx", c.s2, 0)
        self.have = [len(st[2]) >= self.n, len(te[2]) >= self.n, len(s1[2]) >= self.n, len(s2[2]) >= self.n]
        print("split", name, "items", self.n, "present student", self.have[0], "teacher", self.have[1], "s1", self.have[2], "s2", self.have[3])
        for k in range(self.n):
            var tag = tags[k]
            if (self.have[0] and st[2][k] != tag) or (self.have[1] and te[2][k] != tag) or (self.have[2] and s1[2][k] != tag) or (self.have[3] and s2[2][k] != tag):
                raise Error("FAIL idx tag order at " + String(k) + ": sel " + tag)
        var zeros = List[Int](length=self.n, fill=0)
        self.hoff = st[0].copy() if self.have[0] else zeros.copy()
        self.aoff = st[1].copy() if self.have[0] else zeros.copy()
        self.arows = st[4].copy() if self.have[0] else zeros.copy()
        self.toff = te[0].copy() if self.have[1] else zeros.copy()
        self.trows = te[3].copy() if self.have[1] else zeros.copy()
        self.s1off = s1[0].copy() if self.have[2] else zeros.copy()
        self.s2off = s2[0].copy() if self.have[3] else zeros.copy()
        for k in range(self.n):
            var T = it.lp[self.idx[k]] + it.lq[self.idx[k]] + it.la[self.idx[k]]
            var la = it.la[self.idx[k]]
            var bad = (self.have[0] and (st[3][k] != T or st[4][k] < la)) or (self.have[2] and s1[3][k] != T) or (self.have[3] and s2[3][k] != T)
            bad = bad or (self.have[1] and (te[3][k] < la or (self.have[0] and te[4][k] != st[4][k])))
            if bad:
                raise Error("FAIL idx rows at " + tags[k] + ": T " + String(T) + " student " + String(st[3][k]) + " s1 " + String(s1[3][k]) + " s2 " + String(s2[3][k]) +
                            " att " + String(st[4][k]) + "/" + String(te[4][k]) + " teacher " + String(te[3][k]))
        var names: List[String] = [dumps + "flash-student-" + name + ".hid", dumps + "flash-teacher-" + name + ".hid",
                                   dumps + "flash-student-" + name + ".att", dumps + "flash-teacher-" + name + ".att",
                                   dumps + s1n + name + ".hid", dumps + s2n + name + ".hid", root + "/weights/ctarget-" + name + ".f32"]
        for p in names:
            try:
                var f = open(p, "r")
                self.fds.append(Int(f._get_raw_fd()))
                self.files.append(f^)
            except:
                self.fds.append(-1)

    def need(self, which: List[Int], what: String) raises:
        for i in which:
            if self.fds[i] < 0:
                raise Error("FAIL " + what + " needs dump file " + String(i) + " (0 student.hid 1 teacher.hid 2 student.att 3 teacher.att 4 s1 5 s2 6 ctarget), missing")




struct Stage(Movable):
    """Pinned host staging for one batch (two micro-batches) plus its int tables."""
    var hf: HostBuffer[DType.float32]
    var hi: HostBuffer[DType.int32]
    var nf: Int
    var ni: Int

    def __init__(out self, ctx: DeviceContext, nf: Int, ni: Int) raises:
        self.nf = nf
        self.ni = ni
        self.hf = ctx.enqueue_create_host_buffer[DType.float32](nf)
        self.hi = ctx.enqueue_create_host_buffer[DType.int32](ni)


struct Loader(Movable):
    var c: Cfg
    var mb: Int
    var maxlp: Int
    var maxla: Int
    var fmax: Int
    var imax: Int
    var fbytes: Int
    var fill_ns: Int

    def __init__(out self, c: Cfg, mb: Int, maxlp: Int, maxla: Int):
        self.c = c
        self.mb = mb
        self.maxlp = maxlp
        self.maxla = maxla
        var lsum = mb * maxlp
        var na = mb * maxla
        self.fmax = lsum * (c.d + c.s1 + c.s2) + na * (4 * c.d + c.qw()) + 2 * mb * c.nz * c.d + 64
        self.imax = 12 * mb + 2 * c.nsrc * lsum + na
        self.fbytes = 0
        self.fill_ns = 0

    def fill(mut self, sp: Split, it: Items, items: List[Int], st: Stage, half: Int, own: Bool) raises -> Tuple[Int, Int]:
        """Pread the micro-batch `items` (indices into sp) into stage half `half`; returns (lsum, na)."""
        var t0 = perf_counter_ns()
        var c = self.c
        var d = c.d
        var mb = len(items)
        var hf = st.hf.unsafe_ptr().unsafe_offset(half * self.fmax)
        var hi = st.hi.unsafe_ptr().unsafe_offset(half * self.imax)
        var lsum = 0
        var na = 0
        for i in range(mb):
            lsum += it.lp[sp.idx[items[i]]]
            na += it.la[sp.idx[items[i]]]
        var o = offsets(c, lsum, na, mb)
        var x0 = F(unsafe_from_address=Int(hf))
        var lo = 0
        var ao = 0
        var bytes = 0
        for i in range(mb):
            var k = items[i]
            var j = sp.idx[k]
            var lp = it.lp[j]
            var la = it.la[j]
            var ttail = sp.trows[k] - la
            var atail = sp.arows[k] - la
            pread(sp.fds[0], x0.unsafe_offset(o[0] + lo * d), lp * d, sp.hoff[k])
            pread(sp.fds[0], x0.unsafe_offset(o[3] + ao * d), la * d, sp.hoff[k] + (lp + it.lq[j] - 1) * d)
            pread(sp.fds[1], x0.unsafe_offset(o[4] + ao * d), la * d, sp.toff[k] + ttail * d)
            pread(sp.fds[2], x0.unsafe_offset(o[5] + ao * c.qw()), la * c.qw(), sp.aoff[k] + atail * c.qw())
            pread(sp.fds[2], x0.unsafe_offset(o[6] + ao * d), la * d, sp.aoff[k] + sp.arows[k] * c.qw() + atail * d)
            pread(sp.fds[3], x0.unsafe_offset(o[7] + ao * d), la * d, sp.aoff[k] + sp.arows[k] * c.qw() + atail * d)
            pread(sp.fds[6], x0.unsafe_offset(o[8] + i * c.nz * d), c.nz * d, k * c.nz * d)
            bytes += (lp * d + 4 * la * d + la * c.qw() + c.nz * d) * 4
            if c.nsrc >= 2:
                pread(sp.fds[4], x0.unsafe_offset(o[1] + lo * c.s1), lp * c.s1, sp.s1off[k])
                bytes += lp * c.s1 * 4
            if c.nsrc >= 3:
                pread(sp.fds[5], x0.unsafe_offset(o[2] + lo * c.s2), lp * c.s2, sp.s2off[k])
                bytes += lp * c.s2 * 4
            for r in range(c.nz * d):
                x0[unsafe_offset=o[9] + ((i + mb - 1) % mb) * c.nz * d + r] = x0[unsafe_offset=o[8] + i * c.nz * d + r]
            var koff = c.nsrc * lo
            hi[unsafe_offset=4 * i] = Int32(i * c.nz)
            hi[unsafe_offset=4 * i + 1] = Int32(c.nz)
            hi[unsafe_offset=4 * i + 2] = Int32(i * c.nz if own else koff)
            hi[unsafe_offset=4 * i + 3] = Int32(c.nz if own else c.nsrc * lp)
            hi[unsafe_offset=4 * mb + 4 * i] = Int32(ao)
            hi[unsafe_offset=4 * mb + 4 * i + 1] = Int32(la)
            hi[unsafe_offset=4 * mb + 4 * i + 2] = Int32(i * c.nz)
            hi[unsafe_offset=4 * mb + 4 * i + 3] = Int32(c.nz)
            hi[unsafe_offset=8 * mb + 4 * i] = Int32(ao)
            hi[unsafe_offset=8 * mb + 4 * i + 1] = Int32(la)
            hi[unsafe_offset=8 * mb + 4 * i + 2] = Int32(((i + 1) % mb) * c.nz)
            hi[unsafe_offset=8 * mb + 4 * i + 3] = Int32(c.nz)
            for s in range(c.nsrc):
                for r in range(lp):
                    hi[unsafe_offset=12 * mb + koff + s * lp + r] = Int32(s * lsum + lo + r)
                    hi[unsafe_offset=12 * mb + c.nsrc * lsum + koff + s * lp + r] = Int32(s)
            for q in range(la):
                hi[unsafe_offset=12 * mb + 2 * c.nsrc * lsum + ao + q] = it.ans[j * (TAIL - 1) + q]
            lo += lp
            ao += la
        self.fbytes += bytes
        self.fill_ns += Int(perf_counter_ns() - t0)
        return (lsum, na)

    def bat(self, dev: F, devi: I, half: Int, lsum: Int, na: Int, head: F, wo47: F) -> Bat:
        var c = self.c
        var mb = self.mb
        var o = offsets(c, lsum, na, mb)
        var f = dev.unsafe_offset(half * self.fmax)
        var t = devi.unsafe_offset(half * self.imax)
        return Bat(lsum, na, f.unsafe_offset(o[0]), f.unsafe_offset(o[1]), f.unsafe_offset(o[2]), f.unsafe_offset(o[3]), f.unsafe_offset(o[4]),
                   f.unsafe_offset(o[5]), f.unsafe_offset(o[6]), f.unsafe_offset(o[7]), f.unsafe_offset(o[8]), head, wo47,
                   t, t.unsafe_offset(4 * mb), t.unsafe_offset(12 * mb), t.unsafe_offset(12 * mb + c.nsrc * lsum), t.unsafe_offset(12 * mb + 2 * c.nsrc * lsum))


def needed(c: Cfg, ra: Bool, rb: Bool, rc: Bool, lam: Float32) -> List[Int]:
    """Dump files a run reads: student hid always; teacher hid for reader A with lam > 0; both att for reader B; sharers by nsrc; ctarget for C."""
    var needs: List[Int] = [0]
    if ra and lam > 0:
        needs.append(1)
    if rb:
        needs.append(2)
        needs.append(3)
    if c.nsrc >= 2:
        needs.append(4)
    if c.nsrc >= 3:
        needs.append(5)
    if rc:
        needs.append(6)
    return needs^


def offsets(c: Cfg, lsum: Int, na: Int, mb: Int) -> List[Int]:
    """Float offsets inside one staged micro-batch: x0 x1 x2 hA hT qB aoS aoT ct ctshift end."""
    var o = List[Int]()
    var sizes: List[Int] = [lsum * c.d, lsum * c.s1, lsum * c.s2, na * c.d, na * c.d, na * c.qw(), na * c.d, na * c.d, mb * c.nz * c.d, mb * c.nz * c.d]
    var acc = 0
    for s in sizes:
        o.append(acc)
        acc += (s + 3) // 4 * 4
    o.append(acc)
    return o^


def upload_stage(ctx: DeviceContext, st: Stage, dev: F, devi: I, nf: Int, ni: Int) raises:
    ctx.enqueue_copy(dst_buf=DeviceBuffer[DType.float32](ctx, dev, nf, owning=False), src_ptr=st.hf.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=DeviceBuffer[DType.int32](ctx, devi, ni, owning=False), src_ptr=st.hi.unsafe_ptr())


def load_f32(ctx: DeviceContext, path: String, dst: F, n: Int, st: Stage) raises:
    var f = open(path, "r")
    var fd = Int(f._get_raw_fd())
    var done = 0
    var p = F(unsafe_from_address=Int(st.hf.unsafe_ptr()))
    while done < n:
        var m = min(st.nf, n - done)
        pread(fd, p, m, done)
        ctx.enqueue_copy(dst_buf=DeviceBuffer[DType.float32](ctx, dst.unsafe_offset(done), m, owning=False), src_ptr=st.hf.unsafe_ptr())
        ctx.synchronize()
        done += m
    _ = f^


def sysval(path: String) -> Int:
    try:
        return Int(String(read_text(path)).strip())
    except:
        return -1


def vram_used() -> Int:
    var best = -1
    for k in range(4):
        best = max(best, sysval("/sys/class/drm/card" + String(k) + "/device/mem_info_vram_used"))
    return best


def rss_peak() raises -> Int:
    var text = read_text("/proc/self/status")
    for line in text.split("\n"):
        if line.startswith("VmHWM"):
            var f = String(line).split()
            return Int(String(f[1]))
    return -1


def cfg_load(root: String) raises -> Cfg:
    var c = Cfg(2560, 20, 6912, 32, 248320, 3, 4096, 2560, 24, 2, 256, 64, 4)
    try:
        var text = read_text(root + "/cfg.txt")
        for line in text.split("\n"):
            var f = String(line).split()
            if len(f) < 2:
                continue
            var k = String(f[0])
            var v = Int(String(f[1]))
            if k == "d":
                c.d = v
            elif k == "nh":
                c.nh = v
            elif k == "ff":
                c.ff = v
            elif k == "nz":
                c.nz = v
            elif k == "vocab":
                c.vocab = v
            elif k == "s1":
                c.s1 = v
            elif k == "s2":
                c.s2 = v
            elif k == "nhb":
                c.nhb = v
            elif k == "nkvb":
                c.nkvb = v
            elif k == "hdb":
                c.hdb = v
            elif k == "qlo":
                c.qlo = v
            elif k == "R":
                c.R = v
    except:
        pass
    return c


def init_params(c: Cfg, seed: Int, ginit: Float32) -> List[Float32]:
    var rng = Rng(seed)
    var s = psizes(c)
    var out = List[Float32]()
    for t in range(NPAR):
        var n = s[t]
        var sc: Float32 = 0.02
        if t == WO or t == W2 or t == WOA:
            sc = 0.02 / sqrt(Float32(2 * c.R))
        if t == Z0:
            sc = 1
        if t >= GM and t <= GAN:
            sc = 0
        if t == GA or t == GB:
            sc = 0
        for _ in range(n):
            var v: Float32 = Float32(Float64(sc) * rng.normal())
            if t >= GM and t <= GAN:
                v = 1
            if t == GA or t == GB:
                v = ginit
            out.append(v)
    return out^


def write_f32(path: String, data: List[Float32], header: String) raises:
    with open(path, "w") as fo:
        if header.byte_length() > 0:
            fo.write(header)
        fo.write_bytes(Span[UInt8](unsafe_ptr=data.unsafe_ptr().unsafe_bitcast[UInt8](), length=len(data) * 4))


def download(ctx: DeviceContext, src: F, n: Int) raises -> List[Float32]:
    var out = List[Float32](length=n, fill=0)
    ctx.enqueue_copy(dst_ptr=out.unsafe_ptr(), src_buf=DeviceBuffer[DType.float32](ctx, src, n, owning=False))
    ctx.synchronize()
    return out^


struct Loss(ImplicitlyCopyable):
    var ce: Float64
    var msea: Float64
    var mseb: Float64
    var msec: Float64
    var cosc: Float64
    var acc: Float64

    def __init__(out self, sc: List[Float32], na: Int, nz: Int):
        self.ce = Float64(sc[S_CE]) / Float64(max(na, 1))
        self.msea = Float64(sc[S_RA2]) / Float64(sc[S_HT2]) if sc[S_HT2] > 0 else 0.0
        self.mseb = Float64(sc[S_RB2]) / Float64(sc[S_AOT2])
        self.msec = Float64(sc[S_RC2]) / Float64(sc[S_CT2])
        self.cosc = Float64(sc[S_COS]) / Float64(nz)
        self.acc = Float64(sc[S_HIT]) / Float64(max(na, 1))

    def parts(self, run: Run) -> Tuple[Float64, Float64, Float64]:
        return (self.ce + Float64(run.lam) * self.msea if run.ra else 0.0, self.mseb if run.rb else 0.0, self.msec + self.cosc if run.rc else 0.0)

    def total(self, run: Run) -> Float64:
        var p = self.parts(run)
        return Float64(run.wa) * p[0] + Float64(run.wb) * p[1] + Float64(run.wc) * p[2]


def read_loss(ctx: DeviceContext, a: Act, na: Int, nz: Int) raises -> Loss:
    return Loss(download(ctx, a.sc, NSC), na, nz)


def fmt(x: Float64) -> String:
    return String(Float64(Int(x * 1e5 + (0.5 if x >= 0 else -0.5))) / 1e5)


def main() raises:
    var av = argv()
    if len(av) < 2:
        raise Error("usage: bridge RUN [k=v ...]")
    var runname = String(av[1])
    var root = getenv("HOME", "") + "/.cache/baro/latent-bridge"
    var steps = 2000
    var bsz = 32
    var seed = 1
    var lr = 3e-4
    var warmup = 100
    var wd: Float32 = 0.01
    var clipn: Float32 = 1
    var lam: Float32 = 1
    var every = 50
    var ckpt = 500
    var out = String("")
    var items = 0
    var evalsplit = String("val")
    var patience = 500
    var ginit: Float32 = 0
    var single = runname == "single"
    var ident = runname == "identical"
    var own = runname == "noxattn"
    var check = String("")
    var gate0 = False
    var skip = 0
    var csplit = String("train")
    for i in range(2, len(av)):
        var kv = String(av[i]).split("=")
        var k = String(kv[0])
        var v = String(kv[1])
        if k == "data":
            root = v
        elif k == "steps":
            steps = Int(v)
        elif k == "b":
            bsz = Int(v)
        elif k == "seed":
            seed = Int(v)
        elif k == "lr":
            lr = Float64(v)
        elif k == "warmup":
            warmup = Int(v)
        elif k == "wd":
            wd = Float32(Float64(v))
        elif k == "clip":
            clipn = Float32(Float64(v))
        elif k == "lam":
            lam = Float32(Float64(v))
        elif k == "every":
            every = Int(v)
        elif k == "ckpt":
            ckpt = Int(v)
        elif k == "out":
            out = v
        elif k == "items":
            items = Int(v)
        elif k == "evalsplit":
            evalsplit = v
        elif k == "patience":
            patience = Int(v)
        elif k == "ginit":
            ginit = Float32(Float64(v))
        elif k == "single":
            single = v == "1"
        elif k == "ident":
            ident = v == "1"
        elif k == "own":
            own = v == "1"
        elif k == "check":
            check = v
        elif k == "gate0":
            gate0 = v == "1"
        elif k == "skip":
            skip = Int(v)
        elif k == "split":
            csplit = v
        else:
            raise Error("FAIL arg: " + String(av[i]))
    var three = runname == "D" or runname == "ALL" or single or ident or own
    var ra = runname != "B" and runname != "C"
    var rb = runname == "B" or runname == "ALL"
    var rc = runname == "C" or runname == "ALL"
    if not (runname == "A" or runname == "B" or runname == "C" or runname == "D" or runname == "ALL" or single or ident or own):
        raise Error("FAIL run: " + runname)
    var c = cfg_load(root)
    c.nsrc = 3 if three else 1
    if ident:
        c.s1 = c.d
        c.s2 = c.d
    if out == "":
        out = root + "/ckpt/" + runname
    var mb = bsz // 2
    var run = Run(ra, rb, rc, single, own, 1, 1, 1, lam, 0.5)
    var it = Items(root + "/data/items.tsv")
    var tr = Split(root, "train", it, c, ident)
    var va = Split(root, evalsplit, it, c, ident)
    var ntrain = tr.n if items == 0 else min(items, tr.n)
    var maxlp = 0
    var maxla = 0
    for k in range(len(it.lp)):
        maxlp = max(maxlp, it.lp[k])
        maxla = max(maxla, it.la[k])
    var np = ntotal(c)
    var nd = ndecay(c)
    print("params run", runname, "nsrc", c.nsrc, "readers", ra, rb, rc, "single", single, "ident", ident, "own", own, "d", c.d, "nh", c.nh, "ff", c.ff,
          "nz", c.nz, "vocab", c.vocab, "s1", c.s1, "s2", c.s2, "R", c.R, "np", np, "ndecay", nd, "steps", steps, "b", bsz, "seed", seed, "lr", lr,
          "warmup", warmup, "wd", wd, "clip", clipn, "lam", lam, "items", ntrain, "eval", evalsplit, "maxlp", maxlp, "maxla", maxla, "ginit", ginit)
    var tw = perf_counter_ns()
    var ctx = DeviceContext()
    var ld = Loader(c, mb, maxlp, maxla)
    var st0 = Stage(ctx, 2 * ld.fmax, 2 * ld.imax)
    var st1 = Stage(ctx, 2 * ld.fmax, 2 * ld.imax)
    var stv = Stage(ctx, 2 * ld.fmax, 2 * ld.imax)
    var devb = ctx.enqueue_create_buffer[DType.float32](2 * ld.fmax)
    var devi = ctx.enqueue_create_buffer[DType.int32](2 * ld.imax)
    var devbv = ctx.enqueue_create_buffer[DType.float32](2 * ld.fmax)
    var deviv = ctx.enqueue_create_buffer[DType.int32](2 * ld.imax)
    var headb = ctx.enqueue_create_buffer[DType.float32](c.vocab * c.d)
    var wob = ctx.enqueue_create_buffer[DType.float32](c.d * c.qw())
    var head = F(unsafe_from_address=Int(headb.unsafe_ptr()))
    var wo47 = F(unsafe_from_address=Int(wob.unsafe_ptr()))
    load_f32(ctx, root + "/weights/head.f32", head, c.vocab * c.d, st0)
    load_f32(ctx, root + "/weights/wo47.f32", wo47, c.d * c.qw(), st0)
    var pb = ctx.enqueue_create_buffer[DType.float32](np)
    var gb = ctx.enqueue_create_buffer[DType.float32](np)
    var mbuf = ctx.enqueue_create_buffer[DType.float32](np)
    var vbuf = ctx.enqueue_create_buffer[DType.float32](np)
    var npart = ceildiv(np, CH)
    var partb = ctx.enqueue_create_buffer[DType.float32](npart)
    var scb = ctx.enqueue_create_buffer[DType.float32](4)
    var init = init_params(c, seed, ginit)
    ctx.enqueue_copy(dst_buf=pb, src_ptr=init.unsafe_ptr())
    ctx.enqueue_memset(mbuf, 0)
    ctx.enqueue_memset(vbuf, 0)
    ctx.synchronize()
    var p = F(unsafe_from_address=Int(pb.unsafe_ptr()))
    var g = F(unsafe_from_address=Int(gb.unsafe_ptr()))
    var mp = F(unsafe_from_address=Int(mbuf.unsafe_ptr()))
    var vp = F(unsafe_from_address=Int(vbuf.unsafe_ptr()))
    var part = F(unsafe_from_address=Int(partb.unsafe_ptr()))
    var sc = F(unsafe_from_address=Int(scb.unsafe_ptr()))
    var pl = ptrs(c, p)
    var gl = ptrs(c, g)
    var gm = Gemm(ctx, 16 * 1024 * 1024)
    var act = Act(ctx, c, mb, maxlp, maxla)
    var one = List[Float32](length=1, fill=1)
    ctx.enqueue_copy(dst_buf=DeviceBuffer[DType.float32](ctx, act.sc.unsafe_offset(S_ONE), 1, owning=False), src_ptr=one.unsafe_ptr())
    var dev = F(unsafe_from_address=Int(devb.unsafe_ptr()))
    var devip = I(unsafe_from_address=Int(devi.unsafe_ptr()))
    var devv = F(unsafe_from_address=Int(devbv.unsafe_ptr()))
    var devivp = I(unsafe_from_address=Int(deviv.unsafe_ptr()))
    var rng = Rng(seed + 7)
    var order = List[Int]()
    for k in range(ntrain):
        order.append(k)
    var cursor = ntrain
    var vitems = List[Int]()
    for k in range(min(bsz, va.n)):
        vitems.append(k)
    if check == "" and not gate0:
        var needs = needed(c, ra, rb, rc, lam)
        tr.need(needs, "training on train")
        va.need(needs, "eval on " + evalsplit)
    var vdims = List[Int]()
    for h in range(2):
        var sub = List[Int]()
        for k in range(h * mb, min(len(vitems), (h + 1) * mb)):
            sub.append(vitems[k])
        var r = ld.fill(va, it, sub, stv, h, own)
        vdims.append(r[0])
        vdims.append(r[1])
    upload_stage(ctx, stv, devv, devivp, 2 * ld.fmax, 2 * ld.imax)
    ctx.synchronize()
    if check != "" or gate0:
        var sub = List[Int]()
        for k in range(mb):
            sub.append(skip + k)
        var cs = Split(root, csplit, it, c, ident)
        var needs: List[Int] = [0, 2]
        if not gate0:
            needs = needed(c, ra, rb, rc, lam)
        cs.need(needs, "check/gate0 on " + csplit)
        var r = ld.fill(cs, it, sub, st0, 0, own)
        upload_stage(ctx, st0, dev, devip, ld.fmax, ld.imax)
        var b = ld.bat(dev, devip, 0, r[0], r[1], head, wo47)
        var run1 = Run(ra, rb, rc, single, own, 1, 1, 1, lam, 1)
        if gate0:
            var runab = Run(True, True, False, single, own, 1, 1, 1, lam, 1)
            forward(ctx, gm, c, runab, pl, act, b, b.tabA, b.ct)
            gm[False, True](ctx, act.hp, head, act.logits, b.na, c.vocab, c.d)
            var la = download(ctx, act.logits, b.na * c.vocab)
            gm[False, True](ctx, b.hA, head, act.logits, b.na, c.vocab, c.d)
            var lb = download(ctx, act.logits, b.na * c.vocab)
            copy(ctx, b.aoS, act.ra, b.na * c.d)
            ctx.enqueue_function[add_k](act.yb, act.ra, Int32(b.na * c.d), grid_dim=grid(b.na * c.d), block_dim=THREADS)
            var ob = download(ctx, act.ra, b.na * c.d)
            var os = download(ctx, b.aoS, b.na * c.d)
            var bad_a = 0
            var bad_b = 0
            for k in range(len(la)):
                bad_a += Int(la[k].to_bits() != lb[k].to_bits())
            for k in range(len(ob)):
                bad_b += Int(ob[k].to_bits() != os[k].to_bits())
            print("GATE0 rows", b.na, "logits_mismatch", bad_a, "of", len(la), "attnout_mismatch", bad_b, "of", len(ob))
            print("GATE0", "PASS" if bad_a == 0 and bad_b == 0 else "FAIL")
            if bad_a != 0 or bad_b != 0:
                raise Error("FAIL gate0")
            _ = (pb.unsafe_ptr(), gb.unsafe_ptr(), mbuf.unsafe_ptr(), vbuf.unsafe_ptr(), partb.unsafe_ptr(), scb.unsafe_ptr(), devb.unsafe_ptr(), devi.unsafe_ptr(),
             devbv.unsafe_ptr(), deviv.unsafe_ptr(), headb.unsafe_ptr(), wob.unsafe_ptr(), st0.hf.unsafe_ptr(), st0.hi.unsafe_ptr(), stv.hf.unsafe_ptr(), act.mem.unsafe_ptr())
            return
        ctx.enqueue_memset(gb, 0)
        forward(ctx, gm, c, run1, pl, act, b, b.tabA, b.ct)
        backward(ctx, gm, c, run1, pl, gl, act, b)
        ctx.synchronize()
        var L = read_loss(ctx, act, b.na, mb * c.nz)
        var pr = L.parts(run1)
        write_f32(check + "/params.f32", init, "")
        write_f32(check + "/grad.f32", download(ctx, g, np), "")
        var nzr = mb * c.nz * c.d
        var nad = b.na * c.d
        var nlm = act.lm * c.d
        var acts: List[String] = ["zn", "oa", "ua", "hp", "dhp", "dua", "doa", "dqa", "dka", "dva", "dzn", "dz", "m", "kk", "vv", "dm", "dkk", "dvv"]
        var aptr: List[F] = [act.zn, act.oa, act.ua, act.hp, act.dhp, act.dua, act.doa, act.dqa, act.dka, act.dva, act.dzn, act.dz, act.m, act.kk, act.vv, act.dm, act.dkk, act.dvv]
        var alen: List[Int] = [nzr, nad, nad, nad, nad, nad, nad, nad, nzr, nzr, nzr, nzr, nlm, nlm, nlm, nlm, nlm, nlm]
        for k in range(len(acts)):
            write_f32(check + "/act-" + acts[k] + ".f32", download(ctx, aptr[k], alen[k]), "")
        var bt = String("")
        for k in range(mb):
            bt += String(cs.idx[sub[k]]) + " "
        with open(check + "/batch.txt", "w") as fo:
            fo.write("items " + bt + "\nnsrc " + String(c.nsrc) + "\nra " + String(Int(ra)) + "\nrb " + String(Int(rb)) + "\nrc " + String(Int(rc)) +
                     "\nsingle " + String(Int(single)) + "\nown " + String(Int(own)) + "\nident " + String(Int(ident)) + "\nsplit " + String(Int(csplit == "val")) + "\nlam " + String(lam) +
                     "\nloss_a " + String(pr[0]) + "\nloss_b " + String(pr[1]) + "\nloss_c " + String(pr[2]) + "\nce " + String(L.ce) +
                     "\nmsea " + String(L.msea) + "\nmseb " + String(L.mseb) + "\nmsec " + String(L.msec) + "\ncosc " + String(L.cosc) + "\n")
        print("CHECK written to", check, "loss", pr[0], pr[1], pr[2])
        _ = (pb.unsafe_ptr(), gb.unsafe_ptr(), mbuf.unsafe_ptr(), vbuf.unsafe_ptr(), partb.unsafe_ptr(), scb.unsafe_ptr(), devb.unsafe_ptr(), devi.unsafe_ptr(),
             devbv.unsafe_ptr(), deviv.unsafe_ptr(), headb.unsafe_ptr(), wob.unsafe_ptr(), st0.hf.unsafe_ptr(), st0.hi.unsafe_ptr(), stv.hf.unsafe_ptr(), act.mem.unsafe_ptr())
        return
    var stages = List[Stage]()
    stages.append(st0^)
    stages.append(st1^)
    var dims = List[Int](length=4, fill=0)
    var ndims = List[Int](length=4, fill=0)
    var cur = 0
    next_batch(rng, order, cursor, ld, tr, it, stages[0], dims, own, ntrain, mb)
    if runname == "ALL":
        upload_stage(ctx, stages[0], dev, devip, 2 * ld.fmax, 2 * ld.imax)
        var b = ld.bat(dev, devip, 0, dims[0], dims[1], head, wo47)
        forward(ctx, gm, c, run, pl, act, b, b.tabA, b.ct)
        ctx.synchronize()
        var L = read_loss(ctx, act, b.na, mb * c.nz)
        var pr = L.parts(run)
        run.wa = Float32(1 / pr[0])
        run.wb = Float32(1 / pr[1])
        run.wc = Float32(1 / pr[2])
        print("gate0 losses A", pr[0], "B", pr[1], "C", pr[2], "weights", run.wa, run.wb, run.wc)
    var b1: Float32 = 0.9
    var b2: Float32 = 0.999
    var ema_g = 0.0
    var ema_s = 0.0
    var best = 1e30
    var best_step = 0
    var t0 = perf_counter_ns()
    var log_ns = 0
    var vram_peak = vram_used()
    var done = 0
    var nrm = List[Float32](length=4, fill=0)
    for step in range(steps):
        var logging = (step + 1) % every == 0 or step == steps - 1
        upload_stage(ctx, stages[cur], dev, devip, 2 * ld.fmax, 2 * ld.imax)
        ctx.enqueue_memset(gb, 0)
        var tl = List[Loss]()
        for h in range(2):
            var b = ld.bat(dev, devip, h, dims[2 * h], dims[2 * h + 1], head, wo47)
            forward(ctx, gm, c, run, pl, act, b, b.tabA, b.ct)
            backward(ctx, gm, c, run, pl, gl, act, b)
            if logging:
                if h == 0:
                    ctx.enqueue_function[sumsq_k](g, part, Int32(np), grid_dim=grid(npart), block_dim=THREADS)
                    ctx.enqueue_function[reduce_k](part, Int32(npart), sc.unsafe_offset(2), Int32(0), grid_dim=1, block_dim=AT)
                ctx.synchronize()
                tl.append(read_loss(ctx, act, b.na, mb * c.nz))
        ctx.enqueue_function[sumsq_k](g, part, Int32(np), grid_dim=grid(npart), block_dim=THREADS)
        ctx.enqueue_function[clip_k](part, Int32(npart), sc, clipn, grid_dim=1, block_dim=256)
        var t = Float64(step + 1)
        var f = min(1.0, t / Float64(max(warmup, 1)))
        if step >= warmup and steps > warmup:
            f = 0.5 * (1 + cos(PI * Float64(step - warmup) / Float64(steps - warmup)))
        var lr_now = Float32(lr * f)
        var c1 = Float32(1 / (1 - Float64(b1) ** t))
        var c2 = Float32(1 / (1 - Float64(b2) ** t))
        ctx.enqueue_function[adamw_k](p, g, mp, vp, sc, Int32(nd), lr_now, b1, b2, Float32(1e-8), wd, c1, c2, grid_dim=grid(nd), block_dim=THREADS)
        ctx.enqueue_function[adamw_k](p.unsafe_offset(nd), g.unsafe_offset(nd), mp.unsafe_offset(nd), vp.unsafe_offset(nd), sc, Int32(np - nd), lr_now, b1, b2,
            Float32(1e-8), Float32(0), c1, c2, grid_dim=grid(np - nd), block_dim=THREADS)
        done = step + 1
        if step + 1 < steps:
            next_batch(rng, order, cursor, ld, tr, it, stages[1 - cur], ndims, own, ntrain, mb)
        ctx.synchronize()
        if logging:
            var l0 = perf_counter_ns()
            ctx.enqueue_copy(dst_ptr=nrm.unsafe_ptr(), src_buf=scb)
            ctx.synchronize()
            var g32 = Float64(nrm[0]) * Float64(nrm[0])
            var g16 = 4 * Float64(nrm[2])
            var gsq = 2 * g32 - g16
            var trs = 32 * (g16 - g32)
            ema_g = 0.9 * ema_g + 0.1 * gsq if step >= every else gsq
            ema_s = 0.9 * ema_s + 0.1 * trs if step >= every else trs
            var vl = List[Loss]()
            var vs = List[Loss]()
            for h in range(2):
                var b = ld.bat(devv, devivp, h, vdims[2 * h], vdims[2 * h + 1], head, wo47)
                forward(ctx, gm, c, run, pl, act, b, b.tabA, b.ct)
                ctx.synchronize()
                vl.append(read_loss(ctx, act, b.na, mb * c.nz))
                forward(ctx, gm, c, run, pl, act, b, b.tabA.unsafe_offset(4 * mb), b.ct.unsafe_offset(mb * c.nz * c.d))
                ctx.synchronize()
                vs.append(read_loss(ctx, act, b.na, mb * c.nz))
            var ga = download(ctx, pl[GA], c.d)
            var gbv = download(ctx, pl[GB], c.nhb)
            var ma = 0.0
            for v in ga:
                ma += Float64(v)
            var mbb = 0.0
            for v in gbv:
                mbb += Float64(v)
            var tp = List[Float64](length=3, fill=0)
            var vp3 = List[Float64](length=3, fill=0)
            var sp3 = List[Float64](length=3, fill=0)
            var tacc = 0.0
            var vacc = 0.0
            var vtot = 0.0
            for h in range(2):
                var a3 = tl[h].parts(run)
                var v3 = vl[h].parts(run)
                var s3 = vs[h].parts(run)
                for k in range(3):
                    tp[k] += (a3[0] if k == 0 else (a3[1] if k == 1 else a3[2])) / 2
                    vp3[k] += (v3[0] if k == 0 else (v3[1] if k == 1 else v3[2])) / 2
                    sp3[k] += (s3[0] if k == 0 else (s3[1] if k == 1 else s3[2])) / 2
                tacc += tl[h].acc / 2
                vacc += vl[h].acc / 2
                vtot += vl[h].total(run) / 2
            var secs = Float64(Int(perf_counter_ns() - t0) - log_ns) / 1e9
            print("step", step + 1, "lr", lr_now, "| A", fmt(tp[0]), "/", fmt(vp3[0]), "B", fmt(tp[1]), "/", fmt(vp3[1]), "C", fmt(tp[2]), "/", fmt(vp3[2]),
                  "| acc", fmt(tacc), "/", fmt(vacc), "| sep", fmt(sp3[0] - vp3[0]), fmt(sp3[1] - vp3[1]), fmt(sp3[2] - vp3[2]),
                  "| gA", fmt(ma / Float64(c.d)), "gB", fmt(mbb / Float64(c.nhb)), "| gnorm", nrm[0], "clip", nrm[1], "| Bsimple", fmt(trs / gsq), "ema", fmt(ema_s / ema_g),
                  "| ms/step", fmt(secs * 1000 / Float64(step + 1)))
            vram_peak = max(vram_peak, vram_used())
            if vtot < best:
                best = vtot
                best_step = step + 1
            log_ns += Int(perf_counter_ns() - l0)
            if step + 1 - best_step >= patience:
                print("STOP val flat since step", best_step)
                break
        if ckpt > 0 and (step + 1) % ckpt == 0 and step + 1 < steps:
            save(ctx, c, p, np, out, runname, step + 1, seed)
        cur = 1 - cur
        for k in range(4):
            dims[k] = ndims[k]
    ctx.synchronize()
    var sec = Float64(Int(perf_counter_ns() - t0) - log_ns) / 1e9
    save(ctx, c, p, np, out, runname, done, seed)
    vram_peak = max(vram_peak, vram_used())
    var gbs = Float64(ld.fbytes) / 1e9 / (Float64(ld.fill_ns) / 1e9)
    print("RESULT run", runname, "steps", done, "train_sec", fmt(sec), "ms_per_step", fmt(sec * 1000 / Float64(max(done, 1))), "samples_per_s", fmt(Float64(done * bsz) / sec),
          "dump_GB", fmt(Float64(ld.fbytes) / 1e9), "dump_GBps", fmt(gbs), "fill_sec", fmt(Float64(ld.fill_ns) / 1e9), "vram_peak_MB", vram_peak >> 20, "rss_peak_MB", rss_peak() >> 10,
          "wall_sec", fmt(Float64(perf_counter_ns() - tw) / 1e9))
    _ = (pb.unsafe_ptr(), gb.unsafe_ptr(), mbuf.unsafe_ptr(), vbuf.unsafe_ptr(), partb.unsafe_ptr(), scb.unsafe_ptr(), devb.unsafe_ptr(), devi.unsafe_ptr(),
         devbv.unsafe_ptr(), deviv.unsafe_ptr(), headb.unsafe_ptr(), wob.unsafe_ptr())
    _ = (stages[0].hf.unsafe_ptr(), stages[1].hf.unsafe_ptr(), stv.hf.unsafe_ptr(), act.mem.unsafe_ptr())


def next_batch(mut rng: Rng, mut order: List[Int], mut cursor: Int, mut ld: Loader, sp: Split, it: Items, st: Stage, mut dm: List[Int], own: Bool, n: Int, mb: Int) raises:
    for h in range(2):
        var sub = List[Int]()
        for _ in range(mb):
            if cursor >= n:
                for k in range(n - 1, 0, -1):
                    var j = rng.below(k + 1)
                    var t = order[k]
                    order[k] = order[j]
                    order[j] = t
                cursor = 0
            sub.append(order[cursor])
            cursor += 1
        var r = ld.fill(sp, it, sub, st, h, own)
        dm[2 * h] = r[0]
        dm[2 * h + 1] = r[1]

def save(ctx: DeviceContext, c: Cfg, p: F, np: Int, outdir: String, runname: String, step: Int, seed: Int) raises:
    var data = download(ctx, p, np)
    var hdr = "baro-bridge v1 run=" + runname + " step=" + String(step) + " seed=" + String(seed) + " np=" + String(np) + " d=" + String(c.d) + " nh=" + String(c.nh) +
              " ff=" + String(c.ff) + " nz=" + String(c.nz) + " vocab=" + String(c.vocab) + " nsrc=" + String(c.nsrc) + " s1=" + String(c.s1) + " s2=" + String(c.s2) +
              " nhb=" + String(c.nhb) + " nkvb=" + String(c.nkvb) + " hdb=" + String(c.hdb) + " qlo=" + String(c.qlo) + " R=" + String(c.R) + " order=" + PNAMES + "\n"
    try:
        makedirs(outdir, exist_ok=True)
    except:
        pass
    var path = outdir + "/step-" + String(step) + ".ckpt"
    write_f32(path, data, hdr)
    print("checkpoint written to", path)
