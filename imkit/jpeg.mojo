# VENDORED COPY of amarbaro/mojo-imkit (Apache-2.0, see imkit/LICENSE), tag v0.9.0 (eb91210),
# ~/Projects/mojo/mojo-imkit, file src/imkit/jpeg.mojo. This repo keeps a real file
# rather than an -I path outside the tree, because a clone must build without anything
# outside it. Sync by hand if the upstream changes; tools/ci-checks.sh compares the two
# against the tag when the upstream is present and says so when they drift.
#
"""Pure Mojo JPEG decoder, no library, no FFI. `jpeg_decode(bytes)` returns (width, height, pixels) with the pixel packing and the
picture-size guard of `png_decode` (r | g << 8 | b << 16 | a << 24, a = 255).

Baseline, extended-sequential and progressive Huffman, 8-bit, 1 or 3 components, any 1 to 4 sampling factors (4:4:4, 4:2:2, 4:4:0,
4:2:0 get libjpeg's "fancy" triangle upsampling, others replicate), restart markers, APPn / COM segments skipped, Adobe transform
flag read. The IDCT is libjpeg's accurate integer one (jidctint) and the YCbCr conversion uses its fixed-point constants, so the
output sits within 1 to 2 levels of libjpeg-turbo. Arithmetic coding, lossless, 12-bit and 4-component (CMYK) files raise.

Every length is checked before it is used and a truncated entropy stream raises: the bit reader feeds zero bits after a marker or
the end of data and raises as soon as one of them is consumed.
Oracle: tests/jpeg_test.sh (ImageMagick writes the fixtures and the golden RGBA, never run time).
"""

comptime MAXPX = 1 << 26  # same value as assets.MAXPX (tests/jpeg_test.mojo checks it): 64 M pixels


@fieldwise_init
struct Comp(Copyable, Movable):
    var id: Int
    var h: Int  # sampling factors
    var v: Int
    var tq: Int  # quantization table id
    var bw: Int  # blocks per row / column of the coefficient grid, padded to whole MCUs
    var bh: Int
    var nbx: Int  # blocks per row / column that hold picture, the walk of a one-component scan
    var nby: Int
    var dw: Int  # samples per row / rows of picture
    var dh: Int
    var coff: Int  # offset of this component in `co` and in `pl`
    var poff: Int
    var dc: Int  # Huffman table ids of the current scan
    var ac: Int
    var pred: Int  # DC predictor
    var q: Int  # offset of the latched quantization table in `qs`, -1 until the first scan


def tr(mut w: List[Int], b: Int, st: Int, sh: Int):
    """libjpeg jidctint 1-D inverse DCT over w[b + k * st], k = 0..7, in place, descaled by `sh` bits."""
    var z2 = w[b + 2 * st]
    var z3 = w[b + 6 * st]
    var z1 = (z2 + z3) * 4433
    var t2 = z1 - z3 * 15137
    var t3 = z1 + z2 * 6270
    var t0 = (w[b] + w[b + 4 * st]) << 13
    var t1 = (w[b] - w[b + 4 * st]) << 13
    var t10 = t0 + t3
    var t13 = t0 - t3
    var t11 = t1 + t2
    var t12 = t1 - t2
    var a0 = w[b + 7 * st]
    var a1 = w[b + 5 * st]
    var a2 = w[b + 3 * st]
    var a3 = w[b + st]
    z1 = a0 + a3
    z2 = a1 + a2
    z3 = a0 + a2
    var z4 = a1 + a3
    var z5 = (z3 + z4) * 9633
    a0 *= 2446
    a1 *= 16819
    a2 *= 25172
    a3 *= 12299
    z1 *= -7373
    z2 *= -20995
    z3 = z3 * -16069 + z5
    z4 = z4 * -3196 + z5
    a0 += z1 + z3
    a1 += z2 + z4
    a2 += z2 + z3
    a3 += z1 + z4
    var r = 1 << (sh - 1)
    w[b] = (t10 + a3 + r) >> sh
    w[b + 7 * st] = (t10 - a3 + r) >> sh
    w[b + st] = (t11 + a2 + r) >> sh
    w[b + 6 * st] = (t11 - a2 + r) >> sh
    w[b + 2 * st] = (t12 + a1 + r) >> sh
    w[b + 5 * st] = (t12 - a1 + r) >> sh
    w[b + 3 * st] = (t13 + a0 + r) >> sh
    w[b + 4 * st] = (t13 - a0 + r) >> sh


def clamp8(v: Int) -> Int:
    return min(max(v, 0), 255)


struct Jpeg(Movable):
    var d: List[UInt8]
    var pos: Int
    var acc: Int  # bit reader: the low `nb` bits of acc are unread, the lowest `vb` of them are virtual zeros past a marker
    var nb: Int
    var vb: Int
    var stop: Bool
    var eobrun: Int
    var ss: Int  # scan parameters
    var se: Int
    var ah: Int
    var al: Int
    var ri: Int  # restart interval
    var adobe: Int  # Adobe transform flag, -1 none
    var jfif: Bool
    var prog: Bool
    var w: Int
    var h: Int
    var hm: Int  # largest sampling factors
    var vm: Int
    var mcux: Int
    var mcuy: Int
    var zz: List[Int]
    var qt: List[Int]  # 4 tables x 64, natural order
    var qs: List[Int]  # latched tables, 64 per component
    var lk: List[Int]  # per Huffman table (DC 0-3, AC 4-7): 9-bit lookahead (len << 8 | symbol), 0 = longer code
    var mc: List[Int]  # largest code of each length, -1 = none
    var vo: List[Int]  # value index = code + vo[length]
    var hv: List[Int]  # symbols
    var hdef: List[Bool]
    var comps: List[Comp]
    var co: List[Int16]  # coefficients, natural order, 64 per block; a baseline file keeps one block here and transforms it at once
    var ws: List[Int]  # IDCT workspace
    var pl: List[UInt8]  # decoded planes

    def __init__(out self, src: Span[UInt8, _]):
        self.d = List[UInt8]()
        self.d.extend(src)
        self.pos = 0
        self.acc = 0
        self.nb = 0
        self.vb = 0
        self.stop = False
        self.eobrun = 0
        self.ss = 0
        self.se = 63
        self.ah = 0
        self.al = 0
        self.ri = 0
        self.adobe = -1
        self.jfif = False
        self.prog = False
        self.w = 0
        self.h = 0
        self.hm = 1
        self.vm = 1
        self.mcux = 0
        self.mcuy = 0
        self.zz = [0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63]
        self.qt = List[Int](length=256, fill=1)
        self.qs = List[Int]()
        self.lk = List[Int](length=8 * 512, fill=0)
        self.mc = List[Int](length=8 * 18, fill=-1)
        self.vo = List[Int](length=8 * 18, fill=0)
        self.hv = List[Int](length=8 * 256, fill=0)
        self.hdef = List[Bool](length=8, fill=False)
        self.comps = List[Comp]()
        self.co = List[Int16]()
        self.ws = List[Int](length=64, fill=0)
        self.pl = List[UInt8]()

    def be16(self, i: Int) -> Int:
        return Int(self.d[i]) << 8 | Int(self.d[i + 1])

    # ---- bit reader ----

    def fill(mut self):
        while self.nb <= 48:
            if self.stop:
                self.acc <<= 8
                self.nb += 8
                self.vb += 8
                continue
            var p = self.pos
            if p >= len(self.d):
                self.stop = True
                continue
            var b = Int(self.d[p])
            if b == 0xFF:
                if p + 1 >= len(self.d):
                    self.stop = True
                    continue
                var c = Int(self.d[p + 1])
                if c == 0xFF:  # fill byte
                    self.pos += 1
                    continue
                if c != 0:  # a marker ends the entropy-coded data
                    self.stop = True
                    continue
                self.pos += 2
            else:
                self.pos += 1
            self.acc = (self.acc << 8) | b
            self.nb += 8

    def take(mut self, n: Int) raises:
        """Drop n bits; raise if that used a virtual one."""
        self.nb -= n
        if self.nb < self.vb:
            raise Error("jpeg: truncated entropy-coded data")
        self.acc &= (1 << self.nb) - 1

    def get(mut self, n: Int) raises -> Int:
        if self.nb < n:
            self.fill()
        var v = (self.acc >> (self.nb - n)) & ((1 << n) - 1)
        self.take(n)
        return v

    def ext(mut self, s: Int) raises -> Int:
        """s bits as a signed magnitude (JPEG EXTEND)."""
        if s == 0:
            return 0
        var v = self.get(s)
        return v if v >= (1 << (s - 1)) else v - (1 << s) + 1

    def huff(mut self, t: Int) raises -> Int:
        if self.nb < 16:
            self.fill()
        var e = self.lk[t * 512 + ((self.acc >> (self.nb - 9)) & 511)]
        var l = e >> 8
        var sym = e & 255
        if e == 0:
            l = 10
            while True:
                if l > 16:
                    raise Error("jpeg: bad Huffman code")
                var code = (self.acc >> (self.nb - l)) & ((1 << l) - 1)
                if code <= self.mc[t * 18 + l]:
                    sym = self.hv[t * 256 + self.vo[t * 18 + l] + code]
                    break
                l += 1
        self.take(l)
        return sym

    def restart(mut self) raises:
        var n = len(self.d)
        while self.pos + 1 < n and not (self.d[self.pos] == 0xFF and self.d[self.pos + 1] != 0 and self.d[self.pos + 1] != 0xFF):
            self.pos += 1
        if self.pos + 1 >= n:
            raise Error("jpeg: truncated, restart marker missing")
        var m = Int(self.d[self.pos + 1])
        if m < 0xD0 or m > 0xD7:
            raise Error("jpeg: restart marker expected, found FF" + hex(m))
        self.pos += 2
        self.acc = 0
        self.nb = 0
        self.vb = 0
        self.stop = False
        self.eobrun = 0
        for i in range(len(self.comps)):
            self.comps[i].pred = 0

    # ---- marker segments ----

    def dqt(mut self, s: Int, e: Int) raises:
        var i = s
        while i < e:
            var pq, tq = Int(self.d[i]) >> 4, Int(self.d[i]) & 15
            i += 1
            if tq > 3 or pq > 1 or i + 64 * (pq + 1) > e:
                raise Error("jpeg: bad DQT")
            for k in range(64):
                self.qt[tq * 64 + self.zz[k]] = self.be16(i + 2 * k) if pq == 1 else Int(self.d[i + k])
            i += 64 * (pq + 1)

    def dht(mut self, s: Int, e: Int) raises:
        var i = s
        while i < e:
            if i + 17 > e:
                raise Error("jpeg: bad DHT")
            var tc, th = Int(self.d[i]) >> 4, Int(self.d[i]) & 15
            var total = 0
            for l in range(1, 17):
                total += Int(self.d[i + l])
            if tc > 1 or th > 3 or total > 256 or i + 17 + total > e:
                raise Error("jpeg: bad DHT")
            var t = tc * 4 + th
            for k in range(512):
                self.lk[t * 512 + k] = 0
            var code, k = 0, 0
            for l in range(1, 17):
                var n = Int(self.d[i + l])
                self.vo[t * 18 + l] = k - code
                for _ in range(n):
                    if l <= 9:
                        var sym = Int(self.d[i + 17 + k])
                        for f in range(code << (9 - l), (code + 1) << (9 - l)):
                            self.lk[t * 512 + f] = l << 8 | sym
                    self.hv[t * 256 + k] = Int(self.d[i + 17 + k])
                    code += 1
                    k += 1
                if code > (1 << l):
                    raise Error("jpeg: Huffman table over-subscribed")
                self.mc[t * 18 + l] = code - 1 if n > 0 else -1
                code <<= 1
            self.hdef[t] = True
            i += 17 + total

    def sof(mut self, prog: Bool, s: Int, e: Int) raises:
        if len(self.comps) > 0:
            raise Error("jpeg: two SOF markers")
        if e - s < 6 or self.d[s] != 8:
            raise Error("jpeg: only 8-bit samples")
        self.prog = prog
        self.h, self.w = self.be16(s + 1), self.be16(s + 3)
        var nf = Int(self.d[s + 5])
        if self.w < 1 or self.h < 1:
            raise Error("jpeg: empty picture")
        if self.w * self.h > MAXPX:  # before any allocation that grows with w * h
            raise Error("jpeg: " + String(self.w) + " x " + String(self.h) + " is too large (" + String(MAXPX) + " pixels at most)")
        if nf == 4:
            raise Error("jpeg: 4 components (CMYK / YCCK) not supported")
        if nf != 1 and nf != 3:
            raise Error("jpeg: " + String(nf) + " components not supported")
        if e - s != 6 + 3 * nf:
            raise Error("jpeg: bad SOF length")
        for i in range(nf):
            var p = s + 6 + 3 * i
            var hh, vv, tq = Int(self.d[p + 1]) >> 4, Int(self.d[p + 1]) & 15, Int(self.d[p + 2])
            if nf == 1:  # a one-component scan is one block per MCU whatever the factors say
                hh, vv = 1, 1
            if hh < 1 or hh > 4 or vv < 1 or vv > 4 or tq > 3:
                raise Error("jpeg: bad component " + String(i))
            self.hm, self.vm = max(self.hm, hh), max(self.vm, vv)
            self.comps.append(Comp(Int(self.d[p]), hh, vv, tq, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1))
        self.mcux = (self.w + 8 * self.hm - 1) // (8 * self.hm)
        self.mcuy = (self.h + 8 * self.vm - 1) // (8 * self.vm)
        var nco, npl = 0, 0
        for i in range(nf):
            var c = self.comps[i].copy()
            c.bw, c.bh = self.mcux * c.h, self.mcuy * c.v
            c.dw, c.dh = (self.w * c.h + self.hm - 1) // self.hm, (self.h * c.v + self.vm - 1) // self.vm
            c.nbx, c.nby = (c.dw + 7) // 8, (c.dh + 7) // 8
            c.coff, c.poff = nco, npl
            nco += c.bw * c.bh * 64
            npl += c.bw * c.bh * 64
            self.comps[i] = c^
        self.co = List[Int16](length=nco if prog else 64, fill=0)
        self.pl = List[UInt8](length=npl, fill=0)
        self.qs = List[Int](length=64 * nf, fill=1)

    def sos(mut self, s: Int, e: Int) raises:
        var nf = len(self.comps)
        if nf == 0:
            raise Error("jpeg: scan before SOF")
        var ns = Int(self.d[s])
        if ns < 1 or ns > nf or e - s != 4 + 2 * ns:
            raise Error("jpeg: bad SOS")
        var sc = List[Int]()
        for i in range(ns):
            var c = -1
            for k in range(nf):
                if self.comps[k].id == Int(self.d[s + 1 + 2 * i]):
                    c = k
            if c < 0 or c in sc:
                raise Error("jpeg: scan names an unknown or repeated component")
            var tt = Int(self.d[s + 2 + 2 * i])
            if (tt >> 4) > 3 or (tt & 15) > 3:
                raise Error("jpeg: bad Huffman table id")
            self.comps[c].dc, self.comps[c].ac = tt >> 4, 4 + (tt & 15)
            sc.append(c)
        var q = 2 * ns
        self.ss, self.se = Int(self.d[s + 1 + q]), Int(self.d[s + 2 + q])
        self.ah, self.al = Int(self.d[s + 3 + q]) >> 4, Int(self.d[s + 3 + q]) & 15
        if not self.prog:
            self.ss, self.se, self.ah, self.al = 0, 63, 0, 0
        elif self.ss > self.se or self.se > 63 or self.al > 13 or (self.ss == 0 and self.se != 0) or (self.ss > 0 and ns != 1) or (self.ah != 0 and self.ah != self.al + 1):
            raise Error("jpeg: bad progressive scan parameters")
        for c in sc:
            if self.comps[c].q < 0:  # a component keeps the table it first scanned with
                self.comps[c].q = 64 * c
                for k in range(64):
                    self.qs[64 * c + k] = self.qt[self.comps[c].tq * 64 + k]
            if (not self.prog or (self.ss == 0 and self.ah == 0)) and not self.hdef[self.comps[c].dc]:
                raise Error("jpeg: DC Huffman table missing")
            if (not self.prog or self.ss > 0) and not self.hdef[self.comps[c].ac]:
                raise Error("jpeg: AC Huffman table missing")
        self.acc, self.nb, self.vb, self.stop, self.eobrun = 0, 0, 0, False, 0
        for c in sc:
            self.comps[c].pred = 0
        self.scan(sc)

    # ---- entropy-coded data ----

    def scan(mut self, sc: List[Int]) raises:
        var one = len(sc) == 1
        var total = self.mcux * self.mcuy
        if one:
            total = self.comps[sc[0]].nbx * self.comps[sc[0]].nby
        for m in range(total):
            if self.ri > 0 and m > 0 and m % self.ri == 0:
                self.restart()
            if one:
                var k = self.comps[sc[0]].copy()
                self.block(sc[0], k.coff + ((m // k.nbx) * k.bw + m % k.nbx) * 64)
            else:
                for c in sc:
                    var k = self.comps[c].copy()
                    for vy in range(k.v):
                        for hx in range(k.h):
                            self.block(c, k.coff + ((m // self.mcux * k.v + vy) * k.bw + m % self.mcux * k.h + hx) * 64)

    def block(mut self, c: Int, o: Int) raises:
        if not self.prog:
            var s = self.huff(self.comps[c].dc)
            if s > 15:
                raise Error("jpeg: bad DC category")
            var df = self.ext(s)
            self.comps[c].pred += df
            for i in range(64):
                self.co[i] = 0
            self.co[0] = Int16(self.comps[c].pred)
            var k = 1
            while k < 64:
                var rs = self.huff(self.comps[c].ac)
                var r, sz = rs >> 4, rs & 15
                if sz == 0:
                    if r != 15:
                        break
                    k += 16
                    continue
                k += r
                if k > 63:
                    raise Error("jpeg: coefficient index past 63")
                self.co[self.zz[k]] = Int16(self.ext(sz))
                k += 1
            self.put(c, o, 0)
        elif self.ss == 0:
            if self.ah == 0:
                var s = self.huff(self.comps[c].dc)
                if s > 15:
                    raise Error("jpeg: bad DC category")
                var df = self.ext(s)
                self.comps[c].pred += df
                self.co[o] = Int16(self.comps[c].pred * (1 << self.al))
            elif self.get(1) == 1:
                self.co[o] = self.co[o] | Int16(1 << self.al)
        elif self.ah == 0:
            if self.eobrun > 0:
                self.eobrun -= 1
                return
            var k = self.ss
            while k <= self.se:
                var rs = self.huff(self.comps[c].ac)
                var r, sz = rs >> 4, rs & 15
                if sz != 0:
                    k += r
                    if k > self.se:
                        raise Error("jpeg: coefficient index past the band")
                    self.co[o + self.zz[k]] = Int16(self.ext(sz) * (1 << self.al))
                elif r == 15:
                    k += 15
                else:
                    self.eobrun = (1 << r) - 1 + (self.get(r) if r > 0 else 0)
                    break
                k += 1
        else:
            self.refine(c, o)

    def refine(mut self, c: Int, o: Int) raises:
        """AC successive-approximation refinement scan of one block (libjpeg decode_mcu_AC_refine)."""
        var p1, m1 = 1 << self.al, -1 << self.al
        var k = self.ss
        if self.eobrun == 0:
            while k <= self.se:
                var rs = self.huff(self.comps[c].ac)
                var r, s = rs >> 4, rs & 15
                if s != 0:
                    if s != 1:
                        raise Error("jpeg: bad refinement symbol")
                    s = p1 if self.get(1) == 1 else m1
                elif r != 15:
                    self.eobrun = 1 << r
                    if r > 0:
                        self.eobrun += self.get(r)
                    break
                while k <= self.se:  # step over the nonzero coefficients (refining them) and r zero ones
                    var i = o + self.zz[k]
                    var cf = Int(self.co[i])
                    if cf != 0:
                        if self.get(1) == 1 and (cf & p1) == 0:
                            self.co[i] = Int16(cf + (p1 if cf >= 0 else m1))
                    else:
                        r -= 1
                        if r < 0:
                            break
                    k += 1
                if s != 0:
                    if k > self.se:
                        raise Error("jpeg: refinement past the band")
                    self.co[o + self.zz[k]] = Int16(s)
                k += 1
        if self.eobrun > 0:
            while k <= self.se:
                var i = o + self.zz[k]
                var cf = Int(self.co[i])
                if cf != 0 and self.get(1) == 1 and (cf & p1) == 0:
                    self.co[i] = Int16(cf + (p1 if cf >= 0 else m1))
                k += 1
            self.eobrun -= 1

    # ---- the file ----

    def parse(mut self) raises:
        var n = len(self.d)
        if n < 4 or self.d[0] != 0xFF or self.d[1] != 0xD8:
            raise Error("jpeg: not a JPEG")
        self.pos = 2
        var scans = 0
        while True:
            while self.pos < n and self.d[self.pos] != 0xFF:  # entropy data left over, or garbage between segments
                self.pos += 1
            while self.pos < n and self.d[self.pos] == 0xFF:
                self.pos += 1
            if self.pos >= n:
                raise Error("jpeg: truncated, no EOI")
            var m = Int(self.d[self.pos])
            self.pos += 1
            if m == 0 or m == 1 or (m >= 0xD0 and m <= 0xD7):
                continue
            if m == 0xD9:
                break
            if self.pos + 2 > n:
                raise Error("jpeg: truncated segment")
            var s, e = self.pos + 2, self.pos + self.be16(self.pos)
            if e < s or e > n:
                raise Error("jpeg: truncated segment")
            self.pos = e
            if m == 0xDB:
                self.dqt(s, e)
            elif m == 0xC4:
                self.dht(s, e)
            elif m == 0xDD:
                if e - s != 2:
                    raise Error("jpeg: bad DRI")
                self.ri = self.be16(s)
            elif m == 0xEE and e - s >= 12 and self.d[s] == 0x41 and self.d[s + 1] == 0x64 and self.d[s + 2] == 0x6F:  # "Ado"
                self.adobe = Int(self.d[s + 11])
            elif m == 0xE0 and e - s >= 5 and self.d[s] == 0x4A and self.d[s + 1] == 0x46:  # "JF"
                self.jfif = True
            elif m == 0xC0 or m == 0xC1 or m == 0xC2:
                self.sof(m == 0xC2, s, e)
            elif m == 0xC3 or (m >= 0xC5 and m <= 0xCF and m != 0xC8 and m != 0xCC):
                raise Error("jpeg: unsupported process, SOF" + hex(m - 0xC0) + " (lossless, hierarchical or arithmetic)")
            elif m == 0xDA:
                self.sos(s, e)
                scans += 1
        if scans == 0:
            raise Error("jpeg: no image data")

    def put(mut self, c: Int, o: Int, so: Int):
        """Dequantize and inverse-transform the block whose coefficients are at co[so] into the plane slot of block offset o."""
        var k = self.comps[c].copy()
        var pw = k.bw * 8
        var bi = (o - k.coff) // 64
        var dst = k.poff + (bi // k.bw) * 8 * pw + (bi % k.bw) * 8
        var qo = k.q if k.q >= 0 else 64 * c
        var flat = True
        for i in range(1, 64):
            if self.co[so + i] != 0:
                flat = False
                break
        if flat:  # DC only: the whole block is one value
            var v = UInt8(clamp8(((Int(self.co[so]) * self.qs[qo] + 4) >> 3) + 128))
            for y in range(8):
                for x in range(8):
                    self.pl[dst + y * pw + x] = v
            return
        for i in range(64):
            self.ws[i] = Int(self.co[so + i]) * self.qs[qo + i]
        for i in range(8):
            tr(self.ws, i, 8, 11)
        for y in range(8):
            tr(self.ws, y * 8, 1, 18)
            for x in range(8):
                self.pl[dst + y * pw + x] = UInt8(clamp8(self.ws[y * 8 + x] + 128))

    def idct(mut self):
        """Progressive files: transform every block once all scans are in, then drop the coefficients."""
        if self.prog:
            for c in range(len(self.comps)):
                var k = self.comps[c].copy()
                for bi in range(k.bw * k.bh):
                    self.put(c, k.coff + bi * 64, k.coff + bi * 64)
        self.co = List[Int16]()

    def up(self, c: Int, y: Int, mut o: List[UInt8]):
        """Row y of component c upsampled to full width into o (w + 2 long), libjpeg's fancy upsampling where it has one."""
        var k = self.comps[c].copy()
        var pw = k.bw * 8
        var r0 = k.poff + (y * k.v // self.vm) * pw
        if k.h == self.hm and k.v == self.vm:
            for x in range(self.w):
                o[x] = self.pl[r0 + x]
            return
        var tall = k.v * 2 == self.vm
        var sy = y >> 1 if tall else y
        var oy = max(sy - 1, 0) if y % 2 == 0 else min(sy + 1, k.dh - 1)  # the nearer other row, used by the 2-high cases
        r0 = k.poff + sy * pw
        var r1 = k.poff + oy * pw
        if k.h * 2 == self.hm and k.dw > 2 and (k.v == self.vm or tall):  # h2v1 and h2v2
            var wa, wb, be, bo = (3, 1, 8, 7) if tall else (4, 0, 4, 8)
            for i in range(k.dw):
                var cur = wa * Int(self.pl[r0 + i]) + wb * Int(self.pl[r1 + i])
                var lo = cur if i == 0 else wa * Int(self.pl[r0 + i - 1]) + wb * Int(self.pl[r1 + i - 1])
                var hi = cur if i == k.dw - 1 else wa * Int(self.pl[r0 + i + 1]) + wb * Int(self.pl[r1 + i + 1])
                o[2 * i] = UInt8((3 * cur + lo + be) >> 4)
                o[2 * i + 1] = UInt8((3 * cur + hi + bo) >> 4)
        elif k.h == self.hm and tall:  # h1v2
            for x in range(self.w):
                o[x] = UInt8((3 * Int(self.pl[r0 + x]) + Int(self.pl[r1 + x]) + (1 if y % 2 == 0 else 2)) >> 2)
        else:  # replicate
            r0 = k.poff + (y * k.v // self.vm) * pw
            for x in range(self.w):
                o[x] = self.pl[r0 + x * k.h // self.hm]

    def pixels(self) -> List[UInt32]:
        var px = List[UInt32](length=self.w * self.h, fill=0)
        var a = List[UInt8](length=self.w + 2, fill=0)
        var b = List[UInt8](length=self.w + 2, fill=0)
        var c = List[UInt8](length=self.w + 2, fill=0)
        var grey = len(self.comps) == 1
        var rgb = not grey and (self.adobe == 0 or (self.adobe < 0 and not self.jfif and self.comps[0].id == 0x52 and self.comps[1].id == 0x47 and self.comps[2].id == 0x42))
        for y in range(self.h):
            self.up(0, y, a)
            if not grey:
                self.up(1, y, b)
                self.up(2, y, c)
            for x in range(self.w):
                var r, g, bl = Int(a[x]), Int(a[x]), Int(a[x])
                if rgb:
                    g, bl = Int(b[x]), Int(c[x])
                elif not grey:
                    var cb, cr = Int(b[x]) - 128, Int(c[x]) - 128
                    r += (91881 * cr + 32768) >> 16
                    g += (-22554 * cb - 46802 * cr + 32768) >> 16
                    bl += (116130 * cb + 32768) >> 16
                px[y * self.w + x] = UInt32(clamp8(r)) | UInt32(clamp8(g)) << 8 | UInt32(clamp8(bl)) << 16 | 0xFF000000
        return px^


def jpeg_decode(d: Span[UInt8, _]) raises -> Tuple[Int, Int, List[UInt32]]:
    """(width, height, pixels) of a Huffman JPEG, 8-bit, 1 or 3 components."""
    var j = Jpeg(d)
    j.parse()
    j.idct()
    return (j.w, j.h, j.pixels())
