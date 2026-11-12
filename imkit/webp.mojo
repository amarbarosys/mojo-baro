# VENDORED COPY of amarbaro/mojo-imkit (Apache-2.0, see imkit/LICENSE), tag v0.9.0 (eb91210),
# ~/Projects/mojo/mojo-imkit, file src/imkit/webp.mojo. This repo keeps a real file
# rather than an -I path outside the tree, because a clone must build without anything
# outside it. Sync by hand if the upstream changes; tools/ci-checks.sh compares the two
# against the tag when the upstream is present and says so when they drift.
#
"""Pure Mojo WebP decoder (no FFI, no system library). `webp_decode(bytes)` returns (width, height, pixels), pixels packed
r | g << 8 | b << 16 | a << 24 like png_decode. Lossless VP8L: every transform, color cache, meta prefix codes. Lossy VP8 key frames
and the ALPH chunk are in the sections below. Animation raises. Test oracle: ImageMagick/libwebp (tests/webp_fixtures.sh).
"""

from imkit.vp8tab import coeff_proba, coeff_update, bmode_proba, dc_table, ac_table

comptime MAXPX = 1 << 26  # the largest picture webp_decode allocates for (same limit as png_decode in assets.mojo)
comptime MAXTAB = 1 << 24  # prefix-code table entries one VP8L image may build (64 MB)


def le16(d: Span[UInt8, _], i: Int) -> Int:
    return Int(d[i]) | Int(d[i + 1]) << 8


def le24(d: Span[UInt8, _], i: Int) -> Int:
    return Int(d[i]) | Int(d[i + 1]) << 8 | Int(d[i + 2]) << 16


def le32(d: Span[UInt8, _], i: Int) -> Int:
    return le24(d, i) | Int(d[i + 3]) << 24


def size_ok(w: Int, h: Int) raises:
    """Checked before any allocation that grows with w * h."""
    if w < 1 or h < 1 or w > MAXPX or h > MAXPX or w * h > MAXPX:
        raise Error("webp: " + String(w) + " x " + String(h) + " is not a picture size (" + String(MAXPX) + " pixels at most)")


# ---------------------------------------------------------------------------------------------------------------------
# VP8L (lossless)
# ---------------------------------------------------------------------------------------------------------------------


struct Bits(Movable):
    """LSB-first bit reader. Reads past the end give zero bits and `over()` turns true."""

    var d: List[UInt8]
    var pos: Int  # next byte to load, counts past the end too
    var val: UInt64
    var nbits: Int

    def __init__(out self, d: Span[UInt8, _]):
        self.d = List[UInt8](d)
        self.pos = 0
        self.val = 0
        self.nbits = 0

    def fill(mut self):
        while self.nbits <= 56:
            var b = UInt64(self.d[self.pos]) if self.pos < len(self.d) else 0
            self.val |= b << UInt64(self.nbits)
            self.pos += 1
            self.nbits += 8

    def bits(mut self, n: Int) -> Int:
        self.fill()
        var v = Int(self.val & ((UInt64(1) << UInt64(n)) - 1))
        self.val >>= UInt64(n)
        self.nbits -= n
        return v

    def over(self) -> Bool:
        return self.pos * 8 - self.nbits > len(self.d) * 8


def build_table(lens: List[Int], mut tab: List[UInt32]) raises -> Int:
    """Append the lookup table of the canonical prefix code with these lengths, entry = len << 16 | symbol. Returns the table
    index width. One used symbol is a zero-bit code; anything else must be a complete code."""
    var cnt = List[Int](length=16, fill=0)
    var used, last = 0, 0
    for s in range(len(lens)):
        if lens[s] > 0:
            cnt[lens[s]] += 1
            used += 1
            last = s
    if used == 0:
        raise Error("webp: empty prefix code")
    if used == 1:
        tab.append(UInt32(last))
        return 0
    var left, maxl = 1, 0
    for l in range(1, 16):
        left = left * 2 - cnt[l]
        if left < 0:
            raise Error("webp: oversubscribed prefix code")
        if cnt[l] > 0:
            maxl = l
    if left != 0:
        raise Error("webp: incomplete prefix code")
    if len(tab) + (1 << maxl) > MAXTAB:
        raise Error("webp: prefix codes too large")
    var next = List[Int](length=17, fill=0)
    var code = 0
    for l in range(1, 16):
        code = (code + cnt[l - 1]) << 1 if l > 1 else 0
        next[l] = code
    var base = len(tab)
    tab.resize(base + (1 << maxl), 0)
    for s in range(len(lens)):
        var l = lens[s]
        if l == 0:
            continue
        var c = next[l]
        next[l] += 1
        var rev = 0  # canonical codes are MSB first, the stream is LSB first
        for k in range(l):
            rev |= ((c >> k) & 1) << (l - 1 - k)
        for k in range(rev, 1 << maxl, 1 << l):
            tab[base + k] = UInt32(l << 16 | s)
    return maxl


struct Codes(Movable):
    """The prefix codes of every meta group: five per group (green + lengths + cache, red, blue, alpha, distance)."""

    var tab: List[UInt32]
    var off: List[Int]
    var nb: List[Int]

    def __init__(out self):
        self.tab = List[UInt32]()
        self.off = List[Int]()
        self.nb = List[Int]()

    def sym(self, mut br: Bits, c: Int) -> Int:
        br.fill()
        var e = self.tab[self.off[c] + Int(br.val & UInt64((1 << self.nb[c]) - 1))]
        var l = Int(e >> 16)
        br.val >>= UInt64(l)
        br.nbits -= l
        return Int(e & 0xFFFF)


def read_code(mut br: Bits, size: Int, mut tab: List[UInt32]) raises -> Int:
    """Read one prefix code of `size` symbols from the stream, append its table, return the index width."""
    var lens = List[Int](length=size, fill=0)
    if br.bits(1) == 1:  # simple: one or two symbols
        var n = br.bits(1) + 1
        var s = br.bits(8 if br.bits(1) == 1 else 1)
        if s >= size:
            raise Error("webp: prefix symbol out of range")
        lens[s] = 1
        if n == 2:
            s = br.bits(8)
            if s >= size:
                raise Error("webp: prefix symbol out of range")
            lens[s] = 1
    else:
        var order: List[Int] = [17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        var cl = List[Int](length=19, fill=0)
        for i in range(br.bits(4) + 4):
            cl[order[i]] = br.bits(3)
        var ct = List[UInt32]()
        var cb = build_table(cl, ct)
        var maxs = size
        if br.bits(1) == 1:
            maxs = 2 + br.bits(2 + 2 * br.bits(3))
            if maxs > size:
                raise Error("webp: prefix length count too large")
        var s, prev = 0, 8
        while s < size and maxs > 0:
            maxs -= 1
            br.fill()
            var e = ct[Int(br.val & UInt64((1 << cb) - 1))]
            br.val >>= UInt64(Int(e >> 16))
            br.nbits -= Int(e >> 16)
            var c = Int(e & 0xFFFF)
            if c < 16:
                lens[s] = c
                s += 1
                if c != 0:
                    prev = c
            else:
                var rep = br.bits(2) + 3 if c == 16 else (br.bits(3) + 3 if c == 17 else br.bits(7) + 11)
                if s + rep > size:
                    raise Error("webp: prefix length run past the alphabet")
                for _ in range(rep):
                    lens[s] = prev if c == 16 else 0
                    s += 1
    if br.over():
        raise Error("webp: truncated")
    return build_table(lens, tab)


def plane_dist(code: Int, xs: Int) -> Int:
    """Distance code -> pixel distance: codes above 120 are plain, the rest are 2-D neighbourhood offsets."""
    if code > 120:
        return code - 120
    var t: List[Int] = [
        0x18, 0x07, 0x17, 0x19, 0x28, 0x06, 0x27, 0x29, 0x16, 0x1A, 0x26, 0x2A, 0x38, 0x05, 0x37, 0x39, 0x15, 0x1B, 0x36, 0x3A,
        0x25, 0x2B, 0x48, 0x04, 0x47, 0x49, 0x14, 0x1C, 0x35, 0x3B, 0x46, 0x4A, 0x24, 0x2C, 0x58, 0x45, 0x4B, 0x34, 0x3C, 0x03,
        0x57, 0x59, 0x13, 0x1D, 0x56, 0x5A, 0x23, 0x2D, 0x44, 0x4C, 0x55, 0x5B, 0x33, 0x3D, 0x68, 0x02, 0x67, 0x69, 0x12, 0x1E,
        0x66, 0x6A, 0x22, 0x2E, 0x54, 0x5C, 0x43, 0x4D, 0x65, 0x6B, 0x32, 0x3E, 0x78, 0x01, 0x77, 0x79, 0x53, 0x5D, 0x11, 0x1F,
        0x64, 0x6C, 0x42, 0x4E, 0x76, 0x7A, 0x21, 0x2F, 0x75, 0x7B, 0x31, 0x3F, 0x63, 0x6D, 0x52, 0x5E, 0x00, 0x74, 0x7C, 0x41,
        0x4F, 0x10, 0x20, 0x62, 0x6E, 0x30, 0x73, 0x7D, 0x51, 0x5F, 0x40, 0x72, 0x7E, 0x61, 0x6F, 0x50, 0x71, 0x7F, 0x60, 0x70,
    ]
    var v = t[code - 1]
    return max((v >> 4) * xs + 8 - (v & 15), 1)


def lz_value(mut br: Bits, p: Int) -> Int:
    """Length or distance from its prefix symbol plus extra bits."""
    if p < 4:
        return p + 1
    var extra = (p - 2) >> 1
    return ((2 + (p & 1)) << extra) + br.bits(extra) + 1


def cache_key(v: UInt32, bits: Int) -> Int:
    return Int((v * UInt32(0x1E35A7BD)) >> UInt32(32 - bits))


def decode_image(mut br: Bits, w: Int, h: Int, main: Bool) raises -> List[UInt32]:
    """One entropy-coded ARGB image of w x h (the main image, or a transform / meta-group sub-image)."""
    var cbits = 0
    if br.bits(1) == 1:
        cbits = br.bits(4)
        if cbits < 1 or cbits > 11:
            raise Error("webp: color cache bits " + String(cbits))
    var gbits, gw, ngroups = 0, 0, 1
    var egrp = List[UInt32]()
    if main and br.bits(1) == 1:
        gbits = br.bits(3) + 2
        gw = (w + (1 << gbits) - 1) >> gbits
        egrp = decode_image(br, gw, (h + (1 << gbits) - 1) >> gbits, False)
        for i in range(len(egrp)):
            egrp[i] = (egrp[i] >> 8) & 0xFFFF
            ngroups = max(ngroups, Int(egrp[i]) + 1)
    var used = List[Int](length=ngroups, fill=-1)  # group -> slot among the groups the image uses, in stream order
    if ngroups == 1:
        used[0] = 0
    else:
        for i in range(len(egrp)):
            used[Int(egrp[i])] = 0
        var nused = 0
        for g in range(ngroups):
            if used[g] == 0:
                used[g] = nused
                nused += 1
    var cs = Codes()
    var sizes: List[Int] = [256 + 24 + (1 << cbits if cbits > 0 else 0), 256, 256, 256, 40]
    for g in range(ngroups):
        var start = len(cs.tab)
        for k in range(5):
            var o = len(cs.tab)
            var nb = read_code(br, sizes[k], cs.tab)
            if used[g] >= 0:
                cs.off.append(o)
                cs.nb.append(nb)
        if used[g] < 0:
            cs.tab.resize(start, 0)
    var total = w * h
    var px = List[UInt32](length=total, fill=0)
    var cache = List[UInt32](length=(1 << cbits) if cbits > 0 else 0, fill=0)
    var pos, x, y, grp = 0, 0, 0, 0
    while pos < total:
        if br.over():
            raise Error("webp: truncated")
        if gbits > 0:
            grp = used[Int(egrp[(y >> gbits) * gw + (x >> gbits)])]
        var s = cs.sym(br, grp * 5)
        if s < 256:
            var r = cs.sym(br, grp * 5 + 1)
            var b = cs.sym(br, grp * 5 + 2)
            var a = cs.sym(br, grp * 5 + 3)
            var v = UInt32(a) << 24 | UInt32(r) << 16 | UInt32(s) << 8 | UInt32(b)
            px[pos] = v
            if cbits > 0:
                cache[cache_key(v, cbits)] = v
            pos += 1
        elif s < 280:
            var n = lz_value(br, s - 256)
            var dist = plane_dist(lz_value(br, cs.sym(br, grp * 5 + 4)), w)
            if dist > pos or n > total - pos:
                raise Error("webp: backward reference out of range")
            for _ in range(n):
                var v = px[pos - dist]
                px[pos] = v
                if cbits > 0:
                    cache[cache_key(v, cbits)] = v
                pos += 1
        else:
            if s - 280 >= len(cache):
                raise Error("webp: color cache index out of range")
            var v = cache[s - 280]
            px[pos] = v
            cache[cache_key(v, cbits)] = v
            pos += 1
        x = pos % w
        y = pos // w
    if br.over():
        raise Error("webp: truncated")
    return px^


def add4(a: UInt32, b: UInt32) -> UInt32:
    """Per-channel a + b mod 256."""
    return ((a & 0xFF00FF00) + (b & 0xFF00FF00)) & 0xFF00FF00 | ((a & 0x00FF00FF) + (b & 0x00FF00FF)) & 0x00FF00FF


def avg2(a: UInt32, b: UInt32) -> UInt32:
    return (((a ^ b) & 0xFEFEFEFE) >> 1) + (a & b)


def chan(v: UInt32, i: Int) -> Int:
    return Int((v >> UInt32(8 * i)) & 255)


def s8(v: Int) -> Int:
    return v - 256 if v >= 128 else v


def clamp8(v: Int) -> UInt32:
    return UInt32(min(max(v, 0), 255))


def predict(mode: Int, l: UInt32, t: UInt32, tr: UInt32, tl: UInt32) -> UInt32:
    if mode == 1:
        return l
    if mode == 2:
        return t
    if mode == 3:
        return tr
    if mode == 4:
        return tl
    if mode == 5:
        return avg2(avg2(l, tr), t)
    if mode == 6:
        return avg2(l, tl)
    if mode == 7:
        return avg2(l, t)
    if mode == 8:
        return avg2(tl, t)
    if mode == 9:
        return avg2(t, tr)
    if mode == 10:
        return avg2(avg2(l, tl), avg2(t, tr))
    if mode == 11:
        var pl, pt = 0, 0
        for i in range(4):
            var p = chan(l, i) + chan(t, i) - chan(tl, i)
            pl += abs(p - chan(l, i))
            pt += abs(p - chan(t, i))
        return l if pl < pt else t
    if mode == 12:
        var v = UInt32(0)
        for i in range(4):
            v |= clamp8(chan(l, i) + chan(t, i) - chan(tl, i)) << UInt32(8 * i)
        return v
    if mode == 13:
        var a = avg2(l, t)
        var v = UInt32(0)
        for i in range(4):
            var d = chan(a, i) - chan(tl, i)
            v |= clamp8(chan(a, i) + (d // 2 if d >= 0 else -((-d) // 2))) << UInt32(8 * i)
        return v
    return 0xFF000000


struct Xform(Movable):
    var kind: Int  # 0 predictor, 1 cross color, 2 subtract green, 3 color indexing
    var bits: Int
    var xs: Int  # image width this transform sees
    var data: List[UInt32]

    def __init__(out self, kind: Int, bits: Int, xs: Int, var data: List[UInt32]):
        self.kind = kind
        self.bits = bits
        self.xs = xs
        self.data = data^


def inverse(x: Xform, mut px: List[UInt32], w: Int, h: Int) raises -> List[UInt32]:
    """Undo one transform on the w x h picture (w is x.xs, or the final width for color indexing's output)."""
    var out = List[UInt32]()
    if x.kind == 0:
        var bw = (w + (1 << x.bits) - 1) >> x.bits
        for y in range(h):
            for c in range(w):
                var i = y * w + c
                var pred = UInt32(0xFF000000)
                if y == 0:
                    if c > 0:
                        pred = px[i - 1]
                elif c == 0:
                    pred = px[i - w]
                else:
                    var mode = Int(x.data[(y >> x.bits) * bw + (c >> x.bits)] >> 8) & 15
                    pred = predict(mode, px[i - 1], px[i - w], px[i - w + 1], px[i - w - 1])
                px[i] = add4(px[i], pred)
    elif x.kind == 1:
        var bw = (w + (1 << x.bits) - 1) >> x.bits
        for y in range(h):
            for c in range(w):
                var i = y * w + c
                var t = x.data[(y >> x.bits) * bw + (c >> x.bits)]
                var g2r, g2b, r2b = s8(Int(t) & 255), s8(Int(t >> 8) & 255), s8(Int(t >> 16) & 255)
                var v = px[i]
                var g = s8(chan(v, 1))
                var r = (chan(v, 2) + ((g2r * g) >> 5)) & 255
                var b = (chan(v, 0) + ((g2b * g) >> 5) + ((r2b * s8(r)) >> 5)) & 255
                px[i] = (v & 0xFF00FF00) | UInt32(r) << 16 | UInt32(b)
    elif x.kind == 2:
        for i in range(len(px)):
            var v = px[i]
            var g = (v >> 8) & 255
            px[i] = (v & 0xFF00FF00) | ((((v >> 16) & 255) + g) & 255) << 16 | ((v & 255) + g) & 255
    else:
        out = List[UInt32](length=w * h, fill=0)
        var per = 1 << x.bits  # indices per packed pixel
        var bpp = 8 >> x.bits
        var xs = (w + per - 1) >> x.bits
        for y in range(h):
            for c in range(w):
                var packed = Int(px[y * xs + (c >> x.bits)] >> 8) & 255
                var idx = (packed >> ((c & (per - 1)) * bpp)) & ((1 << bpp) - 1)
                out[y * w + c] = x.data[idx] if idx < len(x.data) else 0
    return out^


def decode_stream(mut br: Bits, w: Int, h: Int) raises -> List[UInt32]:
    """Transforms + the main image of a VP8L bitstream (a VP8L chunk after its header, or an ALPH payload): w x h ARGB."""
    var xforms = List[Xform]()
    var seen = 0
    var xs = w
    while br.bits(1) == 1:
        var k = br.bits(2)
        if seen & (1 << k) != 0:
            raise Error("webp: a VP8L transform appears twice")
        seen |= 1 << k
        if k == 0 or k == 1:
            var b = br.bits(3) + 2
            var bw = (xs + (1 << b) - 1) >> b
            xforms.append(Xform(k, b, xs, decode_image(br, bw, (h + (1 << b) - 1) >> b, False)))
        elif k == 2:
            xforms.append(Xform(k, 0, xs, List[UInt32]()))
        else:
            var n = br.bits(8) + 1
            var pal = decode_image(br, n, 1, False)
            for i in range(1, n):
                pal[i] = add4(pal[i], pal[i - 1])
            var b = 3 if n <= 2 else (2 if n <= 4 else (1 if n <= 16 else 0))
            xforms.append(Xform(k, b, xs, pal^))
            xs = (xs + (1 << b) - 1) >> b
    var px = decode_image(br, xs, h, True)
    for i in range(len(xforms) - 1, -1, -1):
        var wd = xforms[i].xs
        var res = inverse(xforms[i], px, wd, h)
        if xforms[i].kind == 3:
            px = res^
    return px^


def vp8l_decode(d: Span[UInt8, _]) raises -> Tuple[Int, Int, List[UInt32]]:
    """Decode a VP8L chunk payload."""
    if len(d) < 5 or d[0] != 0x2F:
        raise Error("webp: not a VP8L stream")
    var br = Bits(d)
    _ = br.bits(8)
    var w = br.bits(14) + 1
    var h = br.bits(14) + 1
    var alpha = br.bits(1)
    if br.bits(3) != 0:
        raise Error("webp: VP8L version is not 0")
    size_ok(w, h)
    var px = decode_stream(br, w, h)
    for i in range(len(px)):  # ARGB -> r | g << 8 | b << 16 | a << 24
        var v = px[i]
        px[i] = ((v >> 16) & 255) | (v & 0xFF00) | (v & 255) << 16 | (v & 0xFF000000 if alpha == 1 else 0xFF000000)
    return (w, h, px^)


# ---------------------------------------------------------------------------------------------------------------------
# VP8 (lossy key frame), RFC 6386. The frame is reconstructed whole (intra prediction needs unfiltered neighbours), then
# loop filtered in macroblock order, then converted to RGB with libwebp's fancy chroma upsampling.
# ---------------------------------------------------------------------------------------------------------------------

comptime BPS = 32  # row stride of the one-macroblock work buffer
comptime Y_OFF = BPS + 8
comptime U_OFF = Y_OFF + 16 * BPS + BPS
comptime V_OFF = U_OFF + 16


struct BoolDec(Movable):
    """The VP8 boolean entropy decoder (RFC 6386 section 7), loaded a byte at a time. `eof` turns true once a read needs a byte
    past the end."""

    var d: List[UInt8]
    var pos: Int
    var value: Int
    var range: Int  # range - 1
    var bits: Int
    var eof: Bool

    def __init__(out self, d: Span[UInt8, _]):
        self.d = List[UInt8](d)
        self.pos = 0
        self.value = 0
        self.range = 254
        self.bits = -8
        self.eof = False
        self.load()

    def load(mut self):
        if self.pos < len(self.d):
            self.bits += 8
            self.value = (self.value << 8) | Int(self.d[self.pos])
            self.pos += 1
        elif not self.eof:
            self.value <<= 8
            self.bits += 8
            self.eof = True
        else:
            self.bits = 0

    def bit(mut self, prob: Int) -> Int:
        var r = self.range
        if self.bits < 0:
            self.load()
        var split = (r * prob) >> 8
        var b = 0
        if (self.value >> self.bits) > split:
            b = 1
            r -= split
            self.value -= (split + 1) << self.bits
        else:
            r = split + 1
        var shift = 0
        while (r << shift) < 128:
            shift += 1
        self.range = (r << shift) - 1
        self.bits -= shift
        return b

    def flag(mut self) -> Bool:
        return self.bit(128) == 1

    def uint(mut self, n: Int) -> Int:
        var v = 0
        for _ in range(n):
            v = (v << 1) | self.bit(128)
        return v

    def sint(mut self, n: Int) -> Int:
        var v = self.uint(n)
        return -v if self.flag() else v


def clip8(v: Int) -> Int:
    return min(max(v, 0), 255)


def av2(a: Int, b: Int) -> Int:
    return (a + b + 1) >> 1


def av3(a: Int, b: Int, c: Int) -> Int:
    return (a + 2 * b + c + 2) >> 2


struct Frame(Movable):
    var w: Int
    var h: Int
    var ys: Int  # luma stride, chroma stride is ys // 2
    var y: List[UInt8]
    var u: List[UInt8]
    var v: List[UInt8]

    def __init__(out self, w: Int, h: Int):
        self.w = w
        self.h = h
        self.ys = ((w + 15) >> 4) * 16
        var rows = ((h + 15) >> 4) * 16
        self.y = List[UInt8](length=self.ys * rows, fill=0)
        self.u = List[UInt8](length=self.ys * rows // 4, fill=0)
        self.v = List[UInt8](length=self.ys * rows // 4, fill=0)


def put_all(mut b: List[UInt8], at: Int, size: Int, v: Int):
    for j in range(size):
        for i in range(size):
            b[at + j * BPS + i] = UInt8(v)


def pred_block(mode: Int, mut b: List[UInt8], at: Int, size: Int, mb_x: Int, mb_y: Int):
    """16x16 luma (size 16) or 8x8 chroma (size 8) prediction from the border of the work buffer. Modes: 0 DC, 1 TM, 2 V, 3 H."""
    if mode == 0:
        var sum = 0
        var shift = 3 if size == 8 else 4
        if mb_x > 0 and mb_y > 0:
            for i in range(size):
                sum += Int(b[at - BPS + i]) + Int(b[at + i * BPS - 1])
            put_all(b, at, size, (sum + size) >> (shift + 1))
        elif mb_x > 0:
            for i in range(size):
                sum += Int(b[at + i * BPS - 1])
            put_all(b, at, size, (sum + size // 2) >> shift)
        elif mb_y > 0:
            for i in range(size):
                sum += Int(b[at - BPS + i])
            put_all(b, at, size, (sum + size // 2) >> shift)
        else:
            put_all(b, at, size, 128)
    elif mode == 1:
        for j in range(size):
            for i in range(size):
                b[at + j * BPS + i] = UInt8(clip8(Int(b[at + j * BPS - 1]) + Int(b[at - BPS + i]) - Int(b[at - BPS - 1])))
    elif mode == 2:
        for j in range(size):
            for i in range(size):
                b[at + j * BPS + i] = b[at - BPS + i]
    else:
        for j in range(size):
            for i in range(size):
                b[at + j * BPS + i] = b[at + j * BPS - 1]


def pred4(mode: Int, mut o: List[Int], mut b: List[UInt8], at: Int):
    """One 4x4 luma sub-block. Modes: 0 DC, 1 TM, 2 VE, 3 HE, 4 RD, 5 VR, 6 LD, 7 VL, 8 HD, 9 HU."""
    var x = Int(b[at - BPS - 1])
    var a = Int(b[at - BPS])
    var bb = Int(b[at - BPS + 1])
    var c = Int(b[at - BPS + 2])
    var d = Int(b[at - BPS + 3])
    var e = Int(b[at - BPS + 4])
    var f = Int(b[at - BPS + 5])
    var g = Int(b[at - BPS + 6])
    var h = Int(b[at - BPS + 7])
    var i = Int(b[at - 1])
    var j = Int(b[at - 1 + BPS])
    var k = Int(b[at - 1 + 2 * BPS])
    var l = Int(b[at - 1 + 3 * BPS])
    if mode == 0:
        var dc = (a + bb + c + d + i + j + k + l + 4) >> 3
        for n in range(16):
            o[n] = dc
    elif mode == 1:
        for yy in range(4):
            var lf = i if yy == 0 else (j if yy == 1 else (k if yy == 2 else l))
            o[yy * 4] = clip8(lf + a - x)
            o[yy * 4 + 1] = clip8(lf + bb - x)
            o[yy * 4 + 2] = clip8(lf + c - x)
            o[yy * 4 + 3] = clip8(lf + d - x)
    elif mode == 2:
        for n in range(4):
            o[n] = av3(x, a, bb) if n == 0 else (av3(a, bb, c) if n == 1 else (av3(bb, c, d) if n == 2 else av3(c, d, e)))
            o[4 + n] = o[n]
            o[8 + n] = o[n]
            o[12 + n] = o[n]
    elif mode == 3:
        for n in range(4):
            var hv = av3(x, i, j) if n == 0 else (av3(i, j, k) if n == 1 else (av3(j, k, l) if n == 2 else av3(k, l, l)))
            for m in range(4):
                o[4 * n + m] = hv
    elif mode == 4:  # RD
        o[12] = av3(j, k, l)
        o[13] = av3(i, j, k)
        o[8] = o[13]
        o[14] = av3(x, i, j)
        o[9] = o[14]
        o[4] = o[14]
        o[15] = av3(a, x, i)
        o[10] = o[15]
        o[5] = o[15]
        o[0] = o[15]
        o[11] = av3(bb, a, x)
        o[6] = o[11]
        o[1] = o[11]
        o[7] = av3(c, bb, a)
        o[2] = o[7]
        o[3] = av3(d, c, bb)
    elif mode == 5:  # VR
        o[0] = av2(x, a)
        o[9] = o[0]
        o[1] = av2(a, bb)
        o[10] = o[1]
        o[2] = av2(bb, c)
        o[11] = o[2]
        o[3] = av2(c, d)
        o[12] = av3(k, j, i)
        o[8] = av3(j, i, x)
        o[4] = av3(i, x, a)
        o[13] = o[4]
        o[5] = av3(x, a, bb)
        o[14] = o[5]
        o[6] = av3(a, bb, c)
        o[15] = o[6]
        o[7] = av3(bb, c, d)
    elif mode == 6:  # LD
        o[0] = av3(a, bb, c)
        o[1] = av3(bb, c, d)
        o[4] = o[1]
        o[2] = av3(c, d, e)
        o[5] = o[2]
        o[8] = o[2]
        o[3] = av3(d, e, f)
        o[6] = o[3]
        o[9] = o[3]
        o[12] = o[3]
        o[7] = av3(e, f, g)
        o[10] = o[7]
        o[13] = o[7]
        o[11] = av3(f, g, h)
        o[14] = o[11]
        o[15] = av3(g, h, h)
    elif mode == 7:  # VL
        o[0] = av2(a, bb)
        o[1] = av2(bb, c)
        o[8] = o[1]
        o[2] = av2(c, d)
        o[9] = o[2]
        o[3] = av2(d, e)
        o[10] = o[3]
        o[4] = av3(a, bb, c)
        o[5] = av3(bb, c, d)
        o[12] = o[5]
        o[6] = av3(c, d, e)
        o[13] = o[6]
        o[7] = av3(d, e, f)
        o[14] = o[7]
        o[11] = av3(e, f, g)
        o[15] = av3(f, g, h)
    elif mode == 8:  # HD
        o[0] = av2(i, x)
        o[6] = o[0]
        o[4] = av2(j, i)
        o[10] = o[4]
        o[8] = av2(k, j)
        o[14] = o[8]
        o[12] = av2(l, k)
        o[3] = av3(a, bb, c)
        o[2] = av3(x, a, bb)
        o[1] = av3(i, x, a)
        o[7] = o[1]
        o[5] = av3(j, i, x)
        o[11] = o[5]
        o[9] = av3(k, j, i)
        o[15] = o[9]
        o[13] = av3(l, k, j)
    else:  # HU
        o[0] = av2(i, j)
        o[2] = av2(j, k)
        o[4] = o[2]
        o[6] = av2(k, l)
        o[8] = o[6]
        o[1] = av3(i, j, k)
        o[3] = av3(j, k, l)
        o[5] = o[3]
        o[7] = av3(k, l, l)
        o[9] = o[7]
        o[10] = l
        o[11] = l
        o[12] = l
        o[13] = l
        o[14] = l
        o[15] = l
    for yy in range(4):
        for xx in range(4):
            b[at + yy * BPS + xx] = UInt8(o[yy * 4 + xx])


def mul1(a: Int) -> Int:
    return ((a * 20091) >> 16) + a


def mul2(a: Int) -> Int:
    return (a * 35468) >> 16


def idct_add(c: List[Int], base: Int, mut t: List[Int], mut b: List[UInt8], at: Int):
    """Inverse DCT of the 16 coefficients at c[base:], added to the 4x4 block of the work buffer (t: 16 ints of scratch)."""
    var any = 0
    for i in range(16):
        any |= c[base + i]
    if any == 0:
        return
    for i in range(4):
        var a = c[base + i] + c[base + 8 + i]
        var bb = c[base + i] - c[base + 8 + i]
        var cc = mul2(c[base + 4 + i]) - mul1(c[base + 12 + i])
        var d = mul1(c[base + 4 + i]) + mul2(c[base + 12 + i])
        t[4 * i] = a + d
        t[4 * i + 1] = bb + cc
        t[4 * i + 2] = bb - cc
        t[4 * i + 3] = a - d
    for i in range(4):
        var dc = t[i] + 4
        var a = dc + t[8 + i]
        var bb = dc - t[8 + i]
        var cc = mul2(t[4 + i]) - mul1(t[12 + i])
        var d = mul1(t[4 + i]) + mul2(t[12 + i])
        var o = at + i * BPS
        b[o] = UInt8(clip8(Int(b[o]) + ((a + d) >> 3)))
        b[o + 1] = UInt8(clip8(Int(b[o + 1]) + ((bb + cc) >> 3)))
        b[o + 2] = UInt8(clip8(Int(b[o + 2]) + ((bb - cc) >> 3)))
        b[o + 3] = UInt8(clip8(Int(b[o + 3]) + ((a - d) >> 3)))


def wht(dc: List[Int], mut o: List[Int]):
    """Inverse Walsh-Hadamard transform of the Y2 block: the DC of each of the 16 luma blocks (out[16 * blk])."""
    var t = List[Int](length=16, fill=0)
    for i in range(4):
        var a0 = dc[i] + dc[12 + i]
        var a1 = dc[4 + i] + dc[8 + i]
        var a2 = dc[4 + i] - dc[8 + i]
        var a3 = dc[i] - dc[12 + i]
        t[i] = a0 + a1
        t[8 + i] = a0 - a1
        t[4 + i] = a3 + a2
        t[12 + i] = a3 - a2
    for i in range(4):
        var d = t[4 * i] + 3
        var a0 = d + t[4 * i + 3]
        var a1 = t[4 * i + 1] + t[4 * i + 2]
        var a2 = t[4 * i + 1] - t[4 * i + 2]
        var a3 = d - t[4 * i + 3]
        o[(4 * i) * 16] = (a0 + a1) >> 3
        o[(4 * i + 1) * 16] = (a3 + a2) >> 3
        o[(4 * i + 2) * 16] = (a0 - a1) >> 3
        o[(4 * i + 3) * 16] = (a3 - a2) >> 3


def coeff_consts() -> List[Int]:
    """Band of each position (17, the last a sentinel), zigzag order (16), then the extra-bit probabilities of categories 3 to 6."""
    return [
        0, 1, 2, 3, 6, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 0,
        0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15,
        173, 148, 140, 0, 176, 155, 140, 135, 0, 180, 157, 141, 134, 130, 0, 254, 254, 243, 230, 196, 177, 153, 140, 133, 130, 129, 0,
    ]


def coeffs(mut br: BoolDec, pr: List[Int], ct: List[Int], t: Int, ctx: Int, dq0: Int, dq1: Int, first: Int, mut co: List[Int], base: Int) -> Int:
    """Tokens of one 4x4 block (RFC 6386 section 13), dequantised into co[base:] in raster order. Returns the position of the
    last non-zero coefficient plus one (`first` if none)."""
    var n = first
    var p = ((t * 8 + ct[n]) * 3 + ctx) * 11
    while n < 16:
        if br.bit(pr[p]) == 0:
            return n
        while br.bit(pr[p + 1]) == 0:
            n += 1
            p = ((t * 8 + ct[n]) * 3) * 11
            if n == 16:
                return 16
        var nb = (t * 8 + ct[n + 1]) * 3 * 11
        var v = 1
        if br.bit(pr[p + 2]) == 0:
            p = nb + 11
        else:
            if br.bit(pr[p + 3]) == 0:
                if br.bit(pr[p + 4]) == 0:
                    v = 2
                else:
                    v = 3 + br.bit(pr[p + 5])
            elif br.bit(pr[p + 6]) == 0:
                if br.bit(pr[p + 7]) == 0:
                    v = 5 + br.bit(159)
                else:
                    v = 7 + 2 * br.bit(165)
                    v += br.bit(145)
            else:
                var b1 = br.bit(pr[p + 8])
                var b0 = br.bit(pr[p + 9 + b1])
                var cat = 2 * b1 + b0
                var q = 33 + (0 if cat == 0 else (4 if cat == 1 else (9 if cat == 2 else 15)))
                v = 0
                while ct[q] != 0:
                    v += v + br.bit(ct[q])
                    q += 1
                v += 3 + (8 << cat)
            p = nb + 22
        var sv = -v if br.flag() else v
        co[base + ct[17 + n]] = sv * (dq1 if n > 0 else dq0)
        n += 1
    return 16


struct Quant(Movable):
    var m: List[Int]  # per segment: y1 dc, y1 ac, y2 dc, y2 ac, uv dc, uv ac

    def __init__(out self):
        self.m = List[Int](length=24, fill=0)


def filter_px(mut b: List[UInt8], p: Int, step: Int, kind: Int):
    """kind 2: the simple 2-tap edge filter, 4: inner-edge 4-tap, 6: macroblock-edge 6-tap (libwebp DoFilter2/4/6)."""
    var p2 = Int(b[p - 3 * step]) if kind == 6 else 0
    var p1 = Int(b[p - 2 * step])
    var p0 = Int(b[p - step])
    var q0 = Int(b[p])
    var q1 = Int(b[p + step])
    var q2 = Int(b[p + 2 * step]) if kind == 6 else 0
    if kind == 2:
        var a = 3 * (q0 - p0) + min(max(p1 - q1, -128), 127)
        var a1 = min(max((a + 4) >> 3, -16), 15)
        var a2 = min(max((a + 3) >> 3, -16), 15)
        b[p - step] = UInt8(clip8(p0 + a2))
        b[p] = UInt8(clip8(q0 - a1))
    elif kind == 4:
        var a = 3 * (q0 - p0)
        var a1 = min(max((a + 4) >> 3, -16), 15)
        var a2 = min(max((a + 3) >> 3, -16), 15)
        var a3 = (a1 + 1) >> 1
        b[p - 2 * step] = UInt8(clip8(p1 + a3))
        b[p - step] = UInt8(clip8(p0 + a2))
        b[p] = UInt8(clip8(q0 - a1))
        b[p + step] = UInt8(clip8(q1 - a3))
    else:
        var a = min(max(3 * (q0 - p0) + min(max(p1 - q1, -128), 127), -128), 127)
        var a1 = (27 * a + 63) >> 7
        var a2 = (18 * a + 63) >> 7
        var a3 = (9 * a + 63) >> 7
        b[p - 3 * step] = UInt8(clip8(p2 + a3))
        b[p - 2 * step] = UInt8(clip8(p1 + a2))
        b[p - step] = UInt8(clip8(p0 + a1))
        b[p] = UInt8(clip8(q0 - a1))
        b[p + step] = UInt8(clip8(q1 - a2))
        b[p + 2 * step] = UInt8(clip8(q2 - a3))


def filter_edge(mut b: List[UInt8], start: Int, step: Int, along: Int, size: Int, thresh: Int, ithresh: Int, hev: Int, kind: Int):
    """One edge of `size` samples. `step` crosses the edge, `along` walks it. kind 2 is the simple filter (no ithresh/hev)."""
    var t2 = 2 * thresh + 1
    for n in range(size):
        var p = start + n * along
        var p1 = Int(b[p - 2 * step])
        var p0 = Int(b[p - step])
        var q0 = Int(b[p])
        var q1 = Int(b[p + step])
        if 4 * abs(p0 - q0) + abs(p1 - q1) > t2:
            continue
        if kind == 2:
            filter_px(b, p, step, 2)
            continue
        var p3 = Int(b[p - 4 * step])
        var p2 = Int(b[p - 3 * step])
        var q2 = Int(b[p + 2 * step])
        var q3 = Int(b[p + 3 * step])
        if abs(p3 - p2) > ithresh or abs(p2 - p1) > ithresh or abs(p1 - p0) > ithresh:
            continue
        if abs(q3 - q2) > ithresh or abs(q2 - q1) > ithresh or abs(q1 - q0) > ithresh:
            continue
        if abs(p1 - p0) > hev or abs(q1 - q0) > hev:
            filter_px(b, p, step, 2)
        else:
            filter_px(b, p, step, kind)


def vp8_frame(d: Span[UInt8, _]) raises -> Frame:
    """Decode a VP8 chunk payload (one key frame) to planar YUV 4:2:0, loop filter applied."""
    if len(d) < 10:
        raise Error("webp: VP8 frame header is truncated")
    var tag = le24(d, 0)
    if tag & 1 != 0:
        raise Error("webp: VP8 frame is not a key frame")
    if (tag >> 1) & 7 > 3 or (tag >> 4) & 1 == 0:
        raise Error("webp: VP8 frame is not displayable")
    var plen = tag >> 5
    if d[3] != 0x9D or d[4] != 0x01 or d[5] != 0x2A:
        raise Error("webp: bad VP8 start code")
    var w = le16(d, 6) & 0x3FFF
    var h = le16(d, 8) & 0x3FFF
    size_ok(w, h)
    if plen > len(d) - 10:
        raise Error("webp: VP8 first partition runs past the data")
    var br = BoolDec(d[10 : 10 + plen])
    _ = br.uint(2)  # color space, clamp type
    # segment header
    var use_seg = br.flag()
    var upd_map = False
    var abs_delta = True
    var seg_q: List[Int] = [0, 0, 0, 0]
    var seg_f: List[Int] = [0, 0, 0, 0]
    var seg_p: List[Int] = [255, 255, 255]
    if use_seg:
        upd_map = br.flag()
        if br.flag():
            abs_delta = br.flag()
            for s in range(4):
                seg_q[s] = br.sint(7) if br.flag() else 0
            for s in range(4):
                seg_f[s] = br.sint(6) if br.flag() else 0
        if upd_map:
            for s in range(3):
                seg_p[s] = br.uint(8) if br.flag() else 255
    if br.eof:
        raise Error("webp: VP8 segment header is truncated")
    # filter header
    var simple = br.flag()
    var level = br.uint(6)
    var sharp = br.uint(3)
    var ref0, mode0 = 0, 0
    if br.flag():
        if br.flag():
            for i in range(4):
                if br.flag():
                    var v = br.sint(6)
                    if i == 0:
                        ref0 = v
            for i in range(4):
                if br.flag():
                    var v = br.sint(6)
                    if i == 0:
                        mode0 = v
    var ftype = 0 if level == 0 else (1 if simple else 2)
    if br.eof:
        raise Error("webp: VP8 filter header is truncated")
    # token partitions
    var nparts = 1 << br.uint(2)
    var rest = d[10 + plen :]
    if len(rest) < 3 * (nparts - 1):
        raise Error("webp: VP8 partition sizes are truncated")
    var parts = List[BoolDec]()
    var at = 3 * (nparts - 1)
    if at >= len(rest) and nparts == 1 or 3 * (nparts - 1) >= len(rest):
        raise Error("webp: VP8 token partitions are empty")
    for p in range(nparts):
        var n = len(rest) - at
        if p < nparts - 1:
            n = min(le24(rest, 3 * p), n)
        parts.append(BoolDec(rest[at : at + n]))
        at += n
        if p == nparts - 2 and at >= len(rest):
            raise Error("webp: VP8 last partition is empty")
    # quantizers
    var q0 = br.uint(7)
    var dq: List[Int] = [0, 0, 0, 0, 0]
    for i in range(5):
        dq[i] = br.sint(4) if br.flag() else 0
    var dct = dc_table()
    var act = ac_table()
    var qm = Quant()
    for s in range(4):
        var q = q0
        if use_seg:
            q = seg_q[s] + (0 if abs_delta else q0)
        qm.m[6 * s] = dct[min(max(q + dq[0], 0), 127)]
        qm.m[6 * s + 1] = act[min(max(q, 0), 127)]
        qm.m[6 * s + 2] = dct[min(max(q + dq[1], 0), 127)] * 2
        qm.m[6 * s + 3] = max((act[min(max(q + dq[2], 0), 127)] * 101581) >> 16, 8)
        qm.m[6 * s + 4] = dct[min(max(q + dq[3], 0), 117)]
        qm.m[6 * s + 5] = act[min(max(q + dq[4], 0), 127)]
    _ = br.flag()  # update_proba
    var pr = coeff_proba()
    var upd = coeff_update()
    for i in range(len(pr)):
        if br.bit(upd[i]) == 1:
            pr[i] = br.uint(8)
    var skip_p = -1
    if br.flag():
        skip_p = br.uint(8)
    if br.eof:
        raise Error("webp: VP8 headers are truncated")
    # filter strengths per segment and block kind: limit, inner level, hev threshold
    var fl = List[Int](length=24, fill=0)
    if ftype > 0:
        for s in range(4):
            var base = level
            if use_seg:
                base = seg_f[s] + (0 if abs_delta else level)
            for i4 in range(2):
                var lv = base + ref0 + (mode0 if i4 == 1 else 0)
                lv = min(max(lv, 0), 63)
                if lv > 0:
                    var il = lv
                    if sharp > 0:
                        il = il >> 2 if sharp > 4 else il >> 1
                        il = min(il, 9 - sharp)
                    il = max(il, 1)
                    fl[6 * s + 3 * i4] = 2 * lv + il
                    fl[6 * s + 3 * i4 + 1] = il
                    fl[6 * s + 3 * i4 + 2] = 2 if lv >= 40 else (1 if lv >= 15 else 0)
    # macroblocks
    var f = Frame(w, h)
    var mbw = (w + 15) >> 4
    var mbh = (h + 15) >> 4
    var bprob = bmode_proba()
    var top_modes = List[Int](length=4 * mbw, fill=0)
    var top_y = List[Int](length=4 * mbw, fill=0)
    var top_u = List[Int](length=2 * mbw, fill=0)
    var top_v = List[Int](length=2 * mbw, fill=0)
    var top_dc = List[Int](length=mbw, fill=0)
    var finfo = List[Int](length=4 * mbw * mbh, fill=0)  # per MB: limit, inner level, hev, inner edges
    var wk = List[UInt8](length=BPS * 17 + BPS * 9, fill=0)
    var cf = List[Int](length=384, fill=0)
    var imodes = List[Int](length=16, fill=0)
    var skipflag = 0
    var ct = coeff_consts()
    var scratch = List[Int](length=16, fill=0)
    var scratch2 = List[Int](length=16, fill=0)
    for mb_y in range(mbh):
        var left_modes: List[Int] = [0, 0, 0, 0]
        var left_y: List[Int] = [0, 0, 0, 0]
        var left_u: List[Int] = [0, 0]
        var left_v: List[Int] = [0, 0]
        var left_dc = 0
        var tb = (mb_y & (nparts - 1))
        for j in range(16):
            wk[Y_OFF + j * BPS - 1] = 129
        for j in range(8):
            wk[U_OFF + j * BPS - 1] = 129
            wk[V_OFF + j * BPS - 1] = 129
        if mb_y > 0:
            wk[Y_OFF - BPS - 1] = 129
            wk[U_OFF - BPS - 1] = 129
            wk[V_OFF - BPS - 1] = 129
        else:
            for i in range(21):
                wk[Y_OFF - BPS - 1 + i] = 127
            for i in range(9):
                wk[U_OFF - BPS - 1 + i] = 127
                wk[V_OFF - BPS - 1 + i] = 127
        for mb_x in range(mbw):
            # --- modes (first partition)
            var seg = 0
            if upd_map:
                seg = br.bit(seg_p[1]) if br.bit(seg_p[0]) == 0 else 2 + br.bit(seg_p[2])
            if skip_p >= 0:
                skipflag = br.bit(skip_p)
            var i4 = br.bit(145) == 0
            var ymode = 0
            if not i4:
                ymode = (1 if br.bit(128) == 1 else 3) if br.bit(156) == 1 else (2 if br.bit(163) == 1 else 0)
                for n in range(4):
                    top_modes[4 * mb_x + n] = ymode
                    left_modes[n] = ymode
            else:
                for y in range(4):
                    var lm = left_modes[y]
                    for x in range(4):
                        var o = (top_modes[4 * mb_x + x] * 10 + lm) * 9
                        var m = 0
                        if br.bit(bprob[o]) == 0:
                            m = 0
                        elif br.bit(bprob[o + 1]) == 0:
                            m = 1
                        elif br.bit(bprob[o + 2]) == 0:
                            m = 2
                        elif br.bit(bprob[o + 3]) == 0:
                            if br.bit(bprob[o + 4]) == 0:
                                m = 3
                            else:
                                m = 4 if br.bit(bprob[o + 5]) == 0 else 5
                        elif br.bit(bprob[o + 6]) == 0:
                            m = 6
                        elif br.bit(bprob[o + 7]) == 0:
                            m = 7
                        else:
                            m = 8 if br.bit(bprob[o + 8]) == 0 else 9
                        top_modes[4 * mb_x + x] = m
                        imodes[4 * y + x] = m
                        lm = m
                    left_modes[y] = lm
            var uvmode = 0
            if br.bit(142) == 1:
                uvmode = 2 if br.bit(114) == 0 else (1 if br.bit(183) == 1 else 3)
            if br.eof and mb_x == mbw - 1:
                raise Error("webp: VP8 first partition is truncated")
            # --- residuals (token partition)
            for n in range(384):
                cf[n] = 0
            var any_nz = False
            ref tk = parts[tb]
            if skipflag == 1 and skip_p >= 0:
                for n in range(4):
                    top_y[4 * mb_x + n] = 0
                    left_y[n] = 0
                for n in range(2):
                    top_u[2 * mb_x + n] = 0
                    top_v[2 * mb_x + n] = 0
                    left_u[n] = 0
                    left_v[n] = 0
                if not i4:
                    top_dc[mb_x] = 0
                    left_dc = 0
            else:
                var m0 = 6 * seg
                var first = 0
                var ytype = 3
                if not i4:
                    var dcv = List[Int](length=16, fill=0)
                    var nz = coeffs(tk, pr, ct, 1, top_dc[mb_x] + left_dc, qm.m[m0 + 2], qm.m[m0 + 3], 0, dcv, 0)
                    top_dc[mb_x] = 1 if nz > 0 else 0
                    left_dc = top_dc[mb_x]
                    wht(dcv, cf)
                    first = 1
                    ytype = 0
                for y in range(4):
                    var l = left_y[y]
                    for x in range(4):
                        var nz = coeffs(tk, pr, ct, ytype, l + top_y[4 * mb_x + x], qm.m[m0], qm.m[m0 + 1], first, cf, (4 * y + x) * 16)
                        l = 1 if nz > first else 0
                        top_y[4 * mb_x + x] = l
                    left_y[y] = l
                for ch in range(2):
                    for y in range(2):
                        var l = left_u[y] if ch == 0 else left_v[y]
                        for x in range(2):
                            var tctx = top_u[2 * mb_x + x] if ch == 0 else top_v[2 * mb_x + x]
                            var nz = coeffs(tk, pr, ct, 2, l + tctx, qm.m[m0 + 4], qm.m[m0 + 5], 0, cf, (16 + 4 * ch + 2 * y + x) * 16)
                            l = 1 if nz > 0 else 0
                            if ch == 0:
                                top_u[2 * mb_x + x] = l
                            else:
                                top_v[2 * mb_x + x] = l
                        if ch == 0:
                            left_u[y] = l
                        else:
                            left_v[y] = l
                for n in range(384):
                    if cf[n] != 0:
                        any_nz = True
                        break
            if tk.eof:
                raise Error("webp: VP8 token partition is truncated")
            # --- loop filter strengths of this macroblock
            var fo = 6 * seg + 3 * (1 if i4 else 0)
            var fi = 4 * (mb_y * mbw + mb_x)
            finfo[fi] = fl[fo]
            finfo[fi + 1] = fl[fo + 1]
            finfo[fi + 2] = fl[fo + 2]
            finfo[fi + 3] = 1 if (i4 or any_nz) else 0
            # --- prediction and reconstruction in the work buffer
            if mb_x > 0:
                for j in range(-1, 16):
                    wk[Y_OFF + j * BPS - 1] = wk[Y_OFF + j * BPS + 15]
                for j in range(-1, 8):
                    wk[U_OFF + j * BPS - 1] = wk[U_OFF + j * BPS + 7]
                    wk[V_OFF + j * BPS - 1] = wk[V_OFF + j * BPS + 7]
            if mb_y > 0:
                for i in range(16):
                    wk[Y_OFF - BPS + i] = f.y[(mb_y * 16 - 1) * f.ys + mb_x * 16 + i]
                for i in range(8):
                    wk[U_OFF - BPS + i] = f.u[(mb_y * 8 - 1) * (f.ys // 2) + mb_x * 8 + i]
                    wk[V_OFF - BPS + i] = f.v[(mb_y * 8 - 1) * (f.ys // 2) + mb_x * 8 + i]
            if i4:
                if mb_y > 0:
                    for i in range(4):
                        if mb_x >= mbw - 1:
                            wk[Y_OFF - BPS + 16 + i] = wk[Y_OFF - BPS + 15]
                        else:
                            wk[Y_OFF - BPS + 16 + i] = f.y[(mb_y * 16 - 1) * f.ys + mb_x * 16 + 16 + i]
                for r in range(1, 4):
                    for i in range(4):
                        wk[Y_OFF + (4 * r - 1) * BPS + 16 + i] = wk[Y_OFF - BPS + 16 + i]
                for n in range(16):
                    var o = Y_OFF + (n >> 2) * 4 * BPS + (n & 3) * 4
                    pred4(imodes[n], scratch2, wk, o)
                    idct_add(cf, n * 16, scratch, wk, o)
            else:
                pred_block(ymode, wk, Y_OFF, 16, mb_x, mb_y)
                for n in range(16):
                    idct_add(cf, n * 16, scratch, wk, Y_OFF + (n >> 2) * 4 * BPS + (n & 3) * 4)
            pred_block(uvmode, wk, U_OFF, 8, mb_x, mb_y)
            pred_block(uvmode, wk, V_OFF, 8, mb_x, mb_y)
            for n in range(4):
                idct_add(cf, (16 + n) * 16, scratch, wk, U_OFF + (n >> 1) * 4 * BPS + (n & 1) * 4)
                idct_add(cf, (20 + n) * 16, scratch, wk, V_OFF + (n >> 1) * 4 * BPS + (n & 1) * 4)
            for j in range(16):
                for i in range(16):
                    f.y[(mb_y * 16 + j) * f.ys + mb_x * 16 + i] = wk[Y_OFF + j * BPS + i]
            for j in range(8):
                for i in range(8):
                    f.u[(mb_y * 8 + j) * (f.ys // 2) + mb_x * 8 + i] = wk[U_OFF + j * BPS + i]
                    f.v[(mb_y * 8 + j) * (f.ys // 2) + mb_x * 8 + i] = wk[V_OFF + j * BPS + i]
    # --- loop filter, macroblock raster order
    if ftype > 0:
        var ys = f.ys
        var cs = ys // 2
        for mb_y in range(mbh):
            for mb_x in range(mbw):
                var fi = 4 * (mb_y * mbw + mb_x)
                var limit = finfo[fi]
                if limit == 0:
                    continue
                var il = finfo[fi + 1]
                var hev = finfo[fi + 2]
                var inner = finfo[fi + 3] == 1
                var yo = mb_y * 16 * ys + mb_x * 16
                var uo = mb_y * 8 * cs + mb_x * 8
                if ftype == 1:
                    if mb_x > 0:
                        filter_edge(f.y, yo, 1, ys, 16, limit + 4, 0, 0, 2)
                    if inner:
                        for k in range(1, 4):
                            filter_edge(f.y, yo + 4 * k, 1, ys, 16, limit, 0, 0, 2)
                    if mb_y > 0:
                        filter_edge(f.y, yo, ys, 1, 16, limit + 4, 0, 0, 2)
                    if inner:
                        for k in range(1, 4):
                            filter_edge(f.y, yo + 4 * k * ys, ys, 1, 16, limit, 0, 0, 2)
                else:
                    if mb_x > 0:
                        filter_edge(f.y, yo, 1, ys, 16, limit + 4, il, hev, 6)
                        filter_edge(f.u, uo, 1, cs, 8, limit + 4, il, hev, 6)
                        filter_edge(f.v, uo, 1, cs, 8, limit + 4, il, hev, 6)
                    if inner:
                        for k in range(1, 4):
                            filter_edge(f.y, yo + 4 * k, 1, ys, 16, limit, il, hev, 4)
                        filter_edge(f.u, uo + 4, 1, cs, 8, limit, il, hev, 4)
                        filter_edge(f.v, uo + 4, 1, cs, 8, limit, il, hev, 4)
                    if mb_y > 0:
                        filter_edge(f.y, yo, ys, 1, 16, limit + 4, il, hev, 6)
                        filter_edge(f.u, uo, cs, 1, 8, limit + 4, il, hev, 6)
                        filter_edge(f.v, uo, cs, 1, 8, limit + 4, il, hev, 6)
                    if inner:
                        for k in range(1, 4):
                            filter_edge(f.y, yo + 4 * k * ys, ys, 1, 16, limit, il, hev, 4)
                        filter_edge(f.u, uo + 4 * cs, cs, 1, 8, limit, il, hev, 4)
                        filter_edge(f.v, uo + 4 * cs, cs, 1, 8, limit, il, hev, 4)
    return f^


def mult_hi(v: Int, c: Int) -> Int:
    return (v * c) >> 8


def clip_yuv(v: Int) -> Int:
    return (v >> 6) if (v & ~(255 << 6 | 63)) == 0 else (0 if v < 0 else 255)


def cpx(f: Frame, ch: Int, i: Int) -> Int:
    return Int(f.u[i]) if ch == 0 else Int(f.v[i])


def yuv_rgba(f: Frame, alpha: List[UInt8], has_alpha: Bool) -> List[UInt32]:
    """RGBA pixels from the planes, chroma upsampled with libwebp's 'fancy' 9-3-3-1 filter (its default)."""
    var w, h = f.w, f.h
    var cs = f.ys // 2
    var lastc = (h + 1) // 2 - 1
    var last = (w - 1) >> 1
    var px = List[UInt32](length=w * h, fill=0)
    for y in range(h):
        var ra, rb = 0, 0
        var top = True
        if y == 0:
            pass
        elif y & 1 == 1:
            ra = (y + 1) // 2 - 1
            rb = min((y + 1) // 2, lastc)
        else:
            ra = y // 2 - 1
            rb = y // 2
            top = False
        for x in range(w):
            var uu, vv = 0, 0
            for ch in range(2):
                var res = 0
                var k = (x + 1) >> 1
                if x == 0 or (x & 1 == 1 and k > last):
                    var kk = 0 if x == 0 else last
                    var tl = cpx(f, ch, ra * cs + kk)
                    var l = cpx(f, ch, rb * cs + kk)
                    res = (3 * tl + l + 2) >> 2 if top else (3 * l + tl + 2) >> 2
                else:
                    var tl = cpx(f, ch, ra * cs + k - 1)
                    var t = cpx(f, ch, ra * cs + k)
                    var l = cpx(f, ch, rb * cs + k - 1)
                    var c = cpx(f, ch, rb * cs + k)
                    var avg = tl + t + l + c + 8
                    var d12 = (avg + 2 * (t + l)) >> 3
                    var d03 = (avg + 2 * (tl + c)) >> 3
                    if x & 1 == 1:
                        res = (d12 + tl) >> 1 if top else (d03 + l) >> 1
                    else:
                        res = (d03 + t) >> 1 if top else (d12 + c) >> 1
                if ch == 0:
                    uu = res
                else:
                    vv = res
            var yv = Int(f.y[y * f.ys + x])
            var r = clip_yuv(mult_hi(yv, 19077) + mult_hi(vv, 26149) - 14234)
            var g = clip_yuv(mult_hi(yv, 19077) - mult_hi(uu, 6419) - mult_hi(vv, 13320) + 8708)
            var b = clip_yuv(mult_hi(yv, 19077) + mult_hi(uu, 33050) - 17685)
            var a = Int(alpha[y * w + x]) if has_alpha else 255
            px[y * w + x] = UInt32(r) | UInt32(g) << 8 | UInt32(b) << 16 | UInt32(a) << 24
    return px^


def alpha_plane(d: List[UInt8], w: Int, h: Int) raises -> List[UInt8]:
    """The w x h alpha plane of an ALPH chunk payload: raw or VP8L-compressed, then un-filtered."""
    if len(d) < 2:
        raise Error("webp: ALPH chunk is too short")
    var method = Int(d[0]) & 3
    var filt = (Int(d[0]) >> 2) & 3
    if method > 1 or (Int(d[0]) >> 4) & 3 > 1 or Int(d[0]) >> 6 != 0:
        raise Error("webp: bad ALPH header")
    var a = List[UInt8](length=w * h, fill=0)
    if method == 0:
        if len(d) - 1 < w * h:
            raise Error("webp: ALPH data is truncated")
        for i in range(w * h):
            a[i] = d[1 + i]
    else:
        var br = Bits(Span(d)[1:])
        var px = decode_stream(br, w, h)
        for i in range(w * h):
            a[i] = UInt8((px[i] >> 8) & 255)
    if filt != 0:
        for y in range(h):
            for x in range(w):
                var i = y * w + x
                var pred = 0
                if y == 0:
                    pred = Int(a[i - 1]) if x > 0 else 0
                elif x == 0:
                    pred = Int(a[i - w])
                elif filt == 1:
                    pred = Int(a[i - 1])
                elif filt == 2:
                    pred = Int(a[i - w])
                else:
                    pred = min(max(Int(a[i - 1]) + Int(a[i - w]) - Int(a[i - w - 1]), 0), 255)
                a[i] = UInt8((Int(a[i]) + pred) & 255)
    return a^


def vp8_decode(d: Span[UInt8, _], alph: List[UInt8], has_alpha: Bool) raises -> Tuple[Int, Int, List[UInt32]]:
    var f = vp8_frame(d)
    var alpha = List[UInt8]()
    if has_alpha:
        alpha = alpha_plane(alph, f.w, f.h)
    return (f.w, f.h, yuv_rgba(f, alpha, has_alpha))


# ---------------------------------------------------------------------------------------------------------------------
# container
# ---------------------------------------------------------------------------------------------------------------------


def webp_decode(d: Span[UInt8, _]) raises -> Tuple[Int, Int, List[UInt32]]:
    """(width, height, pixels) of a still WebP: simple lossless, simple lossy, or extended (VP8X) with ALPH."""
    if len(d) < 20 or le32(d, 0) != 0x46464952 or le32(d, 8) != 0x50424557:  # "RIFF" .... "WEBP"
        raise Error("webp: not a WebP file")
    var end = le32(d, 4) + 8
    if end > len(d):
        raise Error("webp: truncated file")
    var pos = 12
    var cw, ch = 0, 0
    var alph = List[UInt8]()
    var has_alph = False
    while pos + 8 <= end:
        var t = le32(d, pos)
        var n = le32(d, pos + 4)
        if pos + 8 + n > end:
            raise Error("webp: truncated chunk")
        var body = d[pos + 8 : pos + 8 + n]
        if t == 0x4C385056:  # "VP8L"
            return vp8l_decode(body)
        if t == 0x58385056:  # "VP8X"
            if n < 10:
                raise Error("webp: VP8X chunk is too short")
            if body[0] & 2 != 0:
                raise Error("webp: animation is not supported")
            cw, ch = le24(body, 4) + 1, le24(body, 7) + 1
        elif t == 0x48504C41:  # "ALPH"
            if not has_alph:
                alph = List[UInt8](body)
                has_alph = True
        elif t == 0x20385056:  # "VP8 "
            var r = vp8_decode(body, alph, has_alph)
            if cw > 0 and (cw != r[0] or ch != r[1]):
                raise Error("webp: VP8X canvas is " + String(cw) + " x " + String(ch) + " but the frame is " + String(r[0]) + " x " + String(r[1]))
            return r^
        elif t == 0x4D494E41 or t == 0x464D4E41:  # "ANIM", "ANMF"
            raise Error("webp: animation is not supported")
        pos += 8 + n + (n & 1)
    raise Error("webp: no image data")
