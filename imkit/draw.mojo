# VENDORED COPY of amarbaro/mojo-imkit (Apache-2.0, see imkit/LICENSE), tag v0.9.0 (eb91210),
# ~/Projects/mojo/mojo-imkit, file src/imkit/draw.mojo. This repo keeps a real file
# rather than an -I path outside the tree, because a clone must build without anything
# outside it. Sync by hand if the upstream changes; tools/ci-checks.sh compares the two
# against the tag when the upstream is present and says so when they drift.
#
"""Draw list and compose kernel (docs/design.md). `Draw` holds one frame's flat command list and bins it into 16 x 16 tiles
on the host (one compact list, an offset per tile); `compose` gives every UI pixel one thread that walks only its tile's
commands in order: anti-aliased rounded rects and soft shadows by SDF, glyphs from font.mojo's atlas, capsule lines,
triangles, alpha "over". The result is a UW x UH straight-alpha RGBA layer that window.mojo blends over the scene.

Command: CW uint32 = kind, x0, y0, x1, y1, r, rgba, aux (floats stored as bits). rect: r = corner radius, aux = blur
(soft shadow); glyph: aux = atlas x | atlas y << 16; line: (x0, y0)-(x1, y1), r = half width; tri: (x0, y0), (x1, y1),
(r, aux) as the third corner; image: r = offset into the app's source pixels (a float holds it exactly below 2^24),
rgba = alpha in the top byte, aux = source width | height << 16, sampled nearest over the rect.
"""
from std.math import isnan, sqrt
from std.memory import bitcast
from std.sys import get_defined_int
from max.gpu import block_dim, block_idx, thread_idx
from layout import TileTensor, row_major
from .font import AH, AW, Font, decode

comptime UW = get_defined_int["UI_W", 1280]()
comptime UH = get_defined_int["UI_H", 720]()
comptime TILE = 16
comptime TX = UW // TILE
comptime TY = UH // TILE
comptime NBIN = get_defined_int["UI_BINS", 32768]()  # tile entries per frame (a command once per tile it touches); past it whole commands drop, counted
comptime NCMD = 4096
comptime CW = 8
comptime RECT = 0
comptime GLYPH = 1
comptime LINE = 2
comptime TRI = 3
comptime IMAGE = 4
comptime GLASS = 8  # rounded rect: the blurred scene behind it (Layer.blur_from), then the tint c over it
comptime BQ = 4  # the blurred backdrop is 1 / BQ of the layer in each direction
comptime QW = UW // BQ
comptime QH = UH // BQ
comptime BR = 6  # blur taps each side at 1 / BQ resolution (triangle weights, two passes)
comptime CLIP = 7  # state: from here the rect (x0, y0)-(x1, y1) clips later commands (aux 0), or the clip ends (aux 1, same rect)
comptime FULL = SIMD[DType.float32, 4](-1e9, -1e9, 1e9, 1e9)
comptime SRECT = 6  # kind word: kind | border px * 4 << 8 | style id << 16; c = fill (gradient start), aux = border rgba
comptime NSTY = 256  # styles per frame (id 0 is none)
comptime RRECT = 5  # kind word: kind | border px * 4 << 8 | glow px << 16 | gradient i8 << 24; aux = border and glow rgba
comptime BIG = Float32(1e6)  # coordinates clamp to +-BIG: Int() of an infinity or 1e30 is undefined, and no layer is that large
comptime ISRC = get_defined_int["UI_IMG", 65536]()  # source pixels for images (RGBA u32, the app fills them; -DUI_IMG= for more)

comptime TUI = TileTensor[DType.uint32, type_of(row_major[UW * UH]()), MutAnyOrigin]
comptime TCMD = TileTensor[DType.uint32, type_of(row_major[NCMD * CW]()), MutAnyOrigin]
comptime TBIN = TileTensor[DType.uint16, type_of(row_major[NBIN]()), MutAnyOrigin]
comptime TOFF = TileTensor[DType.uint32, type_of(row_major[TX * TY + 1]()), MutAnyOrigin]
comptime TATL = TileTensor[DType.uint8, type_of(row_major[AW * AH]()), MutAnyOrigin]
comptime TQ = TileTensor[DType.uint32, type_of(row_major[QW * QH]()), MutAnyOrigin]
comptime TSTY = TileTensor[DType.uint32, type_of(row_major[NSTY * 8]()), MutAnyOrigin]
comptime TSRC = TileTensor[DType.uint32, type_of(row_major[ISRC]()), MutAnyOrigin]


def rgba(r: Int, g: Int, b: Int, a: Int) -> UInt32:
    return UInt32(r) | UInt32(g) << 8 | UInt32(b) << 16 | UInt32(a) << 24


@always_inline
def sane(v: Float32) -> Float32:
    """v clamped to +-BIG (NaN is the caller's to drop)."""
    return min(max(v, -BIG), BIG)


@always_inline
def bits(x: Float32) -> UInt32:
    return bitcast[DType.uint32](x)


@always_inline
def flt(u: UInt32) -> Float32:
    return bitcast[DType.float32](u)

@always_inline
def unpack(c: UInt32) -> SIMD[DType.float32, 4]:
    return SIMD[DType.float32, 4](Float32(c & 255), Float32((c >> 8) & 255), Float32((c >> 16) & 255), Float32(c >> 24))


@always_inline
def over(acc: SIMD[DType.float32, 4], col: SIMD[DType.float32, 4], cov: Float32) -> SIMD[DType.float32, 4]:
    """`col` (0..255 per lane) with coverage cov over the premultiplied accumulation acc (rgb, a)."""
    var sa = cov * col[3] / 255
    return SIMD[DType.float32, 4](col[0], col[1], col[2], 255) / 255 * sa + acc * (1 - sa)


@always_inline
def clipped(px: Float32, py: Float32, c: SIMD[DType.float32, 4]) -> Bool:
    """Is (px, py) outside the clip rect c = x0, y0, x1, y1 (half open)?"""
    return px < c[0] or px >= c[2] or py < c[1] or py >= c[3]


@always_inline
def rsd(px: Float32, py: Float32, x0: Float32, y0: Float32, x1: Float32, y1: Float32, rr: Float32) -> Float32:
    """Signed distance from (px, py) to the rounded box (negative inside)."""
    var qx = abs(px - (x0 + x1) * 0.5) - (x1 - x0) * 0.5 + rr
    var qy = abs(py - (y0 + y1) * 0.5) - (y1 - y0) * 0.5 + rr
    return sqrt(max(qx, 0) * max(qx, 0) + max(qy, 0) * max(qy, 0)) + min(max(qx, qy), 0) - rr


@no_inline
def rrect(acc: SIMD[DType.float32, 4], px: Float32, py: Float32, x0: Float32, y0: Float32, x1: Float32, y1: Float32,
          rr: Float32, c: UInt32, k: UInt32, bc: UInt32) -> SIMD[DType.float32, 4]:
    """One RRECT pixel: glow (outside, bc), fill with a vertical gradient, border ring (bc), by the signed distance d."""
    var bw = Float32((k >> 8) & 255) * 0.25
    var gl = Float32((k >> 16) & 255)
    var gi = Int(k >> 24)
    var gr = Float32(gi - 256 if gi > 127 else gi) / 127
    var d = rsd(px, py, x0, y0, x1, y1, rr)
    var fill = min(max(0.5 - d, Float32(0)), Float32(1))
    var col = unpack(c)
    var s = (0.5 - min(max((py - y0) / max(y1 - y0, 1e-6), Float32(0)), Float32(1))) * 2 * gr  # + lightens toward white, - darkens
    var tg = Float32(255) if s > 0 else Float32(0)
    col += (SIMD[DType.float32, 4](tg, tg, tg, col[3]) - col) * abs(s)
    var a = acc
    if gl > 0 and d > 0:
        var t = min(max(1 - d / gl, Float32(0)), Float32(1))
        a = over(a, unpack(bc), t * t)
    a = over(a, col, fill)
    if bw > 0:
        a = over(a, unpack(bc), fill * min(max(0.5 + d + bw, Float32(0)), Float32(1)))
    return a


@no_inline
def srect(acc: SIMD[DType.float32, 4], px: Float32, py: Float32, x0: Float32, y0: Float32, x1: Float32, y1: Float32,
          rr: Float32, c: UInt32, k: UInt32, bc: UInt32, st: SIMD[DType.uint32, 8]) -> SIMD[DType.float32, 4]:
    """One SRECT pixel with style words st = mode, c1, gx, gy, shadow dx, dy, radius, rgba: shadow (soft SDF of the rect moved
    by dx, dy), fill (mode 1: linear t = (u - .5) gx + (v - .5) gy + .5, 2: radial t = 2 |(u, v) - .5|; u, v in the rect), border."""
    var d = rsd(px, py, x0, y0, x1, y1, rr)
    var u = (px - x0) / max(x1 - x0, 1e-6)
    var v = (py - y0) / max(y1 - y0, 1e-6)
    var t = Float32(0)
    if st[0] & 3 == 1:
        t = (u - 0.5) * flt(st[2]) + (v - 0.5) * flt(st[3]) + 0.5
    elif st[0] & 3 == 2:
        t = 2 * sqrt((u - 0.5) * (u - 0.5) + (v - 0.5) * (v - 0.5))
    var col = unpack(c)
    col += (unpack(st[1]) - col) * min(max(t, Float32(0)), Float32(1))
    var a = acc
    if st[7] >> 24 > 0:
        a = over(a, unpack(st[7]), min(max(0.5 - rsd(px - flt(st[4]), py - flt(st[5]), x0, y0, x1, y1, rr) / (1 + flt(st[6])), Float32(0)), Float32(1)))
    var fill = min(max(0.5 - d, Float32(0)), Float32(1))
    a = over(a, col, fill)
    var bw = Float32((k >> 8) & 255) * 0.25
    if bw > 0:
        a = over(a, unpack(bc), fill * min(max(0.5 + d + bw, Float32(0)), Float32(1)))
    return a


@always_inline
def pack(v: SIMD[DType.float32, 4]) -> UInt32:
    """0..255 per lane to rgba, rounded, alpha forced opaque."""
    var c = min(max(v + 0.5, SIMD[DType.float32, 4](0)), SIMD[DType.float32, 4](255))
    return UInt32(c[0]) | UInt32(c[1]) << 8 | UInt32(c[2]) << 16 | UInt32(255) << 24


def blur_down(dst: TQ, src: TUI):
    """Box average of each BQ x BQ block of the scene."""
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= QW * QH:
        return
    var s = SIMD[DType.float32, 4](0)
    for dy in range(BQ):
        for dx in range(BQ):
            s += unpack(src[((i // QW) * BQ + dy) * UW + (i % QW) * BQ + dx])
    dst[i] = pack(s / Float32(BQ * BQ))


def blur_axis[dx: Int, dy: Int](dst: TQ, src: TQ):
    """One separable pass over the small image, triangle weights BR + 1 - |k|, edges clamped."""
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= QW * QH:
        return
    var s = SIMD[DType.float32, 4](0)
    var w = Float32(0)
    for k in range(-BR, BR + 1):
        var wk = Float32(BR + 1 - abs(k))
        s += unpack(src[min(max(i // QW + k * dy, 0), QH - 1) * QW + min(max(i % QW + k * dx, 0), QW - 1)]) * wk
        w += wk
    dst[i] = pack(s / w)


@no_inline
def backdrop(blr: TQ, px: Float32, py: Float32) -> SIMD[DType.float32, 4]:
    """Bilinear sample of the blurred scene at layer pixel (px, py)."""
    var x = min(max(px / Float32(BQ) - 0.5, Float32(0)), Float32(QW - 1))
    var y = min(max(py / Float32(BQ) - 0.5, Float32(0)), Float32(QH - 1))
    var ix, iy = Int(x), Int(y)
    var jx, jy = min(ix + 1, QW - 1), min(iy + 1, QH - 1)
    var tx, ty = x - Float32(ix), y - Float32(iy)
    return (unpack(blr[iy * QW + ix]) * (1 - tx) + unpack(blr[iy * QW + jx]) * tx) * (1 - ty) + (unpack(blr[jy * QW + ix]) * (1 - tx) + unpack(blr[jy * QW + jx]) * tx) * ty


@no_inline
def glass(acc: SIMD[DType.float32, 4], px: Float32, py: Float32, x0: Float32, y0: Float32, x1: Float32, y1: Float32,
          rr: Float32, c: UInt32, back: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:
    """One GLASS pixel: the backdrop sample `back` (its alpha 0 before any blur ran), then the tint c over it."""
    var fill = min(max(0.5 - rsd(px, py, x0, y0, x1, y1, rr), Float32(0)), Float32(1))
    return over(over(acc, back, fill), unpack(c), fill)


def compose(img: TUI, cmd: TCMD, bin: TBIN, off: TOFF, atlas: TATL, src: TSRC, sty: TSTY, blr: TQ):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= UW * UH:
        return
    var px = Float32(i % UW) + 0.5
    var py = Float32(i // UW) + 0.5
    var t = (i // UW // TILE) * TX + (i % UW) // TILE
    var acc = SIMD[DType.float32, 4](0)  # premultiplied accumulation
    var clip = FULL
    for j in range(Int(off[t]), Int(off[t + 1])):  # this tile's commands, in draw order
        var o = Int(bin[j]) * CW
        var kind = Int(cmd[o] & 255)
        var x0 = flt(cmd[o + 1])
        var y0 = flt(cmd[o + 2])
        var x1 = flt(cmd[o + 3])
        var y1 = flt(cmd[o + 4])
        var rr = flt(cmd[o + 5])
        var c = cmd[o + 6]
        var cov = Float32(0)
        if kind == CLIP:
            clip = FULL if cmd[o + 7] != 0 else SIMD[DType.float32, 4](x0, y0, x1, y1)
            continue
        if clipped(px, py, clip):
            continue
        if kind == GLASS:
            acc = glass(acc, px, py, x0, y0, x1, y1, rr, c, backdrop(blr, px, py))
            continue
        if kind == RRECT:
            acc = rrect(acc, px, py, x0, y0, x1, y1, rr, c, cmd[o], cmd[o + 7])
            continue
        if kind == SRECT:
            var q = Int(cmd[o] >> 16) * 8
            acc = srect(acc, px, py, x0, y0, x1, y1, rr, c, cmd[o], cmd[o + 7], SIMD[DType.uint32, 8](sty[q], sty[q + 1], sty[q + 2], sty[q + 3], sty[q + 4], sty[q + 5], sty[q + 6], sty[q + 7]))
            continue
        if kind == RECT:
            var soft = flt(cmd[o + 7])
            var d = rsd(px, py, x0, y0, x1, y1, rr)
            cov = min(max(0.5 - d / (1 + soft), Float32(0)), Float32(1))
        elif kind == GLYPH:
            if px >= x0 and px < x1 and py >= y0 and py < y1:
                var ax = Int(cmd[o + 7] & 0xFFFF) + Int(px - x0)
                var ay = Int(cmd[o + 7] >> 16) + Int(py - y0)
                cov = Float32(atlas[ay * AW + ax]) / 255
        elif kind == LINE:
            var bx = x1 - x0
            var by = y1 - y0
            var h = min(max(((px - x0) * bx + (py - y0) * by) / max(bx * bx + by * by, 1e-6), Float32(0)), Float32(1))
            var dx = px - x0 - bx * h
            var dy = py - y0 - by * h
            cov = min(max(0.5 - (sqrt(dx * dx + dy * dy) - rr), Float32(0)), Float32(1))
        elif kind == IMAGE:
            if px >= x0 and px < x1 and py >= y0 and py < y1:
                var sw = Int(cmd[o + 7] & 0xFFFF)
                var sh = Int(cmd[o + 7] >> 16)
                var u = min(Int((px - x0) / (x1 - x0) * Float32(sw)), sw - 1)
                var v = min(Int((py - y0) / (y1 - y0) * Float32(sh)), sh - 1)
                var s = src[Int(rr) + v * sw + u]
                c = (s & 0xFFFFFF) | UInt32(Float32(s >> 24) * Float32(c >> 24) / 255) << 24
                cov = 1
        else:  # TRI: largest signed edge distance, inside negative for either winding
            var x2 = rr
            var y2 = flt(cmd[o + 7])
            var wind = Float32(1) if (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0) > 0 else Float32(-1)  # either winding
            var d = Float32(-1e9)
            for e in range(3):
                var ax = x0 if e == 0 else (x1 if e == 1 else x2)
                var ay = y0 if e == 0 else (y1 if e == 1 else y2)
                var cx = x1 if e == 0 else (x2 if e == 1 else x0)
                var cy = y1 if e == 0 else (y2 if e == 1 else y0)
                var ex = cx - ax
                var ey = cy - ay
                d = max(d, wind * ((px - ax) * ey - (py - ay) * ex) / max(sqrt(ex * ex + ey * ey), 1e-6))
            cov = min(max(0.5 - d, Float32(0)), Float32(1))
        acc = over(acc, unpack(c), cov)
    var inv = 1 / max(acc[3], Float32(1e-6))  # straight alpha out (SDL blends straight alpha)
    img[i] = UInt32(min(acc[0] * inv, 1) * 255) | UInt32(min(acc[1] * inv, 1) * 255) << 8 | UInt32(min(acc[2] * inv, 1) * 255) << 16 | UInt32(acc[3] * 255) << 24


struct Draw(Movable):
    var font: Font
    var cmd: List[UInt32]
    var n: Int
    var bins: List[UInt16]
    var off: List[UInt32]  # tile t's commands are bins[off[t]:off[t + 1]]
    var cur: List[UInt32]  # fill cursor per tile (finish)
    var binned: Int  # n at the last finish(); -1 after begin(): Layer.render refuses an unfinished frame
    var dropped: Int
    var sty: List[UInt32]  # this frame's styles, 8 words each (srect); id 0 is none
    var ns: Int
    var cur_clip: SIMD[DType.float32, 4]  # the clip rect in force while drawing (FULL: none)

    def __init__(out self, var font: Font):
        self.font = font^
        self.cmd = List[UInt32](length=NCMD * CW, fill=0)
        self.bins = List[UInt16](length=NBIN, fill=0)
        self.off = List[UInt32](length=TX * TY + 1, fill=0)
        self.cur = List[UInt32](length=TX * TY, fill=0)
        self.sty = List[UInt32](length=NSTY * 8, fill=0)
        self.binned, self.n, self.dropped, self.ns = 0, 0, 0, 1
        self.cur_clip = FULL

    def begin(mut self):
        self.n, self.ns = 0, 1
        self.cur_clip = FULL
        self.binned = -1

    def push(mut self, kind: Int, x0: Float32, y0: Float32, x1: Float32, y1: Float32, r: Float32, c: UInt32, aux: UInt32 = 0):
        var v = SIMD[DType.float32, 4](x0, y0, x1, y1)
        if self.n == NCMD or isnan(v).reduce_or() or r != r:  # full, or NaN: dropped, counted
            self.dropped += 1
            return
        v = min(max(v, -BIG), BIG)
        var o = self.n * CW
        self.cmd[o], self.cmd[o + 1], self.cmd[o + 2], self.cmd[o + 3] = UInt32(kind), bits(v[0]), bits(v[1]), bits(v[2])
        self.cmd[o + 4], self.cmd[o + 5], self.cmd[o + 6], self.cmd[o + 7] = bits(v[3]), bits(r), c, aux
        self.n += 1

    def rect(mut self, x0: Float32, y0: Float32, x1: Float32, y1: Float32, r: Float32, c: UInt32, blur: Float32 = 0):
        if blur != blur:
            self.dropped += 1
            return
        self.push(RECT, x0, y0, x1, y1, r, c, bits(min(blur, BIG)))

    def rrect(mut self, x0: Float32, y0: Float32, x1: Float32, y1: Float32, r: Float32, c: UInt32, border: Float32 = 0,
              bcol: UInt32 = 0, glow: Float32 = 0, grad: Int = 0):
        """Rounded rect with a border of `border` px (0.25 px steps, up to 63.75) in bcol, an outside glow of `glow` px (up
        to 255) in bcol, and a vertical gradient grad in -127..127 (+: top lighter, bottom darker)."""
        var k = RRECT | min(max(Int(border * 4 + 0.5), 0), 255) << 8 | min(max(Int(glow), 0), 255) << 16 | (min(max(grad, -127), 127) & 255) << 24
        self.push(k, x0, y0, x1, y1, r, c, bcol)

    def style(mut self, c1: UInt32 = 0, mode: Int = 0, gx: Float32 = 0, gy: Float32 = 1, sx: Float32 = 0, sy: Float32 = 0,
              sr: Float32 = 0, sc: UInt32 = 0) -> Int:
        """Add a style for this frame and return its id (0, none, when the table is full): fill gradient from the command's
        color to c1 (mode 1 linear along (gx, gy) in rect units, 2 radial) and a shadow of the shape moved by (sx, sy),
        blurred by sr, in sc."""
        if self.ns == NSTY or gx != gx or gy != gy or sx != sx or sy != sy or sr != sr:
            self.dropped += 1
            return 0
        var o = self.ns * 8
        self.sty[o], self.sty[o + 1], self.sty[o + 2], self.sty[o + 3] = UInt32(mode), c1, bits(gx), bits(gy)
        self.sty[o + 4], self.sty[o + 5], self.sty[o + 6], self.sty[o + 7] = bits(sane(sx)), bits(sane(sy)), bits(sane(sr)), sc
        self.ns += 1
        return self.ns - 1

    def srect(mut self, x0: Float32, y0: Float32, x1: Float32, y1: Float32, r: Float32, c: UInt32, style: Int, border: Float32 = 0, bcol: UInt32 = 0):
        """Rounded rect filled and shadowed by a style() id, with an optional border."""
        self.push(SRECT | min(max(Int(border * 4 + 0.5), 0), 255) << 8 | min(max(style, 0), NSTY - 1) << 16, x0, y0, x1, y1, r, c, bcol)

    def clip(mut self, x0: Float32, y0: Float32, x1: Float32, y1: Float32):
        """Clip everything drawn from here to unclip() to the rect (hard edge at pixel centers; a new clip replaces the old,
        which is ended first so a tile only the old one covered does not keep it)."""
        self.unclip()
        self.cur_clip = SIMD[DType.float32, 4](sane(x0), sane(y0), sane(x1), sane(y1))
        self.push(CLIP, x0, y0, x1, y1, 0, 0, 0)

    def unclip(mut self):
        if self.cur_clip[0] > -1e8:
            self.push(CLIP, self.cur_clip[0], self.cur_clip[1], self.cur_clip[2], self.cur_clip[3], 0, 0, 1)
            self.cur_clip = FULL

    def glass(mut self, x0: Float32, y0: Float32, x1: Float32, y1: Float32, r: Float32, tint: UInt32):
        """Rounded rect showing the blurred scene (Layer.set_scene / blur_from) under the tint."""
        self.push(GLASS, x0, y0, x1, y1, r, tint)

    def line(mut self, x0: Float32, y0: Float32, x1: Float32, y1: Float32, w: Float32, c: UInt32):
        self.push(LINE, x0, y0, x1, y1, w * 0.5, c)

    def tri(mut self, x0: Float32, y0: Float32, x1: Float32, y1: Float32, x2: Float32, y2: Float32, c: UInt32):
        if y2 != y2:
            self.dropped += 1
            return
        self.push(TRI, x0, y0, x1, y1, sane(x2), c, bits(sane(y2)))

    def image(mut self, x0: Float32, y0: Float32, x1: Float32, y1: Float32, off: Int, sw: Int, sh: Int, alpha: Int = 255):
        """Draw sw x sh source pixels starting at `off` (the app's RGBA in the source buffer) scaled to the rect.
        A source outside ISRC is dropped and counted."""
        if off < 0 or sw < 1 or sh < 1 or sw > 0xFFFF or sh > 0xFFFF or off + sw * sh > ISRC:
            self.dropped += 1
            return
        self.push(IMAGE, x0, y0, x1, y1, Float32(off), UInt32(alpha) << 24, UInt32(sw) | UInt32(sh) << 16)

    def text(mut self, x: Float32, y: Float32, s: StringSlice, size: Int, c: UInt32) -> Float32:
        """Draw UTF-8 s with its top at y; returns the x after it. A size index the font does not have draws nothing (counted)."""
        if not self.font.has_size(size):
            self.dropped += 1
            return x
        var pen = x
        var base = y + Float32(self.font.ascent[size])
        var b = s.as_bytes()
        var i = 0
        while i < len(b):
            var d = decode(b, i)
            i += d[1]
            var g = self.font.glyph(size, d[0])
            var w = Float32(self.font.g[g + 2])
            if w > 0:
                var gx = pen + Float32(self.font.g[g + 4])
                var gy = base - Float32(self.font.g[g + 5])
                self.push(GLYPH, gx, gy, gx + w, gy + Float32(self.font.g[g + 3]), 0, c,
                          UInt32(self.font.g[g]) | UInt32(self.font.g[g + 1]) << 16)
            pen += Float32(self.font.g[g + 6])
        return pen

    def shadow_pad(self, id: Int) -> Float32:
        var o = id * 8
        return 2 * flt(self.sty[o + 6]) + max(abs(flt(self.sty[o + 4])), abs(flt(self.sty[o + 5]))) if self.sty[o + 7] >> 24 > 0 else Float32(0)

    def span(self, k: Int, clip: SIMD[DType.float32, 4] = FULL) -> SIMD[DType.int32, 4]:
        """Tiles command k may touch: tx0, tx1, ty0, ty1 (half open), from its box, blur, line width or third corner, cut to
        the tiles of the clip rect in force (not for a CLIP command itself)."""
        var o = k * CW
        var kind = Int(self.cmd[o] & 255)
        var x0 = flt(self.cmd[o + 1])
        var y0 = flt(self.cmd[o + 2])
        var x1 = flt(self.cmd[o + 3])
        var y1 = flt(self.cmd[o + 4])
        var pad = min(Float32(2) + (2 * flt(self.cmd[o + 7]) if kind == RECT else (flt(self.cmd[o + 5]) if kind == LINE else (Float32((self.cmd[o] >> 16) & 255) if kind == RRECT else (self.shadow_pad(Int(self.cmd[o] >> 16)) if kind == SRECT else 0)))), BIG)
        var lx, hx, ly, hy = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
        if kind == TRI:
            lx, hx = min(lx, flt(self.cmd[o + 5])), max(hx, flt(self.cmd[o + 5]))
            ly, hy = min(ly, flt(self.cmd[o + 7])), max(hy, flt(self.cmd[o + 7]))
        var r = SIMD[DType.int32, 4](Int32(max(Int(lx - pad) // TILE, 0)), Int32(min(Int(hx + pad) // TILE + 1, TX)),
                                     Int32(max(Int(ly - pad) // TILE, 0)), Int32(min(Int(hy + pad) // TILE + 1, TY)))
        if kind != CLIP and clip[0] > -1e8:
            r = SIMD[DType.int32, 4](max(r[0], Int32(max(Int(clip[0]) // TILE, 0))), min(r[1], Int32(min(Int(clip[2]) // TILE + 1, TX))),
                                     max(r[2], Int32(max(Int(clip[1]) // TILE, 0))), min(r[3], Int32(min(Int(clip[3]) // TILE + 1, TY))))
        return r

    def clip_after(self, k: Int, clip: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:
        """The clip rect in force after command k."""
        var o = k * CW
        if self.cmd[o] & 255 != CLIP:
            return clip
        return FULL if self.cmd[o + 7] != 0 else SIMD[DType.float32, 4](flt(self.cmd[o + 1]), flt(self.cmd[o + 2]), flt(self.cmd[o + 3]), flt(self.cmd[o + 4]))

    def finish(mut self):
        """Bin the commands into tiles, counted, offsets by prefix sum, then filled in draw order. Past NBIN entries the
        remaining commands drop."""
        for t in range(TX * TY + 1):
            self.off[t] = 0
        var total = 0
        var clip = FULL
        for k in range(self.n):
            var s = self.span(k, clip)
            clip = self.clip_after(k, clip)
            var e = max(Int(s[1] - s[0]), 0) * max(Int(s[3] - s[2]), 0)
            if total + e > NBIN:
                self.dropped += self.n - k
                self.n = k
                break
            total += e
            for ty in range(Int(s[2]), Int(s[3])):
                for tx in range(Int(s[0]), Int(s[1])):
                    self.off[ty * TX + tx + 1] += 1
        for t in range(TX * TY):
            self.off[t + 1] += self.off[t]
            self.cur[t] = self.off[t]
        clip = FULL
        for k in range(self.n):
            var s = self.span(k, clip)
            clip = self.clip_after(k, clip)
            for ty in range(Int(s[2]), Int(s[3])):
                for tx in range(Int(s[0]), Int(s[1])):
                    var t = ty * TX + tx
                    self.bins[Int(self.cur[t])] = UInt16(k)
                    self.cur[t] += 1
        self.binned = self.n
