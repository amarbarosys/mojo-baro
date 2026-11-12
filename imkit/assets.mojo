# VENDORED COPY of amarbaro/mojo-imkit (Apache-2.0, see imkit/LICENSE), tag v0.9.0 (eb91210),
# ~/Projects/mojo/mojo-imkit, file src/imkit/assets.mojo. This repo keeps a real file
# rather than an -I path outside the tree, because a clone must build without anything
# outside it. Sync by hand if the upstream changes; tools/ci-checks.sh compares the two
# against the tag when the upstream is present and says so when they drift.
#
"""Offline asset atlas. `pack` reads PNGs and JPEGs (Blender icons, ComfyUI textures) and writes PREFIX.bin (every picture's RGBA bytes
back to back, r | g << 8 | b << 16 | a << 24 per pixel) and PREFIX.txt (one line `name offset width height`, the name is the file
stem, the offset is in pixels). `Assets(PREFIX)` loads both at run time; `Layer.set_assets` puts the pixels in the image pool
and `Ui.icon_img(name, x, y)` draws one. `png_decode` takes 8-bit RGB or RGBA, no interlace, through the system libz (CRCs are
not checked); anything else raises. `image_decode` picks the decoder by the file's first bytes: FF D8 is `jpeg_decode` (imkit.jpeg, pure
Mojo), RIFF....WEBP is not decoded yet, anything else is `png_decode`. Build with tools/pack_atlas.mojo; tools/png_fixture.py is the
PNG test oracle, tests/jpeg_test.sh the JPEG one.
"""
from std.ffi import OwnedDLHandle
from imkit.jpeg import jpeg_decode
from imkit.webp import webp_decode


comptime MAXPX = 1 << 26  # the largest picture png_decode allocates for: 64 M pixels, 256 MB as RGBA


def be32(d: Span[UInt8, _], i: Int) -> Int:
    return Int(d[i]) << 24 | Int(d[i + 1]) << 16 | Int(d[i + 2]) << 8 | Int(d[i + 3])


def png_decode(d: Span[UInt8, _]) raises -> Tuple[Int, Int, List[UInt32]]:
    """(width, height, pixels) of an 8-bit RGB or RGBA non-interlaced PNG."""
    if len(d) < 33 or be32(d, 0) != 0x89504E47 or be32(d, 4) != 0x0D0A1A0A:
        raise Error("png: not a PNG")
    var w, h, kind, pos = 0, 0, 0, 8
    var z = List[UInt8]()
    while pos + 8 <= len(d):
        var n = be32(d, pos)
        if pos + 12 + n > len(d):
            raise Error("png: truncated chunk")
        var t = be32(d, pos + 4)
        if t == 0x49484452:  # IHDR
            if n != 13:
                raise Error("png: IHDR chunk is " + String(n) + " bytes, not 13")
            w, h, kind = be32(d, pos + 8), be32(d, pos + 12), Int(d[pos + 17])
            if d[pos + 16] != 8 or (kind != 2 and kind != 6) or d[pos + 18] != 0 or d[pos + 19] != 0 or d[pos + 20] != 0:
                raise Error("png: only 8-bit RGB / RGBA, no interlace (depth " + String(d[pos + 16]) + ", color type " + String(kind) + ")")
        elif t == 0x49444154:  # IDAT
            z.extend(d[pos + 8 : pos + 8 + n])
        elif t == 0x49454E44:  # IEND
            break
        pos += 12 + n
    var bpp = 3 if kind == 2 else 4
    if w < 1 or h < 1 or len(z) == 0:
        raise Error("png: no image data")
    if w > MAXPX or h > MAXPX or w * h > MAXPX:  # before any allocation that grows with w * h (a 32-bit side overflows w * h)
        raise Error("png: " + String(w) + " x " + String(h) + " is too large (" + String(MAXPX) + " pixels at most)")
    var stride = w * bpp
    var raw = List[UInt8](length=(stride + 1) * h, fill=0)
    var rlen = UInt(len(raw))
    if OwnedDLHandle("libz.so.1").get_function[Int32]("uncompress")(raw.unsafe_ptr(), Pointer(to=rlen), z.unsafe_ptr(), UInt(len(z))) != 0 or Int(rlen) != len(raw):
        raise Error("png: zlib data is not " + String(len(raw)) + " bytes")
    var px = List[UInt32](length=w * h, fill=0)
    var cur = List[Int](length=stride, fill=0)
    var prev = List[Int](length=stride, fill=0)
    for r in range(h):
        var f = Int(raw[r * (stride + 1)])
        for i in range(stride):
            var a = cur[i - bpp] if i >= bpp else 0
            var b, c = prev[i], (prev[i - bpp] if i >= bpp else 0)
            var pred = 0
            if f == 1:
                pred = a
            elif f == 2:
                pred = b
            elif f == 3:
                pred = (a + b) // 2
            elif f == 4:
                var pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)  # |p - a|, |p - b|, |p - c| for p = a + b - c
                pred = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
            elif f != 0:
                raise Error("png: row filter " + String(f))
            cur[i] = (Int(raw[r * (stride + 1) + 1 + i]) + pred) & 255
        for x in range(w):
            px[r * w + x] = UInt32(cur[x * bpp]) | UInt32(cur[x * bpp + 1]) << 8 | UInt32(cur[x * bpp + 2]) << 16 | UInt32(cur[x * bpp + 3] if bpp == 4 else 255) << 24
        swap(cur, prev)
    return (w, h, px^)


def image_decode(d: Span[UInt8, _]) raises -> Tuple[Int, Int, List[UInt32]]:
    """(width, height, pixels) of a JPEG or PNG, told apart by signature."""
    if len(d) >= 12 and d[0] == 0x52 and d[1] == 0x49 and d[2] == 0x46 and d[3] == 0x46 and d[8] == 0x57 and d[9] == 0x45 and d[10] == 0x42 and d[11] == 0x50:
        return webp_decode(d)
    if len(d) >= 2 and d[0] == 0xFF and d[1] == 0xD8:
        return jpeg_decode(d)
    return png_decode(d)


struct Assets(Movable):
    var px: List[UInt32]  # every picture back to back
    var names: List[String]
    var meta: List[Int]  # offset, width, height per name

    def __init__(out self):
        self.px = List[UInt32]()
        self.names = List[String]()
        self.meta = List[Int]()

    def __init__(out self, prefix: String) raises:
        """Load PREFIX.bin and PREFIX.txt written by pack."""
        self = Self()
        with open(prefix + ".bin", "r") as f:
            var b = f.read_bytes()
            for i in range(len(b) // 4):
                self.px.append(UInt32(b[4 * i]) | UInt32(b[4 * i + 1]) << 8 | UInt32(b[4 * i + 2]) << 16 | UInt32(b[4 * i + 3]) << 24)
        with open(prefix + ".txt", "r") as f:
            for line in f.read().split("\n"):
                var p = line.split(" ")
                if len(p) == 4:
                    self.names.append(String(p[0]))
                    self.meta.extend([Int(p[1]), Int(p[2]), Int(p[3])])
                    if Int(p[1]) + Int(p[2]) * Int(p[3]) > len(self.px):
                        raise Error("assets: " + String(p[0]) + " runs past " + prefix + ".bin")

    def find(self, name: StringSlice) -> Int:
        """Index of name, or -1."""
        for i in range(len(self.names)):
            if self.names[i] == name:
                return i
        return -1


def pack(paths: List[String], prefix: String) raises:
    """Decode each PNG or JPEG and write PREFIX.bin and PREFIX.txt (name = file stem)."""
    var bin = List[UInt8]()
    var txt = String()
    for p in paths:
        var parts = p.split("/")
        var stem = String(String(parts[len(parts) - 1]).split(".")[0])
        with open(p, "r") as f:
            var pic = image_decode(Span(f.read_bytes()))
            txt += stem + " " + String(len(bin) // 4) + " " + String(pic[0]) + " " + String(pic[1]) + "\n"
            for v in pic[2]:
                bin.extend([UInt8(v & 255), UInt8((v >> 8) & 255), UInt8((v >> 16) & 255), UInt8(v >> 24)])
    with open(prefix + ".bin", "w") as f:
        f.write_bytes(Span(bin))
    with open(prefix + ".txt", "w") as f:
        f.write(txt)
