# VENDORED COPY of amarbaro/mojo-imkit (Apache-2.0, see imkit/LICENSE), tag v0.9.0 (eb91210),
# ~/Projects/mojo/mojo-imkit, file src/imkit/font.mojo. This repo keeps a real file
# rather than an -I path outside the tree, because a clone must build without anything
# outside it. Sync by hand if the upstream changes; tools/ci-checks.sh compares the two
# against the tag when the upstream is present and says so when they drift.
#
"""Glyph atlas for ui.mojo from a TrueType font, through FreeType (the system libfreetype, OwnedDLHandle): ASCII 32-126
at a few pixel sizes rasterized at start, every other codepoint on first use (UTF-8 in, `decode`), all skyline-packed
(atlas.mojo) into one alpha8 atlas AW x AH; seven int16 of metrics per glyph. A codepoint the font lacks comes from the
fallback font, else draws as '?'; a full atlas also draws '?' (`full` counts them). Rows touched since the last upload are
`dirty0` to `dirty1` (Layer sends only those). Struct offsets: FreeType 2.14 (offsetof on its headers).

    var font = Font("/usr/share/fonts/TTF/Roboto-Medium.ttf", [13, 16, 22], "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc")
    font.width("Pause", 0)        # pixels at size 0
"""
from std.ffi import OwnedDLHandle
from std.sys import get_defined_int
from .atlas import Skyline

comptime AW = 512  # atlas width, pixels
comptime AH = get_defined_int["UI_ATLAS_H", 1024]()  # atlas rows (on the card too; -DUI_ATLAS_H= for more)
comptime FIRST = 32  # ASCII 32..126
comptime NCH = 95
comptime GM = 7  # metrics per glyph: atlas x, atlas y, w, h, left bearing, top bearing, advance


@always_inline
def peek[T: AnyType](addr: Int) -> Pointer[T, MutAnyOrigin]:
    return Pointer[T, MutAnyOrigin](unsafe_from_address=addr)


def decode(b: Span[UInt8, _], i: Int) -> Tuple[Int, Int]:
    """The codepoint at byte i of UTF-8 text and the bytes it takes; a bad, overlong, surrogate or truncated sequence is
    (0xFFFD, 1), so the scan always moves on."""
    var c = Int(b[i])
    if c < 0x80:
        return (c, 1)
    var n = 2 if c >> 5 == 6 else (3 if c >> 4 == 14 else (4 if c >> 3 == 30 else 0))
    if n == 0 or i + n > len(b):
        return (0xFFFD, 1)
    var cp = c & (0x7F >> n)
    for k in range(1, n):
        if Int(b[i + k]) >> 6 != 2:
            return (0xFFFD, 1)
        cp = cp << 6 | (Int(b[i + k]) & 0x3F)
    if cp < (0x80 if n == 2 else (0x800 if n == 3 else 0x10000)) or cp > 0x10FFFF or 0xD800 <= cp < 0xE000:
        return (0xFFFD, 1)
    return (cp, n)


struct Font(Movable):
    var atlas: List[UInt8]  # alpha, AW wide, the rows in use rounded up to 64 (at most AH)
    var height: Int  # rows in use
    var sizes: List[Int]
    var ascent: List[Int]
    var g: List[Int16]  # [size][ASCII char][GM], then one GM per cached other codepoint (ext)
    var ext: Dict[Int, Int]  # codepoint << 8 | size slot -> offset into g (256 size slots at most)
    var sky: Skyline
    var dirty0: Int  # atlas rows rasterized since the last clean(): dirty0 <= row < dirty1
    var dirty1: Int
    var full: Int  # glyphs that did not fit the atlas
    var ft: OwnedDLHandle
    var lib: Int
    var face: Int
    var fb: Int  # fallback face (0: none)

    def __init__(out self, path: String, var sizes: List[Int], fallback: String = "") raises:
        if len(sizes) > 256:
            raise Error("Font: " + String(len(sizes)) + " sizes, 256 at most")
        self.ft = OwnedDLHandle("libfreetype.so.6")
        var lib = 0
        var face = 0
        var fb = 0
        var p = path.copy()
        if self.ft.get_function[Int32]("FT_Init_FreeType")(Pointer(to=lib)) != 0 or \
           self.ft.get_function[Int32]("FT_New_Face")(lib, p.as_c_string_span().ptr(), Int(0), Pointer(to=face)) != 0:
            raise Error("FreeType could not open " + path)
        var q = fallback.copy()
        if fallback.byte_length() > 0 and self.ft.get_function[Int32]("FT_New_Face")(lib, q.as_c_string_span().ptr(), Int(0), Pointer(to=fb)) != 0:
            raise Error("FreeType could not open " + fallback)
        self.lib, self.face, self.fb = lib, face, fb
        self.atlas = List[UInt8]()  # grows by 64 rows as glyphs land (load); the card's copy is AW x AH
        self.height = 0
        self.ascent = List[Int]()
        self.g = List[Int16]()
        self.ext = Dict[Int, Int]()
        self.sky = Skyline(AW, AH)
        self.dirty0, self.dirty1, self.full = AH, 0, 0
        self.sizes = sizes^
        for s in range(len(self.sizes)):
            _ = self.ft.get_function[Int32]("FT_Set_Pixel_Sizes")(face, UInt32(0), UInt32(self.sizes[s]))
            self.ascent.append(Int(peek[Int64](peek[Int](face + 160)[] + 48)[] >> 6))
            for c in range(FIRST, FIRST + NCH):
                if self.load(face, s, c) < 0:
                    raise Error("Font: ASCII at size " + String(self.sizes[s]) + " does not fit the atlas")

    def __deinit__(deinit self):
        try:
            if self.fb != 0:
                _ = self.ft.get_function[Int32]("FT_Done_Face")(self.fb)
            _ = self.ft.get_function[Int32]("FT_Done_Face")(self.face)
            _ = self.ft.get_function[Int32]("FT_Done_FreeType")(self.lib)
        except:
            pass

    def load(mut self, f: Int, s: Int, cp: Int) raises -> Int:
        """Rasterize cp from face f at size slot s into the atlas; the offset of its metrics in g, or -1 if the atlas is full."""
        _ = self.ft.get_function[Int32]("FT_Set_Pixel_Sizes")(f, UInt32(0), UInt32(self.sizes[s]))
        _ = self.ft.get_function[Int32]("FT_Load_Char")(f, UInt(cp), Int32(4))  # FT_LOAD_RENDER
        var slot = peek[Int](f + 152)[]
        var h = Int(peek[UInt32](slot + 152)[])
        var w = Int(peek[UInt32](slot + 156)[])
        var pitch = Int(peek[Int32](slot + 160)[])
        var buf = peek[Int](slot + 168)[]
        var x, y = 0, 0
        if w > 0 and h > 0:
            var at = self.sky.place(w + 1, h + 1)  # 1 px gutter
            if at[0] < 0:
                self.full += 1
                return -1
            x, y = at[0], at[1]
            var rows = min((y + h + 1 + 63) // 64 * 64, AH)  # rows to hold this glyph and its gutter, in steps of 64
            if len(self.atlas) < rows * AW:
                self.atlas.resize(rows * AW, 0)
            for r in range(h):
                for k in range(w):
                    self.atlas[(y + r) * AW + x + k] = peek[UInt8](buf + r * pitch + k)[]
            self.height = max(self.height, y + h + 1)
            self.dirty0, self.dirty1 = min(self.dirty0, y), max(self.dirty1, y + h)
        var m: List[Int] = [x, y, w, h, Int(peek[Int32](slot + 192)[]), Int(peek[Int32](slot + 196)[]), Int(peek[Int64](slot + 128)[] >> 6)]
        for v in m:
            self.g.append(Int16(v))
        return len(self.g) - GM

    def clean(mut self):
        """The atlas rows are on the card."""
        self.dirty0, self.dirty1 = AH, 0

    def glyph(mut self, s: Int, c: Int) -> Int:
        """Index of codepoint c's metrics at size slot s: ASCII from the table, others rasterized once (this font, else the
        fallback) and cached; a glyph neither font has, or one that does not fit the atlas, is '?'."""
        if FIRST <= c < FIRST + NCH:
            return ((s * NCH) + (c - FIRST)) * GM
        var key = c << 8 | s
        var hit = self.ext.get(key)
        if hit:
            return hit.value()
        var g = -1
        try:
            var f = self.face if self.ft.get_function[UInt32]("FT_Get_Char_Index")(self.face, UInt(c)) != 0 else \
                (self.fb if self.fb != 0 and self.ft.get_function[UInt32]("FT_Get_Char_Index")(self.fb, UInt(c)) != 0 else 0)
            if f != 0:
                g = self.load(f, s, c)
        except:
            pass
        if g < 0:
            g = ((s * NCH) + (ord("?") - FIRST)) * GM
        self.ext[key] = g
        return g

    def has_size(self, s: Int) -> Bool:
        """Is s a size slot of this font (an index into the sizes it was built with)?"""
        return 0 <= s < len(self.sizes)

    def width(mut self, text: StringSlice, s: Int) -> Int:
        if not self.has_size(s):
            return 0
        var w, i = 0, 0
        var b = text.as_bytes()
        while i < len(b):
            var d = decode(b, i)
            w += Int(self.g[self.glyph(s, d[0]) + 6])
            i += d[1]
        return w
