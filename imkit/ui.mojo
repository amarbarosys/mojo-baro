# VENDORED COPY of amarbaro/mojo-imkit (Apache-2.0, see imkit/LICENSE), tag v0.9.0 (eb91210),
# ~/Projects/mojo/mojo-imkit, file src/imkit/ui.mojo. This repo keeps a real file
# rather than an -I path outside the tree, because a clone must build without anything
# outside it. Sync by hand if the upstream changes; tools/ci-checks.sh compares the two
# against the tag when the upstream is present and says so when they drift.
#
"""Immediate-mode widgets (docs/design.md). Every frame the host calls widgets with this frame's Input; each widget
returns its interaction (clicked, toggled, dragged) and appends draw commands to a `Draw` (draw.mojo) that finish() bins
for the compose kernel. Colors come from theme.mojo.
"""
from .assets import Assets
from .anim import Anim
from .draw import Draw, TX, TY, NBIN, NCMD, CW, UW, UH, ISRC, AH, rgba
from .font import Font
from .theme import Theme, glass
from .contrast import contrast
from .window import Input

comptime SC_BACKSPACE = 42  # SDL scancode


def num(x: Float32) -> String:
    """x for an axis label: whole above 100, else one decimal; NaN and magnitudes from 1e9 on (Int cannot hold them) as text."""
    if x != x:
        return "nan"
    if abs(x) >= 1e9:
        return ">1e9" if x > 0 else "<-1e9"
    if abs(x) >= 100:
        return String(Int(round(x)))
    var t = Int(round(abs(x) * 10))
    return ("-" if x < 0 and t > 0 else "") + String(t // 10) + "." + String(t % 10)


def cp_floor(b: Span[UInt8, _], i: Int) -> Int:
    """The start of the UTF-8 code point that byte i is in (i itself if it starts one or is past the end)."""
    var k = i
    while 0 < k < len(b) and b[k] & 0xC0 == 0x80:
        k -= 1
    return k


def cp_ceil(b: Span[UInt8, _], i: Int) -> Int:
    """The first code point start at or after byte i (len(b) if none)."""
    var k = i
    while k < len(b) and b[k] & 0xC0 == 0x80:
        k += 1
    return k


struct Ui(Movable):
    var d: Draw
    var anim: Anim  # widget id -> spring (anim.mojo)
    var theme: Theme  # colors and shape by role (theme.mojo); every widget reads it, set_theme switches it at runtime
    var scale: Float32  # layer pixels per logical unit (DPI 1.5: 1.5); widgets, cursor and input are logical
    var inp: Input
    var active: Int  # widget held by the mouse (0: none)
    var over: Bool  # the mouse is over UI this frame: the scene does not take it
    var x: Float32  # layout cursor
    var y: Float32
    var focus: Int  # id of the text field that takes typing (0: none yet; the first field drawn takes it)
    var menu: Int  # id of the open menu (0: none)
    var pop: SIMD[DType.float32, 4]  # the open menu's popup box this frame: x0, y0, w, h
    var in_menu: Bool  # between menu_begin (open) and menu_end
    var vp: SIMD[DType.float32, 4]  # the scroll list's viewport this frame: x0, y0, w, h
    var rh: Float32  # its row height
    var soff: Float32  # its scroll offset
    var srows: Int  # its row count
    var mk: Int  # items drawn so far in the open menu
    var cols: List[Float32]  # the table's column edges this frame, from its left (ncols + 1)
    var field_focused: Bool  # a field drawn this frame has focus: the app sends keys to typing, not to shortcuts
    var assets: Assets  # icon_img's pictures: the app sets it and calls Layer.set_assets with it

    def __init__(out self, var font: Font):
        self.d = Draw(font^)
        self.anim = Anim()
        self.scale = 1
        self.theme = glass()
        self.active, self.over, self.x, self.y = 0, False, 0, 0
        self.focus, self.field_focused = 0, False
        self.assets = Assets()
        self.menu, self.in_menu, self.rh, self.soff, self.srows, self.mk = 0, False, 24, 0, 0, 0
        self.pop = SIMD[DType.float32, 4](0)
        self.cols = List[Float32]()
        self.vp = SIMD[DType.float32, 4](0)
        self.inp = Input(0, 0, 0, 0, 0, 0, 0, 0, SIMD[DType.uint64, 2](0), False)

    def begin(mut self, inp: Input, dt: Float32 = 0):
        """Start a frame: this frame's input and its length in seconds (0: springs hold still)."""
        self.inp = inp.logical(self.scale)  # the mouse arrives in layer pixels
        self.anim.begin(dt)
        self.d.begin()
        self.over = False
        self.field_focused = False

    def set_scale(mut self, scale: Float32):
        """DPI: draw `scale` layer pixels per logical unit. Call before begin(); the font sizes stay in layer pixels, so
        build the Font with sizes times scale for a sharp result."""
        self.scale = scale

    def set_theme(mut self, theme: Theme):
        self.theme = theme.copy()

    def w(self) -> Float32:
        """Width of the layer in logical units."""
        return Float32(UW) / self.scale

    def h(self) -> Float32:
        return Float32(UH) / self.scale

    def tw(mut self, s: StringSlice, size: Int) -> Float32:
        """Logical width of s at font size index `size`."""
        return Float32(self.d.font.width(s, size)) / self.scale

    def asc(self, size: Int) -> Float32:
        """Logical ascent of font size index `size`."""
        return Float32(self.d.font.ascent[size] if self.d.font.has_size(size) else 0) / self.scale

    def rect(mut self, x0: Float32, y0: Float32, x1: Float32, y1: Float32, r: Float32, c: UInt32, blur: Float32 = 0):
        var k = self.scale
        self.d.rect(x0 * k, y0 * k, x1 * k, y1 * k, r * k, c, blur * k)

    def rrect(mut self, x0: Float32, y0: Float32, x1: Float32, y1: Float32, r: Float32, c: UInt32, border: Float32 = 0,
              bcol: UInt32 = 0, glow: Float32 = 0, grad: Int = 0):
        var k = self.scale
        self.d.rrect(x0 * k, y0 * k, x1 * k, y1 * k, r * k, c, border * k, bcol, glow * k, grad)

    def style(mut self, c1: UInt32 = 0, mode: Int = 0, gx: Float32 = 0, gy: Float32 = 1, sx: Float32 = 0, sy: Float32 = 0,
              sr: Float32 = 0, sc: UInt32 = 0) -> Int:
        var k = self.scale  # offsets and radius are logical; the gradient direction is a unit vector
        return self.d.style(c1, mode, gx, gy, sx * k, sy * k, sr * k, sc)

    def srect(mut self, x0: Float32, y0: Float32, x1: Float32, y1: Float32, r: Float32, c: UInt32, style: Int, border: Float32 = 0, bcol: UInt32 = 0):
        var k = self.scale
        self.d.srect(x0 * k, y0 * k, x1 * k, y1 * k, r * k, c, style, border * k, bcol)

    def clip(mut self, x0: Float32, y0: Float32, x1: Float32, y1: Float32):
        var k = self.scale
        self.d.clip(x0 * k, y0 * k, x1 * k, y1 * k)

    def unclip(mut self):
        self.d.unclip()

    def glass(mut self, x0: Float32, y0: Float32, x1: Float32, y1: Float32, r: Float32, tint: UInt32):
        var k = self.scale
        self.d.glass(x0 * k, y0 * k, x1 * k, y1 * k, r * k, tint)

    def icon_img(mut self, name: StringSlice, x: Float32, y: Float32, w: Float32 = 0, h: Float32 = 0, alpha: Int = 255) -> Bool:
        """Draw the asset `name` with its top left at (x, y) and, if given, size w x h (logical), else its own size in layer
        pixels; False (and the drop is counted) if there is no such asset."""
        var i = self.assets.find(name)
        if i < 0:
            self.d.dropped += 1
            return False
        ref m = self.assets.meta
        var k = self.scale
        self.d.image(x * k, y * k, x * k + (Float32(m[3 * i + 1]) if w <= 0 else w * k), y * k + (Float32(m[3 * i + 2]) if h <= 0 else h * k), m[3 * i], m[3 * i + 1], m[3 * i + 2], alpha)
        return True

    def line(mut self, x0: Float32, y0: Float32, x1: Float32, y1: Float32, w: Float32, c: UInt32):
        var k = self.scale
        self.d.line(x0 * k, y0 * k, x1 * k, y1 * k, w * k, c)

    def tri(mut self, x0: Float32, y0: Float32, x1: Float32, y1: Float32, x2: Float32, y2: Float32, c: UInt32):
        var k = self.scale
        self.d.tri(x0 * k, y0 * k, x1 * k, y1 * k, x2 * k, y2 * k, c)

    def image(mut self, x0: Float32, y0: Float32, x1: Float32, y1: Float32, off: Int, sw: Int, sh: Int, alpha: Int = 255):
        var k = self.scale
        self.d.image(x0 * k, y0 * k, x1 * k, y1 * k, off, sw, sh, alpha)

    def text(mut self, x: Float32, y: Float32, s: StringSlice, size: Int, c: UInt32) -> Float32:
        """Draw s with its top at y (logical); the origin snaps to a whole layer pixel. Returns the logical x after it."""
        var k = self.scale
        return self.d.text(round(x * k), round(y * k), s, size, c) / k

    def animate(mut self, id: Int, target: Float32) -> Float32:
        """The spring value for widget id moving toward target (first sight: the target itself)."""
        return self.anim.get(id, target)

    def hovered(self, x0: Float32, y0: Float32, x1: Float32, y1: Float32) -> Bool:
        return x0 <= self.inp.x < x1 and y0 <= self.inp.y < y1

    def clickable(mut self, id: Int, x0: Float32, y0: Float32, x1: Float32, y1: Float32) -> Tuple[Bool, Bool]:
        """(hovered, clicked): a press inside makes id active, a release inside while active is the click."""
        var h = self.hovered(x0, y0, x1, y1)
        if h and self.inp.pressed & 1:
            self.active = id
        return (h, h and self.active == id and self.inp.released & 1 != 0)

    def panel(mut self, x0: Float32, y0: Float32, x1: Float32, y1: Float32, title: StringSlice):
        """Glass panel with a soft shadow and a title; the layout cursor starts under the title."""
        var t = self.theme.copy()
        self.rect(x0 + 2, y0 + 6, x1 + 2, y1 + 10, t.radius + 2, t.shadow, t.blur)
        if t.border > 0:
            self.rect(x0 - t.border, y0 - t.border, x1 + t.border, y1 + t.border, t.radius + t.border, t.border_c)
        self.rect(x0, y0, x1, y1, t.radius, t.panel)
        _ = self.text(x0 + 16, y0 + 12, title, 1, t.text)
        if self.hovered(x0, y0, x1, y1):
            self.over = True
        self.x, self.y = x0 + 16, y0 + 40

    def button(mut self, id: Int, label: StringSlice, w: Float32, h: Float32 = 30, on: Bool = False, color: UInt32 = 0) -> Bool:
        """Pill button at the cursor (also a toggle: on = filled with `color`, the theme accent by default); returns clicked."""
        var x0, y0 = self.x, self.y
        var hc = self.clickable(id, x0, y0, x0 + w, y0 + h)
        var t = self.theme.copy()
        self.rect(x0, y0, x0 + w, y0 + h, h * 0.5 * t.round, (color if color != 0 else t.accent) if on else (t.hover if hc[0] else t.surface))
        var tw = self.tw(label, 0)
        _ = self.text(x0 + (w - tw) * 0.5, y0 + (h - self.asc(0)) * 0.5 - 1, label, 0, self.theme.on_accent if on else self.theme.text)
        self.x += w + 8
        return hc[1]

    def field(mut self, id: Int, mut buf: String, max: Int) -> Bool:
        """Single-line text box (360 x 30) at the cursor; next row. A click gives it focus, and so does the first field drawn
        while none has it. The focused field appends the ASCII 32-126 of this frame's Input.text up to `max` bytes, then
        removes the last code point on Backspace; a long text shows its tail. Returns True if buf changed."""
        var x0, y0, w = self.x, self.y, Float32(360)
        var hc = self.clickable(id, x0, y0, x0 + w, y0 + 30)
        if hc[1] or self.focus == 0:
            self.focus = id
        var on = self.focus == id
        var changed = False
        if on:
            self.field_focused = True
            var add = List[UInt8]()  # collect, then one append: `buf += chr(b)` aborted with a 96 GiB request
            for b in self.inp.text.as_bytes():
                if 32 <= Int(b) <= 126 and buf.byte_length() + len(add) < max:
                    add.append(b)
            if len(add) > 0:
                buf = buf + String(unsafe_from_utf8=Span(add))
                changed = True
            if self.inp.key(SC_BACKSPACE) and buf.byte_length() > 0:
                var keep = cp_floor(buf.as_bytes(), buf.byte_length() - 1)
                var cut = String(unsafe_from_utf8=buf.as_bytes()[:keep])
                buf = cut^
                changed = True
        var t = self.theme.copy()
        var fr = t.radius * 0.6
        self.rect(x0 - 1.5, y0 - 1.5, x0 + w + 1.5, y0 + 31.5, fr + 1, t.accent if on else (t.border_c if t.border > 0 else t.surface))
        self.rect(x0, y0, x0 + w, y0 + 30, fr, t.hover if hc[0] and not on else t.surface)
        var s = 0
        while s < buf.byte_length() and self.tw(StringSlice(unsafe_from_utf8=buf.as_bytes()[s:]), 0) > w - 24:
            s = cp_ceil(buf.as_bytes(), s + 1)
        var pen = self.text(x0 + 10, y0 + (30 - self.asc(0)) * 0.5 - 1, StringSlice(unsafe_from_utf8=buf.as_bytes()[s:]), 0, t.text)
        if on:
            self.rect(pen + 1, y0 + 7, pen + 2.5, y0 + 23, 0.5, t.text)
        self.y += 38
        return changed

    def icon(mut self, id: Int, kind: Int, on: Bool = False) -> Bool:
        """Round icon button at the cursor: 0 play, 1 pause, 2 step, 3 stop; returns clicked."""
        var x0, y0 = self.x, self.y
        var hc = self.clickable(id, x0, y0, x0 + 36, y0 + 36)
        var t = self.theme.copy()
        self.rect(x0, y0, x0 + 36, y0 + 36, 18 * t.round, t.accent if on else (t.hover if hc[0] else t.surface))
        var cx, cy = x0 + 18, y0 + 18
        var g = t.on_accent if on else t.text
        if kind == 0 or kind == 2:
            self.tri(cx - 5, cy - 7, cx - 5, cy + 7, cx + 7, cy, g)
        if kind == 1:
            self.rect(cx - 6, cy - 7, cx - 2, cy + 7, 1.5, g)
            self.rect(cx + 2, cy - 7, cx + 6, cy + 7, 1.5, g)
        if kind == 2:
            self.rect(cx + 7, cy - 7, cx + 10, cy + 7, 1, g)
        if kind == 3:
            self.rect(cx - 6, cy - 6, cx + 6, cy + 6, 2, g)
        self.x += 44
        return hc[1]

    def bar(mut self, label: StringSlice, v: Float32, c: UInt32, w: Float32):
        """Label and a rounded progress bar of v in 0..1 at the cursor; next row."""
        _ = self.text(self.x, self.y, label, 0, self.theme.muted)
        var x0 = self.x + 64
        self.rect(x0, self.y + 4, x0 + w - 64, self.y + 12, 4, self.theme.surface)
        self.rect(x0, self.y + 4, x0 + (w - 64) * min(max(v, 0), 1), self.y + 12, 4, c)
        self.y += 20

    def heading(mut self, s: StringSlice, size: Int, w: Float32):
        """A heading in font slot `size` at the cursor, an accent rule under it as wide as w; next row."""
        var a = self.asc(size)
        _ = self.text(self.x, self.y, s, size, self.theme.text)
        self.rect(self.x, self.y + a * 1.35, self.x + w, self.y + a * 1.35 + 2, 1, self.theme.accent)
        self.y += a * 1.35 + 10

    def caption(mut self, s: StringSlice, size: Int = 0):
        """Muted small text at the cursor (pass the smallest font slot); next row."""
        _ = self.text(self.x, self.y, s, size, self.theme.muted)
        self.y += self.asc(size) * 1.5

    def separator(mut self, w: Float32):
        """A one-pixel rule in the outline colour across w; next row."""
        self.rect(self.x, self.y + 4, self.x + w, self.y + 5, 0, self.theme.border_c | 0x60000000)
        self.y += 10

    def badge(mut self, label: StringSlice, c: UInt32, size: Int = 0) -> Float32:
        """A pill with label on colour c at the cursor; the cursor moves right by its width plus 6. Returns that width."""
        var tw = self.tw(label, size)
        var h = self.asc(size) + 8
        self.rect(self.x, self.y, self.x + tw + 14, self.y + h, h / 2, c)
        var ink = rgba(255, 255, 255, 255) if contrast(rgba(255, 255, 255, 255), c) >= contrast(rgba(0, 0, 0, 255), c) else rgba(0, 0, 0, 255)
        _ = self.text(self.x + 7, self.y + 4, label, size, ink)  # black or white per pill colour (warn/bad are not the accent)
        self.x += tw + 20
        return tw + 14

    def kv(mut self, key: StringSlice, value: StringSlice, w: Float32, size: Int = 0):
        """key muted on the left, value right-aligned at x + w; next row."""
        _ = self.text(self.x, self.y, key, size, self.theme.muted)
        _ = self.text(self.x + w - self.tw(value, size), self.y, value, size, self.theme.text)
        self.y += self.asc(size) * 1.6

    def spark(mut self, vals: List[Float32], lo: Float32, hi: Float32, w: Float32, h: Float32, c: UInt32):
        """Line chart of vals (oldest first) in a w x h box at the cursor, with the last point marked; next row."""
        var x0, y0 = self.x, self.y
        self.rect(x0, y0, x0 + w, y0 + h, 6, self.theme.surface)
        var n = len(vals)
        for k in range(1, n):
            var ya = y0 + h - 3 - (h - 6) * (vals[k - 1] - lo) / max(hi - lo, 1e-6)
            var yb = y0 + h - 3 - (h - 6) * (vals[k] - lo) / max(hi - lo, 1e-6)
            self.line(x0 + w * Float32(k - 1) / Float32(max(n - 1, 1)), ya, x0 + w * Float32(k) / Float32(max(n - 1, 1)), yb, 1.6, c)
        self.y += h + 8
    def slider(mut self, id: Int, v: Float32, lo: Float32, hi: Float32, w: Float32) -> Float32:
        """Horizontal slider of w at the cursor (24 high); returns the new value. A press on it captures the mouse: the
        value follows the mouse x until release, also outside the track. Next row."""
        var x0, y0 = self.x, self.y
        var hc = self.clickable(id, x0 - 8, y0, x0 + w + 8, y0 + 24)
        var cap = self.active == id and (self.inp.down | self.inp.released) & 1 != 0
        var nv = lo + (hi - lo) * min(max((self.inp.x - x0) / w, Float32(0)), Float32(1)) if cap else v
        var f = min(max((nv - lo) / max(hi - lo, 1e-6), Float32(0)), Float32(1))
        var t = self.theme.copy()
        var r = 7 + 2 * self.animate(id, Float32(1) if hc[0] or cap else Float32(0))
        self.rect(x0, y0 + 9, x0 + w, y0 + 15, 3, t.surface)
        self.rect(x0, y0 + 9, x0 + w * f, y0 + 15, 3, t.accent)
        self.rect(x0 + w * f - r, y0 + 12 - r, x0 + w * f + r, y0 + 12 + r, r, t.text)
        self.y += 30
        return nv

    def menu_begin(mut self, id: Int, label: StringSlice, w: Float32, n: Int) -> Bool:
        """Menu header (a button of w) at the cursor and, when open, a popup of n items under it. Returns open: then call
        menu_item n times and menu_end. One menu is open at a time. Draw menus after the other widgets: draw order is z order."""
        var x0, y0 = self.x, self.y
        if self.button(id, label, w, 30, self.menu == id):
            self.menu = 0 if self.menu == id else id
        self.in_menu = self.menu == id
        if self.in_menu:
            self.pop = SIMD[DType.float32, 4](x0, y0 + 34, w, Float32(n) * 28 + 8)
            var t = self.theme.copy()
            self.rect(x0 + 2, y0 + 38, x0 + w + 2, y0 + 46 + Float32(n) * 28, t.radius * 0.6 + 2, t.shadow, t.blur)
            if t.border > 0:
                self.rect(x0 - t.border, y0 + 34 - t.border, x0 + w + t.border, y0 + 42 + Float32(n) * 28 + t.border, t.radius * 0.6 + t.border, t.border_c)
            self.rect(x0, y0 + 34, x0 + w, y0 + 42 + Float32(n) * 28, t.radius * 0.6, t.panel)
            self.over = True
            self.mk = 0
        return self.in_menu

    def menu_item(mut self, label: StringSlice) -> Bool:
        """Next item of the open menu; returns clicked, which also closes the menu."""
        if not self.in_menu:
            return False
        var x0, y0, w = self.pop[0] + 4, self.pop[1] + 4 + Float32(self.mk) * 28, self.pop[2] - 8
        self.mk += 1
        var hc = self.clickable(0x100000 + self.mk, x0, y0, x0 + w, y0 + 28)
        if hc[0]:
            self.rect(x0, y0, x0 + w, y0 + 28, self.theme.radius * 0.4, self.theme.hover)
        _ = self.text(x0 + 10, y0 + (28 - self.asc(0)) * 0.5 - 1, label, 0, self.theme.text)
        if hc[1]:
            self.menu = 0
        return hc[1]

    def menu_end(mut self):
        """Close the open menu on a press outside its popup, or on Escape (scancode 41)."""
        var out = not self.hovered(self.pop[0], self.pop[1] - 34, self.pop[0] + self.pop[2], self.pop[1] + self.pop[3])  # header and popup
        if self.in_menu and (self.inp.key(41) or (self.inp.pressed & 1 != 0 and out)):
            self.menu = 0
        self.in_menu = False

    def scroll_begin(mut self, id: Int, w: Float32, h: Float32, rows: Int, rh: Float32, mut off: Float32) -> Tuple[Int, Int]:
        """Scroll list viewport of w x h at the cursor over `rows` rows of height rh. `off` is the caller's scroll offset
        (logical units, moved by the wheel over the viewport, clamped). Returns (first, end): the rows to draw, the ones
        that touch the viewport (the rest are never emitted; the edge rows are clipped); row k sits at scroll_y(k). Call
        scroll_end after the rows."""
        var x0, y0 = self.x, self.y
        var rl = rh if rh > 0 and rh < Float32(1e6) else Float32(1)  # a height of 0, below 0, NaN or inf reads as 1: Int() of off // 0 is undefined
        var nr = max(rows, 0)
        var hot = self.hovered(x0, y0, x0 + w, y0 + h)
        off = min(max(off - (self.inp.wheel * rl if hot else 0), Float32(0)), max(Float32(nr) * rl - h, Float32(0)))
        self.vp = SIMD[DType.float32, 4](x0, y0, w, h)
        self.rh, self.soff, self.srows = rl, off, nr
        self.rect(x0, y0, x0 + w, y0 + h, self.theme.radius * 0.5, self.theme.surface)
        if hot:
            self.over = True
        self.clip(x0, y0, x0 + w, y0 + h)
        return (Int(off // rl), min(Int(-((-(off + h)) // rl)), nr))  # floor(off / rh), ceil((off + h) / rh)

    def scroll_y(self, k: Int) -> Float32:
        """Top of row k of the current scroll list, in logical units."""
        return self.vp[1] - self.soff + Float32(k) * self.rh

    def table_row(mut self, id: Int, k: Int, on: Bool = False) -> Bool:
        """Row k's hover or selection background (click target) of the current scroll list or table; returns clicked."""
        var x0, y0, w = self.vp[0], self.scroll_y(k), self.vp[2] - 10
        var hc = self.clickable(id, x0, max(y0, self.vp[1]), x0 + w, min(y0 + self.rh, self.vp[1] + self.vp[3]))  # the visible part only
        if on or hc[0]:
            self.rect(x0 + 2, y0, x0 + w, y0 + self.rh, self.theme.radius * 0.4, self.theme.accent if on else self.theme.hover)
        return hc[1]

    def scroll_row(mut self, id: Int, k: Int, label: StringSlice, on: Bool = False) -> Bool:
        """Selectable text row k of the current scroll list (call for the rows scroll_begin returned); returns clicked."""
        var hc = self.table_row(id, k, on)
        _ = self.text(self.vp[0] + 10, self.scroll_y(k) + (self.rh - self.asc(0)) * 0.5 - 1, label, 0, self.theme.on_accent if on else self.theme.text)
        return hc

    def fit(mut self, s: StringSlice, w: Float32) -> Int:
        """Bytes of s (cut on a code point) whose width at size 0 is at most w."""
        var n = s.byte_length()
        while n > 0 and self.tw(StringSlice(unsafe_from_utf8=s.as_bytes()[:n]), 0) > w:
            n = cp_floor(s.as_bytes(), n - 1)
        return n

    def table_begin(mut self, id: Int, heads: List[String], widths: List[Float32], h: Float32, rows: Int, rh: Float32, mut off: Float32) -> Tuple[Int, Int]:
        """Table at the cursor: a header (26 high) of `heads` over columns of `widths`, then a scroll list of `rows` rows
        of rh, h high, with its own offset `off`. Returns (first, end) like scroll_begin. Per visible row call table_row, then
        table_cell for each column; the data stays yours. Call table_end after the rows."""
        var x0, y0 = self.x, self.y
        var t = self.theme.copy()
        self.cols.clear()
        self.cols.append(0)
        for w in widths:
            self.cols.append(self.cols[len(self.cols) - 1] + w)
        var tw = self.cols[len(self.cols) - 1]
        self.rect(x0, y0, x0 + tw, y0 + 26, t.radius * 0.5, t.hover)
        for c in range(min(len(heads), len(widths))):
            var n = self.fit(heads[c], widths[c] - 14)
            _ = self.text(x0 + self.cols[c] + 10, y0 + (26 - self.asc(0)) * 0.5 - 1, StringSlice(unsafe_from_utf8=heads[c].as_bytes()[:n]), 0, t.muted)
        self.y = y0 + 28
        return self.scroll_begin(id, tw, h, rows, rh, off)

    def table_cell(mut self, k: Int, c: Int, s: StringSlice, color: UInt32 = 0):
        """Text of row k, column c, cut to the column width (color 0: the theme text)."""
        if c >= len(self.cols) - 1:
            return
        var n = self.fit(s, self.cols[c + 1] - self.cols[c] - 14)
        _ = self.text(self.vp[0] + self.cols[c] + 10, self.scroll_y(k) + (self.rh - self.asc(0)) * 0.5 - 1,
                      StringSlice(unsafe_from_utf8=s.as_bytes()[:n]), 0, color if color != 0 else self.theme.text)

    def table_end(mut self):
        self.scroll_end()

    def plot(mut self, vals: List[Float32], lo: Float32, hi: Float32, w: Float32, h: Float32, c: UInt32):
        """Line plot of vals (oldest first) with axes in a w x h box at the cursor; next row. Y labels at lo, middle, hi,
        X labels at sample 0 and the last. More samples than plot pixels are decimated to one point per pixel column: the
        sample of the bucket farthest from the previous point, so spikes survive. Values outside lo..hi stick to the edge."""
        var x0, y0 = self.x, self.y
        var t = self.theme.copy()
        var px0, px1, py0, py1 = x0 + 36, x0 + w - 8, y0 + 8, y0 + h - 20
        self.rect(x0, y0, x0 + w, y0 + h, t.radius * 0.5, t.surface)
        for g in range(3):
            var gy = py1 - (py1 - py0) * Float32(g) / 2
            self.line(px0, gy, px1, gy, 1, t.hover)
            var lab = num(lo + (hi - lo) * Float32(g) / 2)
            _ = self.text(px0 - 6 - self.tw(lab, 0), gy - self.asc(0) * 0.5 - 1, lab, 0, t.muted)
        var n = len(vals)
        _ = self.text(px0, py1 + 4, "0", 0, t.muted)
        var last = String(max(n - 1, 0))
        _ = self.text(px1 - self.tw(last, 0), py1 + 4, last, 0, t.muted)
        var cols = min(n, Int(px1 - px0))
        var span = max(hi - lo, 1e-6)
        var pv = Float32(0)
        var pxp = px0
        var pyp = Float32(0)
        for k in range(cols):
            var a = k * n // cols  # this column's samples: [a, b)
            var b = max((k + 1) * n // cols, a + 1)
            var v = vals[a]
            if k > 0:
                for j in range(a, b):
                    if abs(vals[j] - pv) > abs(v - pv):
                        v = vals[j]
            pv = v
            var xx = px0 + (px1 - px0) * Float32(k) / Float32(max(cols - 1, 1))
            var yy = py1 - (py1 - py0) * min(max((v - lo) / span, Float32(0)), Float32(1))
            if k > 0:
                self.line(pxp, pyp, xx, yy, 1.6, c)
            pxp, pyp = xx, yy
        self.y += h + 8

    def scroll_end(mut self):
        """Scrollbar thumb for the list just drawn; moves the cursor to the row under it."""
        var h, full = self.vp[3], Float32(self.srows) * self.rh
        self.unclip()
        if full > h:
            var x1 = self.vp[0] + self.vp[2] - 6
            var th = max(h * h / full, Float32(16))
            var ty = self.vp[1] + (h - th) * self.soff / (full - h)
            self.rect(x1, ty, x1 + 4, ty + th, 2, self.theme.muted)
        self.x, self.y = self.vp[0], self.vp[1] + h + 8

    def finish(mut self):
        """End the frame: a release ends the active widget (after every widget saw it); bin the draw list."""
        if self.inp.released & 1:
            self.active = 0
        self.d.finish()
