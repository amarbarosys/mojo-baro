# VENDORED COPY of amarbaro/mojo-imkit (Apache-2.0, see imkit/LICENSE), tag v0.9.0 (eb91210),
# ~/Projects/mojo/mojo-imkit, file src/imkit/window.mojo. This repo keeps a real file
# rather than an -I path outside the tree, because a clone must build without anything
# outside it. Sync by hand if the upstream changes; tools/ci-checks.sh compares the two
# against the tag when the upstream is present and says so when they drift.
#
"""SDL3 window for an app's frames (docs/design.md): one resizable window, one streaming RGBA texture at the
render size W x H, letterboxed by SDL (mouse coordinates come back in W x H pixels), vsync. SDL3 is the system library,
reached through OwnedDLHandle (no binding, no build flag). Event layouts: SDL 3.4 (offsets checked with offsetof).

    var win = Window("App", W, H, UW, UH)   # UW x UH: an RGBA UI layer blended over the scene (ui.mojo)
    var inp = win.poll()            # every frame: mouse, wheel, buttons, keys, quit (in UI pixels with a layer)
    win.present(Int(out_h.unsafe_ptr()), Int(ui_h.unsafe_ptr()))   # frames in host memory, row-major RGBA
"""
from std.ffi import OwnedDLHandle, external_call

comptime EV_QUIT = 0x100
comptime EV_CLOSE = 0x210
comptime EV_KEY_DOWN = 0x300
comptime EV_KEY_UP = 0x301
comptime EV_TEXT = 0x303
comptime EV_MOTION = 0x400
comptime EV_BUTTON_DOWN = 0x401
comptime EV_BUTTON_UP = 0x402
comptime EV_WHEEL = 0x403
comptime SCRIPT_MOUSE = 0x5C819700  # `which` of push()ed mouse events: SDL gives a real mouse 0 (non-relative mode), touch and pen their own ids


struct Input(Copyable, Movable):
    """One frame's input, in render pixels. Mouse buttons are bits (left 1, middle 2, right 4); keys are SDL scancodes
    below 128 as bits of two words (pressed: went down this frame); text is what the user typed (layout-aware, UTF-8)."""
    var x: Float32
    var y: Float32
    var dx: Float32
    var dy: Float32
    var wheel: Float32
    var down: UInt32
    var pressed: UInt32
    var released: UInt32
    var keys: SIMD[DType.uint64, 2]
    var quit: Bool
    var text: String  # typed text this frame (SDL text input, after text_input(True)); merge() joins it

    def __init__(out self, x: Float32, y: Float32, dx: Float32, dy: Float32, wheel: Float32, down: UInt32, pressed: UInt32,
                 released: UInt32, keys: SIMD[DType.uint64, 2], quit: Bool, text: String = ""):
        self.x, self.y, self.dx, self.dy, self.wheel, self.down, self.pressed, self.released = x, y, dx, dy, wheel, down, pressed, released
        self.keys, self.quit, self.text = keys, quit, text.copy()

    def merge(mut self, o: Input):
        """Fold a later frame's input in: newest position and buttons held, presses / releases / keys OR'd, motion and
        wheel summed (a UI drawn every few iterations still sees every click)."""
        self.x, self.y, self.down = o.x, o.y, o.down
        self.dx += o.dx
        self.dy += o.dy
        self.wheel += o.wheel
        self.pressed |= o.pressed
        self.released |= o.released
        self.keys |= o.keys
        self.quit = self.quit or o.quit
        self.text += o.text

    def key(self, scancode: Int) -> Bool:
        return (self.keys[scancode >> 6] >> UInt64(scancode & 63)) & 1 == 1

    def logical(self, scale: Float32) -> Input:
        """The same input with the mouse in logical units: a UI drawn at `scale` layer pixels per logical unit (DPI 1.5: 1.5)
        gets x, y, dx, dy / scale back. Buttons, keys, wheel and text carry over."""
        var o = self.copy()
        o.x, o.y, o.dx, o.dy = self.x / scale, self.y / scale, self.dx / scale, self.dy / scale
        return o^


def readable(fd: Int, ms: Int) -> Bool:
    """poll(2) one fd for input for up to ms milliseconds."""
    var pfd = SIMD[DType.int32, 2](Int32(fd), 1)  # struct pollfd: fd, events POLLIN (revents in the high half)
    return external_call["poll", Int32](UnsafePointer(to=pfd), UInt64(1), Int32(ms)) > 0


struct Window(Movable):
    var sdl: OwnedDLHandle
    var win: Int
    var ren: Int
    var tex: Int
    var utex: Int  # the UI layer (0: none)
    var w: Int
    var h: Int
    var uw: Int
    var ev: List[UInt8]
    var x: Float32
    var y: Float32
    var down: UInt32
    var held: SIMD[DType.uint64, 2]  # keys down now (scancodes < 128): key_down(); press / release events keep it
    var keep: List[List[UInt8]]  # text of push()ed EV_TEXT events, alive until the next poll() has read them
    var scripted: Bool  # True: only push()ed mouse events count (which = SCRIPT_MOUSE); a real mouse is ignored (scripted checks)

    def __init__(out self, title: String, w: Int, h: Int, uw: Int = 0, uh: Int = 0) raises:
        self.sdl = OwnedDLHandle("libSDL3.so.0")
        self.w, self.h, self.x, self.y, self.down, self.scripted = w, h, 0, 0, 0, False
        self.held = SIMD[DType.uint64, 2](0)
        self.ev = List[UInt8](length=128, fill=0)
        self.keep = List[List[UInt8]]()
        if not self.sdl.get_function[Bool]("SDL_Init")(UInt32(0x20)):  # SDL_INIT_VIDEO
            raise Error("SDL_Init failed")
        var lw, lh = (uw, uh) if uw else (w, h)  # logical size: the UI layer's when there is one
        var scale = max(1, 1280 // lw)  # a small logical size opens at least 1280 wide; SDL scales it
        var t = title.copy()
        self.win = self.sdl.get_function[Int]("SDL_CreateWindow")(t.as_c_string_span().ptr(), Int32(lw * scale), Int32(lh * scale), UInt64(0x20 | 0x2000))
        self.ren = self.sdl.get_function[Int]("SDL_CreateRenderer")(self.win, Int(0))
        if self.win == 0 or self.ren == 0:
            raise Error("SDL window or renderer failed")
        _ = self.sdl.get_function[Bool]("SDL_SetRenderVSync")(self.ren, Int32(1))
        _ = self.sdl.get_function[Bool]("SDL_SetRenderLogicalPresentation")(self.ren, Int32(lw), Int32(lh), Int32(2))  # letterbox
        self.tex = self.sdl.get_function[Int]("SDL_CreateTexture")(self.ren, UInt32(0x16762004), Int32(1), Int32(w), Int32(h))  # RGBA32, streaming
        _ = self.sdl.get_function[Bool]("SDL_SetTextureBlendMode")(self.tex, UInt32(0))  # the scene is opaque (its alpha marks animals)
        self.utex = 0
        self.uw = uw
        if uw:
            self.utex = self.sdl.get_function[Int]("SDL_CreateTexture")(self.ren, UInt32(0x16762004), Int32(1), Int32(uw), Int32(uh))
            _ = self.sdl.get_function[Bool]("SDL_SetTextureBlendMode")(self.utex, UInt32(1))  # SDL_BLENDMODE_BLEND
        if self.tex == 0 or (uw and self.utex == 0):
            raise Error("SDL texture failed")

    def scale(self) raises -> Float32:
        """Display scale of the window (1.0, 1.5, 2.0 ...: pixel density times the OS content scale; 1 if SDL has none)."""
        var s = self.sdl.get_function[Float32]("SDL_GetWindowDisplayScale")(self.win)
        return s if s > 0 else 1

    def drawable(self) raises -> Tuple[Int, Int]:
        """The window's drawable size in physical pixels (0, 0 if SDL cannot tell)."""
        var wh = List[Int32](length=2, fill=0)
        _ = self.sdl.get_function[Bool]("SDL_GetWindowSizeInPixels")(self.win, Int(wh.unsafe_ptr()), Int(wh.unsafe_ptr().unsafe_offset(1)))
        return (Int(wh[0]), Int(wh[1]))

    def poll(mut self) raises -> Input:
        var inp = Input(self.x, self.y, 0, 0, 0, self.down, 0, 0, SIMD[DType.uint64, 2](0), False)
        var e = self.ev.unsafe_ptr()
        while self.sdl.get_function[Bool]("SDL_PollEvent")(e):
            _ = self.sdl.get_function[Bool]("SDL_ConvertEventToRenderCoordinates")(self.ren, e)
            var kind = Int(e.unsafe_bitcast[UInt32]()[0])
            var f = e.unsafe_bitcast[Float32]()
            if self.scripted and EV_MOTION <= kind <= EV_WHEEL and e.unsafe_bitcast[UInt32]()[5] != SCRIPT_MOUSE:
                continue  # a real mouse, touch or pen while a script drives the window
            if kind == EV_QUIT or kind == EV_CLOSE:
                inp.quit = True
            elif kind == EV_MOTION:
                inp.x, inp.y = f[7], f[8]
                inp.dx += f[9]
                inp.dy += f[10]
            elif kind == EV_BUTTON_DOWN or kind == EV_BUTTON_UP:
                var bit = UInt32(1) << UInt32(Int(e[24]) - 1)
                inp.x, inp.y = f[7], f[8]
                if kind == EV_BUTTON_DOWN:
                    inp.down |= bit
                    inp.pressed |= bit
                else:
                    inp.down &= ~bit
                    inp.released |= bit
            elif kind == EV_WHEEL:
                inp.wheel += f[7]
            elif kind == EV_TEXT:  # SDL_TextInputEvent: the char* at offset 24, owned by SDL until the next poll
                var p = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=e.unsafe_bitcast[Int]()[3])
                var n = Int(external_call["strnlen", UInt64](p, UInt64(256)))  # bounded: never scan past 256 bytes
                if inp.text.byte_length() + n <= 1024:  # one frame's typing; drop the rest rather than grow
                    inp.text += String(unsafe_from_utf8=Span(unsafe_ptr=p, length=n))
            elif kind == EV_KEY_DOWN or kind == EV_KEY_UP:
                var sc = Int(e.unsafe_bitcast[UInt32]()[6])
                if sc < 128:
                    var bit = UInt64(1) << UInt64(sc & 63)
                    if kind == EV_KEY_UP:
                        self.held[sc >> 6] &= ~bit
                    elif e[37] == 0:  # a press, not a repeat
                        inp.keys[sc >> 6] |= bit
                        self.held[sc >> 6] |= bit
        self.x, self.y, self.down = inp.x, inp.y, inp.down
        self.keep.clear()
        return inp^

    def key_down(self, scancode: Int) -> Bool:
        """Is the key held now (between its press and release events)?"""
        return scancode < 128 and (self.held[scancode >> 6] >> UInt64(scancode & 63)) & 1 == 1

    def text_input(mut self, on: Bool) raises:
        """Start or stop SDL text events (Input.text); keys keep arriving either way. Off when no text field has focus."""
        if on:
            _ = self.sdl.get_function[Bool]("SDL_StartTextInput")(self.win)
        else:
            _ = self.sdl.get_function[Bool]("SDL_StopTextInput")(self.win)

    def present(mut self, pixels: Int, ui: Int = 0) raises:
        """Show a W x H RGBA frame and, with a layer, the UW x UH UI over it (host memory, row-major, 4 bytes per pixel)."""
        _ = self.sdl.get_function[Bool]("SDL_UpdateTexture")(self.tex, Int(0), pixels, Int32(self.w * 4))
        _ = self.sdl.get_function[Bool]("SDL_RenderClear")(self.ren)
        _ = self.sdl.get_function[Bool]("SDL_RenderTexture")(self.ren, self.tex, Int(0), Int(0))
        if ui and self.utex:
            _ = self.sdl.get_function[Bool]("SDL_UpdateTexture")(self.utex, Int(0), ui, Int32(self.uw * 4))
            _ = self.sdl.get_function[Bool]("SDL_RenderTexture")(self.ren, self.utex, Int(0), Int(0))
        _ = self.sdl.get_function[Bool]("SDL_RenderPresent")(self.ren)

    def push(mut self, kind: Int, a: Int = 0, x: Float32 = 0, y: Float32 = 0, rx: Float32 = 0, ry: Float32 = 0, which: Int = SCRIPT_MOUSE, text: String = "") raises:
        """Queue a synthetic event (tests, scripted checks): motion (x, y, rx, ry), button a down/up at (x, y), wheel
        (x = amount), key a (scancode) down or up, EV_TEXT with `text`. Mouse events carry `which` (SCRIPT_MOUSE; 0 looks like a real mouse)."""
        for k in range(128):
            self.ev[k] = 0
        var e = self.ev.unsafe_ptr()
        e.unsafe_bitcast[UInt32]()[0] = UInt32(kind)
        var f = e.unsafe_bitcast[Float32]()
        if EV_MOTION <= kind <= EV_WHEEL:
            e.unsafe_bitcast[UInt32]()[5] = UInt32(which)
        if kind == EV_MOTION or kind == EV_BUTTON_DOWN or kind == EV_BUTTON_UP:
            e[24] = UInt8(a)
            e[25] = UInt8(1 if kind == EV_BUTTON_DOWN else 0)
            f[7], f[8], f[9], f[10] = x, y, rx, ry
        elif kind == EV_WHEEL:
            f[7] = x
        elif kind == EV_KEY_DOWN or kind == EV_KEY_UP:
            e.unsafe_bitcast[UInt32]()[6] = UInt32(a)
            e[36] = 1 if kind == EV_KEY_DOWN else 0
        elif kind == EV_TEXT:
            var buf = List[UInt8](text.as_bytes())
            buf.append(0)
            self.keep.append(buf^)  # SDL holds this pointer until the event is polled, so the bytes live until poll() returns
            e.unsafe_bitcast[Int]()[3] = Int(self.keep[len(self.keep) - 1].unsafe_ptr())
        _ = self.sdl.get_function[Bool]("SDL_PushEvent")(e)

    def close(mut self) raises:
        if self.utex:
            self.sdl.get_function[NoneType]("SDL_DestroyTexture")(self.utex)
        self.sdl.get_function[NoneType]("SDL_DestroyTexture")(self.tex)
        self.sdl.get_function[NoneType]("SDL_DestroyRenderer")(self.ren)
        self.sdl.get_function[NoneType]("SDL_DestroyWindow")(self.win)
        self.sdl.get_function[NoneType]("SDL_Quit")()
