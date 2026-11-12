"""baro face, skeleton (F0): one window, chat pane on the left, engine panel on the right, placeholder text in both.
It asks the front's /health once a second and shows the answer in the status line. Nothing else yet.

  app [--front HOST:PORT]               window (default front 127.0.0.1:8080); close it or press Escape to exit
      [--fps N] [--frames M] [--stats]  cap the loop at N frames a second (default 30; 0: only the window's vsync); stop after M frames
                                        (SDL_VIDEODRIVER=offscreen runs it with no window on the desktop; face/window-cost.sh does)
  app --shots DIR [--front HOST:PORT]   headless: 3 frames to DIR/face-<n>.ppm (no window, no display), then exit

build: mojo build --target-accelerator amdgpu:gfx1100 -I . -I face face/app.mojo -o .work/face   (tools/baro gui does it)
System libraries, checked at start with a named error: libSDL3.so.0 (window mode only), libfreetype.so.6.
"""
from std.ffi import OwnedDLHandle
from std.os import getenv
from std.sys import argv
from std.time import perf_counter_ns, sleep
from max.gpu.host import DeviceContext
from imkit.draw import UH, UW
from imkit.font import Font
from imkit.layer import Layer
from imkit.theme import amarbaro
from imkit.ui import Ui
from imkit.window import Input, Window
from json import parse
from net import http_get

comptime SPLIT = Float32(840)  # the chat pane ends here; the engine panel starts 20 px later
comptime BG = UInt32(0xFF1A140E)  # opaque ABGR: the amarbaro backdrop under the translucent panels


def need_lib(name: String, pkg: String) raises:
    try:
        _ = OwnedDLHandle(name)
    except:
        raise Error("FAIL missing system library " + name + " (Arch: pacman -S " + pkg + ")")


def health_line(host: String, port: Int) -> String:
    """One word for the status line: what /health said, or why it said nothing."""
    try:
        var r = http_get(host, port, "/health")
        if r.status == 0:
            return "front: unreachable (" + r.why + ")"
        var j = parse(r.body)
        var st = j.get("engine_state").str() if j.has("engine_state") else String("?")
        var q = j.get("queue").int() if j.has("queue") else 0
        if r.status == 200:
            return "front: ok, engine " + st + ", queue " + String(q)
        return "front: HTTP " + String(r.status) + ", engine " + st
    except e:
        return "front: bad reply (" + String(e) + ")"


def draw(mut ui: Ui, inp: Input, status: String, frame: Int):
    ui.begin(inp)
    var t = ui.theme.copy()
    ui.panel(20, 16, SPLIT, Float32(UH - 56), "Chat")
    _ = ui.text(40, 70, "Chat placeholder: the conversation will appear here.", 2, t.text)
    _ = ui.text(40, 110, "Messages, input and streaming arrive in F1.", 1, t.muted)
    ui.panel(SPLIT + 20, 16, Float32(UW - 20), Float32(UH - 56), "Engine")
    _ = ui.text(SPLIT + 40, 70, "Engine placeholder: live state.", 2, t.text)
    _ = ui.text(SPLIT + 40, 110, "The panel arrives in F2.", 1, t.muted)
    _ = ui.text(24, Float32(UH - 40), status, 1, t.text)
    _ = ui.text(Float32(UW - 200), Float32(UH - 40), "frame " + String(frame), 1, t.muted)
    ui.finish()


def write_ppm(path: String, layer: Layer, bg: List[UInt32]) raises:
    """The layer blended (straight alpha) over the backdrop, as a binary PPM (the gate converts it to PNG)."""
    var ppm = List[UInt8]()
    for c in ("P6\n" + String(UW) + " " + String(UH) + "\n255\n").as_bytes():
        ppm.append(c)
    for i in range(UW * UH):
        var l = layer.pixels[i]
        var a = Int(l >> 24)
        var s = bg[i]
        for sh in [0, 8, 16]:
            ppm.append(UInt8((Int((s >> UInt32(sh)) & 255) * (255 - a) + Int((l >> UInt32(sh)) & 255) * a) // 255))
    with open(path, "w") as o:
        o.write_bytes(Span(ppm))


def main() raises:
    var host = String("127.0.0.1")
    var port = 8080
    var shots = String("")
    var fps = 30  # chosen by number, face/window-cost.sh: 60 fps cost 6 to 7 of 20 prompts about 13%, 30 and 15 cost 1
    var max_frames = 0
    var stats = False
    var a = argv()
    var i = 1
    while i < len(a):
        var k = String(a[i])
        if k == "--front" and i + 1 < len(a):
            var hp = String(a[i + 1]).split(":")
            host = String(hp[0])
            if len(hp) > 1:
                port = Int(hp[1])
            i += 2
        elif k == "--fps" and i + 1 < len(a):
            fps = Int(String(a[i + 1]))
            i += 2
        elif k == "--frames" and i + 1 < len(a):
            max_frames = Int(String(a[i + 1]))
            i += 2
        elif k == "--stats":
            stats = True
            i += 1
        elif k == "--shots" and i + 1 < len(a):
            shots = String(a[i + 1])
            i += 2
        else:
            raise Error("FAIL usage: app [--front HOST:PORT] [--shots DIR] [--fps N] [--frames M] [--stats]")
    need_lib("libfreetype.so.6", "freetype2")
    if not shots:
        need_lib("libSDL3.so.0", "sdl3")
    var ui = Ui(Font(getenv("IMKIT_FONT", "/usr/share/fonts/TTF/Roboto-Medium.ttf"), [13, 16, 22]))
    ui.set_theme(amarbaro())
    var layer = Layer(DeviceContext(), ui)
    var bg = List[UInt32](length=UW * UH, fill=BG)
    var keys = SIMD[DType.uint64, 2](0)
    if shots:
        var status = health_line(host, port)
        for f in range(1, 4):
            draw(ui, Input(0, 0, 0, 0, 0, 0, 0, 0, keys, False), status, f)
            layer.render(ui)
            if ui.d.dropped:
                raise Error("FAIL face: " + String(ui.d.dropped) + " draw commands dropped")
            write_ppm(shots + "/face-" + String(f) + ".ppm", layer, bg)
            print("frame", f, "commands", ui.d.n, "status:", status)
        return
    var win = Window("baro", UW, UH, UW, UH)
    var status = health_line(host, port)
    var last = Int(perf_counter_ns())
    var frame = 0
    var period = 1_000_000_000 // fps if fps > 0 else 0
    var next = Int(perf_counter_ns())
    var t0 = next
    while True:
        var inp = win.poll()
        if inp.quit or (inp.keys[0] >> 41) & 1 == 1:  # scancode 41: Escape
            break
        if Int(perf_counter_ns()) - last > 1_000_000_000:
            status = health_line(host, port)
            last = Int(perf_counter_ns())
        frame += 1
        if stats and frame % 120 == 0:  # measured frame rate, read back by face/window-cost.sh
            print("fps", Float64(frame) * 1e9 / Float64(Int(perf_counter_ns()) - t0), "frames", frame, flush=True)
        draw(ui, inp, status, frame)
        layer.render(ui)
        win.present(Int(bg.unsafe_ptr()), Int(layer.pixels.unsafe_ptr()))
        if max_frames > 0 and frame >= max_frames:
            break
        if period > 0:
            next += period
            var now = Int(perf_counter_ns())
            if next > now:
                sleep(Float64(next - now) / 1e9)
            else:
                next = now  # behind: do not burst to catch up
    win.close()
    print("frames", frame)
