"""F4 gate renderer: every settings tab as one headless frame (no window, no display), PPM files for the OCR gate.

  settings_gate DIR       writes DIR/tab-<n>.ppm for n = 1..5 (Sampling, Engine, Model, Front, Presets), plus tab-2a.ppm (Engine with
                          Advanced on, a staged BARO_TMAX and its help open) and tab-1b.ppm (Sampling basic, nothing staged)
face/settings_gate.sh builds this, runs it under gpu-wait, and reads the frames back with tesseract.
"""
from std.sys import argv
from imkit.draw import UH, UW
from imkit.font import Font
from imkit.layer import Layer
from imkit.theme import amarbaro
from imkit.ui import Ui
from imkit.window import Input
from max.gpu.host import DeviceContext
from app import write_ppm, BG
from settings import Settings
from widgets import settings_pane, settings_bar
from presets_ui import PresetsUi, presets_pane, model_pane
from dash import Dash
from apply import plan


def frame(mut ui: Ui, mut layer: Layer, mut s: Settings, mut p: PresetsUi, mut d: Dash, bg: List[UInt32], path: String) raises:
    var keys = SIMD[DType.uint64, 2](0)
    for _ in range(2):  # a second frame settles the springs
        ui.begin(Input(0, 0, 0, 0, 0, 0, 0, 0, keys, False))
        if s.tab == 5:
            presets_pane(ui, s, p, d, "127.0.0.1", 8080, 20, 52, Float32(UW - 20), Float32(UH - 56))
        elif s.tab == 3:
            model_pane(ui, s, p, d, 20, 52, Float32(UW - 20), Float32(UH - 56))
        else:
            settings_pane(ui, s, 20, 52, Float32(UW - 20), Float32(UH - 56))
        settings_bar(ui, s, 20, 12)
        ui.finish()
        layer.render(ui)
    if ui.d.dropped:
        raise Error("FAIL settings_gate: " + String(ui.d.dropped) + " draw commands dropped")
    write_ppm(path, layer, bg)


def main() raises:
    var a = argv()
    if len(a) != 2:
        raise Error("FAIL usage: settings_gate DIR")
    var dir = String(a[1])
    var ui = Ui(Font("/usr/share/fonts/TTF/Roboto-Medium.ttf", [13, 16, 22]))
    ui.set_theme(amarbaro())
    var layer = Layer(DeviceContext(), ui)
    var bg = List[UInt32](length=UW * UH, fill=BG)
    var s = Settings.load("face/testdata/settings-sample.toml")
    var p = PresetsUi()
    var d = Dash()
    p.model_path = "models/example.gguf"
    p.pack_path = "/cache/packs/example"
    d.model = "models/example.gguf"
    d.pack = "/cache/packs/example"
    d.limits = "tmax 32768 seqs 1 spec_k 2 tier 0"
    var current = Dict[String, String]()
    var staged = Dict[String, String]()
    current["BARO_TMAX"] = "16384"
    staged["BARO_TMAX"] = "32768"
    var plan_keys = List[String]()
    var restart_keys = List[String]()
    plan_keys.append("BARO_TMAX")
    restart_keys.append("BARO_TMAX")
    var ap = plan(current, staged, plan_keys, restart_keys)
    ap.verdict = "FITS: gate fixture"
    p.plan_text = ap.text(current, staged)
    p.confirm = True
    s.tab = 1
    frame(ui, layer, s, p, d, bg, dir + "/tab-1b.ppm")
    s.tab = 2
    s.advanced = True
    s.stage(2, "32768")
    s.sel = 2
    frame(ui, layer, s, p, d, bg, dir + "/tab-2a.ppm")
    s.advanced = False
    s.staged = Dict[String, String]()
    s.sel = -1
    for t in range(1, 6):
        s.tab = t
        frame(ui, layer, s, p, d, bg, dir + "/tab-" + String(t) + ".ppm")
    print("PASS settings_gate: 7 frames in", dir)
