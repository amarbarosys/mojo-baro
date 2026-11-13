"""Presets and model panels for the baro face."""
from std.ffi import external_call
from std.os.path import exists
from std.pathlib import Path
from imkit.ui import Ui
from apply import ApplyPlan, plan, with_fit, apply_action, restart_cmd
from chat import wrap
from dash import Dash, short
from presets import Preset, load_presets, user_presets_path, apply_preset, save_user_preset, delete_user_preset
from settings import Settings
from widgets import control

comptime ID_PRESET = 9500
comptime ID_SAVE_NAME = 9600
comptime ID_SAVE_SUMMARY = 9601
comptime ID_SAVE = 9602
comptime ID_DELETE = 9603
comptime ID_APPLY = 9604
comptime ID_CONFIRM = 9605
comptime ID_CANCEL = 9606
comptime ID_MODEL_PATH = 9700
comptime ID_PACK = 9701
comptime ID_MODEL_LOOKUP = 9702


def shell_arg(s: String) -> String:
    var out = String("'")
    for c in s.codepoints():
        if c == Codepoint.ord("'"):
            out += "'\\''"
        else:
            out += String(c)
    return out + "'"


struct PresetsUi(Movable):
    var presets: List[Preset]
    var selected: Int
    var name: String
    var summary: String
    var message: String
    var confirm: Bool
    var blocked: Bool
    var plan_text: String
    var model_path: String
    var pack_path: String

    def __init__(out self) raises:
        self.presets = load_presets("docs/settings.toml")
        self.selected = -1
        self.name, self.summary, self.message = "", "", ""
        self.confirm = False
        self.blocked = False
        self.plan_text, self.model_path, self.pack_path = "", "", ""
        var user_path = user_presets_path()
        if exists(user_path):
            var users = load_presets(user_path)
            for i in range(len(users)):
                if users[i].group == "user":
                    self.presets.append(users[i].copy())


def preset_keys(s: Settings) -> List[String]:
    var out = List[String]()
    for it in s.items:
        out.append(it.key.copy())
    return out^


def apply_plan_text(s: Settings, pack: String) raises -> ApplyPlan:
    var keys = List[String]()
    var restart = List[String]()
    for it in s.items:
        keys.append(it.key.copy())
        if it.restart:
            restart.append(it.key.copy())
    var p = plan(s.applied, s.staged, keys, restart)
    return with_fit(p^, pack, ".work/face-apply-fit.txt")


def presets_pane(mut ui: Ui, mut s: Settings, mut p: PresetsUi, mut d: Dash, host: String, port: Int, x0: Float32, y0: Float32, x1: Float32, y1: Float32) raises:
    var t = ui.theme.copy()
    ui.panel(x0, y0, x1, y1, "Presets")
    var y = y0 + 38
    for group in [String("speed"), String("task"), String("fit"), String("user")]:
        _ = ui.text(x0 + 16, y, group, 1, t.muted)
        y += 24
        ui.x, ui.y = x0 + 16, y
        var slot = 0
        for i in range(len(p.presets)):
            if p.presets[i].group != group:
                continue
            ui.x = x0 + 16 + Float32(slot % 3) * 215
            ui.y = y + Float32(slot / 3) * 34
            if ui.button(ID_PRESET + i, p.presets[i].name, 136, 26, on=p.selected == i):
                p.selected = i
                p.message = p.presets[i].summary.copy()
                p.confirm = False
            ui.x += 142
            if ui.button(ID_PRESET + 256 + i, "Stage", 62, 26):
                _ = apply_preset(p.presets[i], s.staged)
                p.selected = i
                p.message = "Preset staged. Review changes with Apply."
                p.confirm = False
            slot += 1
        y += 34 * Float32((slot + 2) / 3) + 4
    ui.x, ui.y = x0 + 16, y
    _ = ui.text(ui.x, ui.y, p.message, 0, t.text)
    y += 24
    ui.x, ui.y = x0 + 16, y
    var name = p.name.copy()
    if ui.field(ID_SAVE_NAME, name, 210):
        p.name = name
    ui.x = x0 + 240
    var summary = p.summary.copy()
    if ui.field(ID_SAVE_SUMMARY, summary, 280):
        p.summary = summary
    ui.x = x0 + 16
    ui.y += 34
    if ui.button(ID_SAVE, "Save staged preset", 180, 28):
        try:
            save_user_preset(user_presets_path(), p.name, p.summary, s.staged, preset_keys(s))
            p.presets = load_presets("docs/settings.toml")
            var users = load_presets(user_presets_path())
            for i in range(len(users)):
                if users[i].group == "user":
                    p.presets.append(users[i].copy())
            p.message = "User preset saved."
        except e:
            p.message = "Save failed: " + String(e)
    ui.x += 192
    if ui.button(ID_DELETE, "Delete selected user preset", 220, 28, on=p.selected >= 0 and p.presets[p.selected].group == "user"):
        try:
            if p.selected >= 0 and delete_user_preset(user_presets_path(), p.presets[p.selected].name):
                p.message = "User preset deleted."
                p.selected = -1
            else:
                p.message = "No user preset selected."
        except e:
            p.message = "Delete failed: " + String(e)
    y += 42
    ui.x, ui.y = x0 + 16, y
    if p.confirm:
        _ = ui.text(ui.x, ui.y, "Confirm Apply", 1, t.warn)
        ui.y += 24
        for line in wrap(ui, p.plan_text, x1 - x0 - 38):
            _ = ui.text(ui.x, ui.y, line, 0, t.text)
            ui.y += 18
        ui.x = x0 + 16
        ui.y += 10
        if ui.button(ID_CONFIRM, "Confirm", 100, 28, on=not p.blocked, color=t.good if not p.blocked else t.muted) and not p.blocked:
            var ap = apply_plan_text(s, p.pack_path)
            if len(ap.restart) > 0:
                var model = p.model_path if p.model_path else d.model
                var start = "setsid " + restart_cmd(model, port, s.staged, ap.restart) + " > .work/gui-front.log 2>&1 & echo $! > .work/gui-front.pid"
                p.message = apply_action("bash face/stop-engine.sh", start, host, port, 90, d.fail)
            else:
                p.message = "front: ok, live settings applied; fail " + String(d.fail)
            if p.message.startswith("front: ok"):
                for k in s.staged:
                    s.applied[k] = s.staged[k]
                s.staged = Dict[String, String]()
                try:
                    d.refresh(host, port)
                except:
                    pass
            p.confirm = False
        ui.x += 112
        if ui.button(ID_CANCEL, "Cancel", 100, 28):
            p.confirm = False
    elif ui.button(ID_APPLY, "Apply staged settings", 190, 30, on=s.staged_count() > 0):
        var ap = apply_plan_text(s, p.pack_path)
        p.plan_text = ap.text(s.applied, s.staged)
        p.blocked = ap.blocked()
        p.confirm = True


def model_pane(mut ui: Ui, mut s: Settings, mut p: PresetsUi, d: Dash, x0: Float32, y0: Float32, x1: Float32, y1: Float32) raises:
    var t = ui.theme.copy()
    ui.panel(x0, y0, x1, y1, "Model")
    ui.x, ui.y = x0 + 16, y0 + 42
    _ = ui.text(ui.x, ui.y, "GGUF path", 1, t.muted)
    ui.y += 26
    var model = p.model_path.copy()
    if ui.field(ID_MODEL_PATH, model, 440):
        p.model_path = model
    ui.x += 450
    if ui.button(ID_MODEL_LOOKUP, "Find pack", 100, 28):
        var cmd = "tools/baro pack-path " + shell_arg(p.model_path) + " > .work/face-model-path.txt 2>&1"
        _ = external_call["system", Int32](cmd.as_c_string_slice().unsafe_ptr())
        var result = Path(".work/face-model-path.txt").read_text()
        if result.startswith("PACK FOUND: "):
            p.pack_path = String(result[byte=12:].split("\n")[0])
            p.message = "Pack found. No build started."
        else:
            p.pack_path = ""
            p.message = String(result.split("\n")[0])
    ui.x, ui.y = x0 + 16, y0 + 90
    _ = ui.text(ui.x, ui.y, "Pack path", 1, t.muted)
    ui.x += 100
    var pack = p.pack_path.copy()
    if ui.field(ID_PACK, pack, 440):
        p.pack_path = pack
    _ = ui.text(x0 + 16, y0 + 132, "Model card", 1, t.muted)
    _ = ui.text(x0 + 16, y0 + 158, "model " + short(d.model if d.model else p.model_path, 42), 0, t.text)
    _ = ui.text(x0 + 16, y0 + 182, "pack " + short(d.pack if d.pack else p.pack_path, 42), 0, t.text)
    _ = ui.text(x0 + 16, y0 + 206, d.limits if d.limits else String("limits unavailable"), 0, t.text)
    _ = ui.text(x0 + 16, y0 + 240, p.message, 0, t.warn)
    var y = y0 + 290
    for i in range(len(s.items)):
        if not s.shown(i):
            continue
        var it = s.items[i].copy()
        _ = ui.text(x0 + 16, y + 4, it.label, 1, t.text)
        control(ui, s, i, x0 + 240, y)
        _ = ui.text(x0 + 16, y + 36, it.summary, 0, t.muted)
        y += 66
