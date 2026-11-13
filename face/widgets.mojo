"""baro face, settings widgets (F4): the tab bar and the settings pane, drawn on imkit primitives.

  settings_bar(ui, s, x, y)                    the tab row: Chat, Sampling, Engine, Model, Front, Presets
  settings_pane(ui, s, x0, y0, x1, y1)         one group's settings: a control and a one-line summary per row, the help panel
                                               of the selected row on the right, a Basic / Advanced toggle, staged marks

A control per type: slider (int and float with a min and max), toggle button (bool), one button per choice (enum), text
field (path, string, and a number with no range). imkit has no tab bar, dropdown inside a scroll list, or tooltip: built
here from buttons and the row click (reported to the imkit owner in the lane report).
"""
from imkit.ui import Ui
from chat import wrap
from settings import Settings, NTABS, tab_name, tab_group

comptime ID_TAB = 9290  # + tab index
comptime ID_ADV = 9299
comptime ID_ROW = 9300  # + 10 * item + sub: sub 0 control, 1..8 enum choices, 9 row select
comptime RH = Float32(86)
comptime SLIDER_W = Float32(260)


def settings_bar(mut ui: Ui, mut s: Settings, x: Float32, y: Float32):
    ui.x, ui.y = x, y
    for t in range(NTABS):
        if ui.button(ID_TAB + t, tab_name(t), 84, 28, on=s.tab == t):
            s.tab = t
            s.off, s.sel = 0, -1


def num_text(v: Float64, is_int: Bool) -> String:
    """A slider value as the TOML spells it: whole number for int, at most two decimals for float."""
    if is_int:
        return String(Int(v + 0.5 if v >= 0 else v - 0.5))
    return String(Float64(Int(v * 100 + (0.5 if v >= 0 else -0.5))) / 100)


def num(s: String, dflt: Float64) -> Float64:
    try:
        return Float64(s)
    except:
        return dflt


def control(mut ui: Ui, mut s: Settings, i: Int, x: Float32, y: Float32):
    """The control of item i at (x, y); a change stages the value and selects the row."""
    var it = s.items[i].copy()  # a copy: staging below mutates s
    var id = ID_ROW + 10 * i
    var cur = s.value(i)
    ui.x, ui.y = x, y
    if it.type == "bool":
        if ui.button(id, "on" if cur == "true" else "off", 70, 26, on=cur == "true"):
            s.stage(i, "false" if cur == "true" else "true")
            s.sel = i
    elif it.type == "enum":
        for c in range(min(len(it.choices), 8)):
            if ui.button(id + 1 + c, it.choices[c], 24 + 9 * Float32(it.choices[c].byte_length()), 26, on=cur == it.choices[c]):
                s.stage(i, it.choices[c])
                s.sel = i
    elif (it.type == "int" or it.type == "float") and it.ranged:
        var v = num(cur, it.lo)
        var nv = Float64(ui.slider(id, Float32(v), Float32(it.lo), Float32(it.hi), SLIDER_W))
        var txt = num_text(nv, it.type == "int")
        _ = ui.text(x + SLIDER_W + 20, y + 4, cur, 0, ui.theme.text)
        if txt != cur and ui.active == id:
            s.stage(i, txt)
            s.sel = i
    else:
        var buf = cur.copy()
        if ui.field(id, buf, 300):
            s.stage(i, buf)
            s.sel = i


def help_panel(mut ui: Ui, s: Settings, x: Float32, y: Float32, w: Float32, h: Float32):
    var t = ui.theme.copy()
    ui.rect(x, y, x + w, y + h, t.radius * 0.5, t.surface)
    if s.sel < 0 or s.sel >= len(s.items):
        _ = ui.text(x + 14, y + 14, "Select a setting for its help.", 0, t.muted)
        return
    ref it = s.items[s.sel]
    ui.x, ui.y = x + 14, y + 12
    ui.heading(it.label, 1, w - 28)
    _ = ui.text(ui.x, ui.y, it.summary, 0, t.text)
    ui.y += 28
    for line in wrap(ui, it.help, w - 28):
        if ui.y > y + h - 150:
            break
        _ = ui.text(ui.x, ui.y, line, 0, t.text)
        ui.y += 20
    ui.y = y + h - 134
    ui.separator(w - 28)
    var rng = String("")
    if it.type == "enum":
        for c in it.choices:
            rng += ("," if rng.byte_length() > 0 else "") + c
    elif it.ranged:
        rng = num_text(it.lo, it.type == "int") + " to " + num_text(it.hi, it.type == "int")
    else:
        rng = "any " + it.type
    ui.kv("Key", it.key, w - 28)
    ui.kv("Default", it.default if it.default.byte_length() > 0 else String("(empty)"), w - 28)
    ui.kv("Range", rng, w - 28)
    ui.kv("Restart", "yes, Apply restarts the engine" if it.restart else "no, takes effect at once", w - 28)
    ui.kv("Status", it.status, w - 28)


def settings_pane(mut ui: Ui, mut s: Settings, x0: Float32, y0: Float32, x1: Float32, y1: Float32):
    var t = ui.theme.copy()
    ui.panel(x0, y0, x1, y1, tab_name(s.tab))
    var rows = List[Int]()
    if tab_group(s.tab).byte_length() == 0:
        _ = ui.text(x0 + 16, y0 + 44, "Presets: built in the next step (F5).", 0, t.muted)
        return
    for i in range(len(s.items)):
        if s.shown(i):
            rows.append(i)
    ui.x, ui.y = x0 + 16, y0 + 36
    if ui.button(ID_ADV, "Advanced" if s.advanced else "Basic", 100, 26, on=s.advanced):
        s.advanced = not s.advanced
        s.off = 0
    _ = ui.text(x0 + 140, y0 + 40, String(len(rows)) + " settings, " + String(s.staged_count()) + " staged, " + String(s.restart_count()) + " need a restart", 0, t.muted)
    var lw = (x1 - x0) * 0.6
    var top = y0 + 72
    var lh = y1 - 12 - top
    ui.x, ui.y = x0 + 16, top
    var rng = ui.scroll_begin(ID_ROW - 1, lw - 16, lh, len(rows), RH, s.off)
    for k in range(rng[0], rng[1]):
        var i = rows[k]
        var it = s.items[i].copy()
        var ry = ui.scroll_y(k)
        var hc = ui.clickable(ID_ROW + 10 * i + 9, x0 + 20, max(ry, top), x0 + 16 + lw - 28, min(ry + 24, top + lh))
        if hc[1]:
            s.sel = i
        if s.sel == i:
            ui.rect(x0 + 18, ry + 2, x0 + 20, ry + RH - 6, 1, t.accent)
        _ = ui.text(x0 + 28, ry + 4, it.label, 1, t.text)
        var lx = x0 + 28 + ui.tw(it.label, 1) + 10
        ui.x, ui.y = lx, ry + 4
        if it.status != "stable":
            _ = ui.badge(it.status, t.warn, 0)
        if s.is_staged(i):
            _ = ui.text(lx + (Float32(110) if it.status != "stable" else Float32(0)), ry + 6, "staged, restart" if it.restart else "staged", 0, t.accent)
        control(ui, s, i, x0 + 28, ry + 30)
        _ = ui.text(x0 + 28, ry + 62, it.summary, 0, t.muted)
    ui.scroll_end()
    if len(rows) == 0:
        _ = ui.text(x0 + 28, top + 14, "No settings in this view. Advanced shows more.", 0, t.muted)
    help_panel(ui, s, x0 + 16 + lw, top, x1 - x0 - lw - 32, lh)
