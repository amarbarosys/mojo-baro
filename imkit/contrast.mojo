# VENDORED COPY of amarbaro/mojo-imkit (Apache-2.0, see imkit/LICENSE), tag v0.9.0 (eb91210),
# ~/Projects/mojo/mojo-imkit, file src/imkit/contrast.mojo. This repo keeps a real file
# rather than an -I path outside the tree, because a clone must build without anything
# outside it. Sync by hand if the upstream changes; tools/ci-checks.sh compares the two
# against the tag when the upstream is present and says so when they drift.
#
"""WCAG 2 contrast of packed rgba colours (r | g << 8 | b << 16 | a << 24, alpha ignored) and the theme gate the style
presets must pass: text/panel, text/surface and on_accent/accent >= 4.5, muted/panel >= 3, good/warn/bad vs panel >= 3,
surface/panel >= 1.25 (a control is visible on its panel) and hover/surface >= 1.1."""
from std.math import pow
from .theme import Theme


def lin(c: UInt32) -> Float64:
    var s = Float64(c & 255) / 255.0
    return s / 12.92 if s <= 0.04045 else pow((s + 0.055) / 1.055, 2.4)


def lum(c: UInt32) -> Float64:
    return 0.2126 * lin(c) + 0.7152 * lin(c >> 8) + 0.0722 * lin(c >> 16)


def contrast(a: UInt32, b: UInt32) -> Float64:
    var x = lum(a)
    var y = lum(b)
    return (max(x, y) + 0.05) / (min(x, y) + 0.05)


def r2(x: Float64) -> String:
    return String(Float64(Int(x * 100)) / 100)


def theme_faults(t: Theme) -> String:
    """Empty when the theme passes, else one `role a/b ratio` entry per failed pair."""
    var out = String()
    if contrast(t.text, t.panel | 0xFF000000) < 4.5:
        out += "text/panel " + r2(contrast(t.text, t.panel)) + " "
    if contrast(t.on_accent, t.accent) < 4.5:
        out += "on_accent/accent " + r2(contrast(t.on_accent, t.accent)) + " "
    if contrast(t.muted, t.panel) < 3.0:
        out += "muted/panel " + r2(contrast(t.muted, t.panel)) + " "
    if contrast(t.text, t.surface) < 4.5:
        out += "text/surface " + r2(contrast(t.text, t.surface)) + " "
    if contrast(t.surface, t.panel) < 1.25:
        out += "surface/panel " + r2(contrast(t.surface, t.panel)) + " "
    if contrast(t.hover, t.surface) < 1.1:
        out += "hover/surface " + r2(contrast(t.hover, t.surface)) + " "
    for c in [t.good, t.warn, t.bad]:
        if contrast(c, t.panel) < 3.0:
            out += "status/panel " + r2(contrast(c, t.panel)) + " "
    return out
