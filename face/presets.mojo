"""Presets for the face settings (F5): read [[preset]] tables with the vendored toml package, apply one to the staged
values, save / delete user presets in ~/.config/mojo-baro/presets.toml (same shape as docs/settings.toml's [[preset]]).
Staged values are Dict[String, String]: key -> value text ("16384", "true", "0.7", "path"). A user value that looks
like an int, float or bool is written bare, anything else quoted, so a string setting holding "true" reads back as a bool.
"""
from std.collections import Dict
from std.os import getenv, makedirs
from std.pathlib import Path
from toml import parse, TomlValue


struct Preset(Copyable, Movable):
    var name: String
    var group: String  # speed | task | fit | user
    var summary: String
    var keys: List[String]  # file order, so the apply plan reads the same every time
    var vals: Dict[String, String]

    def __init__(out self, name: String, group: String, summary: String):
        self.name = name
        self.group = group
        self.summary = summary
        self.keys = List[String]()
        self.vals = Dict[String, String]()

    def __init__(out self, *, copy: Self):
        self.name = copy.name
        self.group = copy.group
        self.summary = copy.summary
        self.keys = copy.keys.copy()
        self.vals = copy.vals.copy()

    def put(mut self, k: String, v: String):
        if k not in self.vals:
            self.keys.append(k)
        self.vals[k] = v


def user_presets_path() -> String:
    var home = getenv("HOME", "")
    return getenv("MOJO_BARO_PRESETS", home + "/.config/mojo-baro/presets.toml")


def lit(v: TomlValue) raises -> String:
    if v.is_bool():
        return String("true") if v.as_bool() else String("false")
    if v.is_int():
        return String(v.as_int())
    if v.is_float():
        return String(v.as_float())
    return v.as_string()


def presets_from_text(text: String) raises -> List[Preset]:
    """Every [[preset]] of a TOML text. A preset without a name, or a non-table `values`, is an error."""
    var out = List[Preset]()
    var doc = parse(text)
    if "preset" not in doc:
        return out^
    var arr = doc["preset"].as_array()
    for i in range(len(arr)):
        var t = arr[i].as_table()
        if "name" not in t:
            raise Error("FAIL presets: preset " + String(i) + " has no name")
        var p = Preset(
            t["name"].as_string(),
            t["group"].as_string() if "group" in t else String("user"),
            t["summary"].as_string() if "summary" in t else String(""),
        )
        if "values" in t:
            var vt = t["values"].as_table()
            for e in vt.items():
                p.put(e.key, lit(e.value))
        out.append(p^)
    return out^


def load_presets(path: String) raises -> List[Preset]:
    """A presets file; a missing file is no presets."""
    if not Path(path).exists():
        return List[Preset]()
    return presets_from_text(Path(path).read_text())


def apply_preset(p: Preset, mut staged: Dict[String, String]) raises -> Int:
    """Stage every value of the preset; returns how many values changed what was staged."""
    var n = 0
    for k in p.keys:
        var v = p.vals[k]
        if k not in staged or staged[k] != v:
            n += 1
        staged[k] = v
    return n


def is_bare(v: String) -> Bool:
    if v == "true" or v == "false":
        return True
    try:
        _ = Int(v)
        return True
    except:
        pass
    try:
        _ = Float64(v)
        return v != "nan" and v != "inf" and v != "-inf" and v != "+inf"
    except:
        return False


def q(s: String) -> String:
    var o = String('"')
    for c in s.codepoints():
        if c == Codepoint.ord('"') or c == Codepoint.ord("\\"):
            o += "\\"
        o += String(c)
    return o + '"'


def to_text(ps: List[Preset]) raises -> String:
    var o = String("")
    for p in ps:
        o += "[[preset]]\nname = " + q(p.name) + "\ngroup = " + q(p.group) + "\nsummary = " + q(p.summary) + "\nvalues = { "
        var first = True
        for k in p.keys:
            if not first:
                o += ", "
            first = False
            var v = p.vals[k]
            o += q(k) + " = " + (v if is_bare(v) else q(v))
        o += " }\n\n"
    return o


def write_presets(path: String, ps: List[Preset]) raises:
    var cut = path.rfind("/")
    if cut > 0:
        makedirs(String(path[byte=:cut]), exist_ok=True)
    with open(path, "w") as f:
        f.write(to_text(ps))


def save_user_preset(path: String, name: String, summary: String, staged: Dict[String, String], keys: List[String]) raises:
    """Save the staged values under `name` (replace a user preset of that name). `keys` fixes the written order."""
    if name.byte_length() == 0:
        raise Error("FAIL presets: a preset needs a name")
    var ps = load_presets(path)
    var p = Preset(name, "user", summary)
    for k in keys:
        if k in staged:
            p.put(k, staged[k])
    var out = List[Preset]()
    var done = False
    for old in ps:
        if old.name == name:
            out.append(p.copy())
            done = True
        else:
            out.append(old.copy())
    if not done:
        out.append(p^)
    write_presets(path, out)


def delete_user_preset(path: String, name: String) raises -> Bool:
    """True when a preset of that name existed and is gone."""
    var ps = load_presets(path)
    var out = List[Preset]()
    for old in ps:
        if old.name != name:
            out.append(old.copy())
    if len(out) == len(ps):
        return False
    write_presets(path, out)
    return True


comptime DEFAULT_PRESET = "Balanced"  # owner decision 2026-11-12: spec k=2, KV f32 if it fits else int8, 32K, temp 0.7, top_p 0.9


def startup_staged(ps: List[Preset], mut staged: Dict[String, String]) raises -> Bool:
    """Stage the default preset at face start; False when the preset list has no Balanced (F3 not merged yet)."""
    for p in ps:
        if p.name == DEFAULT_PRESET:
            _ = apply_preset(p, staged)
            return True
    return False
