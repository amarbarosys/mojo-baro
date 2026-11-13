"""baro face, settings model (F4): the settings table read with the vendored toml/ package, plus staged values.

  var s = Settings.load("docs/settings.toml")     every [[setting]] of the plan's contract; raises with a named error
  s.stage(i, "0.9")                                a value that differs from the applied one is marked staged
  s.sampling_json()                                `"temperature":0.9` for the chat request (sampling.* keys that differ from default)

No drawing here (face/widgets.mojo draws). Values are Strings in the TOML's own spelling: "16384", "0.7", "true", "f32".
"""
from std.os.path import exists
from toml import parse, TomlValue

def tab_name(i: Int) -> String:
    return String("Chat" if i == 0 else "Sampling" if i == 1 else "Engine" if i == 2 else "Model" if i == 3 else "Front" if i == 4 else "Presets")


def tab_group(i: Int) -> String:
    """Settings group a tab shows ("" = no settings table: Chat, Presets)."""
    return String("sampling" if i == 1 else "engine" if i == 2 else "model" if i == 3 else "front" if i == 4 else "")


comptime NTABS = 6


def _s(v: TomlValue) raises -> String:
    """Any scalar in the file as the text a control edits."""
    if v.is_string():
        return v.as_string()
    if v.is_int():
        return String(v.as_int())
    if v.is_float():
        return String(v.as_float())
    if v.is_bool():
        return String("true") if v.as_bool() else String("false")
    raise Error("not a scalar")


def _opt(t: Dict[String, TomlValue], k: String, dflt: String) raises -> String:
    return _s(t[k]) if k in t else dflt


struct Setting(Copyable, Movable):
    var key: String
    var group: String
    var label: String
    var type: String  # int | float | bool | enum | path | string
    var default: String
    var lo: Float64
    var hi: Float64
    var ranged: Bool
    var choices: List[String]
    var restart: Bool
    var level: String  # basic | advanced
    var status: String  # stable | experimental | bench-only
    var summary: String
    var help: String

    def __init__(out self, t: Dict[String, TomlValue]) raises:
        for need in ["key", "group", "label", "type", "summary"]:
            if String(need) not in t:
                raise Error("FAIL settings: a [[setting]] has no '" + String(need) + "'")
        self.key, self.group, self.label, self.type = _s(t["key"]), _s(t["group"]), _s(t["label"]), _s(t["type"])
        self.default = _s(t["default"]) if "default" in t else ""  # absent = unset (engine default or no file)
        self.ranged = "min" in t and "max" in t
        self.lo = Float64(_s(t["min"])) if self.ranged else 0.0
        self.hi = Float64(_s(t["max"])) if self.ranged else 0.0
        self.choices = List[String]()
        if "choices" in t:
            for c in t["choices"].as_array():
                self.choices.append(_s(c))
        self.restart = _opt(t, "restart", "false") == "true"
        self.level, self.status = _opt(t, "level", "basic"), _opt(t, "status", "stable")
        self.summary, self.help = _s(t["summary"]), _opt(t, "help", "unmeasured")
        if self.type not in ["int", "float", "bool", "enum", "path", "string"]:
            raise Error("FAIL settings: " + self.key + " has unknown type '" + self.type + "'")
        if self.type == "enum" and len(self.choices) == 0:
            raise Error("FAIL settings: enum " + self.key + " has no choices")


struct Settings(Movable):
    var items: List[Setting]
    var applied: Dict[String, String]  # key -> value the engine runs with (F5 fills it); absent: the default
    var staged: Dict[String, String]  # key -> value waiting for Apply (or, for restart = false, in use already)
    var tab: Int  # index into TABS
    var advanced: Bool
    var sel: Int  # selected item (the help panel shows it), -1 none
    var off: Float32  # scroll offset of the list

    def __init__(out self):
        self.items = List[Setting]()
        self.applied = Dict[String, String]()
        self.staged = Dict[String, String]()
        self.tab, self.advanced, self.sel, self.off = 0, False, -1, 0

    @staticmethod
    def load(path: String) raises -> Self:
        var text: String
        try:
            with open(path, "r") as f:
                text = f.read()
        except e:
            raise Error("FAIL settings: cannot read " + path + ": " + String(e))
        return Self.from_text(text)

    @staticmethod
    def load_default() raises -> Self:
        """docs/settings.toml (F3's file) when it exists, else the contract fixture. A file that exists but is broken raises."""
        return Self.load("docs/settings.toml" if exists("docs/settings.toml") else "face/testdata/settings-sample.toml")

    @staticmethod
    def from_text(text: String) raises -> Self:
        var s = Self()
        var root = parse(text)
        if "setting" not in root:
            raise Error("FAIL settings: no [[setting]] table")
        for e in root["setting"].as_array():
            var item = Setting(e.as_table())
            for o in s.items:
                if o.key == item.key:
                    raise Error("FAIL settings: duplicate key " + item.key)
            s.items.append(item^)
        return s^

    def base(self, i: Int) -> String:
        """The value the engine has: applied, else the default."""
        ref it = self.items[i]
        return self.applied.get(it.key, it.default)

    def value(self, i: Int) -> String:
        ref it = self.items[i]
        return self.staged.get(it.key, self.base(i))

    def is_staged(self, i: Int) -> Bool:
        return self.items[i].key in self.staged

    def stage(mut self, i: Int, v: String):
        """Stage v; a value equal to the applied one drops the mark (a slider back at its start is not a change)."""
        var k = self.items[i].key.copy()
        if v == self.base(i):
            if k in self.staged:
                _ = self.staged.pop(k, String(""))
        else:
            self.staged[k] = v

    def staged_count(self) -> Int:
        var n = 0
        for i in range(len(self.items)):
            n += 1 if self.is_staged(i) else 0
        return n

    def restart_count(self) -> Int:
        """Staged changes that need an engine restart."""
        var n = 0
        for i in range(len(self.items)):
            n += 1 if self.is_staged(i) and self.items[i].restart else 0
        return n

    def shown(self, i: Int) -> Bool:
        ref it = self.items[i]
        return it.group == tab_group(self.tab) and (self.advanced or it.level == "basic")

    def sampling_json(self) -> String:
        """`"temperature":0.9,"stream":false` for the chat request body: sampling.* settings whose value is not the default
        (the front's own defaults stay untouched otherwise). Empty string when none."""
        var out = String("")
        for i in range(len(self.items)):
            ref it = self.items[i]
            if not it.key.startswith("sampling.") or self.value(i) == it.default:
                continue
            var v = self.value(i)
            var lit = String('"') + v + '"' if it.type in ["enum", "string", "path"] else v.copy()
            out += ("," if out.byte_length() > 0 else "") + '"' + String(it.key[byte=9:]) + '":' + lit
        return out^
