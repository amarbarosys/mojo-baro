"""Apply flow for the face settings (F5). plan() compares the staged values with the running ones and the restart flags
of docs/settings.toml, and says in plain text what restarts, what applies live, and the pack-fit verdict for a staged
BARO_TMAX. The restart itself is a command line (restart_cmd) for the face to run through `tools/baro serve`; the done
line's fail word comes from the next run (dash.mojo), shown by the caller.
"""
from std.collections import Dict
from std.ffi import external_call
from std.pathlib import Path
from std.time import sleep
from toml import parse
from json import parse as parse_json
from net import http_get


def restart_keys(settings_text: String) raises -> List[String]:
    """Keys of the [[setting]] tables with restart = true."""
    var out = List[String]()
    var doc = parse(settings_text)
    if "setting" not in doc:
        return out^
    var arr = doc["setting"].as_array()
    for i in range(len(arr)):
        var t = arr[i].as_table()
        if "key" in t and "restart" in t and t["restart"].as_bool():
            out.append(t["key"].as_string())
    return out^


def verdict_of(fit_output: String) -> String:
    """The FITS / TIGHT / NO line of pack-fit's output; 'no verdict' when it printed none (or a FAIL line)."""
    for ln in fit_output.split("\n"):
        var s = String(ln)
        if s.startswith("FITS") or s.startswith("TIGHT") or s.startswith("NO:"):
            return s
        if s.startswith("FAIL"):
            return "no verdict (" + s + ")"
    return String("no verdict (pack-fit printed none)")


def fit_cmd(pack: String, tmax: Int, out_path: String) -> String:
    return "pack-fit '" + pack + "' --tmax " + String(tmax) + " > '" + out_path + "' 2>&1"


def run_fit(pack: String, tmax: Int, out_path: String) -> String:
    """Run pack-fit (CPU, reads the pack header and the card) and return its verdict line."""
    try:
        var cmd = fit_cmd(pack, tmax, out_path)
        _ = external_call["system", Int32](cmd.as_c_string_slice().unsafe_ptr())
        return verdict_of(Path(out_path).read_text())
    except e:
        return "no verdict (" + String(e) + ")"


struct ApplyPlan(Copyable, Movable):
    var restart: List[String]  # changed keys that need an engine restart
    var live: List[String]  # changed keys that apply to the next request
    var tmax: Int  # staged BARO_TMAX, -1 when it is not staged
    var verdict: String  # pack-fit line, "" when no fit check applies

    def __init__(out self):
        self.restart = List[String]()
        self.live = List[String]()
        self.tmax = -1
        self.verdict = String("")

    def __init__(out self, *, copy: Self):
        self.restart = copy.restart.copy()
        self.live = copy.live.copy()
        self.tmax = copy.tmax
        self.verdict = copy.verdict

    def needs_restart(self) -> Bool:
        return len(self.restart) > 0

    def blocked(self) -> Bool:
        """Apply is refused when pack-fit says NO."""
        return self.verdict.startswith("NO:")

    def text(self, current: Dict[String, String], staged: Dict[String, String]) raises -> String:
        if len(self.restart) == 0 and len(self.live) == 0:
            return String("Nothing changed: staged values equal the running ones.")
        var o = String("")
        if len(self.restart) > 0:
            o += "Engine restarts for " + String(len(self.restart)) + " change(s):\n"
            for k in self.restart:
                o += "  " + k + ": " + current.get(k, "(default)") + " -> " + staged[k] + "\n"
        else:
            o += "No engine restart.\n"
        if len(self.live) > 0:
            o += "Applies to the next request, " + String(len(self.live)) + " change(s):\n"
            for k in self.live:
                o += "  " + k + ": " + current.get(k, "(default)") + " -> " + staged[k] + "\n"
        if self.verdict.byte_length() > 0:
            o += "VRAM fit at BARO_TMAX " + String(self.tmax) + ": " + self.verdict + "\n"
        return o


def plan(current: Dict[String, String], staged: Dict[String, String], keys: List[String], restart: List[String]) raises -> ApplyPlan:
    """Changed = staged value differs from the running one (a key never run counts as changed). `keys` fixes the order."""
    var p = ApplyPlan()
    for k in keys:
        if k not in staged:
            continue
        if k in current and current[k] == staged[k]:
            continue
        var needs = False
        for r in restart:
            if r == k:
                needs = True
        if needs:
            p.restart.append(k)
        else:
            p.live.append(k)
    if "BARO_TMAX" in staged and "BARO_TMAX" in p.restart:
        try:
            p.tmax = Int(staged.get("BARO_TMAX", "-1"))
        except:
            p.tmax = -1
    return p^


def with_fit(var p: ApplyPlan, pack: String, out_path: String) raises -> ApplyPlan:
    """Add the pack-fit verdict when BARO_TMAX is staged as a restart change and a pack is known."""
    if p.tmax > 0 and pack.byte_length() > 0:
        p.verdict = run_fit(pack, p.tmax, out_path)
    elif p.tmax > 0:
        p.verdict = String("no verdict (no pack chosen)")
    return p^


def restart_cmd(model: String, port: Int, staged: Dict[String, String], restart: List[String]) raises -> String:
    """Shell line: env of every staged engine knob, then tools/baro serve. The caller stops the old engine first."""
    var env = String("")
    for k in restart:
        if k in staged and k.startswith("BARO_"):
            env += k + "='" + staged[k] + "' "
    var quoted = String("'")
    for c in model.codepoints():
        if c == Codepoint.ord('\''):
            quoted += "'\\''"
        else:
            quoted += String(c)
    quoted += "'"
    return "env " + env + "tools/baro serve " + quoted + " --port " + String(port)


def apply_action(stop_command: String, start_command: String, host: String, port: Int, attempts: Int, fail_word: Int) -> String:
    """Stop the owned engine, start its replacement, and poll the front's /health endpoint."""
    var stop = stop_command.copy()
    var rc = external_call["system", Int32](stop.as_c_string_slice().unsafe_ptr())
    if rc != 0:
        return "FAIL stop command " + String(rc) + "; fail " + String(fail_word)
    var start = start_command.copy()
    rc = external_call["system", Int32](start.as_c_string_slice().unsafe_ptr())
    if rc != 0:
        return "FAIL restart command " + String(rc) + "; fail " + String(fail_word)
    for _ in range(attempts):
        try:
            var r = http_get(host, port, "/health", 500)
            if r.status == 200:
                var state = String("?")
                try:
                    var j = parse_json(r.body)
                    if j.has("engine_state"):
                        state = j.get("engine_state").str()
                except:
                    pass
                return "front: ok, engine " + state + "; fail " + String(fail_word)
        except:
            pass
        sleep(1.0)
    return "FAIL /health timeout; fail " + String(fail_word)
