"""Tests settings.mojo with no engine, no window: contract fixture load, every field read, staging and un-staging, restart
count, sampling JSON, and the refusals (duplicate key, missing field, bad type).
usage: mojo build -I . -I face face/settings_test.mojo"""
from settings import Settings


def check(ok: Bool, what: String) raises:
    if not ok:
        raise Error("FAIL settings_test: " + what)


def refused(text: String, word: String) raises:
    try:
        _ = Settings.from_text(text)
    except e:
        check(word in String(e), "refusal names '" + word + "', got: " + String(e))
        return
    raise Error("FAIL settings_test: bad file accepted (" + word + ")")


def main() raises:
    var real = Settings.load("docs/settings.toml")
    check(len(real.items) == 182, "real docs/settings.toml loads all 182 entries, got " + String(len(real.items)))
    var s = Settings.load("face/testdata/settings-sample.toml")
    check(len(s.items) == 5, "5 entries")
    check(s.items[0].key == "sampling.temperature" and s.items[0].type == "float" and s.items[0].ranged, "temperature")
    check(s.items[0].lo == 0.0 and s.items[0].hi == 2.0 and not s.items[0].restart, "temperature range, no restart")
    check(s.items[2].key == "BARO_TMAX" and s.items[2].lo == 16384.0 and s.items[2].restart, "BARO_TMAX floor and restart")
    check(s.items[3].choices[1] == "int8" and s.items[3].status == "experimental", "enum choices, status")
    check("Cost: KV cache" in s.items[2].help, "multi-line help kept")
    check(s.value(0) == "0.7" and s.staged_count() == 0, "defaults")
    s.stage(0, "0.9")
    s.stage(2, "32768")
    check(s.is_staged(0) and s.value(2) == "32768" and s.staged_count() == 2 and s.restart_count() == 1, "staged")
    check(s.sampling_json() == '"temperature":0.9', "sampling json " + s.sampling_json())
    s.stage(0, "0.7")
    check(not s.is_staged(0) and s.staged_count() == 1 and s.sampling_json() == "", "back to the default un-stages")
    s.stage(1, "false")
    check(s.sampling_json() == '"stream":false', "bool json")
    s.applied["BARO_TMAX"] = "32768"  # the engine now runs it: staging the same value is no change
    s.stage(2, "32768")
    check(not s.is_staged(2) and s.base(2) == "32768", "applied value is the new base")
    s.tab = 2
    check(s.shown(2) and not s.shown(3) and not s.shown(0), "engine tab shows engine entries only")
    s.tab = 1
    check(s.shown(0) and not s.shown(1), "basic view hides advanced")
    s.advanced = True
    check(s.shown(1), "advanced view shows them")
    refused("[[setting]]\nkey='a'\n", "no 'group'")
    refused("x = 1\n", "no [[setting]]")
    var one = '[[setting]]\nkey="a"\ngroup="engine"\nlabel="A"\ntype="int"\ndefault=1\nsummary="s"\n'
    refused(one + one, "duplicate key a")
    refused(one.replace('type="int"', 'type="blob"'), "unknown type")
    try:
        _ = Settings.load("face/testdata/none.toml")
        raise Error("FAIL settings_test: missing file accepted")
    except e:
        check("cannot read" in String(e), "missing file named")
    print("PASS settings_test: load, staging, restart count, sampling json, 5 refusals")
