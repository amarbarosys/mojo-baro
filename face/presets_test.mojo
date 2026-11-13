"""Tests presets.mojo and apply.mojo: fixture load, preset apply, user preset save / load / replace / delete round trip
(including a dotted key and a quote in a summary), apply plan text, pack-fit verdict parsing. No engine, no GPU, no window.
usage: mojo build -I . -I face face/presets_test.mojo"""
from std.collections import Dict
from std.pathlib import Path
from presets import startup_staged, Preset, load_presets, presets_from_text, apply_preset, save_user_preset, delete_user_preset, to_text
from apply import restart_keys, plan, verdict_of, restart_cmd, fit_cmd, apply_action


def check(ok: Bool, what: String) raises:
    if not ok:
        raise Error("FAIL presets_test: " + what)


def main() raises:
    var fx = Path("face/testdata/presets-sample.toml").read_text()
    var ps = presets_from_text(fx)
    check(len(ps) == 2 and ps[0].name == "Fast" and ps[1].group == "speed", "fixture presets")
    check(len(ps[1].keys) == 3 and ps[1].keys[0] == "BARO_TMAX" and ps[1].keys[2] == "sampling.temperature", "value order")
    check(ps[1].vals["BARO_MEGA"] == "true" and ps[1].vals["BARO_TMAX"] == "65536", "value text")
    var rk = restart_keys(fx)
    check(len(rk) == 2 and rk[0] == "BARO_TMAX" and rk[1] == "BARO_MEGA", "restart keys")
    var st0 = Dict[String, String]()
    check(not startup_staged(ps, st0) and len(st0) == 0, "no Balanced in the fixture: nothing staged")
    var bal = Preset("Balanced", "speed", "")
    bal.put("BARO_TMAX", "32768")
    var withbal = List[Preset]()
    withbal.append(bal^)
    check(startup_staged(withbal, st0) and st0["BARO_TMAX"] == "32768", "Balanced staged at start")
    # apply to staged
    var staged = Dict[String, String]()
    staged["BARO_TMAX"] = "16384"
    check(apply_preset(ps[1], staged) == 3 and staged["BARO_TMAX"] == "65536", "apply changed 3")
    check(apply_preset(ps[1], staged) == 0, "second apply changes nothing")
    # round trip of a user preset file
    var path = ".work/presets-test/presets.toml"
    if Path(path).exists():
        _ = delete_user_preset(path, "Mine")
        _ = delete_user_preset(path, "Other")
    check(len(load_presets(".work/presets-test/none.toml")) == 0, "missing file is empty")
    var s2 = Dict[String, String]()
    s2["BARO_TMAX"] = "32768"
    s2["sampling.temperature"] = "0.35"
    s2["BARO_MODEL"] = "a \"b\" c"
    var order = List[String]()
    order.append("BARO_TMAX")
    order.append("sampling.temperature")
    order.append("BARO_MODEL")
    save_user_preset(path, "Mine", 'say "hi"', s2, order)
    save_user_preset(path, "Other", "", s2, order)
    var back = load_presets(path)
    check(len(back) == 2 and back[0].name == "Mine" and back[0].group == "user", "saved two")
    check(back[0].summary == 'say "hi"' and back[0].vals["sampling.temperature"] == "0.35", "dotted key + quote")
    check(back[0].vals["BARO_MODEL"] == 'a "b" c' and back[0].keys[2] == "BARO_MODEL", "string value + order")
    s2["BARO_TMAX"] = "40000"
    save_user_preset(path, "Mine", "v2", s2, order)  # replace, not duplicate
    back = load_presets(path)
    check(len(back) == 2 and back[0].vals["BARO_TMAX"] == "40000" and back[0].summary == "v2", "replace in place")
    check(to_text(back) == to_text(load_presets(path)), "text stable")
    check(delete_user_preset(path, "Mine") and len(load_presets(path)) == 1, "delete")
    check(not delete_user_preset(path, "Mine"), "delete twice says no")
    # apply plan
    var cur = Dict[String, String]()
    cur["BARO_TMAX"] = "16384"
    cur["BARO_MEGA"] = "true"
    cur["sampling.temperature"] = "0.7"
    var keys = ps[1].keys.copy()
    var pl = plan(cur, staged, keys, rk)
    check(len(pl.restart) == 1 and pl.restart[0] == "BARO_TMAX" and len(pl.live) == 1 and pl.tmax == 65536, "plan split")
    pl.verdict = "FITS: headroom 3.2 GB"
    var t = pl.text(cur, staged)
    print(t)
    check("Engine restarts for 1 change(s)" in t and "BARO_TMAX: 16384 -> 65536" in t, "plan text restart")
    check("sampling.temperature: 0.7 -> 0.8" in t and "FITS: headroom 3.2 GB" in t, "plan text live + fit")
    check(not pl.blocked(), "FITS does not block")
    pl.verdict = "NO: over the pool by 1.2 GB"
    check(pl.blocked(), "NO blocks")
    var same = plan(staged, staged, keys, rk)
    check(not same.needs_restart() and same.text(staged, staged).startswith("Nothing changed"), "no change")
    check(verdict_of("pack.bin 1 GB\ntotal 20\nTIGHT: headroom 0.4 GB\n") == "TIGHT: headroom 0.4 GB", "verdict tight")
    check(verdict_of("FAIL pack: no pack.bin under x\n").startswith("no verdict"), "verdict fail")
    check(restart_cmd("m.gguf", 8080, staged, rk) == "env BARO_TMAX='65536' BARO_MEGA='true' tools/baro serve 'm.gguf' --port 8080", "restart cmd")
    check(fit_cmd("p", 4, "o") == "pack-fit 'p' --tmax 4 > 'o' 2>&1", "fit cmd")
    var fake_health = apply_action("true", "printf fake > .work/fake-restart-command", "127.0.0.1", 1, 1, 0)
    check(Path(".work/fake-restart-command").read_text() == "fake", "fake restart command ran")
    check(fake_health.startswith("FAIL /health timeout"), "fake restart waits for health without an engine")
    print("PASS presets_test: fixture, apply, user round trip, plan text, verdict")
