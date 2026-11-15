# operate-audit: writes docs/OPERATE-AUDIT.md from docs/settings.toml (every setting) and docs/operate-capabilities.toml
# (capabilities, flags, best standards), and checks them. TOML via the vendored toml/ package.
# build: mojo build tools/operate-audit.mojo -I . -o .work/operate-audit
#
# usage: .work/operate-audit                       write docs/OPERATE-AUDIT.md
#        .work/operate-audit --check               exit 1 with `FAIL <step>: <reason>` when the file is stale or the data disagrees
#        .work/operate-audit --root DIR [--check]  another tree (the falsifier in tools/operate-audit-check.sh uses a copy)
# Checks: every capability and flag has all its fields, status words are known, every settings name a capability lists is a
# settings key or a name the code reads, every flag's setting exists. The "read, not documented" list is a scan of the
# production directories (serve kernels tools face embed cpu): a quoted name after a call parenthesis (getenv, env::var, env_home), get_defined, $BARO_ and -D BARO_.
from std.sys import argv, exit
from std.subprocess import run
from std.os.path import exists
from std.collections import Dict
from std.builtin.sort import sort
from toml import parse, TomlValue



def fail(step: String, why: String) raises:
    print("FAIL " + step + ": " + why)
    exit(1)


def read(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def sh(cmd: String) raises -> List[String]:
    var out = List[String]()
    for l in String(run(cmd)).split("\n"):
        var s = String(l.strip())
        if s != "":
            out.append(s)
    return out^


def tstr(t: Dict[String, TomlValue], k: String) raises -> String:
    return t[k].as_string() if k in t and t[k].is_string() else ""


def has(xs: List[String], x: String) -> Bool:
    for y in xs:
        if y == x:
            return True
    return False


def row(cells: List[String]) -> String:
    var o = String("|")
    for c in cells:
        o += " " + String(c.replace("|", "\\|").replace("\n", " ")) + " |"
    return o + "\n"


def head(names: List[String]) -> String:
    var sep = List[String]()
    for _ in names:
        sep.append("---")
    return row(names) + row(sep)


def dval(t: Dict[String, TomlValue]) raises -> String:
    if "default" not in t:
        return "unset"
    var v = t["default"].copy()
    if v.is_bool():
        return "`true`" if v.as_bool() else "`false`"
    if v.is_string():
        return "`" + v.as_string() + "`" if v.as_string() != "" else "empty"
    return "`" + (String(v.as_int()) if v.is_int() else String(v.as_float())) + "`"


def gui_of(group: String, level: String) -> String:
    var tab = String("MISSING")
    if group == "sampling":
        tab = "Sampling"
    elif group == "engine":
        tab = "Engine"
    elif group == "model":
        tab = "Model"
    elif group == "front":
        tab = "Front"
    return tab if tab == "MISSING" else tab + " tab" + ("" if level == "basic" else ", Advanced toggle")


def scan_reads(r: String) raises -> Dict[String, String]:
    """BARO_ name -> first file:line that reads it."""
    var found = Dict[String, String]()
    var cmd = "cd '" + r + "' && grep -rnoE --include='*.mojo' --include='*.rs' --include='*.sh' --include=baro --include=baro-multi-serve --exclude-dir=target --exclude-dir=tests --exclude-dir=testdata "
    cmd += "'(\\(\"|get_defined_[a-z]+\\[\"|\\$\\{?|-D ?)BARO_[A-Z0-9_]+' serve kernels tools face embed cpu tools/baro tools/baro-multi-serve 2>/dev/null"
    for ln in sh(cmd):
        var parts = ln.split(":")
        var i = ln.find("BARO_")
        if len(parts) < 3 or i < 0:
            continue
        var name = String(ln[byte=i:])
        if not name.endswith("_") and name not in found:
            found[name] = String(parts[0]) + ":" + String(parts[1])
    return found^


def need(t: Dict[String, TomlValue], fields: String, step: String, who: String) raises:
    for f in fields.split(" "):
        if String(f) not in t:
            fail(step, who + " lacks " + String(f))


def render(r: String) raises -> String:
    var sets = parse(read(r + "/docs/settings.toml"))["setting"].as_array()
    var cdoc = parse(read(r + "/docs/operate-capabilities.toml"))
    var caps = cdoc["capability"].as_array()
    var flags = cdoc["flag"].as_array()
    var reads = scan_reads(r)
    var keys = List[String]()
    for s in sets:
        keys.append(tstr(s.as_table(), "key"))
    var cap_of = Dict[String, Int]()
    for i in range(len(caps)):
        var t = caps[i].as_table()
        need(t, "id name enable default best status gui check source settings", "capability", tstr(t, "id"))
        if String(" WORKS PARTIAL OFF FAILS ").find(" " + tstr(t, "status") + " ") < 0:
            fail("capability", tstr(t, "id") + " status " + tstr(t, "status"))
        for k in t["settings"].as_array():
            var n = k.as_string()
            if not has(keys, n) and n not in reads:
                fail("capability", tstr(t, "id") + " names " + n + ": not in docs/settings.toml and not read by the code")
            if n not in cap_of:
                cap_of[n] = i
    for f in flags:
        var t = f.as_table()
        need(t, "tool flag meaning setting gui", "flag", tstr(t, "flag"))
        var s = tstr(t, "setting")
        if s != "" and s != "any env key" and not has(keys, s):
            fail("flag", tstr(t, "flag") + " names setting " + s)
    var o = String("# mojo-baro operate audit\n\n")
    o += "Generated by `tools/operate-audit.mojo` from `docs/settings.toml` (every setting) and `docs/operate-capabilities.toml` (capabilities, flags,\n"
    o += "best standards). Edit those, then rebuild and run the tool; `--check` fails when this file is stale. Status words are those of `docs/CAPABILITIES.md`:\n"
    o += "WORKS, PARTIAL, OFF (works, default off), FAILS. \"Best standard\" is the repo's own measured recommendation with its source; where none was\n"
    o += "measured the row says so.\n\n## 1. Capabilities\n\n"
    o += head(["capability", "how to turn it on", "default", "best standard (measured reason, source)", "status", "in GUI?", "check"])
    var missing = String("")
    var nm = 0
    for c in caps:
        var t = c.as_table()
        o += row(["**" + tstr(t, "name") + "**", tstr(t, "enable"), tstr(t, "default"), tstr(t, "best") + " [" + tstr(t, "source") + "]", tstr(t, "status"), tstr(t, "gui"), tstr(t, "check")])
        if tstr(t, "gui").startswith("MISSING"):
            missing += "- **" + tstr(t, "name") + "**: " + tstr(t, "gui") + "\n"
            nm += 1
    o += "\n## 2. Every setting\n\n" + String(len(sets)) + " entries of `docs/settings.toml`. Kind: env (engine, read at run time), build (`-D` flag, rebuild), server-env (baro-serve),\n"
    o += "shell (`tools/baro`), request (a request field), external (read by MAX). Internal = status bench-only (benches, tests, probes).\n"
    o += "In GUI: the face lists every setting of group sampling, engine, model and front, Basic or behind the Advanced toggle (`face/settings.mojo:157-159`).\n\n"
    o += head(["setting", "kind", "default", "restart", "status", "best standard", "in GUI?", "check"])
    var dead = List[String]()
    for s in sets:
        var t = s.as_table()
        var key = tstr(t, "key")
        var status = tstr(t, "status")
        var kind = tstr(t, "kind")
        var best = String("internal, no standard") if status == "bench-only" else String("none recorded")
        var chk = String("-")
        if key in cap_of:
            var ct = caps[cap_of[key]].as_table()
            best = tstr(ct, "name") + ": see section 1"
            chk = tstr(ct, "check")
        o += row(["`" + key + "`", kind, dval(t), "yes" if t["restart"].as_bool() else "no", status + (" (internal)" if status == "bench-only" else ""), best, gui_of(tstr(t, "group"), tstr(t, "level")), chk])
        if (kind == "env" or kind == "build" or kind == "server-env" or kind == "shell") and status != "bench-only" and key not in reads:
            dead.append(key + " (" + kind + ", " + status + ")")
    o += "\n## 3. Command-line flags\n\n" + head(["tool", "flag", "what it does", "same as setting", "in GUI?"])
    for f in flags:
        var t = f.as_table()
        o += row([tstr(t, "tool"), "`" + tstr(t, "flag") + "`", tstr(t, "meaning"), "`" + tstr(t, "setting") + "`" if tstr(t, "setting") != "" else "none", tstr(t, "gui")])
    var undoc = List[String]()
    for e in reads.items():
        if not has(keys, e.key):
            undoc.append(e.key)
    sort(undoc)
    sort(dead)
    o += "\n## 4. Gaps\n\n### Read by the code, not in `docs/settings.toml` (" + String(len(undoc)) + ")\n\n"
    o += "Found by the scan described at the top of `tools/operate-audit.mojo` (production directories). `tools/settings-check` covers `serve/*.mojo`, `kernels/`,\n"
    o += "`tools/*.mojo`, `serve/src/*.rs` and `tools/baro`, but not `serve/src/bin/`, `face/`, `cpu/`, `embed/` or other shell scripts.\n\n" + head(["name", "read at"])
    for n in undoc:
        o += row(["`" + n + "`", reads[n]])
    o += "\n### Documented, not read by the scanned code (" + String(len(dead)) + ")\n\n"
    o += "Settings of kind env, build, server-env or shell, not bench-only, with no read in the scanned directories (a bench, a lab or MAX itself may read them).\n\n"
    for d in dead:
        o += "- `" + d + "`\n"
    o += "- none\n" if len(dead) == 0 else ""
    o += "\n### Documented, read, and does nothing\n\n"
    for c in caps:
        if tstr(c.as_table(), "id") == "penalties":
            o += "- " + tstr(c.as_table(), "name") + ": " + tstr(c.as_table(), "best") + "\n"
    o += "\n### Capabilities missing from the GUI (" + String(nm) + ")\n\n" + missing
    return o^


def render_caps(r: String) raises -> String:
    var caps = parse(read(r + "/docs/operate-capabilities.toml"))["capability"].as_array()
    var o = String("# Capabilities, settings and best standards\n\nGenerated by `tools/operate-audit.mojo` from `docs/operate-capabilities.toml`; do not edit. The full table with the GUI column is `docs/OPERATE-AUDIT.md` in the repo.\n\n")
    for c in caps:
        var t = c.as_table()
        o += "## " + tstr(t, "id") + ": " + tstr(t, "name") + "\n\n"
        o += "- Enable: " + tstr(t, "enable") + "\n- Default: " + tstr(t, "default") + "\n- Best standard: " + tstr(t, "best") + "\n"
        o += "- Status: " + tstr(t, "status") + "\n- In the GUI: " + tstr(t, "gui") + "\n- Check: " + tstr(t, "check") + "\n- Source: " + tstr(t, "source") + "\n\n"
    return o^


def main() raises:
    var args = argv()
    var r = String(run("dirname -- \"$(dirname -- \"$(realpath -- '" + String(args[0]) + "')\")\"").strip())
    var check = False
    var i = 1
    while i < len(args):
        if String(args[i]) == "--check":
            check = True
        elif String(args[i]) == "--root" and i + 1 < len(args):
            r = String(args[i + 1])
            i += 1
        else:
            print("usage: operate-audit [--root DIR] [--check]")
            exit(1)
        i += 1
    var text = render(r)
    var caps_text = render_caps(r)
    var path = r + "/docs/OPERATE-AUDIT.md"
    var cpath = r + "/skills/mojo-baro-operate/references/capabilities.md"
    if check:
        if not exists(path) or read(path) != text:
            fail("stale", "docs/OPERATE-AUDIT.md differs from the data: rebuild and run .work/operate-audit")
        if not exists(cpath) or read(cpath) != caps_text:
            fail("stale", "skills/mojo-baro-operate/references/capabilities.md differs from the data: rebuild and run .work/operate-audit")
        print("PASS operate-audit")
        return
    with open(path, "w") as f:
        f.write(text)
    with open(cpath, "w") as f:
        f.write(caps_text)
    print("wrote docs/OPERATE-AUDIT.md (" + String(len(text.split("\n"))) + " lines) and the skill's references/capabilities.md")
