# profile: named run profiles, Mojo port of tools/profile.py (the oracle).
# TOML via DataBooth/mojo-toml, vendored at repo root as toml/ (build with -I .).
# usage: profile check | env NAME | build NAME | get NAME FIELD
from std.sys import argv, exit, stderr
from std.os.path import exists
from std.subprocess import run
from toml import parse, TomlValue


def root() raises -> String:
    # the binary lives in <repo>/.work/, the source in <repo>/tools/
    var exe = String(run("realpath -- '" + String(argv()[0]) + "'").strip())
    return String(run("dirname -- \"$(dirname -- '" + exe + "')\"").strip())


def read(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def str_of(v: TomlValue) raises -> String:
    if v.is_string():
        return v.as_string()
    if v.is_int():
        return String(v.as_int())
    if v.is_bool():
        return "True" if v.as_bool() else "False"
    return String(v.as_float())


def problems(name: String, d: Dict[String, TomlValue], tmpl: Dict[String, TomlValue]) raises -> List[String]:
    var out = List[String]()
    var meta = d["profile"].as_table() if "profile" in d else Dict[String, TomlValue]()
    var pname = str_of(meta["name"]) if "name" in meta else "None"
    if pname != name:
        out.append("[profile] name '" + pname + "' must equal the file name '" + name + "'")
    var model = str_of(meta["model"]) if "model" in meta else "None"
    if model != "qwen35" and model != "qwen35moe" and model != "spark":
        out.append("[profile] model '" + model + "' not one of ('qwen35', 'qwen35moe', 'spark')")
    var source = str_of(meta["source"]) if "source" in meta else "file"
    if source != "file" and source != "checkout":
        out.append("[profile] source must be file or checkout")
    var desc = String(str_of(meta["description"]).strip()) if "description" in meta else ""
    if desc == "":
        out.append("[profile] description is empty")
    for section in ["env", "build"]:
        if section not in d:
            continue
        var sec = d[section].as_table()
        for item in sec.items():
            var k = item.key
            if k not in tmpl:
                out.append("[" + section + "] " + k + ": not in docs/settings-template.toml")
                continue
            var t = tmpl[k].as_table()
            var kind = t["kind"].as_string()
            var ok_kind = (kind == "build") if section == "build" else (kind == "env" or kind == "server-env" or kind == "external")
            if not ok_kind:
                var home = "build" if kind == "build" else "env"
                out.append("[" + section + "] " + k + ": template kind is '" + kind + "', belongs under [" + home + "]")
                continue
            var models = t["models"].as_array()
            var applies = False
            var shown = String("[")
            for i in range(len(models)):
                var m = models[i].as_string()
                if m == "all" or m == model:
                    applies = True
                shown += ("" if i == 0 else ", ") + "'" + m + "'"
            shown += "]"
            if not applies:
                out.append("[" + section + "] " + k + ": applies to " + shown + ", not " + model)
            elif k == "BARO_STATE_HMAC_KEY":
                out.append("[" + section + "] " + k + ": a secret, never in a profile")
    return out^


def run_main() raises:
    var args = argv()
    var r = root()
    var tmpl = parse(read(r + "/docs/settings-template.toml"))["settings"].as_table()
    var cmd = String(args[1]) if len(args) > 1 else "list"
    if cmd == "check":
        var files = String(run("cd '" + r + "/profiles' && LC_ALL=C ls -1 -- *.toml 2>/dev/null").strip())
        var n = 0
        var bad = 0
        for f in files.split("\n"):
            var fname = String(f)
            if fname == "":
                continue
            n += 1
            var stem = String(fname.removesuffix(".toml"))
            var errs = problems(stem, parse(read(r + "/profiles/" + fname)), tmpl)
            for e in errs:
                print("  " + fname + ": " + e)
                bad += 1
        print(("FAIL" if bad > 0 else "PASS") + " profiles: " + String(n) + " checked, " + String(bad) + " problem(s)")
        if bad > 0:
            exit(1)
        return
    if len(args) < 3 or (cmd != "env" and cmd != "build" and cmd != "get"):
        print("usage: profile check | env NAME | build NAME | get NAME FIELD", file=stderr)
        exit(1)
    var name = String(args[2])
    var path = r + "/profiles/" + name + ".toml"
    if not exists(path):
        print("FAIL profile: no " + path, file=stderr)
        exit(1)
    var d = parse(read(path))
    var errs = problems(name, d, tmpl)
    if len(errs) > 0:
        var msg = String("FAIL profile " + name + ": ")
        for i in range(len(errs)):
            msg += ("" if i == 0 else "; ") + errs[i]
        print(msg, file=stderr)
        exit(1)
    if cmd == "get":
        var field = String(args[3]) if len(args) > 3 else ""
        var meta = d["profile"].as_table()
        print(str_of(meta[field]) if field in meta else ("file" if field == "source" else ""))
        return
    var line = String("")
    if cmd in d:
        var sec = d[cmd].as_table()
        for item in sec.items():
            var word = (item.key + "=" + str_of(item.value)) if cmd == "env" else ("-D " + item.key + "=" + str_of(item.value))
            line += (" " if line != "" else "") + word
    print(line)


def main():
    try:
        run_main()
    except e:
        print("Traceback: " + String(e), file=stderr)
        exit(1)
