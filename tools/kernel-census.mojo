# Copyright 2026 amarbaro.com. Licensed under the Apache License, Version 2.0.
"""Kernel census: every `amar_*` device kernel under kernels/ must be reachable
from the engine registry (serve/registry.mojo), a bench, or a test. Emits
docs/KERNELS.md and exits non-zero on an orphan. Run from the repo root.

usage: mojo run tools/kernel-census.mojo [--check]   (--check: no write, just gate)
"""
from std.os import listdir
from std.os.path import exists
from std.sys import argv, exit

comptime REG = "serve/registry.mojo"


def is_name(c: UInt8) -> Bool:
    return (c >= 97 and c <= 122) or (c >= 48 and c <= 57) or c == 95


def is_word(c: UInt8) -> Bool:
    return is_name(c) or (c >= 65 and c <= 90) or c >= 128


def is_ws(c: UInt8) -> Bool:
    return c == 32 or (c >= 9 and c <= 13)


def read(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def sub(s: String, a: Int, b: Int) -> String:
    return String(unsafe_from_utf8=s.as_bytes()[a:b])


def at(s: String, i: Int, lit: String) -> Bool:
    var b = s.as_bytes()
    var l = lit.as_bytes()
    if i + len(l) > len(b):
        return False
    for k in range(len(l)):
        if b[i + k] != l[k]:
            return False
    return True


def name_end(s: String, i: Int) -> Int:
    var b = s.as_bytes()
    var j = i
    while j < len(b) and is_name(b[j]):
        j += 1
    return j


def squash(s: String) -> String:
    var b = s.as_bytes()
    var out = String()
    var i = 0
    while i < len(b):
        while i < len(b) and is_ws(b[i]):
            i += 1
        var j = i
        while j < len(b) and not is_ws(b[j]):
            j += 1
        if j > i:
            if out.byte_length() > 0:
                out += " "
            out += sub(s, i, j)
        i = j
    return out


def line_starts(s: String) -> List[Int]:
    var b = s.as_bytes()
    var out: List[Int] = [0]
    for i in range(len(b)):
        if b[i] == 10:
            out.append(i + 1)
    return out^


def def_body(s: String, i: Int) -> Int:
    """Offset just past `def ` or `fn ` at i, or -1."""
    if at(s, i, "def "):
        return i + 4
    if at(s, i, "fn "):
        return i + 3
    return -1


def params_of(src: String, name: String) -> String:
    var b = src.as_bytes()
    for ls in line_starts(src):
        var p = def_body(src, ls)
        if p < 0 or not at(src, p, name + "["):
            continue
        var q = p + name.byte_length() + 1
        var e = q
        while e < len(b) and b[e] != 93:
            e += 1
        if e < len(b):
            return squash(sub(src, q, e))
    return ""


def has_word(src: String, w: String) -> Bool:
    var b = src.as_bytes()
    var n = w.byte_length()
    var i = src.find(w)
    while i >= 0:
        var before = i == 0 or not is_word(b[i - 1])
        var after = i + n >= len(b) or not is_word(b[i + n])
        if before and after:
            return True
        i = src.find(w, i + 1)
    return False


def sort_strs(mut xs: List[String]):
    for i in range(1, len(xs)):
        var j = i
        while j > 0 and xs[j] < xs[j - 1]:
            xs.swap_elements(j, j - 1)
            j -= 1


def glob(dir: String, prefix: String, test: Bool) raises -> List[String]:
    var out = List[String]()
    for e in listdir(dir):
        if e.endswith(".mojo") and e.startswith(prefix) and (
            e.startswith("test_") == test
        ):
            out.append(dir + "/" + e)
    sort_strs(out)
    return out^


def glob_suffix(dir: String, suffix: String) raises -> List[String]:
    var out = List[String]()
    for e in listdir(dir):
        if e.endswith(suffix):
            out.append(dir + "/" + e)
    sort_strs(out)
    return out^


def join(xs: List[String], sep: String) -> String:
    var out = String()
    for i in range(len(xs)):
        if i > 0:
            out += sep
        out += xs[i]
    return out


def basename(p: String) -> String:
    return sub(p, p.rfind("/") + 1, p.byte_length())


def docline(src: String) -> String:
    var b = src.as_bytes()
    var i = 0
    while True:
        while i < len(b) and is_ws(b[i]):
            i += 1
        if i < len(b) and b[i] == 35:
            while i < len(b) and b[i] != 10:
                i += 1
            if i == len(b):
                return ""
            i += 1
        else:
            break
    if not at(src, i, '"""'):
        return ""
    var q = i + 3
    var e1 = src.find("\n\n", q)
    var e2 = src.find('"""', q)
    var e = e1 if e1 >= 0 and (e2 < 0 or e1 < e2) else e2
    if e < 0:
        return ""
    return squash(sub(src, q, e))


def main() raises:
    var names = List[String]()
    var file = Dict[String, String]()
    var params = Dict[String, String]()
    var used = Dict[String, List[String]]()
    for f in glob("kernels", "", False):
        var src = read(f)
        for ls in line_starts(src):
            var p = def_body(src, ls)
            if p < 0 or not at(src, p, "amar_"):
                continue
            var e = name_end(src, p + 5)
            if e == p + 5:
                continue
            var name = sub(src, p, e)
            if name not in file:
                names.append(name)
            file[name] = basename(f)
            params[name] = params_of(src, name)
            used[name] = List[String]()

    var users: List[String] = [
        REG,
        "serve/engine.mojo",
        "serve/spark.mojo",
        "serve/q4x.mojo",
        "serve/window.mojo",
        "kernels/amarbaro.mojo",
    ]
    users += glob("bench", "", False)
    users += glob("kernels", "test_", True)
    users += glob_suffix("kernels", "_harness.mojo")
    for u in users:
        var src = read(u)
        for name in names:
            if has_word(src, name):
                used[name].append(u)

    var reg = read(REG)
    var roles = Dict[String, List[String]]()
    for ls in line_starts(reg):
        if not at(reg, ls, "comptime "):
            continue
        var x0 = ls + 9
        var x1 = name_end(reg, x0)
        if x1 == x0 or not at(reg, x1, " = amar_"):
            continue
        var k0 = x1 + 3
        var k1 = name_end(reg, k0 + 5)
        if k1 == k0 + 5:
            continue
        var k = sub(reg, k0, k1)
        if k not in roles:
            roles[k] = List[String]()
        roles[k].append(sub(reg, x0, x1))
    var rb = reg.as_bytes()
    var i = reg.find("amar_")
    while i >= 0:
        var e = name_end(reg, i + 5)
        if e > i + 5 and e < len(rb) and rb[e] == 91:
            var k = sub(reg, i, e)
            if k not in roles:
                roles[k] = List[String]()
            i = reg.find("amar_", e + 1)
        else:
            i = reg.find("amar_", i + 1)

    var order = names.copy()
    for a in range(1, len(order)):
        var j = a
        while j > 0:
            var x = order[j]
            var y = order[j - 1]
            if file[x] < file[y] or (file[x] == file[y] and x < y):
                order.swap_elements(j, j - 1)
                j -= 1
            else:
                break

    var out = String(
        "# Kernel census\n\n",
        "Generated by `tools/kernel-census.mojo`; do not hand-edit. Every `amar_*` kernel\n",
        "must be reachable from `serve/registry.mojo`, a bench, or a test.\n\n",
        "| kernel | file | template params | registry roles | used by |\n",
        "|---|---|---|---|---|\n",
    )
    for name in order:
        var r = join(roles[name], ", ") if name in roles else String()
        if r == "" and name == "amar_matmul_skinny_q8row":
            r = "gemm_q8 dispatch"
        out += String(
            "| `", name, "` | `", file[name], "` | `", params[name], "` | ",
            r, " | ", join(used[name], ", "), " |\n",
        )
    out += "\n## Tests\n\n"
    out += "Every `kernels/test_*.mojo`, the gate script that runs it, and its first docstring line.\n\n"
    out += "| test | run by | covers |\n|---|---|---|\n"
    var gates = List[String]()
    var gtext = List[String]()
    for g in ["run-tests.sh", "tools/merge-gate.sh", "tools/mega-gate.sh"]:
        if exists(g):
            gates.append(g)
            gtext.append(read(g))
    for t in glob("kernels", "test_", True):
        var tn = basename(t)
        var stem = sub(tn, 0, tn.byte_length() - 5)
        var by = List[String]()
        for gi in range(len(gates)):
            var s = gtext[gi]
            var p = s.find(stem)
            while p >= 0:
                var e = p + stem.byte_length()
                if e >= s.byte_length():
                    break
                var c = Int(s.as_bytes()[e])
                if not ((c >= 48 and c <= 57) or (c >= 65 and c <= 90) or (c >= 97 and c <= 122) or c == 95):
                    break
                p = s.find(stem, e)
            if p >= 0:
                by.append(gates[gi])
        var bys = join(by, ", ") if len(by) > 0 else String("manual")
        out += String("| `", tn, "` | ", bys, " | ", docline(read(t)), " |\n")

    var check = False
    for a in argv():
        if a == "--check":
            check = True
    var stale = False
    if not check:
        with open("docs/KERNELS.md", "w") as f:
            f.write(out)
    else:
        stale = read("docs/KERNELS.md") != out

    var in_reg = 0
    var orphans = List[String]()
    for name in names:
        if join(used[name], "").find("registry.mojo") >= 0:
            in_reg += 1
        if len(used[name]) == 0:
            orphans.append(name)
    print(len(names), "kernels,", in_reg, "in registry,", len(orphans), "orphans")
    for n in orphans:
        print("ORPHAN:", n, file[n])
    if stale:
        print("STALE: docs/KERNELS.md does not match the kernels on disk;")
        print("       regenerate: ./.work/kernel-census (no --check), then commit docs/KERNELS.md")
    if len(orphans) > 0 or stale:
        exit(1)
