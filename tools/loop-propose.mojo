"""Proposer step of the self-optimising loop (bench/loop-protocol.md): the served model reads the engine sources of
an iteration plus the engine's BARO_PROFILE shares and proposes ONE change per identity framing.

Usage: .work/loop-propose MODEL.gguf ITER [--n 4] [--region auto|attn|ssm|ffn|head] [--profile .work/profile-run.log]
         [--endpoint http://127.0.0.1:8083] [--mega|--no-mega] [--start I] [--max-tokens N]
         [--prepped] [--format sr|diff] [--repair N] [--mojo-rules]
Writes .work/loop/<ITER>/{meta.json,FILES,src/,prompt.md,cand-<i>.raw.md,cand-<i>.diff,cand-<i>.predict}.
Run from the repo root (tools/loop-run.sh does). Build: ./.venv/bin/mojo build tools/loop-propose.mojo -I . -o .work/loop-propose

The output-format block is a placeholder skeleton, not a worked example: iterations 002-005 saw 8 of 14 candidates
echo the example back. Iteration 009 (arm b) removes the three setup defects found after 001-007:
--format sr   SEARCH/REPLACE blocks, the diff is made here with `diff -u` (iteration 002 lost 4/4 to hunk headers);
--repair N    apply + CPU build each answer with the gate's own harness (tools/loop-harness.sh), send the error back
              up to N times (001-007 were one shot);
--mojo-rules  the repo's Mojo syntax correction layer goes into the system prompt (the 9B and 27B predate Mojo 1.x).
--endpoint soft:MODE  arm (c): MODEL is run locally by tools/soft-think (Soft Thinking, MODE soft or discrete, extra
              soft-think flags may follow MODE) instead of a llama-server endpoint.
--prepped     use src/ from tools/loop-prep.py instead of extracting the gguf.
The model is reached with curl (stdin-free, request body from a file); the reply is read with grammar/json_value.
"""
from std.subprocess import run
from std.sys import argv, exit, stderr
from std.os import getenv
from std.pathlib import Path
from grammar.json_value import JSONDoc, parse_json_bytes

comptime RULES = """You are optimising the inference engine embedded inside your own model file
(Qwythos-9B, AMD RX 7900 XTX gfx1100, Mojo). Decode is GPU-bound and
bandwidth-bound: single-token decode reads every weight once, so wins come from
fewer bytes moved, fewer kernel launches, and fused math -- not from FLOPs.
Propose exactly ONE change, one concern, as a unified diff against the files
shown (paths relative to the source dir, e.g. `--- a/matmul_skinny.mojo`). Rules:
- Only edit the files shown. Never touch anything else.
- Kernel files carry zero comments and zero docstrings.
- Output must stay bit-identical: the gate compares 64 greedy tokens.
- Do not change timing, printing, or profiling code.
- End with one line `PREDICT: <signed percent>` = your predicted change in tok/s_gen.
- Every symbol you reference must already exist in the sources shown. If your
  change needs a new kernel, you must write its full body in the same diff.
- Source files are given inside <file path="..."> tags. That is INPUT framing.
  Never use it in your answer -- your diff must use `--- a/<path>` / `+++ b/<path>`
  headers and @@ hunks, or it is discarded unparsed.

Output format, exactly:

<= 8 lines of rationale, then

```diff
--- a/<one of the files shown>
+++ b/<the same file>
@@ -<first line>,<count> +<first line>,<count> @@
 <an unchanged line, copied verbatim from the file>
-<the exact line you remove>
+<the line you put in its place>
 <an unchanged line, copied verbatim from the file>
```

then one line `PREDICT: <signed percent>`.

Every line inside the fence is a real line of the file shown or your real
replacement. No `...`, no angle-bracket placeholders left in, no name that is
absent from the binding table and the files shown."""

comptime SR_FORMAT = """Output format, exactly: <= 8 lines of rationale, then one or more edit blocks, then one line
`PREDICT: <signed percent>`. An edit block replaces one exact, contiguous run of lines of one file:

<<<<<<< SEARCH <file name, one of the files shown>
<the lines to replace, copied verbatim from the file, enough of them to occur exactly once>
=======
<the replacement lines>
>>>>>>> REPLACE

No diff, no line numbers, no `...`. Every SEARCH text must occur exactly once in the named file."""


@fieldwise_init
struct Res(Copyable, Movable):
    var out: String
    var rc: Int


@fieldwise_init
struct Span(Copyable, Movable):
    var a: Int
    var b: Int


@fieldwise_init
struct Reply(Copyable, Movable):
    var text: String
    var finish: String
    var tokens: Int
    var thinking_only: Bool


def die(msg: String):
    print("FAIL loop-propose:", msg, file=stderr)
    exit(1)


def q(s: String) -> String:
    return "'" + s.replace("'", "'\\''") + "'"


def sh(cmd: String) raises -> Res:
    var o = run(cmd + "\necho __rc=$?")
    var k = o.rfind("__rc=")
    return Res(String(o[byte=:k]), Int(atol(String(o[byte=k + 5:]).strip())))


def rd(p: String) raises -> String:
    with open(p, "r") as f:
        return f.read()


def wr(p: String, s: String) raises:
    with open(p, "w") as f:
        f.write(s)


def lines_of(s: String) -> List[String]:
    var out = List[String]()
    for l in s.split("\n"):
        out.append(String(l))
    return out^


def join(ls: List[String], a: Int, b: Int) -> String:
    var o = String()
    for i in range(a, b):
        if i > a:
            o += "\n"
        o += ls[i]
    return o


def blen(s: String) -> Int:
    return s.byte_length()


def count(s: String, sub: String) -> Int:
    var n = 0
    var i = s.find(sub)
    while i >= 0:
        n += 1
        i = s.find(sub, i + max(blen(sub), 1))
    return n


def fmt1(x: Float64) -> String:
    var t = Int(x * 10 + 0.5)
    return String(t // 10) + "." + String(t % 10)


def json_str(s: String) -> String:
    comptime HEX = "0123456789abcdef"
    var o = List[UInt8]()
    o.append(34)
    for b in s.as_bytes():
        if b == 34 or b == 92:
            o.append(92)
            o.append(b)
        elif b == 10:
            o.extend("\\n".as_bytes())
        elif b == 13:
            o.extend("\\r".as_bytes())
        elif b == 9:
            o.extend("\\t".as_bytes())
        elif b < 32:
            o.extend("\\u00".as_bytes())
            o.append(HEX.as_bytes()[Int(b >> 4)])
            o.append(HEX.as_bytes()[Int(b & 15)])
        else:
            o.append(b)
    o.append(34)
    return String(unsafe_from_utf8=o^)


def is_word(c: UInt8) -> Bool:
    return (c >= 48 and c <= 57) or (c >= 65 and c <= 90) or (c >= 97 and c <= 122) or c == 95


def identities() raises -> List[List[String]]:
    var rows = List[List[String]]()
    for l in lines_of(rd(getenv("HOME") + "/Brain/identities/identities-index.md")):
        var p = l.split("|")
        if len(p) < 6 or not l.startswith("| "):
            continue
        var num = String(p[1].strip())
        var cell = String(p[2].strip())
        if blen(num) != 2 or not num.as_bytes()[0] >= 48 or not num.as_bytes()[0] <= 57:
            continue
        if not cell.startswith("[[") or not cell.endswith("]]"):
            continue
        rows.append([String(cell[byte=2 : blen(cell) - 2]), String(p[4].strip())])
    return rows^


def bindings(src_dir: String) raises -> String:
    var txt = String()
    for f in ["registry.mojo", "engine.mojo"]:
        var p = src_dir + "/" + f
        if not Path(p).exists():
            continue
        var rows = String()
        for l in lines_of(rd(p)):
            if not l.startswith("comptime "):
                continue
            var e = l.find(" = amar_")
            if e < 0:
                continue
            var name = String(l[byte=9:e])
            var k = e + 3
            var kb = l.as_bytes()
            var j = k
            while j < len(kb) and is_word(kb[j]):
                j += 1
            rows += "\n  " + name + " = " + String(l[byte=k:j])
        if blen(rows):
            txt += ("\nkernel bindings defined in " + f + " -- these names, and ONLY these, "
                    + "are callable as ctx.enqueue_function[<name>]:" + rows + "\n")
    return txt


def slice_region(body: String, region: String, src_dir: String, body_file: String) raises -> String:
    var a: String
    var b: String
    if region == "attn" or region == "ssm":
        a = "# -- attention / ssm sub-block --"
        b = "# -- ffn sub-block --"
    elif region == "ffn":
        a = "# -- ffn sub-block --"
        b = "# -- head --"
    else:
        a = "# -- head --"
        b = "# advance by the window"
    var ls = lines_of(body)
    var ia = -1
    var ib = -1
    for i in range(len(ls)):
        if ia < 0 and a in ls[i]:
            ia = i
        elif ia >= 0 and b in ls[i]:
            ib = i
            break
    if ia < 0 or ib < 0:
        raise Error("region markers for " + region + " not found in " + body_file)
    var tbl = bindings(src_dir)
    if not blen(String(tbl.strip())):
        raise Error("no kernel binding table found; refusing to prompt blind")
    return (tbl + "\n" + body_file + " lines " + String(ia + 1) + "-" + String(ib) + " (target region `" + region
            + "`):\n" + join(ls, ia, ib))


def def_span(ls: List[String], name: String) raises -> Span:
    var d = -1
    for k in range(len(ls)):
        if ls[k].startswith("def " + name + "["):
            d = k
            break
    if d < 0:
        raise Error("no def " + name + " in mega.mojo")
    var i = d
    while i > 0 and ls[i - 1].startswith("@"):
        i -= 1
    var j = len(ls)
    for k in range(d + 1, len(ls)):
        if ls[k].startswith("@") or ls[k].startswith("def "):
            j = k
            break
    while j > i and not blen(String(ls[j - 1].strip())):
        j -= 1
    return Span(i, j)


def mega_def(region: String) -> String:
    if region == "attn":
        return "attn_phases"
    if region == "ssm":
        return "ssm_phases"
    if region == "ffn":
        return "ffn_phases"
    return "mega_body"


def slice_mega(src: String, region: String) raises -> String:
    var ls = lines_of(src)
    var h0 = def_span(ls, "ssm_phases").a
    var s = def_span(ls, mega_def(region))
    return ("\nYou are editing device code inside the persistent decode megakernel "
            + "(kernels/mega.mojo): one workgroup per row block, `grid_barrier` between "
            + "phases, every weight read once per token. There are no kernel launches to "
            + "add or remove here; wins are fewer bytes, fewer barriers, fewer passes. "
            + "Only the helpers shown and the names already used in the excerpt are callable.\n"
            + "\nmega.mojo, target region `" + region + "` = `" + mega_def(region) + "` (lines " + String(s.a + 1)
            + "-" + String(s.b) + "); the excerpt below is lines 1-" + String(h0) + " (helpers) and " + String(s.a + 1)
            + "-" + String(s.b) + " of that file:\n" + '\n<file path="mega.mojo">\n'
            + join(ls, 0, h0) + "\n\n" + join(ls, s.a, s.b) + "\n</file>")


def mojo_rules() raises -> String:
    var ls = lines_of(rd(".claude/skills/mojo-syntax/SKILL.md"))
    var take = String()
    for head in ["## Removed syntax", "## `var` is required", "## `def` is the only", "## Mojo functions that raise",
                 "## `comptime` replaces", "## Memory and pointer types"]:
        var i = -1
        for k in range(len(ls)):
            if ls[k].startswith(head):
                i = k
                break
        if i < 0:
            raise Error("mojo-syntax skill has no section " + head)
        var j = len(ls)
        for k in range(i + 1, len(ls)):
            if ls[k].startswith("## "):
                j = k
                break
        take += ("\n" if blen(take) else "") + join(ls, i, j)
    return "\n\nMojo changed after your training. These corrections override what you remember:\n\n" + take


def strip_think(s: String) -> String:
    var o = s
    var a = o.find("<think>")
    while a >= 0:
        var b = o.find("</think>", a)
        if b < 0:
            break
        o = String(o[byte=:a]) + String(o[byte=b + 8:])
        a = o.find("<think>")
    return o


def predict(body: String) -> String:
    var k = body.find("PREDICT:")
    if k < 0:
        return "none"
    var bs = body.as_bytes()
    var i = k + 8
    while i < len(bs) and (bs[i] == 32 or bs[i] == 9):
        i += 1
    var j = i
    if j < len(bs) and (bs[j] == 43 or bs[j] == 45):
        j += 1
    var d0 = j
    while j < len(bs) and ((bs[j] >= 48 and bs[j] <= 57) or bs[j] == 46):
        j += 1
    return String(body[byte=i:j]) if j > d0 else "none"


def ask(endpoint: String, model: String, outd: String, system: String, ident: String, max_tokens: Int, hist: List[String]) raises -> Reply:
    # RULES + corpus go in the system message, identical across branches, so the prefix is prefilled once and reused
    # (cache_prompt); the framing varies only in the user turn. hist: alternating assistant / user repair turns.
    var msgs = ('[{"role":"system","content":' + json_str(system) + '},{"role":"user","content":'
                + json_str("Framing for this attempt: " + ident + "\n\nPropose your one change now, in the output format given.") + "}")
    for i in range(len(hist)):
        msgs += ',{"role":"' + ("assistant" if i % 2 == 0 else "user") + '","content":' + json_str(hist[i]) + "}"
    wr(outd + "/request.json", '{"messages":' + msgs + '],"temperature":0.6,"max_tokens":' + String(max_tokens)
       + ',"cache_prompt":true}')
    var r: Res
    if endpoint.startswith("soft:"):
        # arm (c): tools/soft-think (Soft Thinking on libllama), mode and extra flags after "soft:" ("soft:soft --tau 0.05")
        var sr = sh(".work/soft-think/soft-think " + q(model) + " " + q(outd + "/request.json") + " " + q(outd + "/reply.json")
                    + " --mode " + String(endpoint[byte=5:]) + " --max-tokens " + String(max_tokens) + " >> " + q(outd + "/soft-think.log") + " 2>&1")
        if sr.rc != 0:
            raise Error("soft-think failed (" + String(sr.rc) + "), see " + outd + "/soft-think.log")
        r = Res(rd(outd + "/reply.json"), 0)
    else:
        r = sh("curl -sS --max-time 3600 -H 'Content-Type: application/json' --data-binary @" + q(outd + "/request.json")
               + " " + q(endpoint + "/v1/chat/completions") + " 2>&1")
    if r.rc != 0:
        raise Error("curl failed (" + String(r.rc) + "): " + String(r.out[byte=: min(300, blen(r.out))]))
    var doc = parse_json_bytes(List[UInt8](r.out.as_bytes()))
    var ch = doc.get_field(doc.root, "choices")
    if ch < 0 or len(doc.get(ch).arr) == 0:
        raise Error("no choices in reply: " + String(r.out[byte=: min(300, blen(r.out))]))
    var c0 = doc.get(ch).arr[0]
    var msg = doc.get_field(c0, "message")
    var text = String()
    var thinking_only = False
    var ci = doc.get_field(msg, "content")
    if ci >= 0 and doc.get(ci).is_string():
        text = doc.get(ci).s
    if not blen(String(text.strip())):
        var ri = doc.get_field(msg, "reasoning_content")
        if ri >= 0 and doc.get(ri).is_string():
            text = doc.get(ri).s
            thinking_only = True
    var fi = doc.get_field(c0, "finish_reason")
    var finish = doc.get(fi).s if fi >= 0 and doc.get(fi).is_string() else "none"
    var ti = doc.get_field(doc.get_field(doc.root, "usage"), "completion_tokens")
    return Reply(text, finish, Int(doc.get(ti).n) if ti >= 0 else -1, thinking_only)


def apply_sr(text: String, src_dir: String, work: String) raises -> Res:
    """SEARCH/REPLACE blocks -> Res(diff, 0) or Res(error for the model, 1)."""
    var names = List[String]()
    var bodies = List[String]()
    var at = text.find("<<<<<<< SEARCH ")
    var n = 0
    while at >= 0:
        n += 1
        var h = text.find("\n", at)
        var sep = text.find("=======\n", h) if h >= 0 else -1
        var end = text.find(">>>>>>> REPLACE", sep) if sep >= 0 else -1
        if end < 0:
            return Res("Edit block " + String(n) + " is not closed by ======= and >>>>>>> REPLACE lines.", 1)
        var f = String(text[byte=at + 15 : h].strip())
        # tolerate `mega.mojo`, "mega.mojo", a/mega.mojo (iteration 011: backticks rejected a block before its SEARCH
        # text was even checked)
        while f.startswith("`") or f.startswith('"') or f.startswith("'"):
            var f1 = String(f[byte=1:])
            f = f1
        while f.endswith("`") or f.endswith('"') or f.endswith("'"):
            var f2 = String(f[byte=: blen(f) - 1])
            f = f2
        if f.startswith("a/") or f.startswith("b/"):
            var f3 = String(f[byte=2:])
            f = f3
        var old = String(text[byte=h + 1 : sep])
        var new = String(text[byte=sep + 8 : end])
        if old.endswith("\n"):
            var o2 = String(old[byte=: blen(old) - 1])
            old = o2
        if new.endswith("\n"):
            var n2 = String(new[byte=: blen(new) - 1])
            new = n2
        if not Path(src_dir + "/" + f).exists():
            return Res("Edit block " + String(n) + " names `" + f + "`, which is not one of the files shown.", 1)
        var k = -1
        for i in range(len(names)):
            if names[i] == f:
                k = i
        if k < 0:
            names.append(f)
            bodies.append(rd(src_dir + "/" + f))
            k = len(names) - 1
        var c = count(bodies[k], old) if blen(old) else 0
        if c != 1:
            return Res("Edit block " + String(n) + ": its SEARCH text occurs " + String(c) + " times in `" + f
                       + "`, it must occur exactly once. Copy the lines verbatim.", 1)
        bodies[k] = bodies[k].replace(old, new)
        at = text.find("<<<<<<< SEARCH ", end)
    if n == 0:
        return Res("No edit block found. Answer with <<<<<<< SEARCH <file> / ======= / >>>>>>> REPLACE blocks.", 1)
    _ = sh("rm -rf " + q(work) + " && cp -r " + q(src_dir) + " " + q(work))
    var diff = String()
    for i in range(len(names)):
        wr(work + "/" + names[i], bodies[i])
        var d = sh("diff -u --label " + q("a/" + names[i]) + " --label " + q("b/" + names[i]) + " "
                   + q(src_dir + "/" + names[i]) + " " + q(work + "/" + names[i]))
        diff += d.out
    return Res(diff, 0)


def build_check(outd: String, work: String, entry: String) raises -> String:
    """Build the candidate exactly as tools/loop-gate.sh does; "" or the first compiler errors."""
    var main = outd + "/" + entry if entry.startswith("harness/") else work + "/" + String(entry[byte=4:])
    var r = sh("./.venv/bin/mojo build " + q(main) + " -I " + q(work) + " -I " + q(outd + "/harness") + " -o "
               + q(work + "/engine") + " 2>&1")
    if r.rc == 0:
        return ""
    var errs = String()
    var nerr = 0
    for l in lines_of(r.out):
        if "error:" in l and nerr < 12:
            errs += l + "\n"
            nerr += 1
    return errs if nerr else String(r.out[byte=max(0, blen(r.out) - 2000):])


def main() raises:
    var av = argv()
    if len(av) < 3:
        die("usage: loop-propose MODEL.gguf ITER [options], see tools/loop-propose.mojo")
    var model = String(av[1])
    var it = String(av[2])
    var n = 4
    var start = 0
    var max_tokens = 4096
    var repair = 0
    var mega = -1
    var region = String("auto")
    var profile = String(".work/profile-run.log")
    var endpoint = String("http://127.0.0.1:8083")
    var fmt = String("diff")
    var prepped = False
    var with_rules = False
    var i = 3
    while i < len(av):
        var a = String(av[i])
        var v = String(av[i + 1]) if i + 1 < len(av) else String()
        if a == "--n":
            n = Int(atol(v)); i += 1
        elif a == "--start":
            start = Int(atol(v)); i += 1
        elif a == "--max-tokens":
            max_tokens = Int(atol(v)); i += 1
        elif a == "--repair":
            repair = Int(atol(v)); i += 1
        elif a == "--region":
            region = v; i += 1
        elif a == "--profile":
            profile = v; i += 1
        elif a == "--endpoint":
            endpoint = v; i += 1
        elif a == "--format":
            fmt = v; i += 1
        elif a == "--mega":
            mega = 1
        elif a == "--no-mega":
            mega = 0
        elif a == "--prepped":
            prepped = True
        elif a == "--mojo-rules":
            with_rules = True
        else:
            die("unknown argument " + a)
        i += 1
    if fmt != "sr" and fmt != "diff":
        die("--format must be sr or diff")
    if not Path("tools/loop-harness.sh").exists():
        die("run from the repo root")
    var out = ".work/loop/" + it
    var src = out + "/src"
    var commit: String
    if prepped:
        if not Path(out + "/meta.json").exists():
            die("--prepped but no " + out + "/meta.json (run tools/loop-prep.py first)")
        var md = parse_json_bytes(List[UInt8](rd(out + "/meta.json").as_bytes()))
        commit = md.get(md.get_field(md.root, "baro.kernel.commit")).s
    else:
        _ = sh("mkdir -p " + q(src))
        var r = sh("./.venv/bin/python3 tools/gguf-extract.py " + q(model) + " --meta")
        if r.rc != 0:
            die("gguf-extract failed: " + String(r.out[byte=: min(300, blen(r.out))]))
        var md = parse_json_bytes(List[UInt8](r.out.as_bytes()))
        commit = md.get(md.get_field(md.root, "baro.kernel.commit")).s
        var files = md.get(md.get_field(md.root, "baro.kernel.files")).s.split(",")
        var flist = String()
        for f in files:
            var fs = String(f)
            flist += fs + "\n"
            _ = sh("mkdir -p \"$(dirname " + q(src + "/" + fs) + ")\"")
            wr(src + "/" + fs, md.get(md.get_field(md.root, "baro.kernel.src." + fs)).s)
        wr(out + "/FILES", flist)
        var root = md.get(md.root)
        var mj = String("{")
        for k in range(len(root.obj_keys)):
            var vk = md.get(root.obj_vals[k])
            if root.obj_keys[k].startswith("baro.") and vk.is_string():
                mj += ("," if blen(mj) > 1 else "") + "\n " + json_str(root.obj_keys[k]) + ": " + json_str(vk.s)
        wr(out + "/meta.json", mj + "\n}")
    var names = List[String]()
    var secs = List[Float64]()
    var shares = List[Float64]()
    var mnames = List[String]()
    var msecs = List[Float64]()
    var mshares = List[Float64]()
    for l in lines_of(rd(profile)):
        var w = l.split()
        if len(w) == 4 and w[0] == "profile:":
            names.append(String(w[1])); secs.append(atof(String(w[2]))); shares.append(atof(String(w[3])))
        if l.startswith("mega profile (last token, us): ssm sub-blocks ") and len(w) >= 17:
            var tot = atof(String(w[16]))
            var vals = [atof(String(w[7])), atof(String(w[10])), atof(String(w[12])), atof(String(w[14]))]
            var keys = ["ssm", "attn", "ffn", "head"]
            mnames.clear(); msecs.clear(); mshares.clear()
            for k in range(4):
                mnames.append(keys[k]); msecs.append(vals[k] / 1e6); mshares.append(vals[k] / tot)
    if len(mnames):
        names = mnames^; secs = msecs^; shares = mshares^
    if not len(names):
        die("no profile lines in " + profile)
    if region == "auto":
        var best = 0
        for k in range(len(names)):
            if shares[k] > shares[best]:
                best = k
        region = names[best]
    var prof_txt = String()
    for k in range(len(names)):
        prof_txt += ("\n" if k else "") + "  " + names[k] + ": " + fmt1(secs[k] * 1000) + " ms  (" + fmt1(shares[k] * 100) + "%)"
    var body_file = "window.mojo" if Path(src + "/window.mojo").exists() else "engine.mojo"
    var use_mega = mega == 1 or (mega == -1 and Path(src + "/mega.mojo").exists() and body_file == "window.mojo")
    var prompt = ("Engine source commit in this gguf: " + commit + "\nGPU time per decode run, by sub-block ("
                  + ("BARO_PROFILE=5, megakernel phases" if use_mega else "BARO_PROFILE=1") + "):\n" + prof_txt
                  + "\nTarget region: `" + region + "` (largest share).\n\n")
    if use_mega:
        prompt += slice_mega(rd(src + "/mega.mojo"), region)
    else:
        prompt += slice_region(rd(src + "/" + body_file), region, src, body_file)
        var kf: List[String] = ["elementwise.mojo", "matmul_skinny.mojo"]
        if region == "attn":
            kf = ["attn.mojo", "elementwise.mojo", "matmul_skinny.mojo"]
        elif region == "ssm":
            kf = ["ssm.mojo", "elementwise.mojo", "matmul_skinny.mojo"]
        for f in kf:
            prompt += '\n\n<file path="' + f + '">\n' + rd(src + "/" + f) + "\n</file>"
    wr(out + "/prompt.md", prompt)
    var rules = String(RULES)
    if fmt == "sr":
        rules = (String(rules[byte=: rules.find("Output format, exactly:")])
                 .replace("as a unified diff against the files shown (paths relative to the source dir, e.g. `--- a/matmul_skinny.mojo`)",
                          "as edit blocks against the files shown")
                 .replace("your diff must use `--- a/<path>` / `+++ b/<path>`\n  headers and @@ hunks, or it is discarded unparsed.",
                          "name files bare in edit blocks.")) + SR_FORMAT
    if with_rules:
        rules += mojo_rules()
    var system = rules + "\n\n" + prompt
    var entry = String()
    if repair:
        var hr = sh("tools/loop-harness.sh " + q(out))
        if hr.rc != 0:
            die("loop-harness failed: " + hr.out)
        var e0 = hr.out.find("entry=")
        entry = String(hr.out[byte=e0 + 6 :].split()[0])
    var ids = identities()
    var stop = min(len(ids), start + n)
    var idn = String()
    for k in range(start, stop):
        idn += ("," if k > start else "") + ids[k][0]
    print("region=" + region, "mega=" + String(use_mega), "prompt_chars=" + String(blen(prompt)), "identities=[" + idn + "]")
    for c in range(stop - start):
        var name = ids[start + c][0]
        var hist = List[String]()
        var log = String()
        var diff = String()
        var err = String()
        var body = String()
        var rep = Reply(String(), String("none"), -1, False)
        var cw = out + "/cand-" + String(c) + ".work"
        for attempt in range(repair + 1):
            rep = ask(endpoint, model, out, system, ids[start + c][1], max_tokens, hist)
            log += ("--- attempt " + String(attempt) + " finish=" + rep.finish + " completion_tok=" + String(rep.tokens)
                    + (" thinking_only" if rep.thinking_only else "") + "\n" + rep.text + "\n\n")
            body = strip_think(rep.text)
            if fmt == "sr":
                var r = apply_sr(body, src, cw)
                diff = r.out if r.rc == 0 else String()
                err = r.out if r.rc else String()
            else:
                var f0 = body.find("```diff\n")
                var f1 = body.find("```", f0 + 8) if f0 >= 0 else -1
                diff = String(body[byte=f0 + 8 : f1]) if f1 >= 0 else String()
                err = String() if f1 >= 0 else String("No ```diff fence found.")
            if not blen(err) and repair and fmt == "sr":
                var be = build_check(out, cw, entry)
                err = "Your change applied but does not compile. Compiler errors:\n" + be if blen(be) else String()
            print("  cand-" + String(c) + " attempt " + String(attempt) + ": " + ("ok" if not blen(err) else String(lines_of(err)[0][byte=: min(150, blen(lines_of(err)[0]))])))
            if not blen(err) or attempt == repair:
                break
            # A thinking-only reply goes back as one line: 6000 chars of its own rumination as the assistant turn made
            # the 27B echo it verbatim on every repair (iteration 009, cand-1).
            hist.append("(My previous reply ended while still thinking; it contained no edit block.)" if rep.thinking_only
                        else String(body[byte=max(0, blen(body) - 6000):]))
            hist.append(err + "\n\nFix it and answer again in the same output format, the whole change.")
        wr(out + "/cand-" + String(c) + ".raw.md", "identity: " + name + "\n\n" + log)
        wr(out + "/cand-" + String(c) + ".diff", diff)
        var pr = predict(body)
        wr(out + "/cand-" + String(c) + ".predict", pr + "\n")
        print("cand-" + String(c), "[" + name + "]", "diff=" + ("yes" if blen(diff) else "NO"), "lines=" + String(count(diff, "\n")),
              "predict=" + pr, "finish=" + rep.finish, "completion_tok=" + String(rep.tokens))
        if rep.finish == "length":
            print("  cand-" + String(c) + ": budget exhausted at " + String(max_tokens)
                  + " tokens -- no diff here is a harness result, not a proposer result")
