"""Stage-0 simulation for bench/ngram-tree-protocol.md: n-gram draft tree beside the k=2 MTP window.

usage: mojo build -I serve tools/ngram-tree-sim.mojo -o SIM; SIM TRACE_DIR PROMPT_DIR [OFF]
TRACE_DIR holds bench/mtp-prompts.sh logs run with BARO_SPEC_DBG=1 and K=2 (p*.B2.log); PROMPT_DIR
holds the p*.tokens. Per traced window: generated index g = pos - n_prompt + OFF, MTP yield
1 + n_acc, tree yield 1 + longest ngram_propose_tree branch prefix matching gen[g:], rows m plus
the branch tokens (capped at 8), cost 23.1 + 5.0 * (rows - 3) ms. Prints one line per prompt and
arm, then the median tok/s per arm over all prompts and over the code prompts.
"""
from std.sys import argv
from std.os import listdir
from ngram import ngram_propose_tree


def ints(line: String, start: Int) raises -> List[Int]:
    var out = List[Int]()
    var parts = line.split(" ")
    for i in range(start, len(parts)):
        if parts[i].byte_length() > 0:
            out.append(Int(parts[i]))
    return out^


def field(line: String, key: String) raises -> Int:
    var parts = line.split(" ")
    for i in range(len(parts)):
        if parts[i].startswith(key):
            return Int(String(parts[i])[byte=key.byte_length():])
    raise Error("no " + key + " in: " + line)


def cost(rows: Int) -> Float64:
    return 23.1 + 5.0 * Float64(rows - 3)


def median(var x: List[Float64]) -> Float64:
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            if x[j] < x[i]:
                var t = x[i]
                x[i] = x[j]
                x[j] = t
    var n = len(x)
    if n == 0:
        return 0.0
    return x[n // 2] if n % 2 == 1 else 0.5 * (x[n // 2 - 1] + x[n // 2])


def is_code(name: String) -> Bool:
    return "py" in name or "code" in name or "fib" in name or "json" in name or "sql" in name or "rust" in name or "bash" in name


def main() raises:
    var args = argv()
    var trace = String(args[1])
    var pdir = String(args[2])
    var off = Int(args[3]) if len(args) > 3 else 0
    var arms_b: List[Int] = [0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1]
    var arms_l: List[Int] = [0, 2, 3, 4, 2, 3, 4, 2, 3, 4, 3, 3, 3]
    var arms_w: List[Int] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 4, 6]
    var na = len(arms_b)
    var all_tps = List[List[Float64]]()
    var code_tps = List[List[Float64]]()
    for _ in range(na):
        all_tps.append(List[Float64]())
        code_tps.append(List[Float64]())
    var names = listdir(pdir)
    for ni in range(len(names)):
        var f = names[ni]
        if not f.endswith(".tokens"):
            continue
        var name = String(f[byte = 0 : f.byte_length() - 7])
        var prompt = ints(open(pdir + "/" + f, "r").read().replace("\n", " "), 0)
        var log = open(trace + "/" + name + ".B2.log", "r").read()
        var gen = List[Int]()
        var wins = List[String]()
        for ln in log.splitlines():
            var s = String(ln)
            if s.startswith("GENERATED:"):
                gen = ints(s, 1)
            elif s.startswith("win pos="):
                wins.append(s)
        if len(gen) == 0 or len(wins) == 0:
            raise Error("no GENERATED or win lines in " + name + ".B2.log (BARO_SPEC_DBG=1, K=2?)")
        var n_prompt = len(prompt)
        for a in range(na):
            var toks = 0
            var ms = 0.0
            for w in range(len(wins)):
                var m = field(wins[w], "m=")
                var n_acc = field(wins[w], "n_acc=")
                var g = field(wins[w], "pos=") - n_prompt + off
                if g < 0 or g >= len(gen):
                    continue
                var y = 1 + n_acc
                var rows = m
                if arms_b[a] > 0:
                    var hist = prompt.copy()
                    for i in range(g):
                        hist.append(gen[i])
                    var tree = ngram_propose_tree(hist, arms_l[a], arms_b[a], arms_w[a])
                    var best = 0
                    for bi in range(len(tree)):
                        var k = 0
                        while k < len(tree[bi]) and g + k < len(gen) and tree[bi][k] == gen[g + k]:
                            k += 1
                        best = max(best, k)
                        rows += len(tree[bi])
                    rows = min(rows, 8)
                    y = 1 + max(n_acc, best)
                toks += y
                ms += cost(rows)
            var tps = 1000.0 * Float64(toks) / ms
            all_tps[a].append(tps)
            if is_code(name):
                code_tps[a].append(tps)
            print(name, "B", arms_b[a], "L", arms_l[a], "windows", len(wins), "tokens", toks, "tok/s", tps)
    var base_all = median(all_tps[0].copy())
    var base_code = median(code_tps[0].copy())
    for a in range(na):
        var ma = median(all_tps[a].copy())
        var mc = median(code_tps[a].copy())
        print("ARM B", arms_b[a], "L", arms_l[a], "W", arms_w[a], "all", ma, "x", ma / base_all, "code", mc, "x", mc / base_code if base_code > 0 else 0.0, "n_code", len(code_tps[a]))
