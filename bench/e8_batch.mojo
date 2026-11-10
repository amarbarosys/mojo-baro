# Run the E8 evaluator in batches, scoring after each one.
# Mojo port of bench/e8_batch.py. Same batching, skip, stop-after and merge behaviour;
# the running tally reads the scorer's own per-batch output (<out>.json, correct_exact)
# so bench/e8_score.py stays the single scoring implementation.
#
# Batches are stratified so each carries the same type ratio as the whole set.
# Already-finished batches (<prefix>-bNN.raw.json exists) are skipped, so an
# interrupted run resumes by re-invoking with the same --prefix.
#
# Env always wins over defaults: BARO_E8_TMAX / BARO_E8_ANS_MAX are read, then
# --tmax / --ans-max override them. The Python version once reset a caller's
# BARO_E8_TMAX to its own default (E12-long, rc=250 on the first 8k item).
#
# build: ./.venv/bin/mojo build bench/e8_batch.mojo -I . -o .work/e8_batch
# usage: .work/e8_batch [--batches N] [--prefix P] [--arms a,b] [--tmax T] [--ans-max A]
#                       [--stop-after N] [--dry-run]

from std.ffi import external_call
from std.os import getenv, setenv
from std.os.path import exists
from std.sys import argv
from std.time import perf_counter_ns

from grammar.json_value import JSONDoc, parse_json_file


def system(cmd: String) -> Int:
    """Run a shell command, streaming its output; return its exit code."""
    var c = cmd + "\0"
    var st = external_call["system", Int32](c.unsafe_ptr())
    return Int((st >> 8) & 0xFF) if st >= 0 else -1


def field_str(doc: JSONDoc, idx: Int, key: String) -> String:
    var vi = doc.get_field(idx, key)
    return doc.get(vi).s if vi >= 0 else String("")


def pad2(i: Int) -> String:
    return ("0" if i < 10 else "") + String(i)


def fixed1(x: Float64) -> String:
    var n = Int(x * 10.0 + 0.5)
    return String(n // 10) + "." + String(n % 10)


def read_text(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def items_span(raw: String) raises -> Tuple[Int, Int]:
    """Byte range of the items array's contents in a harness raw dump.

    bench_latent_handoff.mojo writes `...,"items":[<items>]}` with nothing after
    the items array, so the contents run from after `"items":[` to before the final `]}`.
    """
    var key = String("\"items\":[")
    var start = raw.find(key)
    var end = raw.byte_length() - 2
    if start < 0 or String(unsafe_from_utf8=raw.as_bytes()[end:]) != "]}":
        raise Error("raw dump is not in the harness's items layout")
    return (start + key.byte_length(), end)


def main() raises:
    var args = argv()
    var n_batches = 4
    var prefix = String("results/e8/round4")
    var arms_s = String("0,T,L8-raw,L8-soft,L32-soft")
    var ans_max = getenv("BARO_E8_ANS_MAX", "512")
    var tmax = getenv("BARO_E8_TMAX", "1088")
    var stop_after = 0
    var dry_run = False
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--dry-run":
            dry_run = True
            i += 1
            continue
        if i + 1 >= len(args):
            raise Error("missing value for " + a)
        var v = String(args[i + 1])
        if a == "--batches":
            n_batches = atol(v)
        elif a == "--prefix":
            prefix = v
        elif a == "--arms":
            arms_s = v
        elif a == "--ans-max":
            ans_max = v
        elif a == "--tmax":
            tmax = v
        elif a == "--stop-after":
            stop_after = atol(v)
        else:
            raise Error("unknown arg " + a)
        i += 2

    var tasks_path = getenv("BARO_E8_TASKS", "bench/data/e8_tasks.json")
    var tasks = parse_json_file(tasks_path)
    var troot = tasks.get(tasks.root).arr.copy()

    # stratified_batches: types in sorted order, ids round-robin within each type.
    var types = List[String]()
    for j in range(len(troot)):
        var t = field_str(tasks, troot[j], "type")
        var seen = False
        for k in range(len(types)):
            if types[k] == t:
                seen = True
        if not seen:
            types.append(t)
    for x in range(1, len(types)):
        var v = types[x]
        var y = x - 1
        while y >= 0 and types[y] > v:
            types[y + 1] = types[y]
            y -= 1
        types[y + 1] = v
    var batches = List[List[String]]()
    for _ in range(n_batches):
        batches.append(List[String]())
    for k in range(len(types)):
        var c = 0
        for j in range(len(troot)):
            if field_str(tasks, troot[j], "type") == types[k]:
                batches[c % n_batches].append(field_str(tasks, troot[j], "id"))
                c += 1

    var sizes = String("")
    for b in range(n_batches):
        sizes += ("" if b == 0 else ", ") + String(len(batches[b]))
    print(len(troot), "tasks ->", n_batches, "batches (" + sizes + " items), arms", arms_s)
    if dry_run:
        return

    # gpu-wait does not forward the caller's env; latent-handoff.sh forwards these explicitly.
    _ = setenv("BARO_E8_ANS_MAX", ans_max)
    _ = setenv("BARO_E8_TMAX", tmax)
    print("env: BARO_E8_TMAX=" + tmax, " BARO_E8_ANS_MAX=" + ans_max, " tasks=" + tasks_path)

    var arms = List[String]()
    for a in arms_s.split(","):
        arms.append(String(a))
    var ok = List[Int]()
    var tot = List[Int]()
    for _ in range(len(arms)):
        ok.append(0)
        tot.append(0)
    var done = 0
    var t_start = perf_counter_ns()

    for b in range(n_batches):
        var out = prefix + "-b" + pad2(b + 1)
        var raw = out + ".raw.json"
        if exists(raw):
            print("\n=== batch", String(b + 1) + "/" + String(n_batches) + ": already done, skipping ===")
        else:
            print("\n=== batch", String(b + 1) + "/" + String(n_batches) + ":", len(batches[b]), "items ===")
            var ids = String("")
            for k in range(len(batches[b])):
                ids += ("" if k == 0 else ",") + batches[b][k]
            var t0 = perf_counter_ns()
            var rc = system("bash bench/latent-handoff.sh --ids " + ids + " --arms " + arms_s + " --out " + out)
            if rc != 0 or not exists(raw):
                print("batch", b + 1, "FAILED (rc=" + String(rc) + "); stopping")
                raise Error("batch failed")
            print("batch", b + 1, "took", fixed1(Float64(perf_counter_ns() - t0) / 60e9), "min")

        var scored = parse_json_file(out + ".json")
        var items = scored.get(scored.get_field(scored.root, "items")).arr.copy()
        for it in range(len(items)):
            done += 1
            var arr = scored.get(scored.get_field(items[it], "arms")).arr.copy()
            for q in range(len(arr)):
                var name = field_str(scored, arr[q], "arm")
                for z in range(len(arms)):
                    if String(arms[z]) == name:
                        tot[z] += 1
                        var ce = scored.get_field(arr[q], "correct_exact")
                        if ce >= 0 and scored.get(ce).b:
                            ok[z] += 1
        print("--- running tally after batch", String(b + 1) + ":", done, "items ---")
        var hi = 0
        var lo = 1 << 30
        for z in range(len(arms)):
            if tot[z] > 0:
                print("   ", String(arms[z]), String(ok[z]) + "/" + String(tot[z]), fixed1(100.0 * Float64(ok[z]) / Float64(tot[z])) + "%")
                hi = max(hi, ok[z])
                lo = min(lo, ok[z])
        if hi >= lo:
            print("    spread across arms:", hi - lo, "items")

        if stop_after > 0 and b + 1 >= stop_after and b + 1 < n_batches:
            print("\nstopping after batch", b + 1, "as asked;", n_batches - b - 1, "batches left.")
            print("continue with: .work/e8_batch --batches", n_batches, "--prefix", prefix)
            return

    # Merge: batch 1's header, every batch's items, same layout the harness writes.
    var head = read_text(prefix + "-b01.raw.json")
    var hs = items_span(head)
    var merged = String(unsafe_from_utf8=head.as_bytes()[: hs[0]])
    for b in range(n_batches):
        var r = read_text(prefix + "-b" + pad2(b + 1) + ".raw.json")
        var sp = items_span(r)
        if b > 0:
            merged += ","
        merged += String(unsafe_from_utf8=r.as_bytes()[sp[0] : sp[1]])
    merged += "]}"
    var mprefix = prefix + "-merged"
    with open(mprefix + ".raw.json", "w") as f:
        f.write(merged)
    var rc = system("python3 bench/e8_score.py " + mprefix + ".raw.json " + tasks_path + " " + mprefix + ".json " + mprefix + ".md")
    if rc != 0:
        raise Error("e8_score.py failed on the merged dump")
    print("\ntotal", fixed1(Float64(perf_counter_ns() - t_start) / 60e9), "min")
    print("merged:", mprefix + ".json", mprefix + ".md")
