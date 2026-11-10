"""WeightStore gate (big-models P1, CPU only, no GPU minute).

1. Bytes: every tensor bound through any tier equals its bytes in the shard mapping (memcmp), on every bind.
2. Counts: walking the layers in order for 3 passes, a sweep over pinned prefix x ring depth must read exactly what
   the tier rules predict (streamed layers S: pass 1 reads S; later passes read S again if S > ring slots, else 0) and
   every other bind must count as resident. A wrong count is a slot evicted early or a hit claimed that was not one.
3. Experts: batches of 8 distinct experts through 16 and 64 LRU slots, bytes on every bind, binds = resident + reads,
   and a repeated batch is fully resident.

Build: ./.venv/bin/mojo build kernels/test_wstore.mojo -I serve -I tools -o .work/test_wstore
Run:   .work/test_wstore DENSE.gguf MOE.gguf   (Llama-3.2-1B Q4_K_M and the RegesCore bake on this box)
"""
from std.ffi import external_call
from std.sys import argv, exit
from gguf_reader import GGUFModel
from wstore import WeightStore

def check(cond: Bool, what: String) -> Int:
    if not cond:
        print("FAIL", what)
    return 0 if cond else 1


def bytes_ok(ws: WeightStore, layer: Int, base: Int) -> Bool:
    ref g = ws.trunk[layer]
    for i in range(len(g.ext)):
        if external_call["memcmp", Int32](g.data(base, i), g.ext[i].maddr, g.ext[i].length) != 0:
            return False
    return True


def blocks(path: String) raises -> Int:
    var g = GGUFModel(path)
    for item in g.ints.items():
        if item.key.endswith(".block_count"):
            return item.value
    raise Error("no block_count in " + path)


def dense_case(path: String, nl: Int, pin: Int, nslot: Int, mmap_only: Bool) raises -> Int:
    var ws = WeightStore(path, nl, pin, nslot, 0, mmap_only)
    var good = True
    for _ in range(3):
        for l in range(nl):
            var base = ws.bind_layer(l)
            good = good and bytes_ok(ws, l, base)
    var tag = "dense pin " + String(pin) + " ring " + String(nslot) + (" mmap" if mmap_only else "")
    var f = check(good, tag + ": bytes differ from the mapping")
    var s = 0 if mmap_only else nl - pin
    var want = 0 if mmap_only else (s + (2 * s if s > nslot else 0))
    f += check(ws.t_reads == want, tag + ": reads " + String(ws.t_reads) + " want " + String(want))
    f += check(ws.t_binds == ws.t_resident + ws.t_reads + ws.t_mmap, tag + ": binds do not split into resident + reads + mmap")
    print("  ", tag, "|", ws.report())
    return f


def moe_case(path: String, nl: Int, slots: Int) raises -> Int:
    var ws = WeightStore(path, nl, nl, 1, slots, False)
    var good = True
    var f = 0
    var x = UInt64(12345)
    for tok in range(4):
        for l in range(4):
            var ids = List[Int]()
            while len(ids) < 8:
                x = x * 6364136223846793005 + 1442695040888963407
                var e = Int((x >> 33) % 64)
                if e not in ids:
                    ids.append(e)
            var bases = ws.bind_experts(l, ids)
            for i in range(len(ids)):
                var g = ws.expert(l, ids[i])
                for k in range(len(g.ext)):
                    good = good and external_call["memcmp", Int32](g.data(bases[i], k), g.ext[k].maddr, g.ext[k].length) == 0
            if tok == 3:
                var before = ws.e_resident
                _ = ws.bind_experts(l, ids)
                f += check(ws.e_resident - before == 8, "experts slots " + String(slots) + ": a repeated batch was not fully resident")
    f += check(good, "experts slots " + String(slots) + ": bytes differ from the mapping")
    f += check(ws.e_binds == ws.e_resident + ws.e_reads, "experts slots " + String(slots) + ": binds != resident + reads")
    print("   experts slots", slots, "|", ws.report())
    return f


def main() raises:
    var a = argv()
    if len(a) != 3:
        print("usage: test_wstore DENSE.gguf MOE.gguf")
        exit(2)
    var dense = String(a[1])
    var moe = String(a[2])
    var nl = blocks(dense)
    var fails = dense_case(dense, nl, 0, 0, True)
    fails += dense_case(dense, nl, nl, 0, False)
    for pin in [0, nl // 2, nl - 1]:
        for nslot in [1, 2, 3, nl]:
            fails += dense_case(dense, nl, pin, nslot, False)
    var ml = blocks(moe)
    fails += moe_case(moe, ml, 16)
    fails += moe_case(moe, ml, 64)
    if fails:
        print(fails, "FAILURES")
        exit(1)
    print("PASS test_wstore")
