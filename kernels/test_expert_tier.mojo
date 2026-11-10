"""B4 stage 2b gate 1: the tier's own LRU reproduces the offline replay.

`bench/moe-locality.py` replayed an LRU over the router trace and reported
76.7% at capacity 64 and 80.5% at 256 (`bench/moe-locality-protocol.md`,
`1aa06d5`). Those numbers are what the tier was built on, so the tier's LRU
has to BE that LRU: this test replays `serve/expert_tier.mojo::LayerLru` over
the same trace rows and compares.

If the two disagree, the live hit rate measures a policy nobody predicted and
gate 3's comparison is meaningless, which is why this runs before any GPU
minute.

Build: ./.venv/bin/mojo build kernels/test_expert_tier.mojo -I kernels -I serve -o .work/test_expert_tier
Trace: .work/b1/experts.txt (BARO_EXPERTS, 20 prompts x 64 tokens x 40 layers).
Skips cleanly when the trace is absent, so a cold checkout does not fail here.
"""
from std.collections import List
from std.sys import exit

from expert_tier import LayerLru

comptime TRACE = ".work/b1/experts.txt"
comptime N_LAYERS = 40
comptime EXPECT_64 = 0.767
comptime EXPECT_256 = 0.805
comptime TOL = 0.002


@fieldwise_init
struct Row(Copyable, Movable):
    var layer: Int
    var ids: List[Int]


def read_trace(path: String) raises -> List[List[Row]]:
    """Blocks of rows, one block per prompt, split on the trace's own
    `# trace ...` header lines. The replay resets its cache per block, and so
    does this, because a request starts with a cold tier."""
    var blocks = List[List[Row]]()
    var cur = List[Row]()
    with open(path, "r") as f:
        for line in f.read().splitlines():
            if line.byte_length() == 0:
                continue
            if line.startswith("#"):
                if len(cur) > 0:
                    blocks.append(cur^)
                    cur = List[Row]()
                continue
            var parts = line.split(" ")
            if len(parts) < 3:
                continue
            var ids = List[Int]()
            for i in range(2, len(parts)):
                var v = Int(parts[i])
                if v >= 0:
                    ids.append(v)
            cur.append(Row(Int(parts[1]), ids^))
    if len(cur) > 0:
        blocks.append(cur^)
    return blocks^


def replay(blocks: List[List[Row]], cap: Int) raises -> Tuple[Int, Int]:
    var hits = 0
    var refs = 0
    for bi in range(len(blocks)):
        var lru = List[LayerLru]()
        for _ in range(N_LAYERS):
            lru.append(LayerLru(cap))
        ref block = blocks[bi]
        for ri in range(len(block)):
            ref row = block[ri]
            for j in range(len(row.ids)):
                refs += 1
                var r = lru[row.layer].touch(row.ids[j])
                if r[1]:
                    hits += 1
    return (hits, refs)


def main() raises:
    var blocks: List[List[Row]]
    try:
        blocks = read_trace(TRACE)
    except:
        print("SKIP: no trace at", TRACE, "(run BARO_EXPERTS first)")
        return
    var rows = 0
    for i in range(len(blocks)):
        rows += len(blocks[i])
    print("trace:", len(blocks), "prompts,", rows, "rows")
    var fails = 0
    var caps = List[Int]()
    caps.append(64)
    caps.append(256)
    var wants = List[Float64]()
    wants.append(EXPECT_64)
    wants.append(EXPECT_256)
    for ci in range(len(caps)):
        var cap = caps[ci]
        var want = wants[ci]
        var r = replay(blocks, cap)
        var rate = Float64(r[0]) / Float64(r[1])
        var ok = abs(rate - want) <= TOL
        print(
            "  ", "PASS" if ok else "FAIL", "cap", cap, ": hit rate", rate,
            " replay said", want, " hits", r[0], "of", r[1],
        )
        if not ok:
            fails += 1
    if fails == 0:
        print("PASS: expert tier LRU reproduces bench/moe-locality.py")
    else:
        print("FAIL:", fails, "capacity/capacities disagree with the replay")
        exit(1)
