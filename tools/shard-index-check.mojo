"""p8 step 1 check: index the qwen4exp shards, print count, bytes, types.

usage: shard-index-check FIRST_SHARD
"""
from std.sys import argv
from shard_index import ShardIndex


def main() raises:
    var ix = ShardIndex(argv()[1])
    var total = 0
    var by = Dict[Int, Int]()
    for e in ix.t.items():
        total += e.value.nbytes
        by[e.value.gtype] = by.get(e.value.gtype, 0) + e.value.nbytes
    print("tensors", len(ix.t), "bytes", total, "shards", len(ix.maps))
    for e in by.items():
        print("  type", e.key, "bytes", e.value)
    var r = ix["blk.0.ffn_down_exps.weight"]
    var p = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=r.addr)
    var s = 0
    for i in range(64):
        s += Int(p[i])
    print("blk.0.ffn_down_exps type", r.gtype, "dims", r.dims[0], r.dims[1], r.dims[2], "first64 sum", s)
