"""Prefill GEMM tile table (bench/tile-tune-protocol.md stage 1).

`<packdir>/tune.txt`, one row per line: `N K m_max variant`, variant 0..5 as in
bench/bench_tile_sweep.mojo (wmma16x256 wmma32x256 wmma64x128 wmma128x128
lds64x128 lds128x128). First row with matching N, K and m <= m_max wins; no
row, no file, or BARO_TUNE=0 -> -1, the registry rule. Flat List[Int], 4 per row.
Blank and # lines are skipped; any other malformed row raises at boot.
"""
from std.os import getenv
from std.os.path import exists


def load_tune(packdir: String) raises -> List[Int]:
    var t = List[Int]()
    var path = packdir + "/tune.txt"
    if getenv("BARO_TUNE", "1") == "0" or not exists(path):
        print("TUNE rows 0", "(BARO_TUNE=0)" if getenv("BARO_TUNE", "1") == "0" else "(no tune.txt)")
        return t^
    with open(path, "r") as f:
        for line in f.read().splitlines():
            if line.byte_length() == 0 or line.startswith("#"):
                continue
            var parts = line.split(" ")
            if len(parts) != 4:
                raise Error("tune.txt row needs 4 single-space fields: '" + line + "'")
            for i in range(4):
                t.append(Int(parts[i]))
            if t[len(t) - 1] < 0 or t[len(t) - 1] > 5:
                raise Error("tune.txt variant out of range: " + line)
    print("TUNE rows", len(t) // 4, "from", path)
    return t^


def pick(t: List[Int], n: Int, k: Int, m: Int) -> Int:
    for i in range(0, len(t), 4):
        if t[i] == n and t[i + 1] == k and m <= t[i + 2]:
            return t[i + 3]
    return -1
