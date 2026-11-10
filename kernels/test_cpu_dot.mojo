"""Step 0c: one q4 row dot, CPU emulation (cpu/ops.mojo) vs amar_matmul_skinny_q4rowb on real pack rows.

usage: test_cpu_dot PACKDIR
Exit 0 only if every output is bit-equal. Prints the arm line, the flags and the mismatch count.
"""
from std.memory import bitcast
from std.sys import argv
from std.math import ceildiv
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from matmul_skinny import amar_matmul_skinny_q4rowb, ROW_WAVES, ROW_THREADS
from ops import q4_row_dot

comptime bf16 = DType.bfloat16
comptime f32 = DType.float32
comptime NR = 512
comptime H = 4096
comptime TOKS = 4


def tensor_at(idx: String, name: String) raises -> List[Int]:
    for l in idx.split("\n"):
        var p = l.split(" ")
        if len(p) == 4 and String(p[0]) == name:
            return [atol(String(p[2])), atol(String(p[3]))]
    raise Error("FAIL index: no tensor " + name)


def read_at(path: String, off: Int, n: Int) raises -> List[UInt8]:
    with open(path, "r") as f:
        _ = f.seek(off)
        return f.read_bytes(n)


def run[K: Int](dir: String, idx: String, name: String, toks: Array[Int, 4], ctx: DeviceContext) raises -> Int:
    comptime nb = K // 32
    comptime q_layout = row_major[NR, K // 2]()
    comptime s_layout = row_major[NR, nb]()
    comptime a_layout = row_major[1, K]()
    comptime p_layout = row_major[1, 1, NR]()
    var tn = tensor_at(idx, name)
    var off = tn[0]
    var n = tn[1]
    var rows = n // K
    var bad = 0
    var total = 0
    for seg in range(2):
        var r0 = 0 if seg == 0 else rows - NR
        var qb = read_at(dir + "/pack.bin", off + r0 * (K // 2), NR * (K // 2))
        var sb = read_at(dir + "/pack.bin", off + n // 2 + r0 * nb * 2, NR * nb * 2)
        var q_d = ctx.enqueue_create_buffer[DType.uint8](NR * (K // 2))
        var s_d = ctx.enqueue_create_buffer[DType.float16](NR * nb)
        ctx.enqueue_copy(q_d, qb.unsafe_ptr())
        ctx.enqueue_copy(s_d, sb.unsafe_ptr().unsafe_bitcast[Float16]())
        for t in toks:
            var ab = List[BFloat16](length=K, fill=0)
            for c in range(K // H):
                var e = read_at(dir + "/pack.bin", (t + c) * H * 2, H * 2)
                for i in range(H):
                    ab[c * H + i] = e.unsafe_ptr().unsafe_bitcast[BFloat16]().unsafe_offset(i).unsafe_load()
            var a_d = ctx.enqueue_create_buffer[bf16](K)
            var p_d = ctx.enqueue_create_buffer[f32](NR)
            ctx.enqueue_copy(a_d, ab.unsafe_ptr())
            var A = TileTensor(a_d, a_layout)
            var Q = TileTensor(q_d, q_layout)
            var S = TileTensor(s_d, s_layout)
            var P = TileTensor(p_d, p_layout)
            ctx.enqueue_function[amar_matmul_skinny_q4rowb[2, 1, type_of(a_layout), type_of(q_layout), type_of(s_layout), type_of(p_layout)]](
                A, Q, S, P, Int32(1), Int32(NR), Int32(K), grid_dim=ceildiv(NR, ROW_WAVES), block_dim=ROW_THREADS,
            )
            var got = List[Float32](length=NR, fill=0)
            ctx.enqueue_copy(got.unsafe_ptr(), p_d)
            ctx.synchronize()
            for r in range(NR):
                var w = q4_row_dot(ab.unsafe_ptr(), qb.unsafe_ptr().unsafe_offset(r * (K // 2)), sb.unsafe_ptr().unsafe_bitcast[Float16]().unsafe_offset(r * nb), nb)
                total += 1
                if bitcast[DType.uint32, 1](w) != bitcast[DType.uint32, 1](got[r]):
                    if bad < 3:
                        print("  mismatch", name, "seg", seg, "tok", t, "row", r, "cpu", w, "gpu", got[r])
                    bad += 1
    print(name, "K", K, "mismatches", bad, "of", total)
    return bad


def main() raises:
    var a = argv()
    if len(a) != 2:
        raise Error("usage: test_cpu_dot PACKDIR")
    var dir = String(a[1])
    var idx: String
    with open(dir + "/index.txt", "r") as f:
        idx = f.read()
    var ctx = DeviceContext()
    var toks = [1000, 5000, 20000, 100000]
    print("arm gpu=amar_matmul_skinny_q4rowb[UNROLL=2,MR=1] cpu=cpu/ops.mojo:q4_row_dot pack=" + dir)
    var bad = run[4096](dir, idx, "blk.3.attn_output.weight", toks, ctx)
    bad += run[4096](dir, idx, "blk.3.ffn_gate.weight", toks, ctx)
    bad += run[12288](dir, idx, "blk.3.ffn_down.weight", toks, ctx)
    if bad != 0:
        raise Error("MISMATCH " + String(bad))
    print("PASS test_cpu_dot: cpu == gpu bitwise")
