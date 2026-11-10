"""Kernel receipt for bench/llama-stint-protocol.md run 10 (not an arm): the kq prefill GEMM routes on real Ornith rows at
m = 1024, cold weights (NBUF rotating copies of each tensor, > the 96 MB Infinity Cache): run 9's kq_dequant and kq_gemm
(f16 scratch + amar_matmul_wmma_pipe) timed apart, and kq_gemm_lds (amar_matmul_prefill_kq, kernels/matmul_prefill_kq.mojo, decode in the B
staging). Median of REPEATS, each ITERS launches, us per GEMM. Shapes: ffn_gate Q4_K [12288, 4096], ffn_down Q6_K
[4096, 12288].

usage: bench_kq_prefill PACKDIR   (an engine-pack --kq of the Ornith Q4_K_M file)
"""
from std.math import ceildiv
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major
from gemv_kq import kq_bytes, GT_Q4K, GT_Q6K
from prefill_rows import kq_dequant, kq_gemm, kq_gemm_lds
from tio import median

comptime bf16 = DType.bfloat16
comptime f16 = DType.float16
comptime f32 = DType.float32
comptime u8 = DType.uint8
comptime MP = 1024
comptime NBUF = 4
comptime ITERS = 8
comptime REPEATS = 7


def tensor_off(idx: String, name: String) raises -> Int:
    for l in idx.split("\n"):
        var p = l.split(" ")
        if len(p) == 4 and String(p[0]) == name:
            return atol(String(p[2]))
    raise Error("FAIL index: no tensor " + name)


def shape[N: Int, K: Int, GT: Int](ctx: DeviceContext, pack: String, idx: String, name: String) raises:
    var nbytes = N * (K // 256) * kq_bytes[GT]()
    var w_h = ctx.enqueue_create_host_buffer[u8](nbytes)
    var w_d = ctx.enqueue_create_buffer[u8](NBUF * nbytes)
    var a_d = ctx.enqueue_create_buffer[bf16](MP * K)
    var x16_d = ctx.enqueue_create_buffer[f16](MP * K)
    var ws_d = ctx.enqueue_create_buffer[f16](NBUF * N * K)
    var c_d = ctx.enqueue_create_buffer[f32](MP * N)
    ctx.synchronize()
    with open(pack + "/pack.bin", "r") as f:
        _ = f.seek(tensor_off(idx, name))
        var raw = f.read_bytes(nbytes)
        for i in range(nbytes):
            w_h[i] = raw[i]
    for b in range(NBUF):
        ctx.enqueue_copy(dst_buf=DeviceBuffer[u8](ctx, w_d.unsafe_ptr().unsafe_offset(b * nbytes), nbytes, owning=False), src_buf=w_h)
    a_d.enqueue_fill(Scalar[bf16](0.5))
    ctx.synchronize()
    var At = rebind[TileTensor[bf16, type_of(row_major[MP, K]()), MutAnyOrigin]](TileTensor(a_d, row_major[MP, K]()))
    var Ct = rebind[TileTensor[f32, type_of(row_major[MP, N]()), MutAnyOrigin]](TileTensor(c_d, row_major[MP, N]()))
    var t_deq = List[Float64]()
    var t_pipe = List[Float64]()
    var t_lds = List[List[Float64]]()
    comptime for ui in range(4):
        t_lds.append(List[Float64]())
    for rep in range(REPEATS + 1):
        var t0 = perf_counter_ns()
        for i in range(ITERS):
            var b = i % NBUF
            kq_dequant[N, K](ctx, Int(w_d.unsafe_ptr()) + b * nbytes, GT, Int(ws_d.unsafe_ptr()) + b * N * K * 2)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        for i in range(ITERS):
            var b = i % NBUF
            kq_gemm[N, K, MP, bf16](ctx, Int(a_d.unsafe_ptr()), Int(x16_d.unsafe_ptr()), Int(ws_d.unsafe_ptr()) + b * N * K * 2, Int(c_d.unsafe_ptr()), MP)
        ctx.synchronize()
        var t2 = perf_counter_ns()
        if rep > 0:
            t_deq.append(Float64(t1 - t0) / ITERS / 1e3)
            t_pipe.append(Float64(t2 - t1) / ITERS / 1e3)
        comptime for ui in range(4):
            comptime U = 1 << ui
            var t3 = perf_counter_ns()
            for i in range(ITERS):
                var b = i % NBUF
                kq_gemm_lds[N, K, False, U, U](ctx, At, Int(w_d.unsafe_ptr()) + b * nbytes, GT, Ct, MP)
            ctx.synchronize()
            if rep > 0:
                t_lds[ui].append(Float64(perf_counter_ns() - t3) / ITERS / 1e3)
    var d = median(t_deq.copy())
    var p = median(t_pipe.copy())
    print(name, "kind", GT, "N", N, "K", K, "m", MP, " dequant", d, "us  pipe", p, "us  dequant+pipe", d + p, "us")
    comptime for ui in range(4):
        var l = median(t_lds[ui].copy())
        print("  lds U", 1 << ui, " ", l, "us  (dequant+pipe)/lds", (d + p) / l, " pipe/lds", p / l)


def main() raises:
    var a = argv()
    if len(a) != 2:
        raise Error("usage: bench_kq_prefill PACKDIR")
    var pack = String(a[1])
    var idx: String
    with open(pack + "/index.txt", "r") as f:
        idx = f.read()
    var ctx = DeviceContext()
    shape[12288, 4096, GT_Q4K](ctx, pack, idx, "blk.0.ffn_gate.weight")
    shape[4096, 4096, GT_Q4K](ctx, pack, idx, "blk.0.attn_gate.weight")
    shape[8192, 4096, GT_Q6K](ctx, pack, idx, "blk.0.attn_qkv.weight")
    shape[4096, 12288, GT_Q6K](ctx, pack, idx, "blk.0.ffn_down.weight")
