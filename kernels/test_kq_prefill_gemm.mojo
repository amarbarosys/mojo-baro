"""Kq batched prefill GEMM on real Ornith Q4_K / Q6_K rows at m = 1, 17, 128, 1024: run 9's dequant + WMMA (kernels/prefill_rows.mojo
kq_dequant + kq_gemm), run 10's in-staging decode (kq_gemm_lds -> amar_matmul_prefill_kq, kernels/matmul_prefill_kq.mojo, the path serve/window.mojo
gemm_pw_kq launches), next to the m=1 GEMV per row (amar_gemv_kq as window.gemm_kq launches it). The ACC=True instance of the
lds kernel must double its own output bit for bit. Every arm is dumped for tools/kq-prefill-ref.py, the fp64 numpy oracle over the same
dequantised bytes (bar |err| <= 2^-8 |ref| + 2^-19 sum|a w|, see the oracle); this binary fails only on a NaN or a missing tensor.

usage: test_kq_prefill_gemm PACKDIR OUTDIR   (PACKDIR: an engine-pack --kq of a Q4_K_M qwen35 dense file)
"""
from std.math import ceildiv
from std.sys import argv
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major
from gemv_kq import amar_gemv_kq, kq_bytes, GT_Q4K, GT_Q6K
from matmul_skinny import ROW_WAVES, ROW_THREADS
from prefill_rows import kq_dequant, kq_gemm, kq_gemm_lds
from q4x_kernels import UP

comptime bf16 = DType.bfloat16
comptime f16 = DType.float16
comptime f32 = DType.float32
comptime u8 = DType.uint8
comptime MP = 1024


def tensor_at(idx: String, name: String) raises -> Tuple[Int, Int, Int]:
    for l in idx.split("\n"):
        var p = l.split(" ")
        if len(p) == 4 and String(p[0]) == name:
            var kind = 12 if String(p[1]) == "q4_k" else (14 if String(p[1]) == "q6_k" else 0)
            return (kind, atol(String(p[2])), atol(String(p[3])))
    raise Error("FAIL index: no tensor " + name)


def lcg(mut st: UInt64) -> Float32:
    st = st * 6364136223846793005 + 1442695040888963407
    return Float32(Int(st >> 40)) / Float32(1 << 23) - 1


def dump(path: String, p: Int, nbytes: Int) raises:
    with open(path, "w") as f:
        f.write_bytes(Span[UInt8](unsafe_ptr=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=p), length=nbytes))


def run_case[N: Int, K: Int, GT: Int](ctx: DeviceContext, pack: String, idx: String, name: String, tag: String, odir: String) raises:
    var t = tensor_at(idx, name)
    if t[0] != GT or t[2] != N * K:
        raise Error("FAIL " + name + ": kind " + String(t[0]) + " elements " + String(t[2]) + ", want " + String(GT) + " " + String(N * K))
    var nbytes = N * (K // 256) * kq_bytes[GT]()
    var w_h = ctx.enqueue_create_host_buffer[u8](nbytes)
    var w_d = ctx.enqueue_create_buffer[u8](nbytes)
    var a_h = ctx.enqueue_create_host_buffer[bf16](MP * K)
    var a_d = ctx.enqueue_create_buffer[bf16](MP * K)
    var x16_d = ctx.enqueue_create_buffer[f16](MP * K)
    var ws_d = ctx.enqueue_create_buffer[f16](N * K)
    var ws_h = ctx.enqueue_create_host_buffer[f16](N * K)
    var c_d = ctx.enqueue_create_buffer[f32](MP * N)
    var c_h = ctx.enqueue_create_host_buffer[f32](MP * N)
    var g_d = ctx.enqueue_create_buffer[f32](MP * N)
    var g_h = ctx.enqueue_create_host_buffer[f32](MP * N)
    var l_d = ctx.enqueue_create_buffer[f32](MP * N)
    var l_h = ctx.enqueue_create_host_buffer[f32](MP * N)
    var l2_h = ctx.enqueue_create_host_buffer[f32](MP * N)
    ctx.synchronize()
    with open(pack + "/pack.bin", "r") as f:
        _ = f.seek(t[1])
        var raw = f.read_bytes(nbytes)
        for i in range(nbytes):
            w_h[i] = raw[i]
    var st = UInt64(11 + N)
    for i in range(MP * K):
        a_h[i] = (lcg(st) * 4).cast[bf16]()
    ctx.enqueue_copy(dst_buf=w_d, src_buf=w_h)
    ctx.enqueue_copy(dst_buf=a_d, src_buf=a_h)
    c_d.enqueue_fill(0)
    g_d.enqueue_fill(0)
    kq_dequant[N, K](ctx, Int(w_d.unsafe_ptr()), GT, Int(ws_d.unsafe_ptr()))
    ctx.enqueue_copy(dst_buf=ws_h, src_buf=ws_d)
    ctx.synchronize()
    dump(odir + "/" + tag + ".w.bin", Int(w_h.unsafe_ptr()), nbytes)
    dump(odir + "/" + tag + ".a.bf16", Int(a_h.unsafe_ptr()), MP * K * 2)
    dump(odir + "/" + tag + ".ws.f16", Int(ws_h.unsafe_ptr()), N * K * 2)
    comptime a_l = row_major[1, K]()
    comptime c_l = row_major[N]()
    comptime b_l = row_major[1]()
    var ob = TileTensor(DeviceBuffer[bf16](ctx, a_d.unsafe_ptr(), 1, owning=False), b_l)
    var At = rebind[TileTensor[bf16, type_of(row_major[MP, K]()), MutAnyOrigin]](TileTensor(a_d, row_major[MP, K]()))
    var Lt = rebind[TileTensor[f32, type_of(row_major[MP, N]()), MutAnyOrigin]](TileTensor(l_d, row_major[MP, N]()))
    for m in [1, 17, 128, 1024]:
        kq_gemm[N, K, MP, bf16](ctx, Int(a_d.unsafe_ptr()), Int(x16_d.unsafe_ptr()), Int(ws_d.unsafe_ptr()), Int(c_d.unsafe_ptr()), m)
        l_d.enqueue_fill(0)
        kq_gemm_lds[N, K, False](ctx, At, Int(w_d.unsafe_ptr()), GT, Lt, m)
        ctx.enqueue_copy(dst_buf=l_h, src_buf=l_d)
        kq_gemm_lds[N, K, True](ctx, At, Int(w_d.unsafe_ptr()), GT, Lt, m)
        ctx.enqueue_copy(dst_buf=l2_h, src_buf=l_d)
        for r in range(m):
            var a = TileTensor(DeviceBuffer[bf16](ctx, a_d.unsafe_ptr().unsafe_offset(r * K), K, owning=False), a_l)
            var c = TileTensor(DeviceBuffer[f32](ctx, g_d.unsafe_ptr().unsafe_offset(r * N), N, owning=False), c_l)
            ctx.enqueue_function[amar_gemv_kq[GT, GT, 0, type_of(a_l), type_of(c_l), type_of(b_l)]](
                a, w_d.unsafe_ptr(), c, ob, Int32(N), Int32(K), Int32(N), Int32(0), grid_dim=ceildiv(N, ROW_WAVES), block_dim=ROW_THREADS)
        ctx.enqueue_copy(dst_buf=c_h, src_buf=c_d)
        ctx.enqueue_copy(dst_buf=g_h, src_buf=g_d)
        ctx.synchronize()
        var dmax = Float32(0)
        var lmax = Float32(0)
        var gmax = Float32(0)
        var nan = 0
        var acc_bad = 0
        for i in range(m * N):
            var d = abs(c_h[i] - g_h[i])
            if d != d or c_h[i] != c_h[i] or l_h[i] != l_h[i]:
                nan += 1
            dmax = max(dmax, d)
            lmax = max(lmax, abs(l_h[i] - g_h[i]))
            gmax = max(gmax, abs(g_h[i]))
            if l2_h[i] != 2 * l_h[i]:
                acc_bad += 1
        dump(odir + "/" + tag + ".m" + String(m) + ".wmma.f32", Int(c_h.unsafe_ptr()), m * N * 4)
        dump(odir + "/" + tag + ".m" + String(m) + ".gemv.f32", Int(g_h.unsafe_ptr()), m * N * 4)
        dump(odir + "/" + tag + ".m" + String(m) + ".lds.f32", Int(l_h.unsafe_ptr()), m * N * 4)
        print(tag, name, "kind", GT, "N", N, "K", K, "m", m, " max|wmma-gemv|", dmax, " max|lds-gemv|", lmax, " max|gemv|", gmax, " nan", nan, " acc!=2x", acc_bad)
        if nan != 0:
            raise Error("FAIL " + tag + " m " + String(m) + ": NaN in a GEMM output")
        if acc_bad != 0:
            raise Error("FAIL " + tag + " m " + String(m) + ": ACC instance is not 2x the plain lds output on " + String(acc_bad) + " elements")
    with open(odir + "/" + tag + ".meta", "w") as f:
        f.write(String(GT) + " " + String(N) + " " + String(K) + " " + String(MP) + " 1 17 128 1024\n")


def main() raises:
    var a = argv()
    if len(a) != 3:
        raise Error("usage: test_kq_prefill_gemm PACKDIR OUTDIR")
    var pack = String(a[1])
    var out = String(a[2])
    var idx: String
    with open(pack + "/index.txt", "r") as f:
        idx = f.read()
    var ctx = DeviceContext()
    run_case[12288, 4096, GT_Q4K](ctx, pack, idx, "blk.0.ffn_gate.weight", "gate_q4k", out)
    run_case[32, 4096, GT_Q4K](ctx, pack, idx, "blk.0.ssm_alpha.weight", "alpha_q4k", out)
    run_case[4096, 12288, GT_Q6K](ctx, pack, idx, "blk.0.ffn_down.weight", "down_q6k", out)
    print("dumped", out, "-> tools/kq-prefill-ref.py", out)
