"""Byte test: amar_matmul_wmma_pipe_gelu (bias + GELU + fp16 in the epilogue) equals the default WMMA pipe GEMM with
f32 C followed by the same bias + GELU + fp16 cast as a separate kernel (newera bge_gpu k_gelu). bge's up-GEMM shape."""
from std.math import erf
from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext
from layout import Idx, TileTensor, row_major
from matmul_wmma_pipe import amar_matmul_wmma_pipe, amar_matmul_wmma_pipe_gelu, BLK_M, BLK_N, NTHREADS

comptime M = 256
comptime N = 1536
comptime K = 384
comptime FP = MutPointer[Scalar[DType.float32], MutAnyOrigin]
comptime HP = MutPointer[Scalar[DType.float16], MutAnyOrigin]


def k_gelu_ref(h: FP, bias: FP, hh: HP, n: Int32):
    var i = Int(block_idx.x) * 256 + Int(thread_idx.x)
    if i < Int(n):
        var u = h[unsafe_offset=i] + bias[unsafe_offset=i % N]
        hh[unsafe_offset=i] = (u * 0.5 * (1 + erf(u * 0.7071067811865476))).cast[DType.float16]()


def main() raises:
    var ctx = DeviceContext()
    var ah = ctx.enqueue_create_host_buffer[DType.float16](M * K)
    var wh = ctx.enqueue_create_host_buffer[DType.float16](N * K)
    var bh = ctx.enqueue_create_host_buffer[DType.float32](N)
    ctx.synchronize()
    for i in range(M * K):
        ah[i] = Float16(Float32((i * 37) % 17 - 8) / 16)
    for i in range(N * K):
        wh[i] = Float16(Float32((i * 53) % 23 - 11) / 64)
    for i in range(N):
        bh[i] = Float32(i % 7 - 3) / 4
    var a = ctx.enqueue_create_buffer[DType.float16](M * K)
    var w = ctx.enqueue_create_buffer[DType.float16](N * K)
    var b = ctx.enqueue_create_buffer[DType.float32](N)
    var c32 = ctx.enqueue_create_buffer[DType.float32](M * N)
    var ref16 = ctx.enqueue_create_buffer[DType.float16](M * N)
    var got16 = ctx.enqueue_create_buffer[DType.float16](M * N)
    ctx.enqueue_copy(dst_buf=a, src_buf=ah)
    ctx.enqueue_copy(dst_buf=w, src_buf=wh)
    ctx.enqueue_copy(dst_buf=b, src_buf=bh)
    var la = row_major(Idx[M], Idx[K])
    var lb = row_major(Idx[N], Idx[K])
    var lc = row_major(Idx[M], Idx[N])
    comptime k32 = amar_matmul_wmma_pipe[type_of(la), type_of(lb), type_of(lc), 4, 2, 2, 4, 1, 0]
    comptime kg = amar_matmul_wmma_pipe_gelu[type_of(la), type_of(lb), type_of(lc), 4, 2, 2, 4, 1]
    var g = (N // BLK_N, M // BLK_M)
    ctx.enqueue_function[k32](TileTensor(a, la), TileTensor(w, lb), TileTensor(c32, lc), Int32(M), Int32(N), Int32(K),
                              grid_dim=g, block_dim=NTHREADS)
    ctx.enqueue_function[k_gelu_ref](FP(unsafe_from_address=Int(c32.unsafe_ptr())), FP(unsafe_from_address=Int(b.unsafe_ptr())),
                                     HP(unsafe_from_address=Int(ref16.unsafe_ptr())), Int32(M * N),
                                     grid_dim=(M * N + 255) // 256, block_dim=256)
    ctx.enqueue_function[kg](TileTensor(a, la), TileTensor(w, lb), TileTensor(got16, lc),
                             FP(unsafe_from_address=Int(b.unsafe_ptr())), Int32(M), Int32(N), Int32(K),
                             grid_dim=g, block_dim=NTHREADS)
    var rh = ctx.enqueue_create_host_buffer[DType.float16](M * N)
    var gh = ctx.enqueue_create_host_buffer[DType.float16](M * N)
    ctx.enqueue_copy(dst_buf=rh, src_buf=ref16)
    ctx.enqueue_copy(dst_buf=gh, src_buf=got16)
    ctx.synchronize()
    var bad = 0
    var nz = 0
    for i in range(M * N):
        if rh[i].to_bits() != gh[i].to_bits():
            if bad < 5:
                print("MISMATCH at", i, "ref", rh[i], "got", gh[i])
            bad += 1
        if rh[i] != 0:
            nz += 1
    if bad or nz < M * N // 2:
        raise Error("FAIL wmma_pipe_gelu: ", bad, " byte mismatch(es), ", nz, " nonzero of ", M * N)
    print("PASS wmma_pipe_gelu byte-equal to wmma_pipe f32 + k_gelu,", M, "x", N, "x", K, ",", nz, "nonzero")
