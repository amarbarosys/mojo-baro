from std.sys import llvm_intrinsic
from std.sys.info import _is_amd_rdna3, _is_amd_rdna4
from max.gpu import lane_id
from max.gpu.host import DeviceContext

comptime R4 = _is_amd_rdna4()
comptime KF = 8 if R4 else 16
comptime F16x = SIMD[DType.float16, KF]
comptime Acc = SIMD[DType.float32, 8]


@always_inline
def wmma(a: F16x, b: F16x, c: Acc) -> Acc:
    return llvm_intrinsic["llvm.amdgcn.wmma.f32.16x16x16.f16", Acc](a, b, c)


@always_inline
def load_frag(p: UnsafePointer[Float16, MutAnyOrigin], lane: Int, k: Int, ld: Int) -> F16x:
    var base = (lane % 16) * ld + k
    comptime if R4:
        var h = 4 * (lane // 16)
        return rebind[F16x](p.load[width=4](base + h).join(p.load[width=4](base + 8 + h)))
    else:
        return rebind[F16x](p.load[width=16](base))


@always_inline
def store_acc(p: UnsafePointer[Float32, MutAnyOrigin], acc: Acc, lane: Int, ld: Int):
    var h = lane // 16
    comptime for i in range(8):
        var row = (i + 8 * h) if R4 else (2 * i + h)
        p.store(row * ld + lane % 16, acc[i])


def wmma_gemm_16x16(
    a: UnsafePointer[Float16, MutAnyOrigin],
    b: UnsafePointer[Float16, MutAnyOrigin],
    c: UnsafePointer[Float32, MutAnyOrigin],
    k_dim: Int32, lda: Int32, ldb: Int32, ldc: Int32,
):
    comptime assert _is_amd_rdna3() or R4, "RDNA3/RDNA4 only"
    var lane = Int(lane_id())
    var acc = Acc(0)
    for k in range(0, Int(k_dim), 16):
        acc = wmma(load_frag(a, lane, k, Int(lda)), load_frag(b, lane, k, Int(ldb)), acc)
    store_acc(c, acc, lane, Int(ldc))


def main() raises:
    comptime K = 64
    var ctx = DeviceContext()
    var a_h = ctx.enqueue_create_host_buffer[DType.float16](16 * K)
    var b_h = ctx.enqueue_create_host_buffer[DType.float16](16 * K)
    var c_h = ctx.enqueue_create_host_buffer[DType.float32](256)
    ctx.synchronize()
    for r in range(16):
        for k in range(K):
            a_h[r * K + k] = Float16((r * 3 + k) % 7 - 3)
            b_h[r * K + k] = Float16((r + 2 * k) % 5 - 2 + (r % 3))
    var a_d = ctx.enqueue_create_buffer[DType.float16](16 * K)
    var b_d = ctx.enqueue_create_buffer[DType.float16](16 * K)
    var c_d = ctx.enqueue_create_buffer[DType.float32](256)
    ctx.enqueue_copy(dst_buf=a_d, src_buf=a_h)
    ctx.enqueue_copy(dst_buf=b_d, src_buf=b_h)
    ctx.enqueue_function[wmma_gemm_16x16](a_d.unsafe_ptr(), b_d.unsafe_ptr(), c_d.unsafe_ptr(), Int32(K), Int32(K), Int32(K), Int32(16), grid_dim=1, block_dim=32)
    ctx.enqueue_copy(dst_buf=c_h, src_buf=c_d)
    ctx.synchronize()
    var bad = 0
    var want01 = Float32(0)
    for r in range(16):
        for c in range(16):
            var want = Float32(0)
            var mag = Float32(0)
            for k in range(K):
                var pr = Float32(a_h[r * K + k]) * Float32(b_h[c * K + k])
                want += pr
                mag += abs(pr)
            if r == 0 and c == 1:
                want01 = want
            if abs(c_h[r * 16 + c] - want) > mag * Float32(1.9073486328125e-06):
                bad += 1
    print("C[0,1] gpu", c_h[1], "want", want01)
    if bad != 0:
        raise Error("FAIL wmma-gfx: " + String(bad) + "/256 mismatch")
    print("PASS wmma-gfx: 256/256 within 2^-19 sum|ab|, K=64")
