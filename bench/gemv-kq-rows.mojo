"""R rows per wave for amar_gemv_kq (prototype, kernels/gemv_kq.mojo untouched): each wave runs R rows of a Q4_K GEMV
with independent accumulators, so the loads of all R rows can be in flight together. One row per wave keeps 864 B of
weights per wave at K=1536 (vibethinker's FFN up, 8960 rows) and reaches 260 GB/s, latency-bound.
Gate: C byte-equal to amar_gemv_kq on random weights for every R. Then device time per R (rocprofv3 --kernel-trace, tools/gemv-rows-times.py; this program prints only the gate).
"""
from std.memory import bitcast
from max.gpu import block_idx, lane_id, thread_idx, WARP_SIZE
from max.gpu.host import DeviceBuffer, DeviceContext
from layout import TileTensor, TensorLayout, row_major
from gemv_kq import amar_gemv_kq, kq_bytes, kq_lane_part, kq_tree16, kq_wave_sum, GT_Q4K
from matmul_skinny import ROW_WAVES, ROW_THREADS
from q4x_kernels import UP

comptime bf16 = DType.bfloat16
comptime f32 = DType.float32
comptime ROWS = 8960
comptime K = 1536
comptime SETS = 16
comptime REPS = 9
comptime WARM = 200  # x SETS launches, about 3 s


def gemv_rows[R: Int, ALayout: TensorLayout, CLayout: TensorLayout](
    A: TileTensor[bf16, ALayout, MutAnyOrigin], W: UP, C: TileTensor[f32, CLayout, MutAnyOrigin], n: Int32, k_dim: Int32
):
    comptime assert C.flat_rank == 1
    var k = Int(k_dim)
    var lane = Int(lane_id())
    var row0 = (Int(block_idx.x) * ROW_WAVES + Int(thread_idx.x) // WARP_SIZE) * R
    var acc = SIMD[f32, 16 * R](0)  # the R row accumulators side by side
    comptime for r in range(R):
        if row0 + r < Int(n):
            acc = acc.insert[offset=16 * r](kq_lane_part[GT_Q4K, bf16](A, W, (row0 + r) * (k // 256) * kq_bytes[GT_Q4K](), k, lane))
    comptime for r in range(R):
        var total = kq_wave_sum(kq_tree16(acc.slice[16, offset=16 * r]()))
        if lane == 0 and row0 + r < Int(n):
            C[row0 + r] = rebind[C.ElementType](total)


def launch[R: Int](ctx: DeviceContext, X: TileTensor, w: DeviceBuffer[DType.uint8], C: TileTensor) raises:
    ctx.enqueue_function[gemv_rows[R, type_of(X).LayoutType, type_of(C).LayoutType]](
        rebind[TileTensor[bf16, type_of(X).LayoutType, MutAnyOrigin]](X), w.unsafe_ptr(),
        rebind[TileTensor[f32, type_of(C).LayoutType, MutAnyOrigin]](C), Int32(ROWS), Int32(K),
        grid_dim=(ROWS + ROW_WAVES * R - 1) // (ROW_WAVES * R), block_dim=ROW_THREADS,
    )


def main() raises:
    var ctx = DeviceContext()
    var nbytes = ROWS * (K // 256) * kq_bytes[GT_Q4K]()
    var w_h = ctx.enqueue_create_host_buffer[DType.uint8](nbytes)
    var x_h = ctx.enqueue_create_host_buffer[bf16](K)
    ctx.synchronize()
    var s: UInt64 = 7
    for i in range(nbytes):
        s = s * 6364136223846793005 + 1442695040888963407
        w_h[i] = UInt8(s >> 56)
    for i in range(nbytes // 144):  # finite f16 scales: d and dmin with exponent bits below inf
        w_h[i * 144 + 1] = w_h[i * 144 + 1] & 0x3B
        w_h[i * 144 + 3] = w_h[i * 144 + 3] & 0x3B
    for i in range(K):
        s = s * 6364136223846793005 + 1442695040888963407
        x_h[i] = (Float32(Int(s >> 40) % 2001 - 1000) / 1000).cast[bf16]()
    var ws = List[DeviceBuffer[DType.uint8]]()
    for _ in range(SETS):
        ws.append(ctx.enqueue_create_buffer[DType.uint8](nbytes))
        ctx.enqueue_copy(dst_buf=ws[len(ws) - 1], src_buf=w_h)
    var x_d = ctx.enqueue_create_buffer[bf16](K)
    ctx.enqueue_copy(dst_buf=x_d, src_buf=x_h)
    var c_d = ctx.enqueue_create_buffer[f32](ROWS)
    var o_d = ctx.enqueue_create_buffer[bf16](ROWS)
    var ref_h = ctx.enqueue_create_host_buffer[f32](ROWS)
    var got_h = ctx.enqueue_create_host_buffer[f32](ROWS)
    var X = TileTensor(x_d, row_major[1, K]())
    var C = TileTensor(c_d, row_major[ROWS]())
    var O = TileTensor(o_d, row_major[ROWS]())
    ctx.enqueue_function[amar_gemv_kq[GT_Q4K, GT_Q4K, 0, type_of(X).LayoutType, type_of(C).LayoutType, type_of(O).LayoutType, bf16]](
        X, ws[0].unsafe_ptr(), C, O, Int32(ROWS), Int32(K), Int32(ROWS), Int32(nbytes),
        grid_dim=(ROWS + ROW_WAVES - 1) // ROW_WAVES, block_dim=ROW_THREADS,
    )
    ctx.enqueue_copy(dst_buf=ref_h, src_buf=c_d)
    ctx.synchronize()
    comptime for ri in range(4):
        comptime R = 1 << ri
        c_d.enqueue_fill(0)
        launch[R](ctx, X, ws[0], C)
        ctx.enqueue_copy(dst_buf=got_h, src_buf=c_d)
        ctx.synchronize()
        var bad = 0
        for i in range(ROWS):
            if bitcast[DType.uint32](got_h[i]) != bitcast[DType.uint32](ref_h[i]):
                bad += 1
        print("R", R, "mismatches", bad, "of", ROWS)
        if bad:
            raise Error("FAIL rows R=" + String(R))
    # timing: warm the clocks first (the memory clock steps up after ~1 s of load and halves the time),
    # then every round runs amar_gemv_kq, R 1, 2, 4, 8 in turn, each over all SETS copies
    for _ in range(WARM):
        for i in range(SETS):
            launch[4](ctx, X, ws[i], C)
    ctx.synchronize()
    for _ in range(REPS):  # each variant sweeps all copies: none rereads weights another left in the cache
        for i in range(SETS):
            ctx.enqueue_function[amar_gemv_kq[GT_Q4K, GT_Q4K, 0, type_of(X).LayoutType, type_of(C).LayoutType, type_of(O).LayoutType, bf16]](
                X, ws[i].unsafe_ptr(), C, O, Int32(ROWS), Int32(K), Int32(ROWS), Int32(nbytes),
                grid_dim=(ROWS + ROW_WAVES - 1) // ROW_WAVES, block_dim=ROW_THREADS,
            )
        for i in range(SETS):
            launch[1](ctx, X, ws[i], C)
        for i in range(SETS):
            launch[2](ctx, X, ws[i], C)
        for i in range(SETS):
            launch[4](ctx, X, ws[i], C)
        for i in range(SETS):
            launch[8](ctx, X, ws[i], C)
    ctx.synchronize()
    print("ok gemv-kq-rows: R 1 2 4 8 byte-equal to amar_gemv_kq,", nbytes, "weight bytes x", SETS, "copies")
