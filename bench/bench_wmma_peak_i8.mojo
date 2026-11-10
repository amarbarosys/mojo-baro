"""WMMA issue-rate receipt: v_wmma_i32_16x16x16_iu8 against
v_wmma_f32_16x16x16_bf16, same register-only loop (kernels/wmma_peak.mojo
structure, NACC independent accumulators, no memory in the timed loop),
interleaved runs, ratio reported. bench/prefill-protocol.md R5: the int8
prefill kernel's issue model assumes iu8 is 2x the bf16 WMMA rate; this
measures it. NUM_CU is this box's XTX (arm-defining, see bench/wmma-peak.sh).
"""
from max.gpu import block_idx, thread_idx
from std.sys import has_accelerator
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from layout import TileTensor, TensorLayout, row_major
from layout.tensor_core import mma
from tio import median

comptime NACC = 8
comptime ITERS = 4096
comptime WAVES_PER_BLOCK = 4
comptime BLOCKS_PER_CU = 4
comptime NUM_CU = 96
comptime NUM_BLOCKS = NUM_CU * BLOCKS_PER_CU
comptime NTHREADS = WAVES_PER_BLOCK * 32
comptime NOUT = NUM_BLOCKS * NTHREADS
comptime c_layout = row_major[NOUT, 1]()
comptime REPEATS = 7


def peak_kernel[INT8: Bool, CLayout: TensorLayout](
    C: TileTensor[DType.float32, CLayout, MutAnyOrigin],
):
    comptime assert C.flat_rank == 2
    comptime ADT = DType.int8 if INT8 else DType.bfloat16
    comptime CDT = DType.int32 if INT8 else DType.float32
    var tid = thread_idx.x
    var lane = Int(tid) % 32
    var warp = Int(tid) // 32
    var global_wave = Int(block_idx.x) * WAVES_PER_BLOCK + warp

    var a = SIMD[ADT, 16](0)
    var b = SIMD[ADT, 16](0)
    comptime for i in range(16):
        a[i] = Scalar[ADT]((lane + i + global_wave) % 5 - 2)
        b[i] = Scalar[ADT]((lane * 2 + i - global_wave) % 3 - 1)

    var acc = SIMD[CDT, NACC * 8](0)
    comptime for j in range(NACC):
        comptime for i in range(8):
            acc[j * 8 + i] = Scalar[CDT]((j + i + lane) % 4)

    for _ in range(ITERS):
        comptime for j in range(NACC):
            var c = SIMD[CDT, 8](0)
            comptime for i in range(8):
                c[i] = acc[j * 8 + i]
            var d = SIMD[CDT, 8](0)
            mma(d, a, b, c)
            comptime for i in range(8):
                acc[j * 8 + i] = d[i]

    var s = Scalar[CDT](0)
    comptime for j in range(NACC):
        comptime for i in range(8):
            s += acc[j * 8 + i]
    C[global_wave * 32 + lane, 0] = rebind[C.ElementType](s.cast[DType.float32]())


def run[INT8: Bool](ctx: DeviceContext, C: TileTensor[DType.float32, type_of(c_layout), MutAnyOrigin]) raises -> Float64:
    ctx.synchronize()
    var t0 = perf_counter_ns()
    ctx.enqueue_function[peak_kernel[INT8, type_of(c_layout)]](C, grid_dim=NUM_BLOCKS, block_dim=NTHREADS)
    ctx.synchronize()
    return Float64(perf_counter_ns() - t0) / 1.0e6


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()
    var cd = ctx.enqueue_create_buffer[DType.float32](NOUT)
    ctx.synchronize()
    var C = TileTensor(cd, c_layout)
    var w0 = perf_counter_ns()
    while Float64(perf_counter_ns() - w0) / 1.0e9 < 5.0:
        _ = run[False](ctx, C)
        _ = run[True](ctx, C)
    var bf = List[Float64]()
    var i8 = List[Float64]()
    for _ in range(REPEATS):
        bf.append(run[False](ctx, C))
        i8.append(run[True](ctx, C))
    comptime TOTAL_MMA = Float64(NUM_BLOCKS) * Float64(WAVES_PER_BLOCK) * Float64(ITERS) * Float64(NACC)
    var mb = median(bf.copy())
    var mi = median(i8.copy())
    print("grid", NUM_BLOCKS, "block", NTHREADS, "nacc", NACC, "iters", ITERS, "repeats", REPEATS)
    print("bf16 ms", mb, "min", min(bf[0], min(bf[1], bf[2])), "TFLOP/s", TOTAL_MMA * 8192.0 / (mb * 1.0e9))
    print("iu8  ms", mi, "min", min(i8[0], min(i8[1], i8[2])), "TOP/s", TOTAL_MMA * 8192.0 / (mi * 1.0e9))
    print("iu8 / bf16 issue-rate ratio:", mb / mi)
