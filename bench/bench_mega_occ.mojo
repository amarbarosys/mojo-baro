from max.gpu import block_idx, thread_idx, grid_dim, lane_id, WARP_SIZE
from max.gpu.primitives import warp
from std.atomic import Atomic, Ordering
from std.math import ceildiv
from std.sys import has_accelerator, llvm_intrinsic, argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from max.gpu.sync import barrier
from layout import TileTensor, TensorLayout, row_major
from matmul_skinny import ROW_WAVES, ROW_THREADS

comptime u32 = DType.uint32
comptime f32 = DType.float32
comptime bf16 = DType.bfloat16
comptime i8 = DType.int8
comptime f16 = DType.float16
comptime N = 4096
comptime K = 4096
comptime ITERS = 50
comptime SPIN_LIMIT = 1 << 22

def grid_barrier(ctr: MutPointer[Scalar[u32], MutAnyOrigin], gen: MutPointer[Scalar[u32], MutAnyOrigin], fail: MutPointer[Scalar[u32], MutAnyOrigin]) -> Bool:
    barrier()
    if thread_idx.x == 0:
        var nb = UInt32(grid_dim.x)
        var g = Atomic[Scalar[u32], scope="agent"].load[ordering=Ordering.ACQUIRE](gen)
        if Atomic[Scalar[u32], scope="agent"].fetch_add[ordering=Ordering.ACQUIRE_RELEASE](ctr, 1) == nb - 1:
            Atomic[Scalar[u32], scope="agent"].store[ordering=Ordering.RELAXED](ctr, 0)
            Atomic[Scalar[u32], scope="agent"].store[ordering=Ordering.RELEASE](gen, g + 1)
        else:
            var spins = 0
            while Atomic[Scalar[u32], scope="agent"].load[ordering=Ordering.ACQUIRE](gen) == g:
                llvm_intrinsic["llvm.amdgcn.s.sleep", NoneType](Int32(1))
                spins += 1
                if spins > SPIN_LIMIT or Atomic[Scalar[u32], scope="agent"].load[ordering=Ordering.RELAXED](fail) != 0:
                    Atomic[Scalar[u32], scope="agent"].store[ordering=Ordering.RELEASE](fail, 1)
                    break
    barrier()
    return Atomic[Scalar[u32], scope="agent"].load[ordering=Ordering.ACQUIRE](fail) == 0


def probe_k[ALayout: TensorLayout, QLayout: TensorLayout, SLayout: TensorLayout, OLayout: TensorLayout, CLayout: TensorLayout](
    A: TileTensor[bf16, ALayout, MutAnyOrigin],
    Q: TileTensor[i8, QLayout, MutAnyOrigin],
    S: TileTensor[f16, SLayout, MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    Ctr: TileTensor[u32, CLayout, MutAnyOrigin],
    iters: Int32,
):
    comptime assert A.flat_rank == 2 and Q.flat_rank == 2 and S.flat_rank == 2 and O.flat_rank == 1 and Ctr.flat_rank == 1
    comptime UNROLL = 4
    comptime QV = 16
    comptime STEP = WARP_SIZE * QV
    var lane = Int(lane_id())
    var wave = Int(thread_idx.x) // WARP_SIZE
    var ngroups = ceildiv(N, ROW_WAVES)
    var Qv = Q.vectorize[1, QV]()
    var Av = A.vectorize[1, QV]()
    var ctr = Ctr.ptr
    var gen = Ctr.ptr.unsafe_offset(1)
    var fail = Ctr.ptr.unsafe_offset(2)
    if Atomic[Scalar[u32], scope="agent"].load[ordering=Ordering.ACQUIRE](fail) != 0:
        return
    for _ in range(Int(iters)):
        var g = Int(block_idx.x)
        while g < ngroups:
            var row = g * ROW_WAVES + wave
            if row < N:
                var acc = SIMD[f32, QV](0)
                var kk = 0
                while kk + UNROLL * STEP <= K:
                    var qs = Array[SIMD[i8, QV], UNROLL](uninitialized=True)
                    var ds = Array[Scalar[f16], UNROLL](uninitialized=True)
                    comptime for u in range(UNROLL):
                        var kb = kk + u * STEP
                        qs[u] = rebind[SIMD[i8, QV]](Qv[row, kb // QV + lane])
                        ds[u] = rebind[Scalar[f16]](S[row, (kb + lane * QV) // 32])
                    comptime for u in range(UNROLL):
                        var kb = kk + u * STEP
                        var w = qs[u].cast[f32]() * ds[u].cast[f32]()
                        var a = rebind[SIMD[bf16, QV]](Av[0, kb // QV + lane]).cast[f32]()
                        acc += w * a
                    kk += UNROLL * STEP
                var total = warp.sum(acc.reduce_add())
                if lane == 0:
                    O[row] = rebind[O.ElementType](total)
            g += Int(grid_dim.x)
        if not grid_barrier(ctr, gen, fail):
            return


def main() raises:
    comptime assert has_accelerator()
    var ctx = DeviceContext()
    var G = Int(String(argv()[1]))
    comptime a_layout = row_major[1, K]()
    comptime q_layout = row_major[N, K]()
    comptime s_layout = row_major[N, K // 32]()
    comptime o_layout = row_major[N]()
    comptime c_layout = row_major[3]()
    var a = ctx.enqueue_create_buffer[bf16](K)
    var q = ctx.enqueue_create_buffer[i8](N * K)
    var s = ctx.enqueue_create_buffer[f16](N * K // 32)
    var o = ctx.enqueue_create_buffer[f32](N)
    var c = ctx.enqueue_create_buffer[u32](3)
    ctx.enqueue_memset(a, 0); ctx.enqueue_memset(q, 0); ctx.enqueue_memset(s, 0); ctx.enqueue_memset(o, 0); ctx.enqueue_memset(c, 0)
    ctx.synchronize()
    comptime k = probe_k[type_of(a_layout), type_of(q_layout), type_of(s_layout), type_of(o_layout), type_of(c_layout)]
    var t = perf_counter_ns()
    ctx.enqueue_function[k](TileTensor(a, a_layout), TileTensor(q, q_layout), TileTensor(s, s_layout), TileTensor(o, o_layout), TileTensor(c, c_layout), Int32(ITERS), grid_dim=G, block_dim=ROW_THREADS)
    ctx.synchronize()
    var dt = Float64(perf_counter_ns() - t) / 1e3
    var flag = ctx.enqueue_create_host_buffer[u32](3)
    ctx.enqueue_copy(dst_buf=flag, src_buf=c)
    ctx.synchronize()
    if flag[2] != 0:
        print("G", G, "NOT-RESIDENT: barrier spin limit hit after", dt, "us")
        return
    print("G", G, "completed", ITERS, "phase+barrier iters in", Float64(perf_counter_ns() - t) / 1e3, "us")
