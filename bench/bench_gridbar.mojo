"""Stage 2 of bench/launch-fusion-protocol.md: grid-wide barrier cost on this box.

Sense-reversing atomic barrier (agent-scope counter + generation word), no
cooperative launch: grid must fit resident blocks. One block-256 thread 0 per
block spins with s_sleep; ITERS barriers per launch, kernel time / ITERS.
Launch floor (b) from stage 0 is re-measured in the same run for the ratio.
"""

from max.gpu import block_idx, thread_idx, grid_dim
from std.atomic import Atomic, Ordering
from std.sys import has_accelerator, llvm_intrinsic
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from max.gpu.sync import barrier
from layout import TileTensor, TensorLayout, row_major

comptime u32 = DType.uint32
comptime f32 = DType.float32
comptime ITERS = 2000
comptime ctr_layout = row_major[2]()
comptime sink_layout = row_major[4096]()


def empty_k():
    pass


def gridbar_k[CLayout: TensorLayout, SLayout: TensorLayout](
    Ctr: TileTensor[u32, CLayout, MutAnyOrigin],
    Sink: TileTensor[f32, SLayout, MutAnyOrigin],
    iters: Int32,
):
    comptime assert Ctr.flat_rank == 1 and Sink.flat_rank == 1
    var acc: Scalar[f32] = 0
    var nb = UInt32(grid_dim.x)
    var ctr = Ctr.ptr
    var gen = Ctr.ptr.unsafe_offset(1)
    for _ in range(Int(iters)):
        acc += Scalar[f32](thread_idx.x) * 1e-9
        barrier()
        if thread_idx.x == 0:
            var g = Atomic[Scalar[u32], scope="agent"].load[ordering=Ordering.ACQUIRE](gen)
            if Atomic[Scalar[u32], scope="agent"].fetch_add[ordering=Ordering.ACQUIRE_RELEASE](ctr, 1) == nb - 1:
                Atomic[Scalar[u32], scope="agent"].store[ordering=Ordering.RELAXED](ctr, 0)
                Atomic[Scalar[u32], scope="agent"].store[ordering=Ordering.RELEASE](gen, g + 1)
            else:
                while Atomic[Scalar[u32], scope="agent"].load[ordering=Ordering.ACQUIRE](gen) == g:
                    llvm_intrinsic["llvm.amdgcn.s.sleep", NoneType](Int32(1))
        barrier()
    if thread_idx.x == 0:
        Sink[Int(block_idx.x)] = rebind[Sink.ElementType](acc)


def time_us(ctx: DeviceContext, n: Int, t0: Int) raises -> Float64:
    ctx.synchronize()
    return Float64(perf_counter_ns() - t0) / 1e3 / Float64(n)


def main() raises:
    comptime assert has_accelerator()
    var ctx = DeviceContext()
    var ctr = ctx.enqueue_create_buffer[u32](2)
    var sink = ctx.enqueue_create_buffer[f32](4096)
    ctx.enqueue_memset(ctr, 0)
    ctx.enqueue_memset(sink, 0)
    ctx.synchronize()
    var Ctr = TileTensor(ctr, ctr_layout)
    var Sink = TileTensor(sink, sink_layout)
    comptime k = gridbar_k[type_of(ctr_layout), type_of(sink_layout)]

    for _ in range(200):
        ctx.enqueue_function[empty_k](grid_dim=96, block_dim=256)
    ctx.synchronize()
    var t = perf_counter_ns()
    for _ in range(ITERS):
        ctx.enqueue_function[empty_k](grid_dim=96, block_dim=256)
    print("launch floor g96 b256:", time_us(ctx, ITERS, t), "us/launch")

    for g in [96, 192, 288, 384]:
        ctx.enqueue_function[k](Ctr, Sink, Int32(ITERS), grid_dim=g, block_dim=256)
        ctx.synchronize()
        t = perf_counter_ns()
        ctx.enqueue_function[k](Ctr, Sink, Int32(ITERS), grid_dim=g, block_dim=256)
        print("atomic barrier grid", g, ":", time_us(ctx, ITERS, t), "us/barrier")
