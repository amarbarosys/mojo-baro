from max.gpu import block_idx, global_idx, lane_id, thread_idx, WARP_SIZE
from max.gpu.primitives import warp
from std.math import cos, rsqrt, sin
from bexp import exp, log
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from layout import TileTensor, TensorLayout, row_major, stack_allocation
from model import HD, NROT, YARN_LOW, YARN_HIGH, FREQ_BASE, FREQ_SCALE, MSCALE

comptime f32 = DType.float32
comptime i32 = DType.int32


@always_inline
def mrope_cs(P: MutPointer[Scalar[i32], MutAnyOrigin], t: Int, j: Int) -> SIMD[f32, 2]:
    var p = Int(P[unsafe_offset=3 * t + (j % 3)])
    var theta_ex = Float32(p) * exp(Float32(-2 * j) / Float32(NROT) * log(FREQ_BASE))
    var theta_in = FREQ_SCALE * theta_ex
    var ramp = (Float32(j) - YARN_LOW) / max(YARN_HIGH - YARN_LOW, 0.001)
    ramp = min(max(ramp, 0), 1)
    var theta = theta_in * (1 - ramp) + theta_ex * ramp
    return SIMD[f32, 2](cos(theta) * MSCALE, sin(theta) * MSCALE)


def amar_rope_mrope[
    XLayout: TensorLayout
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    P: MutPointer[Scalar[i32], MutAnyOrigin],
    pos: Int32,
    nh: Int32,
):
    comptime assert X.flat_rank == 2
    var h = block_idx.x
    var r = Int(block_idx.y)
    var j = Int(thread_idx.x)
    var cs = mrope_cs(P, Int(pos) + r, j)
    var row = r * Int(nh) + h
    var x0 = rebind[Scalar[f32]](X[row, j])
    var x1 = rebind[Scalar[f32]](X[row, j + NROT // 2])
    X[row, j] = rebind[X.ElementType](x0 * cs[0] - x1 * cs[1])
    X[row, j + NROT // 2] = rebind[X.ElementType](x0 * cs[1] + x1 * cs[0])


def amar_head_rmsnorm_mrope[
    XLayout: TensorLayout, GLayout: TensorLayout
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    G: TileTensor[f32, GLayout, MutAnyOrigin],
    eps: Float32,
    P: MutPointer[Scalar[i32], MutAnyOrigin],
    pos: Int32,
    nh: Int32,
):
    comptime assert X.flat_rank == 2 and G.flat_rank == 1
    var row = Int(block_idx.x)
    var d = thread_idx.x
    var v = rebind[Scalar[f32]](X[row, d])
    var ssq = warp.sum(v * v)
    var sums = stack_allocation[f32, address_space = AddressSpace.SHARED](
        row_major[HD // WARP_SIZE]()
    )
    if lane_id() == 0:
        sums[d // WARP_SIZE] = rebind[sums.ElementType](ssq)
    barrier()
    var total: Float32 = 0
    comptime for w in range(HD // WARP_SIZE):
        total += rebind[Scalar[f32]](sums[w])
    X[row, d] = rebind[X.ElementType](
        v * rsqrt(total / Float32(HD) + eps) * rebind[Scalar[f32]](G[d])
    )
    barrier()
    var j = Int(d)
    if j >= NROT // 2:
        return
    var r = row // Int(nh)
    var cs = mrope_cs(P, Int(pos) + r, j)
    var x0 = rebind[Scalar[f32]](X[row, j])
    var x1 = rebind[Scalar[f32]](X[row, j + NROT // 2])
    X[row, j] = rebind[X.ElementType](x0 * cs[0] - x1 * cs[1])
    X[row, j + NROT // 2] = rebind[X.ElementType](x0 * cs[1] + x1 * cs[0])


def amar_inject_rows[
    OLayout: TensorLayout, ILayout: TensorLayout, MLayout: TensorLayout
](
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    Img: TileTensor[f32, ILayout, MutAnyOrigin],
    Map: TileTensor[i32, MLayout, MutAnyOrigin],
    pos: Int32,
    n: Int32,
):
    comptime assert O.flat_rank == 2 and Img.flat_rank == 2 and Map.flat_rank == 1
    var idx = global_idx.x
    if idx >= Int(n):
        return
    var k = Int(rebind[Scalar[i32]](Map[Int(pos) + Int(block_idx.y)]))
    if k < 0:
        return
    O[block_idx.y, idx] = rebind[O.ElementType](Img[k, idx])
