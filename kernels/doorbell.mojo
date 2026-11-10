from std.atomic import Atomic, Ordering, fence
from std.memory import UnsafePointer, bitcast
from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.sync import barrier

from moe import TOPK


def tier_bell_out(
    ids: UnsafePointer[Int32, MutAnyOrigin],
    wt: UnsafePointer[Float32, MutAnyOrigin],
    x: UnsafePointer[BFloat16, MutAnyOrigin],
    bell: UnsafePointer[Int32, MutAnyOrigin],
    bell_x: UnsafePointer[BFloat16, MutAnyOrigin],
    seq: Int32,
    nx: Int32,
):
    var t = thread_idx.x
    if t < TOPK:
        bell[1 + t] = ids[t]
        bell[1 + TOPK + t] = bitcast[DType.int32](wt[t])
    var i = t
    while i < Int(nx):
        bell_x[i] = x[i]
        i += block_dim.x
    barrier()
    if t == 0:
        fence[Ordering.RELEASE]()
        Atomic[Int32].store[ordering=Ordering.RELEASE](bell, seq)


def tier_bell_in(
    plan: UnsafePointer[Int32, MutAnyOrigin],
    flag: UnsafePointer[Int32, MutAnyOrigin],
    seq: Int32,
    idx: UnsafePointer[Int32, MutAnyOrigin],
    he: UnsafePointer[Int32, MutAnyOrigin],
    wt: UnsafePointer[Float32, MutAnyOrigin],
):
    var t = thread_idx.x
    if t == 0:
        while Atomic[Int32].load[ordering=Ordering.ACQUIRE](flag) < seq:
            pass
    barrier()
    if t < TOPK:
        idx[t] = Atomic[Int32].load[ordering=Ordering.ACQUIRE](plan + t)
        he[t] = Atomic[Int32].load[ordering=Ordering.ACQUIRE](plan + TOPK + t)
        wt[t] = bitcast[DType.float32](Atomic[Int32].load[ordering=Ordering.ACQUIRE](plan + 2 * TOPK + t))


def tier_wait_add(
    part: UnsafePointer[Float32, MutAnyOrigin],
    flag: UnsafePointer[Int32, MutAnyOrigin],
    seq: Int32,
    xres: UnsafePointer[Float32, MutAnyOrigin],
    n: Int32,
):
    var t = thread_idx.x
    if t == 0:
        while Atomic[Int32].load[ordering=Ordering.ACQUIRE](flag) < seq:
            pass
    barrier()
    var i = block_idx.x * block_dim.x + t
    if i < Int(n):
        xres[i] = xres[i] + Atomic[Float32].load[ordering=Ordering.ACQUIRE](part + i)
