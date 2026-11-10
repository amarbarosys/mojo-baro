from layout import TileTensor, TensorLayout, row_major
from max.gpu import block_idx, global_idx

comptime f32 = DType.float32
comptime bf16 = DType.bfloat16

comptime REALIGN_SPLIT = 32


def amar_realign_gather[
    TLayout: TensorLayout, PLayout: TensorLayout, PartLayout: TensorLayout
](
    Table: TileTensor[bf16, TLayout, MutAnyOrigin],
    Probs: TileTensor[f32, PLayout, MutAnyOrigin],
    Part: TileTensor[f32, PartLayout, MutAnyOrigin],
    vocab: Int32,
    h: Int32,
):
    comptime assert Table.flat_rank == 2 and Probs.flat_rank == 2 and Part.flat_rank == 2

    var hh = global_idx.x
    if hh >= Int(h):
        return
    var V = Int(vocab)
    var split = Int(block_idx.y)
    var per = (V + REALIGN_SPLIT - 1) // REALIGN_SPLIT
    var v0 = split * per
    var v1 = min(v0 + per, V)

    var acc: Float32 = 0
    var v = v0
    while v < v1:
        var p = rebind[Scalar[f32]](Probs[0, v])
        var w = rebind[Scalar[bf16]](Table[v, hh]).cast[f32]()
        acc += p * w
        v += 1
    Part[split, hh] = rebind[Part.ElementType](acc)


def amar_realign_reduce[
    PartLayout: TensorLayout, ELayout: TensorLayout
](
    Part: TileTensor[f32, PartLayout, MutAnyOrigin],
    E: TileTensor[f32, ELayout, MutAnyOrigin],
    h: Int32,
):
    comptime assert Part.flat_rank == 2 and E.flat_rank == 1

    var hh = global_idx.x
    if hh >= Int(h):
        return
    var acc: Float32 = 0
    comptime for s in range(REALIGN_SPLIT):
        acc += rebind[Scalar[f32]](Part[s, hh])
    E[hh] = rebind[E.ElementType](acc)
