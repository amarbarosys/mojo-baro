# All GGUF tensor data in one VRAM buffer, uploaded once from the mmapped file, unchanged (quantised blocks stay as stored).
# The index (names, types, dims, file offsets) is mojo-baro's ShardIndex; a tensor's device address is base + off - data_start.
from max.gpu.host import DeviceContext, DeviceBuffer
from shard_index import ShardIndex, TensorRef

struct Weights(Movable):
    var idx: ShardIndex
    var buf: DeviceBuffer[DType.uint8]
    var data_start: Int

    def __init__(out self, ctx: DeviceContext, path: String) raises:
        self.idx = ShardIndex(path)
        self.data_start = self.idx.gguf.data_start
        var n = self.idx.lens[0] - self.data_start
        self.buf = ctx.enqueue_create_buffer[DType.uint8](n)
        ctx.enqueue_copy(dst_buf=self.buf, src_ptr=Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=self.idx.maps[0] + self.data_start))
        ctx.synchronize()

    def nbytes(self) -> Int:
        return len(self.buf)

    def ref(self, name: String) raises -> TensorRef:
        return self.idx[name]

    def addr(self, name: String) raises -> Int:
        return Int(self.buf.unsafe_ptr()) + self.idx[name].off - self.data_start
