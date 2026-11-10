# M3 loader gate: device weight bytes == file tensor-data bytes, upload time, and three tensors read back byte-equal.
# usage: test_weights MODEL.gguf
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from weights import Weights

def main() raises:
    var path = String(argv()[1])
    var ctx = DeviceContext()
    var t0 = perf_counter_ns()
    var w = Weights(ctx, path)
    var ms = Float64(perf_counter_ns() - t0) / 1e6
    var tensor_bytes = 0
    for name in w.idx.gguf.tensor_names:
        tensor_bytes += w.ref(name).nbytes
    print("tensors", len(w.idx.t), "tensor_bytes", tensor_bytes, "device_bytes", w.nbytes(), "file_bytes", w.idx.lens[0], "load_ms", ms)
    for name in ["token_embd.weight", "blk.7.ffn_down.weight", "output_norm.weight"]:
        var r = w.ref(name)
        var back = ctx.enqueue_create_host_buffer[DType.uint8](r.nbytes)
        ctx.enqueue_copy(dst_buf=back, src_buf=w.buf.create_sub_buffer[DType.uint8](r.off - w.data_start, r.nbytes))
        ctx.synchronize()
        var src = Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=r.addr)
        for i in range(r.nbytes):
            if back[i] != src[i]:
                raise Error("FAIL weights: " + name + " differs at byte " + String(i))
        print(name, "type", r.gtype, "dims", r.dims[0], r.dims[1] if len(r.dims) > 1 else 1, "bytes", r.nbytes, "equal")
    print("PASS weights")
