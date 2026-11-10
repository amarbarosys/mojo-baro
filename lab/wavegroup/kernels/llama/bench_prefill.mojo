# prefill timing: 511 tokens through a Pass[512] forward, three times; usage: bench_prefill MODEL.gguf (build -D WMMA_TB=1 -D WMMA_C_F16=0)
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from attn import KVT, KVHSTR
from model import N_LAYERS, NKVH
from weights import Weights
from llama import Pass, forward

def main() raises:
    var ctx = DeviceContext()
    var w = Weights(ctx, String(argv()[1]))
    comptime PAGES = 4
    comptime NKVE = PAGES * N_LAYERS * NKVH * KVHSTR
    var kc = ctx.enqueue_create_buffer[KVT](NKVE)
    var vc = ctx.enqueue_create_buffer[KVT](NKVE)
    var tab = ctx.enqueue_create_buffer[DType.int32](PAGES)
    var toks = ctx.enqueue_create_buffer[DType.int32](511)
    with tab.map_to_host() as h:
        for p in range(PAGES):
            h[p] = Int32(p)
    with toks.map_to_host() as h:
        for i in range(511):
            h[i] = Int32(1000 + i * 7)
    var Kc = TileTensor(kc, row_major[NKVE]())
    var Vc = TileTensor(vc, row_major[NKVE]())
    var tp = Pointer[Int32, MutAnyOrigin](unsafe_from_address=Int(tab.unsafe_ptr()))
    var kp = Pointer[Int32, MutAnyOrigin](unsafe_from_address=Int(toks.unsafe_ptr()))
    for _ in range(3):
        var t0 = perf_counter_ns()
        forward(ctx, w, Pass[512](ctx), kp, 0, 511, Kc, Vc, tp, Float32(1e-5))
        ctx.synchronize()
        print("prefill 511 tokens ms", Float64(perf_counter_ns() - t0) / 1e6)
