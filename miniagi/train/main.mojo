"""Train the parity model: main INIT DATA STEPS LR OUT [T=512] [fast=0].

fast=1: forward weight GEMMs on fp16 WMMA (miniagi/train/fast.mojo); backward and attention stay fp32.

INIT raw f32 params (NP), DATA raw bytes, consecutive T-byte windows, one AdamW step each.
OUT: one line per step, `step loss bits` (bits = f32 hex, for bit-identical rerun checks).
Raw pointers do not keep a DeviceBuffer alive: ASAP destruction freed pb/gb/tb at their last
named use and the Model arena reused that memory (loss 32 vs 5.6). Keep-alive line after the sync.
"""
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from ops import *
from model import *

def main() raises:
    var a = argv()
    if len(a) > 7 and String(a[7]) == "1":
        run[True]()
    else:
        run[False]()


def run[fast: Bool]() raises:
    var a = argv()
    var ctx = DeviceContext()
    var raw = open(String(a[1]), "r").read_bytes()
    var data = open(String(a[2]), "r").read_bytes()
    var steps = Int(String(a[3]))
    var lr = Float32(Float64(String(a[4])))
    var T = Int(String(a[6])) if len(a) > 6 else 512
    var pb = ctx.enqueue_create_buffer[DType.float32](NP)
    var gb = ctx.enqueue_create_buffer[DType.float32](NP)
    var mb = ctx.enqueue_create_buffer[DType.bfloat16](NP)
    var vb = ctx.enqueue_create_buffer[DType.bfloat16](NP)
    var tb = ctx.enqueue_create_buffer[DType.int32](2 * T)
    var init = List[Float32](length=NP, fill=0)
    var src = raw.unsafe_ptr().unsafe_bitcast[Float32]()
    for i in range(NP):
        init[i] = src[i]
    upload(ctx, pb, init)
    ctx.enqueue_memset(mb, 0)
    ctx.enqueue_memset(vb, 0)
    var p = F(unsafe_from_address=Int(pb.unsafe_ptr()))
    var g = F(unsafe_from_address=Int(gb.unsafe_ptr()))
    var tok = I(unsafe_from_address=Int(tb.unsafe_ptr()))
    var m = Model[fast](ctx, T)
    var win = List[Int32](length=2 * T, fill=0)
    var log = String()
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for s in range(steps):
        for i in range(T):
            win[i] = Int32(data[s * T + i])
            win[T + i] = Int32(data[s * T + i + 1])
        upload(ctx, tb, win)
        ctx.enqueue_memset(gb, 0)
        var loss = m.forward(ctx, p, tok, tok.unsafe_offset(T))
        m.backward(ctx, p, g, tok)
        var t = Float64(s + 1)
        ctx.enqueue_function[adamw_k](p, g, B(unsafe_from_address=Int(mb.unsafe_ptr())), B(unsafe_from_address=Int(vb.unsafe_ptr())),
            Int32(NP), lr, Float32(0.9), Float32(0.95), Float32(1e-8), Float32(0.1),
            Float32(1 / (1 - 0.9**t)), Float32(1 / (1 - 0.95**t)), grid_dim=grid(NP), block_dim=THREADS)
        log += String(s, " ", loss, " ", hex(Int(loss.to_bits())), "\n")
    ctx.synchronize()
    var sec = Float64(perf_counter_ns() - t0) / 1e9
    print("fast", fast, "steps", steps, "T", T, "sec", sec, "bytes/s", Float64(steps * T) / sec, "arena_MB", Float64(m.mem.top * 4) / 1e6)
    _ = (pb.unsafe_ptr(), gb.unsafe_ptr(), tb.unsafe_ptr(), mb.unsafe_ptr(), vb.unsafe_ptr())
    with open(String(a[5]), "w") as f:
        f.write(log)
    print("written to", String(a[5]))
