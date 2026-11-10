"""ENT / NEU rows (nlook.mojo) vs the float64 torch oracle (tools/neuron_oracle.py): loss and every gradient block.

Case dir = argv[1]. exact (fp32 GEMMs): max|err| <= 1e-4 * max|want| per block; bf16 GEMMs: 3e-2 (the bf16 input rounding floor).
Also checks that a second same-input run gives byte-identical gradients.
"""
from std.sys import argv
from max.gpu.host import DeviceContext, DeviceBuffer
from ops import F, I, upload
from lmodel import *


def load(path: String) raises -> List[Float32]:
    var raw = open(path, "r").read_bytes()
    var src = raw.unsafe_ptr().unsafe_bitcast[Float32]()
    var h = List[Float32](length=len(raw) // 4, fill=0)
    for i in range(len(h)):
        h[i] = src[unsafe_offset=i]
    return h^


def run[exact: Bool](ctx: DeviceContext, CASE: String, tol: Float64) raises -> Bool:
    var params = load(CASE + "params.f32")
    var ids = List[Int32]()
    for x in load(CASE + "ids.f32"):
        ids.append(Int32(Int(x)))
    var tgt = List[Int32]()
    for x in load(CASE + "tgt.f32"):
        tgt.append(Int32(Int(x)))
    var wf = load(CASE + "w.f32")
    var kf = load(CASE + "kinds.f32")
    var want = load(CASE + "grad.f32")
    var wl = Float64(load(CASE + "loss.f32")[0])
    var b = Int(load(CASE + "b.f32")[0])
    var meta = load(CASE + "meta.f32")
    var mt = List[Int32]()
    for x in load(CASE + "mt.f32"):
        mt.append(Int32(Int(x)))
    var S = len(kf)
    var kinds = List[Int]()
    for r in range(S):
        kinds.append(Int(kf[r]))
    var nh = len(mt) // b
    var NT = NP + 2 * D * EB
    var pb = ctx.enqueue_create_buffer[DType.float32](NT)
    var gb = ctx.enqueue_create_buffer[DType.float32](NT)
    var ib = ctx.enqueue_create_buffer[DType.int32](S * b)
    var tb = ctx.enqueue_create_buffer[DType.int32](S * b)
    var wb = ctx.enqueue_create_buffer[DType.float32](S * b)
    upload(ctx, wb, wf)
    upload(ctx, pb, params)
    upload(ctx, ib, ids)
    upload(ctx, tb, tgt)
    var m = Model[exact](ctx, b)
    m.lk = Look(ctx, b, nh, Int(meta[0]), Int(meta[1]), NP, NP + D * EB)
    m.lk.ls = meta[2]
    m.lk.aw = meta[3]
    upload(ctx, m.lk.mt, mt)
    upload(ctx, m.lk.ent, load(CASE + "ent.f32"))
    m.lk.load(ctx, load(CASE + "K.f32"), load(CASE + "V.f32"))
    var p = F(unsafe_from_address=Int(pb.unsafe_ptr()))
    var g = F(unsafe_from_address=Int(gb.unsafe_ptr()))
    var got = List[Float32](length=NT, fill=0)
    var again = List[Float32](length=NT, fill=0)
    var lrow = List[Float32](length=S * b, fill=0)
    var lo = List[Float32](length=nh * 2 * b, fill=0)
    for rep in range(2):
        ctx.enqueue_memset(gb, 0)
        m.step(ctx, p, g, I(unsafe_from_address=Int(ib.unsafe_ptr())), I(unsafe_from_address=Int(tb.unsafe_ptr())), F(unsafe_from_address=Int(wb.unsafe_ptr())), kinds, S, True)
        ctx.enqueue_copy(dst_ptr=got.unsafe_ptr() if rep == 0 else again.unsafe_ptr(), src_buf=gb)
        ctx.enqueue_copy(dst_ptr=lrow.unsafe_ptr(), src_buf=DeviceBuffer[DType.float32](ctx, m.lrow, S * b, owning=False))
        ctx.enqueue_copy(dst_ptr=lo.unsafe_ptr(), src_buf=m.lk.lo)
        ctx.synchronize()
    _ = (pb.unsafe_ptr(), gb.unsafe_ptr(), ib.unsafe_ptr(), tb.unsafe_ptr(), wb.unsafe_ptr())
    var loss: Float64 = 0
    for x in lrow:
        loss += Float64(x)
    for h in range(nh):
        var s: Float64 = 0
        for i in range(b):
            s += Float64(lo[h * 2 * b + i])
        loss += Float64(m.lk.aw) * s / Float64(b)
    var ok = abs(loss - wl) <= tol * abs(wl)
    print("PASS" if ok else "FAIL", "loss exact" if exact else "loss bf16", loss, "want", wl)
    var same = True
    for i in range(NT):
        same = same and got[i] == again[i]
    print("PASS" if same else "FAIL", "two same-input runs byte-identical gradients" if exact else "two runs identical (bf16)")
    ok = ok and same
    var names = ["embed", "layer0", "layer1", "final_gain", "pause", "W_a", "W_b"]
    var bounds = [0, V * D, V * D + LAYER, GF, PAUSE, NP, NP + D * EB, NT]
    for s in range(7):
        var err: Float64 = 0
        var mx: Float64 = 0
        for i in range(bounds[s], bounds[s + 1]):
            err = max(err, abs(Float64(got[i]) - Float64(want[i])))
            mx = max(mx, abs(Float64(want[i])))
        var good = err <= tol * mx
        ok = ok and good
        print("PASS" if good else "FAIL", "grad", names[s], "exact" if exact else "bf16", "max|err|", err, "max|want|", mx)
    return ok


def main() raises:
    var ctx = DeviceContext()
    var cdir = String(argv()[1]) + "/"
    var a = run[True](ctx, cdir, 1e-4)
    var c = run[False](ctx, cdir, 3e-2)
    if not (a and c):
        raise Error("FAIL neuron parity")
    print("PASS neuron parity")
