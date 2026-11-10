"""KV-segment latent model vs the torch recompute oracle (tools/miniagi_latent_oracle.py): loss and every grad.

Case dir = argv[1] (default .work/miniagi/latent-case/); argv[2], if given, is a path prefix: grads of the fp32 and bf16 runs go to PREFIX.f32 / PREFIX.bf16 as raw f32, and dL/dz (rows r * b + i, V wide) to PREFIX.f32.dz / PREFIX.bf16.dz. Row weights come from row_weights and must equal the oracle's w exactly.
exact (fp32 GEMMs): max|err| <= 1e-4 * max|want|. bf16 GEMMs: same check at 3e-2, the bf16 input rounding floor.
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
        h[i] = src[i]
    return h^


def run[exact: Bool](ctx: DeviceContext, CASE: String, tol: Float64, dump: String = "") raises -> Bool:
    var params = load(CASE + "params.f32")
    var idsf = load(CASE + "ids.f32")
    var tgtf = load(CASE + "tgt.f32")
    var wf = load(CASE + "w.f32")
    var kf = load(CASE + "kinds.f32")
    var want = load(CASE + "grad.f32")
    var wl = load(CASE + "loss.f32")[0]
    var b = Int(load(CASE + "b.f32")[0])
    var S = len(kf)
    var kinds = List[Int]()
    for r in range(S):
        kinds.append(Int(kf[r]))
    var ids = List[Int32](length=S * b, fill=0)
    var tgt = List[Int32](length=S * b, fill=0)
    for i in range(S * b):
        ids[i] = Int32(Int(idsf[i]))
        tgt[i] = Int32(Int(tgtf[i]))
    var pb = ctx.enqueue_create_buffer[DType.float32](NP)
    var gb = ctx.enqueue_create_buffer[DType.float32](NP)
    var ib = ctx.enqueue_create_buffer[DType.int32](S * b)
    var tb = ctx.enqueue_create_buffer[DType.int32](S * b)
    var wb = ctx.enqueue_create_buffer[DType.float32](S * b)
    var wr = row_weights(kinds, tgt, b, Float64(load(CASE + "aux.f32")[0]))
    var wok = True
    for i in range(S * b):
        wok = wok and wr[i] == wf[i]
    print("PASS" if wok else "FAIL", "row_weights exact vs oracle w")
    upload(ctx, wb, wr)
    upload(ctx, pb, params)
    upload(ctx, ib, ids)
    upload(ctx, tb, tgt)
    ctx.enqueue_memset(gb, 0)
    var m = Model[exact](ctx, b)
    comptime if V > 32:
        m.sort_ids(ctx, ids)
    m.itemp = 1 / load(CASE + "temp.f32")[0]
    var p = F(unsafe_from_address=Int(pb.unsafe_ptr()))
    var g = F(unsafe_from_address=Int(gb.unsafe_ptr()))
    m.step(ctx, p, g, I(unsafe_from_address=Int(ib.unsafe_ptr())), I(unsafe_from_address=Int(tb.unsafe_ptr())), F(unsafe_from_address=Int(wb.unsafe_ptr())), kinds, S, True)
    var got = List[Float32](length=NP, fill=0)
    var lrow = List[Float32](length=S * b, fill=0)
    ctx.enqueue_copy(dst_ptr=got.unsafe_ptr(), src_buf=gb)
    ctx.enqueue_copy(dst_ptr=lrow.unsafe_ptr(), src_buf=DeviceBuffer[DType.float32](ctx, m.lrow, S * b, owning=False))
    ctx.synchronize()
    _ = (pb.unsafe_ptr(), gb.unsafe_ptr(), ib.unsafe_ptr(), tb.unsafe_ptr(), wb.unsafe_ptr())
    if dump:
        with open(dump, "w") as f:
            f.write_bytes(Span[UInt8](unsafe_ptr=got.unsafe_ptr().bitcast[UInt8](), length=NP * 4))
        var dz = List[Float32](length=S * b * V, fill=0)
        ctx.enqueue_copy(dst_ptr=dz.unsafe_ptr(), src_buf=DeviceBuffer[DType.float32](ctx, m.dz, S * b * V, owning=False))
        ctx.synchronize()
        with open(dump + ".dz", "w") as f:
            f.write_bytes(Span[UInt8](unsafe_ptr=dz.unsafe_ptr().bitcast[UInt8](), length=S * b * V * 4))
    var loss: Float64 = 0
    for x in lrow:
        loss += Float64(x)
    var ok = wok and abs(loss - Float64(wl)) <= tol * abs(Float64(wl))
    print("PASS" if ok else "FAIL", "loss exact" if exact else "loss bf16", loss, "want", wl)
    var names = ["embed", "layer0", "layersN", "final_gain", "pause"]
    var bounds = [0, V * D, V * D + LAYER, GF, PAUSE, NP]
    for s in range(5):
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
    var cdir = String(argv()[1]) + "/" if len(argv()) > 1 else String(".work/miniagi/latent-case/")
    var dump = String(argv()[2]) if len(argv()) > 2 else String("")
    var a = run[True](ctx, cdir, 1e-4, dump + ".f32" if dump else dump)
    var c = run[False](ctx, cdir, 3e-2, dump + ".bf16" if dump else dump)
    if not (a and c):
        raise Error("FAIL latent parity")
    print("PASS latent parity")
