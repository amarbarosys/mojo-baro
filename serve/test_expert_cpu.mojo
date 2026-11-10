"""expert_ffn_cpu (BARO_TIER_CPU): scalar-reference parity, then bandwidth over
a 566 MB rotation (40 layers x 8 experts, past the 96 MB L3) so reads are cold.
Target from the plan: 30 GB/s at n=8 (RAM reads ~52 GB/s on this box). CPU only, no pack needed.

Build: ./.venv/bin/mojo build serve/test_expert_cpu.mojo -I serve -o .work/test_expert_cpu
"""
from std.memory import alloc, bitcast
from std.math import exp, round
from std.sys import exit
from std.time import perf_counter_ns
from expert_cpu import expert_ffn_cpu, Q4K, Q4K_BYTES

comptime H = 2048
comptime FFN = 512
comptime NE = 8
comptime LAYERS = 40
comptime EB = FFN * (H // Q4K) * Q4K_BYTES
comptime EBD = H * (FFN // Q4K) * Q4K_BYTES


def fill(p: MutPointer[UInt8, MutUntrackedOrigin], n: Int, mut s: UInt64):
    for i in range(n):
        s = s * 6364136223846793005 + 1442695040888963407
        p[i] = UInt8(s >> 56)
    # Finite scales: d = 0.01, dmin = 0.005 (f16).
    for b in range(n // Q4K_BYTES):
        var o = b * Q4K_BYTES
        p[o] = 0x1F
        p[o + 1] = 0x21
        p[o + 2] = 0x1F
        p[o + 3] = 0x1D


def ref_value(w: MutPointer[UInt8, MutUntrackedOrigin], row_base: Int, k: Int) -> Tuple[Float32, Float32]:
    var base = row_base + (k // Q4K) * Q4K_BYTES
    var grp = (k % Q4K) // 32
    var d = bitcast[DType.float16, 1](SIMD[DType.uint16, 1](UInt16(w[base]) | (UInt16(w[base + 1]) << 8)))[0].cast[DType.float32]()
    var dm = bitcast[DType.float16, 1](SIMD[DType.uint16, 1](UInt16(w[base + 2]) | (UInt16(w[base + 3]) << 8)))[0].cast[DType.float32]()
    var sc: Int
    var mn: Int
    if grp < 4:
        sc = Int(w[base + 4 + grp]) & 0x3F
        mn = Int(w[base + 8 + grp]) & 0x3F
    else:
        sc = (Int(w[base + 12 + grp - 4]) & 0x0F) | ((Int(w[base + 4 + grp % 4]) >> 2) & 0x30)
        mn = (Int(w[base + 12 + grp - 4]) >> 4) | ((Int(w[base + 8 + grp % 4]) >> 2) & 0x30)
    var q = Int(w[base + 16 + (grp // 2) * 32 + (k % 32)])
    q = (q >> 4) if grp % 2 == 1 else (q & 0x0F)
    return (d * Float32(sc) * Float32(q), dm * Float32(mn))


def qdq(v: MutPointer[Float32, MutUntrackedOrigin], n: Int, o: MutPointer[Float32, MutUntrackedOrigin]):
    """int8 per-32 quantize-dequantize, the kernel's activation format."""
    for g in range(n // 32):
        var amax = Float32(0)
        for i in range(32):
            amax = max(amax, abs(v[g * 32 + i]))
        for i in range(32):
            o[g * 32 + i] = round(v[g * 32 + i] * (Float32(127) / amax)) * (amax / 127) if amax > 0 else Float32(0)


def ref_ffn(
    gate: MutPointer[UInt8, MutUntrackedOrigin], up: MutPointer[UInt8, MutUntrackedOrigin],
    down: MutPointer[UInt8, MutUntrackedOrigin], ids: MutPointer[Int32, MutUntrackedOrigin],
    wts: MutPointer[Float32, MutUntrackedOrigin], x: MutPointer[Float32, MutUntrackedOrigin],
    quant: Bool, res: MutPointer[Float32, MutUntrackedOrigin],
):
    """Scalar reference for every 16th output; quant=True applies the kernel's int8 activations."""
    var xa = alloc[Float32](H)
    var hr = alloc[Float32](NE * FFN)
    var ha = alloc[Float32](NE * FFN)
    if quant:
        qdq(x, H, xa)
    else:
        for k in range(H):
            xa[k] = x[k]
    for j in range(NE):
        for r in range(FFN):
            var rb = Int(ids[j]) * EB + r * (H // Q4K) * Q4K_BYTES
            var g = Float32(0)
            var u = Float32(0)
            for k in range(H):
                var a = ref_value(gate, rb, k)
                g += a[0] * xa[k] - a[1] * x[k]
                var c = ref_value(up, rb, k)
                u += c[0] * xa[k] - c[1] * x[k]
            hr[j * FFN + r] = (g / (Float32(1) + exp(-g)) * u).cast[DType.bfloat16]().cast[DType.float32]()
    if quant:
        qdq(hr, NE * FFN, ha)
    else:
        for k in range(NE * FFN):
            ha[k] = hr[k]
    for c in range(0, H, 16):
        var acc = Float32(0)
        for j in range(NE):
            var rb = Int(ids[j]) * EBD + c * (FFN // Q4K) * Q4K_BYTES
            var dot = Float32(0)
            for k in range(FFN):
                var a = ref_value(down, rb, k)
                dot += a[0] * ha[j * FFN + k] - a[1] * hr[j * FFN + k]
            acc += wts[j] * dot
        res[c] = acc


def rel_err(got: MutPointer[Float32, MutUntrackedOrigin], want: MutPointer[Float32, MutUntrackedOrigin]) -> Float32:
    var worst = Float32(0)
    var scale = Float32(0)
    for c in range(0, H, 16):
        worst = max(worst, abs(want[c] - got[c]))
        scale = max(scale, abs(want[c]))
    return worst / scale


def main() raises:
    var gate = alloc[UInt8](LAYERS * NE * EB)
    var up = alloc[UInt8](LAYERS * NE * EB)
    var down = alloc[UInt8](LAYERS * NE * EBD)
    var s = UInt64(7)
    fill(gate, LAYERS * NE * EB, s)
    fill(up, LAYERS * NE * EB, s)
    fill(down, LAYERS * NE * EBD, s)
    var x = alloc[Float32](H)
    for i in range(H):
        x[i] = (Float32((i * 37) % 101 - 50) / 64).cast[DType.bfloat16]().cast[DType.float32]()
    var ids = alloc[Int32](NE)
    var wts = alloc[Float32](NE)
    for j in range(NE):
        ids[j] = Int32(NE - 1 - j)
        wts[j] = Float32(j + 1) / 36
    var hbuf = alloc[Float32](NE * FFN)
    var res = alloc[Float32](H)
    var xq = alloc[Int8](H + NE * FFN)
    var aux = alloc[Float32](2 * (H + NE * FFN) // 32)

    expert_ffn_cpu[H, FFN](gate, up, down, ids, wts, NE, x, hbuf, xq, aux, res)
    var rq = alloc[Float32](H)
    var rf = alloc[Float32](H)
    ref_ffn(gate, up, down, ids, wts, x, True, rq)
    ref_ffn(gate, up, down, ids, wts, x, False, rf)
    var e_kernel = rel_err(res, rq)
    print("parity vs same-int8 reference: rel", e_kernel, "(bar 1e-4)")
    print("int8 activation error vs f32 reference: rel", rel_err(res, rf), "(informational; judged end to end by BARO_FORCE agreement)")
    if not (e_kernel < 1e-4):
        print("FAIL parity: rel err", e_kernel, ">= 1e-4")
        exit(1)

    for n in [8, 2]:
        var reps = 3
        var t0 = perf_counter_ns()
        for _ in range(reps):
            for l in range(LAYERS):
                expert_ffn_cpu[H, FFN](gate + l * NE * EB, up + l * NE * EB, down + l * NE * EBD, ids, wts, n, x, hbuf, xq, aux, res)
        var dt = Float64(perf_counter_ns() - t0) / 1e9
        var gbs = Float64(reps * LAYERS * n * (2 * EB + EBD)) / dt / 1e9
        print("bench n=", n, " us/layer", dt / Float64(reps * LAYERS) * 1e6, " GB/s", gbs)
        if n == 8 and gbs < 30:
            print("FAIL bandwidth: under the 30 GB/s target")
            exit(1)
    print("PASS expert_cpu")
