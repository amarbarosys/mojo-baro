# ffn-unit-rank: rank FFN units by how much more they fire on failed than on kept prompts (sleep round 2, arm S1).
#   ffn-unit-rank MODEL.gguf CAPDIR LABELS OUT.tsv     (LABELS: "request_id 1|0" per line, 1 = failed; kind-match it)
# CAPDIR = spark BARO_CAPTURE output (mojo-neuron docs/format-capture.md, field x = FFN input, f16, one record per
# position, req.i32 = request id per position). LABELS marks each used request failed or kept. Per request the
# LAST position is used (where the first assistant token is chosen). Unit j of layer L: a = silu(x . Wg_j) * (x . Wu_j),
# Wg / Wu = blk.L.ffn_gate / ffn_up (F32). Output: layer, unit, Welch t (fail - kept), mean fail, mean kept (unsorted).
from std.sys import argv
from std.math import exp, sqrt
from std.memory import bitcast
from gguf_reader import GGUFModel


def read_bytes(path: String) raises -> List[UInt8]:
    with open(path, "r") as f:
        return f.read_bytes()


def f16_at(b: List[UInt8], i: Int) -> Float32:  # same bit decode as tools/llama-slot-to-state.mojo f16_at (byte offset 2i)
    var h = UInt16(Int(b[2 * i]) | (Int(b[2 * i + 1]) << 8))
    return bitcast[DType.float16, 1](SIMD[DType.uint16, 1](h)).cast[DType.float32]()


def main() raises:
    var a = argv()
    if len(a) != 5:
        raise Error("usage: ffn-unit-rank MODEL.gguf CAPDIR LABELS OUT.tsv")
    var m = GGUFModel(String(a[1]))
    var cap = String(a[2])
    var lab = Dict[Int, Bool]()  # LABELS: "request_id 1|0" per line (1 = failed); requests not listed are skipped
    with open(String(a[3]), "r") as lf:
        for ln in lf.read().splitlines():
            var w = ln.split(" ")
            if len(w) == 2:
                lab[Int(String(w[0]))] = String(w[1]) == "1"
    var H = m.akv_int("embedding_length", 0)
    var F = m.akv_int("feed_forward_length", 0)
    var NL = m.akv_int("block_count", 0)
    var req = read_bytes(cap + "/req.i32")
    var npos = len(req) // 4
    var last = Dict[Int, Int]()  # request id -> last position
    for p in range(npos):
        var r = Int(req[4 * p]) | (Int(req[4 * p + 1]) << 8) | (Int(req[4 * p + 2]) << 16) | (Int(req[4 * p + 3]) << 24)
        last[r] = p
    var ids = List[Int]()
    for k in last.keys():
        if k in lab:
            ids.append(k)
    var gg = read_bytes(String(a[1]))
    var out = String("layer\tunit\tt\tmean_fail\tmean_kept\n")
    var rows = List[Tuple[Float64, String]]()
    for L in range(NL):
        var x = read_bytes(cap + "/L" + String(L) + ".f16")
        var og = m.data_start + m.tensor_off["blk." + String(L) + ".ffn_gate.weight"]
        var ou = m.data_start + m.tensor_off["blk." + String(L) + ".ffn_up.weight"]
        var wg = gg.unsafe_ptr().bitcast[Float32]() + og // 4
        var wu = gg.unsafe_ptr().bitcast[Float32]() + ou // 4
        var sf = List[Float64](length=F, fill=0.0)
        var sf2 = List[Float64](length=F, fill=0.0)
        var sk = List[Float64](length=F, fill=0.0)
        var sk2 = List[Float64](length=F, fill=0.0)
        var nf = 0
        var nk = 0
        var xv = List[Float32](length=H, fill=0.0)
        for r in ids:
            var p = last[r]
            for i in range(H):
                xv[i] = f16_at(x, p * H + i)
            var fail = lab[r]
            if fail:
                nf += 1
            else:
                nk += 1
            for j in range(F):
                var g: Float32 = 0.0
                var u: Float32 = 0.0
                for i in range(H):
                    g += xv[i] * wg[j * H + i]
                    u += xv[i] * wu[j * H + i]
                var v = Float64(g / (1.0 + exp(-g)) * u)
                if fail:
                    sf[j] += v
                    sf2[j] += v * v
                else:
                    sk[j] += v
                    sk2[j] += v * v
        for j in range(F):
            var mf = sf[j] / Float64(nf)
            var mk = sk[j] / Float64(nk)
            var vf = sf2[j] / Float64(nf) - mf * mf
            var vk = sk2[j] / Float64(nk) - mk * mk
            var t = (mf - mk) / sqrt(vf / Float64(nf) + vk / Float64(nk) + 1e-12)
            rows.append((t, String(L) + "\t" + String(j) + "\t" + String(t) + "\t" + String(mf) + "\t" + String(mk) + "\n"))
        print("layer", L, "fail", nf, "kept", nk)
    for r in rows:
        out += r[1]
    with open(String(a[4]), "w") as f:
        f.write(out)
    print("wrote", len(rows), "units ->", String(a[4]), "(unsorted: sort -t TAB -k3,3gr)")
