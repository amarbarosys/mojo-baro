"""BARO_IMPLANTS for spark (mojo-neuron docs/format-implants.md): rank-k side FFN per layer, applied on the host.

engine.mojo carries its own copy of the loader (serve/harness.mojo) and the hook (serve/window.mojo); this module is the
same arithmetic in one place. Keys are re-normalised on load. delta(): c_i = dot(x, key_i) / |x|,
d = sum_i gate_i * scale * max(c_i - tau, 0) * value_i over rows lo..min(k, klim)."""
from std.math import sqrt


struct Implants(Movable):
    var set_of: List[Int]  # per layer: set index or -1
    var k: List[Int]
    var tau: List[Float32]
    var keys: List[List[Float32]]
    var gate: List[List[Float32]]
    var vals: List[List[Float32]]

    def __init__(out self, n_layers: Int):
        self.set_of = List[Int](length=n_layers, fill=-1)
        self.k = List[Int]()
        self.tau = List[Float32]()
        self.keys = List[List[Float32]]()
        self.gate = List[List[Float32]]()
        self.vals = List[List[Float32]]()

    def load(mut self, path: String, h: Int) raises:
        var ib = open(path, "r").read_bytes()
        if len(ib) < 16 or String(unsafe_from_utf8=ib[0:8]) != "NIMP0001":
            raise Error("BARO_IMPLANTS: not an implants.bin v1 file")
        var u = ib.unsafe_ptr().bitcast[UInt32]()
        var f = ib.unsafe_ptr().bitcast[Float32]()
        if Int(u[2]) != h:
            raise Error("BARO_IMPLANTS: hidden size " + String(Int(u[2])) + " != " + String(h))
        var w = 4
        for si in range(Int(u[3])):
            var layer = Int(u[w])
            var k = Int(u[w + 1])
            if layer >= len(self.set_of) or self.set_of[layer] >= 0 or (w + 3 + 2 * k * h + k) * 4 > len(ib):
                raise Error("BARO_IMPLANTS: bad set " + String(si))
            self.set_of[layer] = si
            self.k.append(k)
            self.tau.append(f[w + 2])
            w += 3
            var kl = List[Float32](length=k * h, fill=0)
            for i in range(k):
                var n2: Float32 = 0
                for j in range(h):
                    n2 += f[w + i * h + j] * f[w + i * h + j]
                var inv = 1.0 / sqrt(n2) if n2 > 0 else Float32(0)
                for j in range(h):
                    kl[i * h + j] = f[w + i * h + j] * inv
            w += k * h
            var gl = List[Float32](length=k, fill=0)
            for i in range(k):
                gl[i] = f[w + i]
            w += k
            var vl = List[Float32](length=k * h, fill=0)
            for i in range(k * h):
                vl[i] = f[w + i]
            w += k * h
            self.keys.append(kl^)
            self.gate.append(gl^)
            self.vals.append(vl^)
        print("BARO_IMPLANTS:", path, Int(u[3]), "sets")

    def delta(self, si: Int, x: List[Float32], lo: Int, klim: Int, scale: Float32, mut d: List[Float32]) -> Bool:
        var h = len(x)
        var xn: Float32 = 0
        for j in range(h):
            xn += x[j] * x[j]
        xn = sqrt(xn)
        for j in range(h):
            d[j] = 0
        var fired = False
        var kk = self.k[si] if klim < 0 else min(self.k[si], klim)
        for i in range(max(lo, 0), kk):
            var dt: Float32 = 0
            for j in range(h):
                dt += x[j] * self.keys[si][i * h + j]
            var ca = dt / xn - self.tau[si]
            if ca > 0:
                fired = True
                var gw = self.gate[si][i] * ca * scale
                for j in range(h):
                    d[j] += gw * self.vals[si][i * h + j]
        return fired
