from std.math import fma, sqrt

comptime f32 = DType.float32


def sign_of(seed: UInt64, i: Int) -> Float64:
    var z = seed + UInt64(i + 1) * 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return 1.0 if ((z ^ (z >> 31)) & 1) == 1 else -1.0


struct Cert(Copyable, Movable):
    var u: List[Float64]
    var n: Int
    var k: Int
    var seed: UInt64

    def __init__(out self, q: Pointer[UInt8, _], s: Pointer[Float16, _], n: Int, k: Int, seed: UInt64):
        self.n = n
        self.k = k
        self.seed = seed
        self.u = List[Float64](length=k, fill=0)
        for i in range(n):
            var r = sign_of(seed, i)
            for b in range(k // 32):
                var bytes = q.unsafe_offset(i * (k // 2) + b * 16).unsafe_load[width=16]()
                var d = s.unsafe_offset(i * (k // 32) + b).unsafe_load().cast[f32]()
                var m8d = d * -8
                for j in range(16):
                    self.u[b * 32 + j] += r * Float64(fma((bytes[j] & 0x0F).cast[f32](), d, m8d))
                    self.u[b * 32 + 16 + j] += r * Float64(fma((bytes[j] >> 4).cast[f32](), d, m8d))

    def check(self, x: List[BFloat16], y: List[Float32]) -> SIMD[DType.float64, 2]:
        var rhs = Float64(0)
        for j in range(self.k):
            rhs += self.u[j] * Float64(x[j].cast[f32]())
        var lhs = Float64(0)
        var yy = Float64(0)
        for i in range(self.n):
            lhs += sign_of(self.seed, i) * Float64(y[i])
            yy += Float64(y[i]) * Float64(y[i])
        return SIMD[DType.float64, 2](abs(lhs - rhs), sqrt(yy) * sqrt(Float64(self.k)) * 5.9604644775390625e-08)
