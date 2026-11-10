"""Weight pack for the CPU tier: index.txt parsed once, pack.bin mapped read-only, tensors by name.

A tensor is (byte offset, dtype, element count) as tools/engine-pack.mojo writes it. q4 stores the
n/2 nibble bytes, then n/32 fp16 scales; q8 stores n int8, then n/32 fp16 scales.
"""
from std.ffi import external_call

comptime PROT_READ = 1
comptime MAP_SHARED = 1


@fieldwise_init
struct Tensor(Copyable, Movable):
    var off: Int
    var dt: String
    var n: Int

    def bytes(self) -> Int:
        if self.dt == "q4":
            return self.n // 2 + self.n // 16
        if self.dt == "q8":
            return self.n + self.n // 16
        return self.n * (2 if self.dt == "bf16" else 4)


struct Pack(Movable):
    var base: Int
    var size: Int
    var t: Dict[String, Tensor]

    def __init__(out self, dir: String) raises:
        self.t = Dict[String, Tensor]()
        with open(dir + "/index.txt", "r") as f:
            for line in f.read().split("\n"):
                var p = line.split(" ")
                if len(p) < 4:
                    continue
                if len(p) > 4:
                    raise Error("FAIL pack: tier tensor " + String(p[0]) + " (experts.bin) is not supported on CPU")
                self.t[String(p[0])] = Tensor(Int(p[2]), String(p[1]), Int(p[3]))
        with open(dir + "/pack.bin", "r") as f:
            self.size = Int(f.seek(0, 2))
            self.base = external_call["mmap", Int](Int(0), self.size, Int32(PROT_READ), Int32(MAP_SHARED), f._get_raw_fd(), Int64(0))
        if self.base == -1:
            raise Error("FAIL pack: mmap " + dir + "/pack.bin")
        for e in self.t.items():
            if e.value.off + e.value.bytes() > self.size:
                raise Error("FAIL pack: " + e.key + " ends past pack.bin")

    def __deinit__(deinit self):
        _ = external_call["munmap", Int32](self.base, self.size)

    def ptr[T: DType](self, name: String, at: Int = 0) raises -> Pointer[Scalar[T], ImmUntrackedOrigin]:
        """Typed view of tensor `name`, `at` bytes into it (q4 scales: at = n // 2)."""
        if name not in self.t:
            raise Error("FAIL pack: no tensor " + name)
        return Pointer[Scalar[T], ImmUntrackedOrigin](unsafe_from_address=self.base + self.t[name].off + at)
