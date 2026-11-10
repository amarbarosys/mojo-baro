"""Pack = index over the GGUF shards themselves (plan p8 step 1).

Each tensor resolves to (host address in an mmap of its shard, ggml type,
dims). Nothing is copied: the page cache is the only RAM copy and the
headers are the index. The ggml type id is the per-tensor tag that picks
the row dot.
"""
from std.ffi import external_call
from std.os import listdir
from gguf_reader import GGUFModel

comptime PROT_READ = 1
comptime MAP_SHARED = 1


@fieldwise_init
struct TensorRef(Copyable, Movable):
    var addr: Int
    var nbytes: Int
    var gtype: Int
    var dims: List[Int]
    var shard: Int
    var off: Int


def ggml_row_bytes(gtype: Int, n: Int) raises -> Int:
    if gtype == 0:
        return n * 4
    if gtype == 1 or gtype == 30:
        return n * 2
    if gtype == 2:
        return n // 32 * 18
    if gtype == 6:
        return n // 32 * 22
    if gtype == 8:
        return n // 32 * 34
    if gtype == 10:
        return n // 256 * 84
    if gtype == 11:
        return n // 256 * 110
    if gtype == 12:
        return n // 256 * 144
    if gtype == 13:
        return n // 256 * 176
    if gtype == 14:
        return n // 256 * 210
    if gtype == 16:
        return n // 256 * 66
    if gtype == 20:
        return n // 32 * 18
    if gtype == 23:
        return n // 256 * 136
    if gtype == 29:
        return n // 256 * 56
    if gtype == 42:
        return n // 64 * 18
    raise Error("shard_index: ggml type " + String(gtype) + " unsupported")


struct ShardIndex(Movable):
    var gguf: GGUFModel
    var t: Dict[String, TensorRef]
    var maps: List[Int]
    var lens: List[Int]
    var paths: List[String]

    def __init__(out self, first_shard: String) raises:
        self.gguf = GGUFModel(first_shard)
        self.t = Dict[String, TensorRef]()
        self.maps = List[Int]()
        self.lens = List[Int]()
        self.paths = List[String]()
        var n = self.gguf.ints.get("split.count", 1)
        for i in range(n):
            var path = first_shard
            if n > 1:
                var tag = "-00001-of-"
                var cut = first_shard.find(tag)
                if cut < 0:
                    raise Error("shard_index: cannot derive shard names from " + first_shard)
                var num = String(i + 1)
                while num.byte_length() < 5:
                    num = "0" + num
                path = String(first_shard[byte=:cut]) + "-" + num + String(first_shard[byte=cut + 6:])
            self.add_file(path, "")

    def add_file(mut self, path: String, prefix: String) raises:
        """Maps one more GGUF file and indexes its tensors named `prefix`* (an MTP head beside its model)."""
        var g = GGUFModel(path)
        var i = len(self.maps)
        var size: Int
        var addr: Int
        with open(path, "r") as f:
            var cpath = path
            var fd = external_call["open64", Int32](cpath.as_c_string_slice().unsafe_ptr(), Int32(0))
            size = Int(f.seek(0, 2)) # stdlib seek, not a raw lseek: a second lseek declaration conflicts with the stdlib one
            addr = external_call["mmap", Int](Int(0), size, Int32(PROT_READ), Int32(MAP_SHARED), fd, Int64(0))
        if addr == -1:
            raise Error("shard_index: mmap failed " + path)
        self.maps.append(addr)
        self.lens.append(size)
        self.paths.append(path)
        for name in g.tensor_names:
            if not name.startswith(prefix):
                continue
            var dims = g.tensor_dims[name].copy()
            var rows = 1
            for d in range(1, len(dims)):
                rows *= dims[d]
            var gt = g.tensor_type[name]
            var nb = ggml_row_bytes(gt, dims[0]) * rows
            var a = addr + g.data_start + g.tensor_off[name]
            if a + nb > addr + size:
                raise Error("shard_index: " + name + " runs past its shard")
            self.t[name] = TensorRef(a, nb, gt, dims^, i, a - addr)

    def __getitem__(self, name: String) raises -> TensorRef:
        if name not in self.t:
            raise Error("shard_index: no tensor " + name)
        return self.t[name].copy()

    def __deinit__(deinit self):
        for i in range(len(self.maps)):
            _ = external_call["munmap", Int32](self.maps[i], self.lens[i])
