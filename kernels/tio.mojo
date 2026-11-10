from std.builtin.sort import sort
from std.memory import unsafe_memcpy


def load_into(path: String, dst: MutPointer[UInt8, MutUntrackedOrigin], size: Int, skip: Int = 0) raises:
    with open(path, "r") as f:
        _ = f.seek(skip)
        var data = f.read_bytes(size)
        if len(data) < size:
            raise Error("size mismatch for " + path)
        unsafe_memcpy(dest=dst, src=data.unsafe_ptr(), count=size)


def load_exact(path: String, dst: MutPointer[UInt8, MutUntrackedOrigin], size: Int) raises:
    with open(path, "r") as f:
        var data = f.read_bytes()
        if len(data) != size:
            raise Error("size mismatch for " + path)
        unsafe_memcpy(dest=dst, src=data.unsafe_ptr(), count=size)


def pack_offset(fam: String, name: String) raises -> Int:
    with open(".work/engine-pack-" + fam + "/index.txt", "r") as f:
        for line in f.read().splitlines():
            var parts = line.split(" ")
            if parts[0] == name:
                if String(parts[1]) != fam:
                    raise Error("not " + fam + ": " + name)
                return Int(parts[2])
    raise Error("missing " + name)


def median(var v: List[Float64]) -> Float64:
    sort(v)
    return v[len(v) // 2]
