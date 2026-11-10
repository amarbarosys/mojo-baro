"""WeightStore: GGUF tensors grouped and placed on host tiers (big-models P1, docs/BIG-MODELS.md).

A group is what one bind needs: a dense layer's trunk tensors, or one expert's gate/up/down slices. Tiers per layer:
MMAP (the shard mapping itself; the OS page cache is the tier), PIN (read once into the arena at open), RING (layers
not pinned, streamed through NSLOT ring slots); experts go to CACHE (LRU slots). Reads are O_DIRECT preads of each
extent's 4 KiB-aligned window, issued in disk-offset order. Counters are per bind and cannot flatter: a bind counts as
resident only if its group was already resident when the bind arrived (kimi-k3's 100%-hit beside 0%-resident trap).
MMAP binds are counted apart, since the page cache decides those.
"""
from std.ffi import external_call, c_ssize_t
from std.memory import Layout, Allocation, alloc, dealloc
from shard_index import ShardIndex, TensorRef

comptime ALIGN = 4096
comptime O_DIRECT = 0o40000
comptime T_MMAP = 0
comptime T_PIN = 1
comptime T_RING = 2


@fieldwise_init
struct Extent(Copyable, Movable):
    var name: String
    var shard: Int
    var off: Int
    var length: Int
    var win: Int
    var win_len: Int
    var pos: Int
    var maddr: Int


@fieldwise_init
struct Group(Copyable, Movable):
    var ext: List[Extent]
    var buf_len: Int

    def data(self, base: Int, i: Int) -> Int:
        """Address of extent i's bytes: base 0 means the MMAP tier (the mapping itself)."""
        if base == 0:
            return self.ext[i].maddr
        return base + self.ext[i].pos + (self.ext[i].off - self.ext[i].win)

    def find(self, name: String) raises -> Int:
        for i in range(len(self.ext)):
            if self.ext[i].name == name:
                return i
        raise Error("wstore: no tensor " + name + " in group")


def make_group(var ext: List[Extent]) -> Group:
    for i in range(1, len(ext)):
        var j = i
        while j > 0 and (ext[j - 1].shard > ext[j].shard or (ext[j - 1].shard == ext[j].shard and ext[j - 1].off > ext[j].off)):
            ext.swap_elements(j - 1, j)
            j -= 1
    var pos = 0
    for i in range(len(ext)):
        ext[i].win = ext[i].off // ALIGN * ALIGN
        ext[i].win_len = (ext[i].off + ext[i].length + ALIGN - 1) // ALIGN * ALIGN - ext[i].win
        ext[i].pos = pos
        pos += ext[i].win_len
    return Group(ext^, pos)


def trunk_group(idx: ShardIndex, layer: Int) raises -> Group:
    var pre = "blk." + String(layer) + "."
    var ext = List[Extent]()
    for item in idx.t.items():
        if item.key.startswith(pre) and "_exps" not in item.key:
            ext.append(Extent(item.key, item.value.shard, item.value.off, item.value.nbytes, 0, 0, 0, item.value.addr))
    if len(ext) == 0:
        raise Error("wstore: no trunk tensors for layer " + String(layer))
    return make_group(ext^)


def n_experts(idx: ShardIndex, layer: Int) raises -> Int:
    var name = "blk." + String(layer) + ".ffn_gate_exps.weight"
    if name not in idx.t:
        return 0
    var r = idx[name]
    return r.dims[len(r.dims) - 1]


def expert_group(idx: ShardIndex, layer: Int, e: Int) raises -> Group:
    var ext = List[Extent]()
    for part in ["gate", "up", "down"]:
        var name = "blk." + String(layer) + ".ffn_" + part + "_exps.weight"
        var r = idx[name]
        var per = r.nbytes // r.dims[len(r.dims) - 1]
        ext.append(Extent(name, r.shard, r.off + e * per, per, 0, 0, 0, r.addr + e * per))
    return make_group(ext^)


struct WeightStore(Movable):
    var idx: ShardIndex
    var fds: List[Int]
    var n_layers: Int
    var trunk: List[Group]
    var tier: List[Int]
    var pin_at: List[Int]
    var nslot: Int
    var slot_bytes: Int
    var ring_at: Int
    var slot_layer: List[Int]
    var ring_next: Int
    var eslots: Int
    var eslot_bytes: Int
    var eslot_at: Int
    var ekey: List[Int]
    var eused: List[Int]
    var eslot_of: Dict[Int, Int]
    var egroups: Dict[Int, Group]
    var clock: Int
    var arena: Allocation[UInt8]
    var t_binds: Int
    var t_resident: Int
    var t_mmap: Int
    var t_reads: Int
    var t_bytes: Int
    var e_binds: Int
    var e_resident: Int
    var e_reads: Int
    var e_bytes: Int
    var pin_bytes: Int

    def __init__(out self, first_shard: String, n_layers: Int, pin_layers: Int, nslot: Int, expert_slots: Int,
                 mmap_only: Bool) raises:
        self.idx = ShardIndex(first_shard)
        self.n_layers = n_layers
        self.fds = List[Int]()
        for p in self.idx.paths:
            var cp = p
            var fd = external_call["open64", Int32](cp.as_c_string_span().ptr(), Int32(O_DIRECT))
            if fd < 0:
                raise Error("wstore: O_DIRECT open failed " + p)
            self.fds.append(Int(fd))
        self.trunk = List[Group]()
        self.tier = List[Int]()
        self.pin_at = List[Int]()
        var pin_total = 0
        self.slot_bytes = 0
        self.eslot_bytes = 0
        for l in range(n_layers):
            self.trunk.append(trunk_group(self.idx, l))
            var t = T_MMAP if mmap_only else (T_PIN if l < pin_layers else T_RING)
            if t == T_RING and nslot < 1:
                raise Error("wstore: layer " + String(l) + " streams but the ring has no slot")
            self.tier.append(t)
            self.pin_at.append(0)
            if t == T_PIN:
                pin_total += self.trunk[l].buf_len
            elif t == T_RING:
                self.slot_bytes = max(self.slot_bytes, self.trunk[l].buf_len)
            if n_experts(self.idx, l) > 0:
                self.eslot_bytes = max(self.eslot_bytes, expert_group(self.idx, l, 0).buf_len)
        self.nslot = 0 if mmap_only else nslot
        self.eslots = 0 if mmap_only else expert_slots
        var total = pin_total + self.nslot * self.slot_bytes + self.eslots * self.eslot_bytes + ALIGN
        self.arena = alloc(Layout[UInt8](count=total))
        var at = (Int(self.arena.unsafe_ptr()) + ALIGN - 1) // ALIGN * ALIGN
        self.ring_at = 0
        self.eslot_at = 0
        self.slot_layer = List[Int](length=self.nslot, fill=-1)
        self.ring_next = 0
        self.ekey = List[Int](length=self.eslots, fill=-1)
        self.eused = List[Int](length=self.eslots, fill=0)
        self.eslot_of = Dict[Int, Int]()
        self.egroups = Dict[Int, Group]()
        self.clock = 0
        self.t_binds = 0
        self.t_resident = 0
        self.t_mmap = 0
        self.t_reads = 0
        self.t_bytes = 0
        self.e_binds = 0
        self.e_resident = 0
        self.e_reads = 0
        self.e_bytes = 0
        self.pin_bytes = 0
        for l in range(n_layers):
            if self.tier[l] == T_PIN:
                self.pin_at[l] = at
                self.pin_bytes += self._read(self.trunk[l], at)
                at += self.trunk[l].buf_len
        self.ring_at = at
        at += self.nslot * self.slot_bytes
        self.eslot_at = at

    def _read(self, g: Group, base: Int) raises -> Int:
        """O_DIRECT read of every extent window into base; returns the data bytes delivered."""
        var data = 0
        for i in range(len(g.ext)):
            ref e = g.ext[i]
            var p = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=base + e.pos)
            var got = 0
            var flen = self.idx.lens[e.shard]
            while got < e.win_len:
                var n = external_call["pread", c_ssize_t](self.fds[e.shard], p.unsafe_offset(got), e.win_len - got, Int64(e.win + got))
                if n <= 0:
                    if e.win + got >= flen:
                        break
                    raise Error("wstore: short read of " + e.name)
                got += Int(n)
            if e.off + e.length > e.win + got:
                raise Error("wstore: window of " + e.name + " ends before its data")
            data += e.length
        return data

    def bind_layer(mut self, layer: Int) raises -> Int:
        """Base address of layer's group (0 = MMAP tier); tensor addresses via self.trunk[layer].data(base, i)."""
        self.t_binds += 1
        var t = self.tier[layer]
        if t == T_MMAP:
            self.t_mmap += 1
            return 0
        if t == T_PIN:
            self.t_resident += 1
            return self.pin_at[layer]
        for s in range(self.nslot):
            if self.slot_layer[s] == layer:
                self.t_resident += 1
                return self.ring_at + s * self.slot_bytes
        var s = self.ring_next
        self.ring_next = (s + 1) % self.nslot
        self.slot_layer[s] = -1
        var base = self.ring_at + s * self.slot_bytes
        self.t_bytes += self._read(self.trunk[layer], base)
        self.slot_layer[s] = layer
        self.t_reads += 1
        return base

    def expert(mut self, layer: Int, e: Int) raises -> Group:
        var key = layer * 65536 + e
        if key not in self.egroups:
            self.egroups[key] = expert_group(self.idx, layer, e)
        return self.egroups[key].copy()

    def bind_experts(mut self, layer: Int, ids: List[Int]) raises -> List[Int]:
        """Slot base per requested expert (0 = MMAP). Misses read in disk-offset order; a slot used by this batch
        is never the victim for another expert of the same batch."""
        self.clock += 1
        var out = List[Int](length=len(ids), fill=0)
        self.e_binds += len(ids)
        if self.eslots == 0:
            self.e_resident += 0
            return out^
        var miss = List[Int]()
        for i in range(len(ids)):
            var key = layer * 65536 + ids[i]
            if key in self.eslot_of:
                var s = self.eslot_of[key]
                self.eused[s] = self.clock
                out[i] = self.eslot_at + s * self.eslot_bytes
                self.e_resident += 1
            else:
                var dup = False
                for j in miss:
                    dup = dup or ids[j] == ids[i]
                if not dup:
                    miss.append(i)
        for a in range(1, len(miss)):
            var b = a
            while b > 0 and self.expert(layer, ids[miss[b - 1]]).ext[0].off > self.expert(layer, ids[miss[b]]).ext[0].off:
                miss.swap_elements(b - 1, b)
                b -= 1
        for i in miss:
            var victim = -1
            for s in range(self.eslots):
                if self.eused[s] == self.clock:
                    continue
                if victim < 0 or self.ekey[s] < 0 or (self.ekey[victim] >= 0 and self.eused[s] < self.eused[victim]):
                    victim = s
                if self.ekey[s] < 0:
                    break
            if victim < 0:
                raise Error("wstore: batch of " + String(len(ids)) + " experts exceeds " + String(self.eslots) + " slots")
            if self.ekey[victim] >= 0:
                _ = self.eslot_of.pop(self.ekey[victim])
            var key = layer * 65536 + ids[i]
            self.ekey[victim] = -1
            var base = self.eslot_at + victim * self.eslot_bytes
            self.e_bytes += self._read(self.expert(layer, ids[i]), base)
            self.ekey[victim] = key
            self.eslot_of[key] = victim
            self.eused[victim] = self.clock
            self.e_reads += 1
            out[i] = base
        for i in range(len(ids)):
            if out[i] == 0:
                out[i] = self.eslot_at + self.eslot_of[layer * 65536 + ids[i]] * self.eslot_bytes
        return out^

    def report(self) -> String:
        return ("wstore: trunk binds " + String(self.t_binds) + " resident " + String(self.t_resident) + " mmap "
                + String(self.t_mmap) + " reads " + String(self.t_reads) + " bytes " + String(self.t_bytes)
                + " | experts binds " + String(self.e_binds) + " resident " + String(self.e_resident) + " reads "
                + String(self.e_reads) + " bytes " + String(self.e_bytes) + " | pinned " + String(self.pin_bytes))

    def __deinit__(deinit self):
        for fd in self.fds:
            _ = external_call["close", Int32](Int32(fd))
        dealloc(self.arena^)
