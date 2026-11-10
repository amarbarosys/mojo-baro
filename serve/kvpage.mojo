"""KV block table for the paged cache (bench/a2-protocol.md, A2 step 1).

The KV pool is page-major in 128-token pages (`kernels/attn.mojo::kv_off`); every
kernel reads the physical page of logical page p from `tab[p]`. One request
owns the whole pool today, so the table is the identity map; `reverse()` is the
gate arm that proves the kernels read it, and `alloc`/`release` are the free
list A3 will hand pages out of.
"""
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer


struct PageTable(Movable):
    var n: Int
    var phys: List[Int]
    var free: List[Int]

    def __init__(out self, n: Int):
        self.n = n
        self.phys = List[Int](capacity=n)
        self.free = List[Int](capacity=n)
        for i in range(n):
            self.phys.append(i)
            self.free.append(i)

    def identity(mut self):
        for i in range(self.n):
            self.phys[i] = i

    def reverse(mut self):
        for i in range(self.n):
            self.phys[i] = self.n - 1 - i

    def alloc(mut self, k: Int) raises -> List[Int]:
        if len(self.free) < k:
            raise Error("kv page table: " + String(k) + " pages requested, " + String(len(self.free)) + " free")
        var out = List[Int](capacity=k)
        for _ in range(k):
            out.append(self.free.pop())
        return out^

    def release(mut self, pages: List[Int]):
        for p in pages:
            self.free.append(p)

    def upload(self, ctx: DeviceContext, mut tab_h: HostBuffer[DType.int32], mut tab_d: DeviceBuffer[DType.int32]) raises:
        for i in range(self.n):
            tab_h[i] = Int32(self.phys[i])
        ctx.enqueue_copy(dst_buf=tab_d, src_buf=tab_h)
