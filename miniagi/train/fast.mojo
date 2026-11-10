from std.math import ceildiv
from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major
from matmul_wmma_lds import amar_matmul_wmma_lds, BLK_M, BLK_N, NTHREADS
from ops import F, THREADS, grid, add_k

comptime H16 = MutPointer[Float16, MutUntrackedOrigin]


def cast_k[tr: Bool](x: F, y: H16, rows32: Int32, cols32: Int32):
    var rows = Int(rows32)
    var cols = Int(cols32)
    var i = global_idx.x
    if i >= rows * cols:
        return
    y[unsafe_offset=(i % cols) * rows + i // cols if tr else i] = x[unsafe_offset=i].cast[DType.float16]()


struct Wmma(Movable):
    var a: DeviceBuffer[DType.float16]
    var b: DeviceBuffer[DType.float16]
    var c: DeviceBuffer[DType.float32]

    def __init__(out self, ctx: DeviceContext, cap: Int) raises:
        self.a = ctx.enqueue_create_buffer[DType.float16](cap)
        self.b = ctx.enqueue_create_buffer[DType.float16](cap)
        self.c = ctx.enqueue_create_buffer[DType.float32](cap)

    def gemm[ta: Bool, tb: Bool](self, ctx: DeviceContext, a: F, b: F, c: F, m: Int, n: Int, k: Int, acc: Bool = False) raises:
        var ah = H16(unsafe_from_address=Int(self.a.unsafe_ptr()))
        var bh = H16(unsafe_from_address=Int(self.b.unsafe_ptr()))
        ctx.enqueue_function[cast_k[ta]](a, ah, Int32(k if ta else m), Int32(m if ta else k), grid_dim=grid(m * k), block_dim=THREADS)
        ctx.enqueue_function[cast_k[tb]](b, bh, Int32(n if tb else k), Int32(k if tb else n), grid_dim=grid(k * n), block_dim=THREADS)
        var la = row_major(m, k)
        var lb = row_major(k, n)
        var lc = row_major(m, n)
        comptime kern = amar_matmul_wmma_lds[type_of(la), type_of(lb), type_of(lc)]
        ctx.enqueue_function[kern](TileTensor(self.a, la), TileTensor(self.b, lb), TileTensor(self.c, lc), Int32(m), Int32(n), Int32(k),
            grid_dim=(ceildiv(n, BLK_N), ceildiv(m, BLK_M)), block_dim=NTHREADS)
        var tmp = F(unsafe_from_address=Int(self.c.unsafe_ptr()))
        if not acc:
            ctx.enqueue_memset(DeviceBuffer[DType.float32](ctx, c, m * n, owning=False), 0)
        ctx.enqueue_function[add_k](tmp, c, Int32(m * n), grid_dim=grid(m * n), block_dim=THREADS)
