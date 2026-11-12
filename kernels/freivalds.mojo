from arch import perm, sudot4
from max.gpu import block_idx, thread_idx, lane_id, WARP_SIZE
from max.gpu.primitives import warp
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.math import ceildiv
from std.memory import bitcast
from std.sys import size_of
from std.sys.intrinsics import llvm_intrinsic

comptime u64 = DType.uint64
comptime i64 = DType.int64
comptime i32 = DType.int32
comptime u32 = DType.uint32
comptime u8 = DType.uint8
comptime FV_P = UInt64((1 << 61) - 1)
comptime FV_M21 = UInt64((1 << 21) - 1)
comptime FV_WAVES = 8
comptime FV_THREADS = FV_WAVES * WARP_SIZE
comptime UP = MutPointer[Scalar[u64], MutAnyOrigin]
comptime BP8 = MutPointer[Scalar[u8], MutAnyOrigin]


@always_inline
def fv_red(x: UInt64) -> UInt64:
    var s = (x & FV_P) + (x >> 61)
    return s - FV_P if s >= FV_P else s


@always_inline
def fv_add(a: UInt64, b: UInt64) -> UInt64:
    var s = a + b
    return s - FV_P if s >= FV_P else s


@always_inline
def fv_signed(s: Int64) -> UInt64:
    if s >= 0:
        return fv_red(s.cast[u64]())
    var m = fv_red((-s).cast[u64]())
    return FV_P - m if m != 0 else UInt64(0)


@always_inline
def fv_shift[S: Int](x: UInt64) -> UInt64:
    comptime L = UInt64(61 - S)
    comptime MASK = (UInt64(1) << L) - 1
    return fv_red((x >> L) + ((x & MASK) << UInt64(S)))


@always_inline
def fv_term(a: Int64, r: UInt64) -> UInt64:
    var l0 = (r & FV_M21).cast[i64]()
    var l1 = ((r >> UInt64(21)) & FV_M21).cast[i64]()
    var l2 = (r >> UInt64(42)).cast[i64]()
    return fv_red(fv_signed(a * l0) + fv_shift[21](fv_signed(a * l1)) + fv_shift[42](fv_signed(a * l2)))


@always_inline
def fv_sel[a: Int, b: Int, c: Int, d: Int]() -> UInt32:
    return UInt32(a | (b << 8) | (c << 16) | (d << 24))


@always_inline
def fv_perm(hi: UInt32, lo: UInt32, sel: UInt32) -> UInt32:
    return perm(hi, lo, sel)


@always_inline
def fv_digit[ND: Int, b: Int, g: Int](w: SIMD[u32, 4]) -> UInt32:
    comptime if ND == 1:
        return w[g]
    elif ND == 2:
        return fv_perm(w[2 * g + 1], w[2 * g], fv_sel[b, 2 + b, 4 + b, 6 + b]())
    else:
        var t0 = fv_perm(w[1], w[0], fv_sel[b, 4 + b, 0, 0]())
        var t1 = fv_perm(w[3], w[2], fv_sel[b, 4 + b, 0, 0]())
        return fv_perm(t1, t0, fv_sel[0, 1, 4, 5]())


@always_inline
def fv_dot[ASIGN: Bool](a: UInt32, d: UInt32, c: Int32) -> Int32:
    return sudot4(ASIGN, bitcast[i32, 1](a), False, bitcast[i32, 1](d), c, False)


@always_inline
def fv_fold[NACC: Int, G: Int](total: UInt64, acc: Array[Int32, NACC * G]) -> UInt64:
    var t = total
    comptime for s in range(NACC):
        var v = Int64(0)
        comptime for g in range(G):
            v += acc[s * G + g].cast[i64]()
        t = fv_add(t, fv_shift[(8 * s) % 61](fv_signed(v)))
    return t


@always_inline
def fv_wave_sum(x: UInt64) -> UInt64:
    var a = x
    comptime for sh in [16, 8, 4, 2, 1]:
        a = fv_add(a, warp.shuffle_down(a, UInt32(sh)))
    return a


@always_inline
def fv_plane_at(j: Int) -> Int:
    return (j // 16) * 128 + (j % 16)


@always_inline
def fv_dots[ND: Int, G: Int](w: SIMD[u32, 4], pb: BP8, mut acc: Array[Int32, (ND + 7) * G]):
    comptime for i in range(8):
        var pl = bitcast[u32, G](pb.unsafe_offset(i * 16).unsafe_load[width=4 * G]())
        comptime for b in range(ND):
            comptime for g in range(G):
                acc[(b + i) * G + g] = fv_dot[b == ND - 1](fv_digit[ND, b, g](w), pl[g], acc[(b + i) * G + g])


@always_inline
def fv_rowchunk[T: DType, ND: Int, G: Int, U: Int](
    base: MutPointer[Scalar[T], MutAnyOrigin], planes: BP8, j0: Int, j1: Int, mut acc: Array[Int32, (ND + 7) * G]
):
    comptime VEC = 4 * G
    comptime STEP = WARP_SIZE * VEC
    var j = j0
    while j + (U - 1) * STEP < j1:
        var ws = Array[SIMD[u32, 4], U](uninitialized=True)
        comptime for u in range(U):
            ws[u] = bitcast[u32, 4](base.unsafe_offset(j + u * STEP).unsafe_load[width=VEC]())
        comptime for u in range(U):
            fv_dots[ND, G](ws[u], planes.unsafe_offset(fv_plane_at(j + u * STEP)), acc)
        j += U * STEP
    while j < j1:
        var w = bitcast[u32, 4](base.unsafe_offset(j).unsafe_load[width=VEC]())
        fv_dots[ND, G](w, planes.unsafe_offset(fv_plane_at(j)), acc)
        j += STEP


@always_inline
def fv_row[T: DType, MODE: Int, U: Int](
    Mp: MutPointer[Scalar[T], MutAnyOrigin], x: UP, planes: BP8, dst: UP, dplanes: BP8, cmp: UP, flag: UP,
    row: Int, n: Int
):
    comptime ND = size_of[Scalar[T]]()
    comptime G = 4 // ND
    comptime VEC = 4 * G
    comptime NACC = ND + 7
    comptime CHUNK = (4096 // ND) * WARP_SIZE * VEC
    var lane = Int(lane_id())
    var base = Mp.unsafe_offset(row * n)
    var total = UInt64(0)
    var nv = (n // VEC) * VEC
    for c0 in range(0, nv, CHUNK):
        var acc = Array[Int32, NACC * G](fill=Int32(0))
        fv_rowchunk[T, ND, G, U](base, planes, c0 + lane * VEC, min(nv, c0 + CHUNK), acc)
        total = fv_fold[NACC, G](total, acc)
    for j in range(nv + lane, n, WARP_SIZE):
        total = fv_add(total, fv_term(base[unsafe_offset=j].cast[i64](), x[unsafe_offset=j]))
    total = fv_wave_sum(total)
    if lane == 0:
        dst[unsafe_offset=row] = total
        comptime if MODE == 1:
            var at = fv_plane_at(row)
            comptime for i in range(8):
                dplanes[unsafe_offset=at + i * 16] = ((total >> UInt64(8 * i)) & 0xFF).cast[u8]()
            if row == 0:
                flag[unsafe_offset=0] = 0
                flag[unsafe_offset=1] = 0
        comptime if MODE == 2:
            flag[unsafe_offset=1] = 1
            if total != cmp[unsafe_offset=row]:
                flag[unsafe_offset=0] = 1


def freivalds_rowdot[T: DType, MODE: Int, U: Int = 4](
    Mp: MutPointer[Scalar[T], MutAnyOrigin], x: UP, planes: BP8, dst: UP, dplanes: BP8, cmp: UP, flag: UP,
    rows: Int32, cols: Int32
):
    var row = Int(block_idx.x) * FV_WAVES + Int(thread_idx.x) // WARP_SIZE
    if row >= Int(rows):
        return
    fv_row[T, MODE, U](Mp, x, planes, dst, dplanes, cmp, flag, row, Int(cols))


def freivalds_rowdot2[T: DType, U: Int = 4](
    B: MutPointer[Scalar[T], MutAnyOrigin], C: MutPointer[Scalar[i32], MutAnyOrigin], x: UP, planes: BP8,
    u: UP, uplanes: BP8, w: UP, flag: UP, krows: Int32, mrows: Int32, cols: Int32
):
    var kb = ceildiv(Int(krows), FV_WAVES)
    var blk = Int(block_idx.x)
    var wave = Int(thread_idx.x) // WARP_SIZE
    if blk < kb:
        var row = blk * FV_WAVES + wave
        if row >= Int(krows):
            return
        fv_row[T, 1, U](B, x, planes, u, uplanes, u, flag, row, Int(cols))
    else:
        var row = (blk - kb) * FV_WAVES + wave
        if row >= Int(mrows):
            return
        fv_row[i32, 0, U](C, x, planes, w, uplanes, w, flag, row, Int(cols))


@always_inline
def fv_stride(n: Int) -> Int:
    return ceildiv(n, 16) * 16


struct Freivalds(Movable):
    var m: Int
    var n: Int
    var k: Int
    var r_h: HostBuffer[u64]
    var r_last_h: HostBuffer[u64]
    var pr_h: HostBuffer[u8]
    var pr_last_h: HostBuffer[u8]
    var u_h: HostBuffer[u64]
    var v_h: HostBuffer[u64]
    var w_h: HostBuffer[u64]
    var f_h: HostBuffer[u64]
    var r_d: DeviceBuffer[u64]
    var u_d: DeviceBuffer[u64]
    var v_d: DeviceBuffer[u64]
    var w_d: DeviceBuffer[u64]
    var f_d: DeviceBuffer[u64]
    var pr_d: DeviceBuffer[u8]
    var pu_d: DeviceBuffer[u8]
    var urandom: FileHandle

    def __init__(out self, ctx: DeviceContext, m: Int, n: Int, k: Int) raises:
        self.m = m
        self.n = n
        self.k = k
        self.r_h = ctx.enqueue_create_host_buffer[u64](n)
        self.r_last_h = ctx.enqueue_create_host_buffer[u64](n)
        self.pr_h = ctx.enqueue_create_host_buffer[u8](8 * fv_stride(n))
        self.pr_last_h = ctx.enqueue_create_host_buffer[u8](8 * fv_stride(n))
        self.u_h = ctx.enqueue_create_host_buffer[u64](k)
        self.v_h = ctx.enqueue_create_host_buffer[u64](m)
        self.w_h = ctx.enqueue_create_host_buffer[u64](m)
        self.f_h = ctx.enqueue_create_host_buffer[u64](2)
        self.r_d = ctx.enqueue_create_buffer[u64](n)
        self.u_d = ctx.enqueue_create_buffer[u64](k)
        self.v_d = ctx.enqueue_create_buffer[u64](m)
        self.w_d = ctx.enqueue_create_buffer[u64](m)
        self.f_d = ctx.enqueue_create_buffer[u64](2)
        self.pr_d = ctx.enqueue_create_buffer[u8](8 * fv_stride(n))
        self.pu_d = ctx.enqueue_create_buffer[u8](8 * fv_stride(k))
        self.urandom = open("/dev/urandom", "r")
        ctx.synchronize()
        self.draw()
        self.upload(ctx)
        ctx.synchronize()

    def draw(mut self) raises:
        var want = 8 * self.n
        var d = self.urandom.read_bytes(want)
        if len(d) != want:
            raise Error("short read from /dev/urandom")
        var p = d.unsafe_ptr().unsafe_bitcast[Scalar[u64]]()
        for j in range(self.n):
            var v = fv_red(p[unsafe_offset=j])
            self.r_last_h[j] = v
            var at = fv_plane_at(j)
            comptime for i in range(8):
                self.pr_last_h[at + i * 16] = ((v >> UInt64(8 * i)) & 0xFF).cast[u8]()

    def upload(mut self, ctx: DeviceContext) raises:
        ctx.enqueue_copy(dst_buf=self.r_d, src_buf=self.r_last_h)
        ctx.enqueue_copy(dst_buf=self.pr_d, src_buf=self.pr_last_h)
        swap(self.r_h, self.r_last_h)
        swap(self.pr_h, self.pr_last_h)

    def launch[T: DType](
        self, ctx: DeviceContext, A: DeviceBuffer[T], B: DeviceBuffer[T], C: DeviceBuffer[i32]
    ) raises:
        ctx.enqueue_function[freivalds_rowdot2[T]](
            B.unsafe_ptr(), C.unsafe_ptr(), self.r_d.unsafe_ptr(), self.pr_d.unsafe_ptr(), self.u_d.unsafe_ptr(),
            self.pu_d.unsafe_ptr(), self.w_d.unsafe_ptr(), self.f_d.unsafe_ptr(),
            Int32(self.k), Int32(self.m), Int32(self.n),
            grid_dim=ceildiv(self.k, FV_WAVES) + ceildiv(self.m, FV_WAVES), block_dim=FV_THREADS,
        )
        ctx.enqueue_function[freivalds_rowdot[T, 2]](
            A.unsafe_ptr(), self.u_d.unsafe_ptr(), self.pu_d.unsafe_ptr(), self.v_d.unsafe_ptr(),
            self.pu_d.unsafe_ptr(), self.w_d.unsafe_ptr(), self.f_d.unsafe_ptr(), Int32(self.m), Int32(self.k),
            grid_dim=ceildiv(self.m, FV_WAVES), block_dim=FV_THREADS,
        )
        ctx.enqueue_copy(dst_buf=self.f_h, src_buf=self.f_d)

    def verdict(self) raises -> Bool:
        if self.f_h[1] != 1:
            raise Error("freivalds: compare kernel did not run")
        return self.f_h[0] == 0

    def run[T: DType](
        mut self, ctx: DeviceContext, A: DeviceBuffer[T], B: DeviceBuffer[T], C: DeviceBuffer[i32]
    ) raises -> Bool:
        self.launch[T](ctx, A, B, C)
        self.draw()
        ctx.synchronize()
        var ok = self.verdict()
        self.upload(ctx)
        return ok

    def read_back(self, ctx: DeviceContext) raises:
        ctx.enqueue_copy(dst_buf=self.u_h, src_buf=self.u_d)
        ctx.enqueue_copy(dst_buf=self.v_h, src_buf=self.v_d)
        ctx.enqueue_copy(dst_buf=self.w_h, src_buf=self.w_d)
        ctx.synchronize()


def freivalds_check[T: DType](
    ctx: DeviceContext, A: DeviceBuffer[T], B: DeviceBuffer[T], C: DeviceBuffer[i32], m: Int, n: Int, k: Int
) raises -> Bool:
    var fv = Freivalds(ctx, m, n, k)
    return fv.run[T](ctx, A, B, C)
