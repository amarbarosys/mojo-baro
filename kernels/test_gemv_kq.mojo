"""amar_gemv_kq (kernels/gemv_kq.mojo) byte test: the native K-quant GEMV the spark engine runs on Q4_K / Q6_K
packs, every epilogue mode (0 store, 1 add, 2 gelu*, 3 silu*), on random Q4_K, Q6_K and split Q4_K/Q6_K tensors
(the fused QKV of a Q4_K_M file: Q rows and K rows Q4_K, V rows Q6_K), odd and even superblock counts, with bf16 and
f32 activations (the engine's ADT; the epilogue output takes the same dtype). The dot and the pre-activation echo are
byte-equal in both dtypes; the f32 gelu/silu epilogue is allowed 3e-7 relative (device exp/tanh vs host libm, 1 ulp;
the bf16 epilogue stays byte-equal).

1. Decode: the lane decoders the kernel reuses (q4x_kernels q4k_s16 / q6k_s16) against tools/ggml_quant.mojo
   dequant_block (gguf-py's operation order): Q6_K bit-equal, Q4_K within one f32 ulp (the kernel fuses q*d - m).
2. GEMV: device bytes equal to the host run of the same lane partials (kq_lane_part), the same 16-wide tree and the
   same xor butterfly, for every output element and epilogue mode. Fails loudly on the first mismatch.
"""
from std.math import exp, fma, tanh
from std.memory import bitcast
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from gemv_kq import amar_gemv_kq, kq_lane_part, kq_tree16, kq_bytes, kq_lane16, GT_Q4K, GT_Q6K
from matmul_skinny import ROW_WAVES, ROW_THREADS
from q4x_kernels import UP
from ggml_quant import dequant_block

comptime f32 = DType.float32
comptime bf16 = DType.bfloat16
comptime u8 = DType.uint8
comptime N = 24
comptime KMAX = 1536


struct Lcg:
    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed

    def next(mut self) -> UInt64:
        self.s = self.s * 6364136223846793005 + 1442695040888963407
        return self.s >> 33

    def byte(mut self) -> UInt8:
        return UInt8(self.next() & 0xFF)

    def unit(mut self) -> Float32:
        return Float32(Int(self.next() % 20001) - 10000) / 10000


def put_f16(mut buf: List[UInt8], at: Int, x: Float32):
    var h = bitcast[DType.uint16](Float16(x))
    buf[at] = UInt8(h & 0xFF)
    buf[at + 1] = UInt8(h >> 8)


def gen_rows(mut rng: Lcg, gt: Int, rows: Int, k: Int, mut buf: List[UInt8]):
    var bs = 144 if gt == GT_Q4K else 210
    for _ in range(rows * (k // 256)):
        var base = len(buf)
        for _ in range(bs):
            buf.append(rng.byte())
        if gt == GT_Q4K:
            put_f16(buf, base, rng.unit() * 0.05)
            put_f16(buf, base + 2, rng.unit() * 0.05)
        else:
            put_f16(buf, base + 208, rng.unit() * 0.05)


def host_epi[ADT: DType](epi: Int, c: Float32, total: Float32) -> Tuple[Float32, Scalar[ADT]]:
    if epi == 0:
        return (total, Scalar[ADT](0))
    if epi == 1:
        return (c + total, Scalar[ADT](0))
    if epi == 2:
        var gelu = 0.5 * c * (1 + tanh(0.7978845608028654 * c * (1 + 0.044715 * c * c)))
        return (c, (gelu * total).cast[ADT]())
    var silu = c / (1 + exp(-c))
    return (c, (silu * total).cast[ADT]())


def host_row[GT: Int, ADT: DType](X: TileTensor[ADT, type_of(row_major[1, KMAX]()), MutAnyOrigin], w: UP, rb: Int, k: Int) -> Float32:
    var v = SIMD[f32, 32](0)
    for lane in range(32):
        v[lane] = kq_tree16(kq_lane_part[GT, ADT](X, w, rb, k, lane))
    for i in range(5):
        var nv = SIMD[f32, 32](0)
        for lane in range(32):
            nv[lane] = v[lane] + v[lane ^ (1 << i)]
        v = nv
    return v[0]


def check_decode[GT: Int](w: UP, base: Int, mut bad: Int):
    var want = List[Float32](length=256, fill=0)
    dequant_block(GT, Int(w.unsafe_offset(base)), want, 0)
    for s in range(16):
        var r = kq_lane16[GT](w, base, s)
        var v = fma(r[0].cast[f32](), SIMD[f32, 16](r[1]), SIMD[f32, 16](-r[2]))
        for j in range(16):
            var e = want[r[3] + j]
            var tol = Float32(0) if GT == GT_Q6K else abs(e) * 3e-7
            if abs(v[j] - e) > tol:
                if bad < 5:
                    print("  decode mismatch gt", GT, "elem", r[3] + j, v[j], "vs", e)
                bad += 1


def run_case[GTA: Int, GTB: Int, ADT: DType](ctx: DeviceContext, name: String, k: Int, split: Int, mut rng: Lcg) raises:
    var wb = List[UInt8]()
    gen_rows(rng, GTA, split, k, wb)
    var offb = len(wb)
    gen_rows(rng, GTB, N - split, k, wb)
    var w_d = ctx.enqueue_create_buffer[u8](len(wb))
    var w_h = ctx.enqueue_create_host_buffer[u8](len(wb))
    var x_d = ctx.enqueue_create_buffer[ADT](KMAX)
    var x_h = ctx.enqueue_create_host_buffer[ADT](KMAX)
    var c_d = ctx.enqueue_create_buffer[f32](N)
    var c_h = ctx.enqueue_create_host_buffer[f32](N)
    var o_d = ctx.enqueue_create_buffer[ADT](N)
    var o_h = ctx.enqueue_create_host_buffer[ADT](N)
    var c0 = List[Float32](length=N, fill=0)
    ctx.synchronize()
    for i in range(len(wb)):
        w_h[i] = wb[i]
    for i in range(KMAX):
        x_h[i] = rng.unit().cast[ADT]()
    for i in range(N):
        c0[i] = rng.unit() * 4
    ctx.enqueue_copy(dst_buf=w_d, src_buf=w_h)
    ctx.enqueue_copy(dst_buf=x_d, src_buf=x_h)
    ctx.synchronize()
    var wp = UP(unsafe_from_address=Int(w_h.unsafe_ptr()))
    var bad = 0
    for b in range(k // 256):
        check_decode[GTA](wp, b * kq_bytes[GTA](), bad)
        check_decode[GTB](wp, offb + b * kq_bytes[GTB](), bad)
    var X_h = TileTensor(x_h, row_major[1, KMAX]())
    var X = TileTensor(x_d, row_major[1, KMAX]())
    var C = TileTensor(c_d, row_major[N]())
    var O = TileTensor(o_d, row_major[N]())
    var want = List[Float32](length=N, fill=0)
    for row in range(N):
        if row < split:
            want[row] = host_row[GTA, ADT](X_h, wp, row * (k // 256) * kq_bytes[GTA](), k)
        else:
            want[row] = host_row[GTB, ADT](X_h, wp, offb + (row - split) * (k // 256) * kq_bytes[GTB](), k)
    comptime for epi in range(4):
        for i in range(N):
            c_h[i] = c0[i]
            o_h[i] = Scalar[ADT](0)
        ctx.enqueue_copy(dst_buf=c_d, src_buf=c_h)
        ctx.enqueue_copy(dst_buf=o_d, src_buf=o_h)
        ctx.enqueue_function[amar_gemv_kq[GTA, GTB, epi, type_of(X).LayoutType, type_of(C).LayoutType, type_of(O).LayoutType, ADT]](
            X, w_d.unsafe_ptr(), C, O, Int32(N), Int32(k), Int32(split), Int32(offb),
            grid_dim=(N + ROW_WAVES - 1) // ROW_WAVES, block_dim=ROW_THREADS,
        )
        ctx.enqueue_copy(dst_buf=c_h, src_buf=c_d)
        ctx.enqueue_copy(dst_buf=o_h, src_buf=o_d)
        ctx.synchronize()
        for row in range(N):
            var e = host_epi[ADT](epi, c0[row], want[row])
            var ob = o_h[row].cast[f32]()
            var eb = e[1].cast[f32]()
            var ob_tol = Float32(0) if ADT == bf16 or epi < 2 else abs(eb) * 3e-7
            if bitcast[DType.uint32](c_h[row]) != bitcast[DType.uint32](e[0]) or (ob_tol == 0 and bitcast[DType.uint32](ob) != bitcast[DType.uint32](eb)) or abs(ob - eb) > ob_tol:
                if bad < 10:
                    print("  gemv mismatch", name, "epi", epi, "row", row, c_h[row], o_h[row], "vs", e[0], e[1])
                bad += 1
    print(name, "act", ADT, "k", k, "split", split, "bytes", len(wb), "mismatches", bad)
    if bad != 0:
        raise Error("FAIL " + name)


def main() raises:
    var ctx = DeviceContext()
    var rng = Lcg(7)
    comptime for a in range(2):
        comptime ADT = bf16 if a == 0 else f32
        run_case[GT_Q4K, GT_Q4K, ADT](ctx, "q4k", 1536, N, rng)
        run_case[GT_Q6K, GT_Q6K, ADT](ctx, "q6k", 1536, N, rng)
        run_case[GT_Q4K, GT_Q6K, ADT](ctx, "q4k/q6k", 1280, 16, rng)
        run_case[GT_Q6K, GT_Q6K, ADT](ctx, "q6k-odd", 768, N, rng)
    print("PASS test_gemv_kq")
