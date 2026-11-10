from std.math import sqrt, tanh
from semantic_latent import read_f32_exact

struct LatentMLPAdapter(Movable):
    var dim_in: Int
    var rank: Int
    var dim_out: Int
    var w_down: List[Float32]
    var w_up: List[Float32]
    var b_up: List[Float32]

    def __init__(out self, dir_path: String, d_in: Int = 2048, r: Int = 256, d_out: Int = 2048) raises:
        self.dim_in = d_in
        self.rank = r
        self.dim_out = d_out
        self.w_down = read_f32_exact(dir_path + "/W_down.f32", d_in * r)
        self.w_up = read_f32_exact(dir_path + "/W_up.f32", r * d_out)
        self.b_up = read_f32_exact(dir_path + "/b_up.f32", d_out)

    def forward_ptr_in_place[mut_origin: MutOrigin](self, ptr: Pointer[Float32, mut_origin], mut scratch_rank: List[Float32], offset: Int = 0):
        var mean_acc = SIMD[DType.float32, 16](0.0)
        for i in range(0, self.dim_in, 16):
            mean_acc += ptr.load[width=16](offset + i)
        var mean = mean_acc.reduce_add() / Float32(self.dim_in)

        var var_acc = SIMD[DType.float32, 16](0.0)
        var m_v = SIMD[DType.float32, 16](mean)
        for i in range(0, self.dim_in, 16):
            var d = ptr.load[width=16](offset + i) - m_v
            var_acc = var_acc.fma(d, d)
        var inv_std = 1.0 / sqrt(var_acc.reduce_add() / Float32(self.dim_in) + 1e-5)

        var inv_v = SIMD[DType.float32, 16](inv_std)
        var w_down_ptr = self.w_down.unsafe_ptr()
        for j in range(self.rank):
            var acc_v = SIMD[DType.float32, 16](0.0)
            var w_off = j * self.dim_in
            for i in range(0, self.dim_in, 16):
                var norm_x = (ptr.load[width=16](offset + i) - m_v) * inv_v
                acc_v = acc_v.fma(norm_x, w_down_ptr.load[width=16](w_off + i))
            var acc = acc_v.reduce_add()
            var arg = 0.79788456 * (acc + 0.044715 * acc * acc * acc)
            scratch_rank[j] = 0.5 * acc * (1.0 + tanh(arg))

        var w_up_ptr = self.w_up.unsafe_ptr()
        for k in range(self.dim_out):
            var acc_v = SIMD[DType.float32, 16](0.0)
            var w_off = k * self.rank
            for j in range(0, self.rank, 16):
                var rank_v = scratch_rank.unsafe_ptr().load[width=16](j)
                acc_v = acc_v.fma(rank_v, w_up_ptr.load[width=16](w_off + j))
            ptr.store(offset + k, self.b_up[k] + acc_v.reduce_add())

    def forward_in_place(self, mut x: List[Float32], mut scratch_rank: List[Float32], offset: Int = 0):
        self.forward_ptr_in_place(x.unsafe_ptr(), scratch_rank, offset)


    def forward_batch_in_place(self, mut buf: List[Float32], positions: Int):
        var scratch_rank = List[Float32](unsafe_uninit_length=self.rank)
        for p in range(positions):
            self.forward_in_place(buf, scratch_rank, p * self.dim_in)

    def forward(self, x: List[Float32]) -> List[Float32]:
        var out = x.copy()
        var scratch = List[Float32](unsafe_uninit_length=self.rank)
        self.forward_in_place(out, scratch, 0)
        return out^


