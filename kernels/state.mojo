from max.gpu import block_idx, thread_idx
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from layout import row_major, stack_allocation
from std.memory import Pointer

comptime KV4_TAIL = 512
comptime KV5_TAIL = 256
comptime KV4_ROWS = 128 * 256
comptime KV4_PACKED = KV4_ROWS // 2
comptime KV4_SRC_BLOCK = KV4_PACKED
comptime KV4_DST_BLOCK = KV4_ROWS + KV4_TAIL


def kv4_value(u: Int) -> Int8:
    var q = u - 8
    if q <= -7:
        return Int8(-127)
    if q == -6:
        return Int8(-109)
    if q == -5:
        return Int8(-91)
    if q == -4:
        return Int8(-73)
    if q == -3:
        return Int8(-54)
    if q == -2:
        return Int8(-36)
    if q == -1:
        return Int8(-18)
    if q == 0:
        return Int8(0)
    if q == 1:
        return Int8(18)
    if q == 2:
        return Int8(36)
    if q == 3:
        return Int8(54)
    if q == 4:
        return Int8(73)
    if q == 5:
        return Int8(91)
    if q == 6:
        return Int8(109)
    return Int8(127)


def amar_kv4_unpack[T16: Bool](
    src_all: Pointer[UInt8, ImmUntrackedOrigin],
    dst: Pointer[Int8, MutUntrackedOrigin],
    nblocks: Int32,
):
    var b = Int(block_idx.x)
    if b >= Int(nblocks):
        return
    var t = Int(thread_idx.x)
    comptime TAIL_IN = KV5_TAIL if T16 else KV4_TAIL
    var tails = src_all
    var packed = src_all.unsafe_offset(Int(nblocks) * TAIL_IN)
    var src = packed.unsafe_offset(b * KV4_SRC_BLOCK)
    var out = dst.unsafe_offset(b * KV4_DST_BLOCK)
    var per = KV4_PACKED // 256
    for i in range(per):
        var j = t * per + i
        var byte = Int(src[j])
        out[2 * j] = kv4_value(byte & 15)
        out[2 * j + 1] = kv4_value(byte >> 4)
    if t < KV4_TAIL // 4:
        var tdst = out.unsafe_offset(KV4_ROWS + 4 * t)
        comptime if T16:
            var h = tails.unsafe_offset(b * KV5_TAIL + 2 * t).unsafe_bitcast[Float16]()[0]
            tdst.unsafe_bitcast[Float32]()[0] = Float32(h)
        else:
            var tsrc = tails.unsafe_offset(b * KV4_TAIL + 4 * t)
            for k in range(4):
                tdst[k] = Int8(tsrc[k])


comptime KV6_TAB = 32 * 256 * 16 * 2
comptime KV6_STREAMS = 64
comptime KV6_SYMS = 512


def amar_kv6_decode(
    src: Pointer[UInt8, ImmUntrackedOrigin],
    dst: Pointer[Int8, MutUntrackedOrigin],
    nblocks: Int32,
):
    var b = Int(block_idx.x)
    var nb = Int(nblocks)
    if b >= nb:
        return
    var t = Int(thread_idx.x)
    var tails = src
    var tables = src.unsafe_offset(nb * KV5_TAIL)
    var offs = tables.unsafe_offset(KV6_TAB)
    var streams = offs.unsafe_offset((nb * KV6_STREAMS + 1) * 4)
    var out = dst.unsafe_offset(b * KV4_DST_BLOCK)
    for k in range(2):
        var i = 2 * t + k
        var h = tails.unsafe_offset(b * KV5_TAIL + 2 * i).unsafe_bitcast[Float16]()[0]
        out.unsafe_offset(KV4_ROWS + 4 * i).unsafe_bitcast[Float32]()[0] = Float32(h)
    var cum = stack_allocation[DType.uint16, address_space = AddressSpace.SHARED](row_major[256 * 16]())
    var tab = tables.unsafe_offset((b % 32) * 256 * 16 * 2).unsafe_bitcast[UInt16]()
    for c in range(4):
        var ch = 4 * t + c
        var acc = 0
        for s in range(16):
            cum[ch * 16 + s] = UInt16(acc)
            acc += Int(tab[ch * 16 + s])
    barrier()
    var si = b * KV6_STREAMS + t
    var op = offs.unsafe_bitcast[UInt32]()
    var p = Int(op[si])
    var x = (Int(streams[p]) << 24) | (Int(streams[p + 1]) << 16) | (Int(streams[p + 2]) << 8) | Int(streams[p + 3])
    p += 4
    var word = 0
    for j in range(KV6_SYMS):
        var ch = 4 * t + (j & 3)
        var slot = x & 4095
        var s = 0
        for k in range(1, 15):
            if slot >= Int(cum[ch * 16 + k]):
                s = k
        var c0 = Int(cum[ch * 16 + s])
        var f = Int(cum[ch * 16 + s + 1]) - c0
        x = f * (x >> 12) + slot - c0
        while x < 65536:
            x = (x << 8) | Int(streams[p])
            p += 1
        var v = Int(kv4_value(s + 1)) & 255
        word |= v << (8 * (j & 3))
        if (j & 3) == 3:
            out.unsafe_offset((j >> 2) * 256 + 4 * t).unsafe_bitcast[UInt32]()[0] = UInt32(word)
            word = 0


def amar_split_bf16_planes(
    src: Pointer[UInt16, ImmUntrackedOrigin],
    dst_hi: Pointer[UInt8, MutUntrackedOrigin],
    dst_lo: Pointer[UInt8, MutUntrackedOrigin],
    n: Int32,
):
    var i = Int(block_idx.x) * 256 + Int(thread_idx.x)
    if i >= Int(n):
        return
    var w = src[i]
    dst_hi[i] = UInt8(w >> 8)
    dst_lo[i] = UInt8(w & 255)


def amar_merge_bf16_planes(
    src_hi: Pointer[UInt8, ImmUntrackedOrigin],
    src_lo: Pointer[UInt8, ImmUntrackedOrigin],
    dst: Pointer[UInt16, MutUntrackedOrigin],
    n: Int32,
):
    var i = Int(block_idx.x) * 256 + Int(thread_idx.x)
    if i >= Int(n):
        return
    dst[i] = (UInt16(src_hi[i]) << 8) | UInt16(src_lo[i])

