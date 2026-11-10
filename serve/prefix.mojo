"""Prefix checkpoints for one sequence (bench/chat-protocol.md M1a, M1b).

A checkpoint is the SSM state after `pos` tokens (one conv ring slot + one
delta ring slot, copied device -> pinned host) keyed by a SHA-256 digest of
a per-pack salt followed by the little-endian i32 token ids `tokens[0:pos]`.
The salt (derived from the pack directory string) means two packs loaded by
two different processes never share a checkpoint's identity even if their
early tokens happen to coincide -- load-bearing once checkpoints move off
this one process (disk persistence, a pool of engines), inert but free
today. The KV pool is position addressed and never copied: for a single
sequence the entries for `[0, pos)` are still in place unless a later
replay overwrote them, which is what `invalidate_above` tracks.

M1b adds role-boundary checkpoints: `serve/src/main.rs` renders each prefix
of the conversation's messages (no generation prompt) and sends the token
length after each one as a `ckpt` hint. A hint position is taken alongside
the M1a period-1024 grid and the prompt-end point; the first hint (the
system prompt, by convention message 0) is pinned against eviction, and a
full chain evicts a periodic-grid point before a role-boundary one. A hint
that does not actually land on a stable tokenization boundary costs nothing
beyond a wasted slot: `lookup`'s hash compare is what decides correctness,
never the hint itself.
"""
from std.collections import Array
from std.bit import rotate_bits_right
from std.builtin.globals import global_constant
from std.os import stat

from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from registry import CONV_SLOT, SSM_SLOT, KVPAGE, KVHSTR, NKVH, N_ATT, KVT, SEQ_CAP

comptime f32 = DType.float32
comptime CKPT_PERIOD = 1024
comptime CKPT_BYTES = (CONV_SLOT + SSM_SLOT) * 4

# ---- SHA-256 (FIPS 180-4) --------------------------------------------------

comptime _SHA_H0: Array[UInt32, 8] = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19]

comptime _SHA_K: Array[UInt32, 64] = [
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
    0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3, 0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
    0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC, 0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
    0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7, 0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
    0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13, 0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
    0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3, 0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
    0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5, 0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
    0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208, 0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2,
]


def _sha_ch(x: UInt32, y: UInt32, z: UInt32) -> UInt32:
    return (x & y) ^ (~x & z)


def _sha_maj(x: UInt32, y: UInt32, z: UInt32) -> UInt32:
    return (x & y) ^ (x & z) ^ (y & z)


def _sha_bsig0(x: UInt32) -> UInt32:
    return rotate_bits_right[2](x) ^ rotate_bits_right[13](x) ^ rotate_bits_right[22](x)


def _sha_bsig1(x: UInt32) -> UInt32:
    return rotate_bits_right[6](x) ^ rotate_bits_right[11](x) ^ rotate_bits_right[25](x)


def _sha_ssig0(x: UInt32) -> UInt32:
    return rotate_bits_right[7](x) ^ rotate_bits_right[18](x) ^ (x >> 3)


def _sha_ssig1(x: UInt32) -> UInt32:
    return rotate_bits_right[17](x) ^ rotate_bits_right[19](x) ^ (x >> 10)


def _sha_compress(mut h: Array[UInt32, 8], block: List[UInt8], off: Int):
    var w = Array[UInt32, 64](fill=UInt32(0))
    for t in range(16):
        var i = off + t * 4
        w[t] = (UInt32(block[i]) << 24) | (UInt32(block[i + 1]) << 16) | (UInt32(block[i + 2]) << 8) | UInt32(block[i + 3])
    for t in range(16, 64):
        w[t] = _sha_ssig1(w[t - 2]) + w[t - 7] + _sha_ssig0(w[t - 15]) + w[t - 16]
    var a = h[0]
    var b = h[1]
    var c = h[2]
    var d = h[3]
    var e = h[4]
    var f = h[5]
    var g = h[6]
    var hh = h[7]
    for t in range(64):
        var t1 = hh + _sha_bsig1(e) + _sha_ch(e, f, g) + global_constant[_SHA_K]()[t] + w[t]
        var t2 = _sha_bsig0(a) + _sha_maj(a, b, c)
        hh = g
        g = f
        f = e
        e = d + t1
        d = c
        c = b
        b = a
        a = t1 + t2
    h[0] += a
    h[1] += b
    h[2] += c
    h[3] += d
    h[4] += e
    h[5] += f
    h[6] += g
    h[7] += hh


def sha256_bytes(data: List[UInt8]) -> List[UInt8]:
    var h = materialize[_SHA_H0]()
    var n = len(data)
    var nfull = n // 64
    for i in range(nfull):
        _sha_compress(h, data, i * 64)
    var rem = n - nfull * 64
    var pad = List[UInt8](unsafe_uninit_length=128)
    for i in range(128):
        pad[i] = 0
    for i in range(rem):
        pad[i] = data[nfull * 64 + i]
    pad[rem] = UInt8(0x80)
    var bitlen = UInt64(n) * 8
    var padlen = 128 if rem >= 56 else 64
    for i in range(8):
        pad[padlen - 1 - i] = UInt8((bitlen >> UInt64(i * 8)) & UInt64(0xFF))
    _sha_compress(h, pad, 0)
    if padlen == 128:
        _sha_compress(h, pad, 64)
    var out = List[UInt8](unsafe_uninit_length=32)
    for i in range(8):
        out[i * 4 + 0] = UInt8((h[i] >> 24) & 0xFF)
        out[i * 4 + 1] = UInt8((h[i] >> 16) & 0xFF)
        out[i * 4 + 2] = UInt8((h[i] >> 8) & 0xFF)
        out[i * 4 + 3] = UInt8(h[i] & 0xFF)
    return out^


def hint_index(hints: List[Int], pos: Int) -> Int:
    # Index of `pos` in a request's M1b role-boundary hints, or -1.
    for i in range(len(hints)):
        if hints[i] == pos:
            return i
    return -1


def next_ckpt_stop(hints: List[Int], pos: Int, limit: Int) -> Int:
    # Smallest checkpoint-relevant position > pos and <= limit: the next
    # CKPT_PERIOD grid point or the next hint, whichever is smaller (limit
    # if neither is closer). A prefill chunk is capped to this so a hint
    # that would otherwise fall inside one big chunk still lands on a real
    # wst.pos value -- step_window only stops at chunk boundaries, and a
    # checkpoint can only be taken between calls, never mid-chunk.
    var stop = limit
    var grid = ((pos // CKPT_PERIOD) + 1) * CKPT_PERIOD
    if grid < stop:
        stop = grid
    for i in range(len(hints)):
        if hints[i] > pos and hints[i] < stop:
            stop = hints[i]
    return stop


def bytes_eq(a: List[UInt8], b: List[UInt8]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def string_bytes(s: String) -> List[UInt8]:
    var out = List[UInt8]()
    for b in s.as_bytes():
        out.append(b)
    return out^


def prefix_hash(salt: List[UInt8], tokens: List[Int], n: Int) -> List[UInt8]:
    # salt || little-endian i32 tokens[0:n].
    var msg = List[UInt8](unsafe_uninit_length=len(salt) + n * 4)
    for i in range(len(salt)):
        msg[i] = salt[i]
    var base = len(salt)
    for i in range(n):
        var v = UInt32(tokens[i])
        msg[base + i * 4 + 0] = UInt8(v & 0xFF)
        msg[base + i * 4 + 1] = UInt8((v >> 8) & 0xFF)
        msg[base + i * 4 + 2] = UInt8((v >> 16) & 0xFF)
        msg[base + i * 4 + 3] = UInt8((v >> 24) & 0xFF)
    return sha256_bytes(msg)


@fieldwise_init
struct Checkpoint(Copyable, Movable):
    var pos: Int
    var hash: List[UInt8]
    var gen: Int
    var valid: Bool
    var pending: Bool
    # M1b retention (Chain.save's eviction order): pinned checkpoints are
    # never evicted (the system prompt, hint index 0); boundary marks a
    # role-boundary checkpoint, evicted only after every periodic-grid one.
    var pinned: Bool
    var boundary: Bool
    var conv_h: HostBuffer[f32]
    var ssm_h: HostBuffer[f32]
    var kv_h: HostBuffer[f32]
    # BARO_KV_SHARE: physical KV pages behind logical pages [0, ceil(pos / KVPAGE)); empty = identity.
    var pages: List[Int32]
    # Restores served; a checkpoint two requests restored (a shared persona) is evicted last.
    var hits: Int

    # B-M4: (valid, pending) is one of free (F,F), pending (F,T), valid (T,F); (T,T) must not exist.
    # Every writer outside this file goes through these two, never through the Bools.
    def set_valid(mut self):
        self.valid = True
        self.pending = False

    def install(mut self, pos: Int, var hash: List[UInt8], gen: Int, pinned: Bool, boundary: Bool):
        # A checkpoint whose payload was written from outside a decode (state file, IPC): all metadata at once, then valid.
        self.pos = pos
        self.hash = hash^
        self.gen = gen
        self.pinned = pinned
        self.boundary = boundary
        self.set_valid()


# ---- P1 CONTRACT 2 identity salt (P1-STATE-API.md, coordinator amendment
# 0c68cb4): <packdir>/identity.json, written by tools/engine-pack.py, is the
# only place `pack_sha256`/`tokenizer_sha256` are computed -- this reads it,
# it never re-derives either hash. A hand-rolled reader, not a JSON library:
# the file has a fixed, known shape (a handful of flat string/int keys), the
# same idiom serve/identity.mojo already uses for GGUF header fields.


def _hex_nibble_or_zero(c: UInt8) -> UInt8:
    # Malformed input degrades to a wrong salt, never a crash or a silent
    # match: a wrong salt only ever costs a spurious "saved from a different
    # pack" refusal, the same failure mode as identity.json being absent.
    if c >= 48 and c <= 57:
        return c - 48
    if c >= 97 and c <= 102:
        return c - 97 + 10
    if c >= 65 and c <= 70:
        return c - 65 + 10
    return 0


def _hex_to_bytes(s: String) -> List[UInt8]:
    var b = s.as_bytes()
    var n = len(b) // 2
    var out = List[UInt8](unsafe_uninit_length=n)
    for i in range(n):
        out[i] = (_hex_nibble_or_zero(b[i * 2]) << 4) | _hex_nibble_or_zero(b[i * 2 + 1])
    return out^


def _json_value_start(text: String, key: String) -> Int:
    # Byte offset of the value right after `"key":`, skipping one optional
    # space (`json.dumps(..., indent=2)`'s separator); -1 if the key is
    # absent. Not a general JSON scanner: identity.json's keys never nest
    # and never repeat, so a plain substring search is exact here.
    var needle = String("\"") + key + String("\":")
    var idx = text.find(needle)
    if idx < 0:
        return -1
    var pos = idx + len(needle.as_bytes())
    var b = text.as_bytes()
    if pos < len(b) and b[pos] == 32:
        pos += 1
    return pos


def _json_string_at(text: String, pos: Int) -> String:
    var b = text.as_bytes()
    if pos < 0 or pos >= len(b) or b[pos] != 34:
        return String("")
    var end = pos + 1
    while end < len(b) and b[end] != 34:
        end += 1
    return String(unsafe_from_utf8=Span(b)[pos + 1:end])


def _json_int_at(text: String, pos: Int) -> Int:
    var b = text.as_bytes()
    if pos < 0:
        return -1
    var end = pos
    while end < len(b) and ((b[end] >= 48 and b[end] <= 57) or b[end] == 45):
        end += 1
    if end == pos:
        return -1
    try:
        return Int(String(unsafe_from_utf8=Span(b)[pos:end]))
    except:
        return -1


def pack_identity_salt(packdir: String) -> List[UInt8]:
    """CONTRACT 2: `sha256(pack_sha256 || tokenizer_sha256)` from
    `<packdir>/identity.json`, only when the file is present, parses, and its
    `pack_bytes`/`pack_mtime_ns` match `packdir/pack.bin` on disk right now
    (a stat, not a re-hash -- P1's whole point is avoiding a 7 GB rehash at
    every engine start). Any miss at all -- missing file, missing field,
    malformed hex, or a stat mismatch -- falls back to today's
    `sha256(packdir)`, never raises."""
    var fallback = sha256_bytes(string_bytes(packdir))
    var text: String
    try:
        var raw: List[UInt8]
        with open(packdir + "/identity.json", "r") as f:
            raw = f.read_bytes()
        text = String(unsafe_from_utf8=Span(raw))
    except:
        return fallback.copy()
    var pack_sha_hex = _json_string_at(text, _json_value_start(text, "pack_sha256"))
    var tok_sha_hex = _json_string_at(text, _json_value_start(text, "tokenizer_sha256"))
    var recorded_bytes = _json_int_at(text, _json_value_start(text, "pack_bytes"))
    var recorded_mtime = _json_int_at(text, _json_value_start(text, "pack_mtime_ns"))
    # tokenizer_sha256 is nullable in the pack identity contract. A missing
    # tokenizer hash must not turn an otherwise portable pack identity into a
    # path-derived salt, because the same pack is routinely mounted at a
    # different path on the receiving server.
    if pack_sha_hex.byte_length() != 64 or (tok_sha_hex.byte_length() != 0 and tok_sha_hex.byte_length() != 64) or recorded_bytes < 0 or recorded_mtime < 0:
        return fallback.copy()
    try:
        var st = stat(packdir + "/pack.bin")
        if st.st_size != recorded_bytes or st.st_mtimespec.as_nanoseconds() != recorded_mtime:
            return fallback.copy()
    except:
        return fallback.copy()
    var msg = _hex_to_bytes(pack_sha_hex)
    if tok_sha_hex.byte_length() == 64:
        for b in _hex_to_bytes(tok_sha_hex):
            msg.append(b)
    return sha256_bytes(msg)


struct Chain(Movable):
    var cap: Int
    var gen: Int
    var salt: List[UInt8]
    var items: List[Checkpoint]
    var npages: Int
    var lives: List[List[Int32]]

    def __init__(out self, ctx: DeviceContext, cap: Int, packdir: String, kv_elements_per_ckpt: Int = 0) raises:
        self.cap = cap
        self.gen = 0
        self.salt = pack_identity_salt(packdir)
        self.items = List[Checkpoint]()
        self.npages = 0
        self.lives = List[List[Int32]]()
        for _ in range(max(2, SEQ_CAP)):
            self.lives.append(List[Int32]())
        for _ in range(cap):
            var c = ctx.enqueue_create_host_buffer[f32](CONV_SLOT)
            var s = ctx.enqueue_create_host_buffer[f32](SSM_SLOT)
            var k = ctx.enqueue_create_host_buffer[f32](kv_elements_per_ckpt)
            self.items.append(Checkpoint(pos=0, hash=List[UInt8](), gen=0, valid=False, pending=False, pinned=False, boundary=False, conv_h=c^, ssm_h=s^, kv_h=k^, pages=List[Int32](), hits=0))
        ctx.synchronize()

    def lookup(self, tokens: List[Int], n: Int) -> Int:
        # Index of the valid checkpoint with the largest pos <= n - 1 whose
        # hash matches tokens[0:pos]; -1 when none. n = len(tokens) normally.
        var best = -1
        for i in range(len(self.items)):
            if not self.items[i].valid or self.items[i].pos < 1 or self.items[i].pos > n - 1:
                continue
            if best >= 0 and self.items[i].pos <= self.items[best].pos:
                continue
            if bytes_eq(prefix_hash(self.salt, tokens, self.items[i].pos), self.items[i].hash):
                best = i
        return best

    def touch(mut self, idx: Int):
        # A restore makes a checkpoint recent: eviction drops the least recently saved or used.
        if idx >= 0:
            self.gen += 1
            self.items[idx].gen = self.gen
            self.items[idx].hits += 1

    def _evict_before(self, i: Int, j: Int) -> Bool:
        # Order of eviction: unshared before shared (restored twice or more), periodic-grid
        # before role-boundary, then least recently saved or restored.
        var si = self.items[i].hits >= 2
        var sj = self.items[j].hits >= 2
        if si != sj:
            return sj
        if self.items[i].boundary != self.items[j].boundary:
            return self.items[j].boundary
        return self.items[i].gen < self.items[j].gen

    def pos_of(self, idx: Int) -> Int:
        return self.items[idx].pos if idx >= 0 else 0

    def invalidate_above(mut self, pos: Int):
        # A replay from `pos` overwrites KV [pos, ..): every later checkpoint
        # loses its KV prefix. A miss (pos 0) drops everything.
        for i in range(len(self.items)):
            if self.items[i].pos > pos:
                self.items[i].valid = False
                self.items[i].pending = False

    def save(
        mut self, ctx: DeviceContext, convstate_d: DeviceBuffer[f32], sstate_d: DeviceBuffer[f32],
        slot: Int, pos: Int, tokens: List[Int], pinned: Bool, boundary: Bool, seq: Int = 0,
    ) raises:
        self.save_kv(ctx, convstate_d, sstate_d, slot, pos, tokens, pinned, boundary, DeviceBuffer[f32](ctx, convstate_d.unsafe_ptr(), 0, owning=False), False, seq)

    def save_kv(
        mut self, ctx: DeviceContext, convstate_d: DeviceBuffer[f32], sstate_d: DeviceBuffer[f32],
        slot: Int, pos: Int, tokens: List[Int], pinned: Bool, boundary: Bool,
        kv_d: DeviceBuffer[f32], save_kv_page: Bool = True, seq: Int = 0,
    ) raises:
        # Device -> pinned host copy of ring slot `slot` and KV pages, stream-ordered.
        if self.cap == 0:
            return
        var idx = -1
        for i in range(len(self.items)):
            if self.items[i].valid and self.items[i].pos == pos:
                idx = i
                break
        if idx < 0:
            for i in range(len(self.items)):
                if not self.items[i].valid and not self.items[i].pending:
                    idx = i
                    break
        if idx < 0:
            for i in range(len(self.items)):
                if self.items[i].pinned:
                    continue
                if idx < 0:
                    idx = i
                    continue
                if self._evict_before(i, idx):
                    idx = i
            if idx < 0:
                idx = 0
                for i in range(1, len(self.items)):
                    if self.items[i].gen < self.items[idx].gen:
                        idx = i
        self.gen += 1
        self.items[idx].pos = pos
        self.items[idx].hash = prefix_hash(self.salt, tokens, pos)
        self.items[idx].gen = self.gen
        self.items[idx].valid = False
        self.items[idx].pending = True
        self.items[idx].pinned = pinned
        self.items[idx].boundary = boundary
        self.items[idx].hits = 0
        self.items[idx].pages = List[Int32]()
        if self.npages > 0:
            for lp in range((pos + KVPAGE - 1) // KVPAGE):
                self.items[idx].pages.append(self.lives[seq][lp])
        var cs = DeviceBuffer[f32](ctx, convstate_d.unsafe_ptr().unsafe_offset(slot * CONV_SLOT), CONV_SLOT, owning=False)
        var ss = DeviceBuffer[f32](ctx, sstate_d.unsafe_ptr().unsafe_offset(slot * SSM_SLOT), SSM_SLOT, owning=False)
        ctx.enqueue_copy(dst_buf=self.items[idx].conv_h, src_buf=cs)
        ctx.enqueue_copy(dst_buf=self.items[idx].ssm_h, src_buf=ss)
        if save_kv_page and len(self.items[idx].kv_h) >= len(kv_d):
            ctx.enqueue_copy(dst_buf=self.items[idx].kv_h, src_buf=kv_d)

    def commit(mut self):
        for i in range(len(self.items)):
            if self.items[i].pending:
                self.items[i].set_valid()

    def install(mut self, idx: Int, pos: Int, var hash: List[UInt8], pinned: Bool, boundary: Bool):
        self.gen += 1
        self.items[idx].install(pos, hash^, self.gen, pinned, boundary)

    def mark_valid(mut self, idx: Int):
        self.items[idx].set_valid()

    def restore(
        self, ctx: DeviceContext, convstate_d: DeviceBuffer[f32], sstate_d: DeviceBuffer[f32], slot: Int, idx: Int
    ) raises:
        self.restore_kv(ctx, convstate_d, sstate_d, slot, idx, DeviceBuffer[f32](ctx, convstate_d.unsafe_ptr(), 0, owning=False), False)

    def restore_kv(
        self, ctx: DeviceContext, convstate_d: DeviceBuffer[f32], sstate_d: DeviceBuffer[f32], slot: Int, idx: Int,
        kv_d: DeviceBuffer[f32], restore_kv_page: Bool = True,
    ) raises:
        var cs = DeviceBuffer[f32](ctx, convstate_d.unsafe_ptr().unsafe_offset(slot * CONV_SLOT), CONV_SLOT, owning=False)
        var ss = DeviceBuffer[f32](ctx, sstate_d.unsafe_ptr().unsafe_offset(slot * SSM_SLOT), SSM_SLOT, owning=False)
        ctx.enqueue_copy(dst_buf=cs, src_buf=self.items[idx].conv_h)
        ctx.enqueue_copy(dst_buf=ss, src_buf=self.items[idx].ssm_h)
        if restore_kv_page and len(self.items[idx].kv_h) >= len(kv_d):
            ctx.enqueue_copy(dst_buf=kv_d, src_buf=self.items[idx].kv_h)

    def page_of(self, idx: Int, lp: Int) -> Int32:
        return self.items[idx].pages[lp] if len(self.items[idx].pages) > 0 else Int32(lp)

    def share_reset(mut self, ctx: DeviceContext, kvtab_h: HostBuffer[DType.int32], mut kvtab_d: DeviceBuffer[DType.int32], tpages: Int) raises:
        # A path that writes KV pages directly (state load, IPC ingest) assumes the identity
        # table and owns the whole pool: every shared page reference goes first.
        for i in range(len(self.items)):
            self.items[i].valid = False
            self.items[i].pending = False
        for seq in range(len(self.lives)):
            self.lives[seq] = List[Int32]()
            if (seq + 1) * tpages > len(kvtab_h):
                continue
            for lp in range(tpages):
                if seq == 0:
                    self.lives[seq].append(Int32(lp))
                kvtab_h[seq * tpages + lp] = Int32(seq * tpages + lp)
        ctx.enqueue_copy(dst_buf=kvtab_d, src_buf=kvtab_h)

    def _alloc(mut self, seq: Int, need: Int, idx: Int) raises:
        # Free pages (no live sequence, no valid or pending checkpoint holds them) appended to
        # `seq` until it has `need`; evicts unpinned checkpoints other than `idx`, oldest first.
        while len(self.lives[seq]) < need:
            var used = List[Bool](length=self.npages, fill=False)
            for lv in self.lives:
                for p in lv:
                    used[Int(p)] = True
            for i in range(len(self.items)):
                if self.items[i].valid or self.items[i].pending:
                    for lp in range((self.items[i].pos + KVPAGE - 1) // KVPAGE):
                        used[Int(self.page_of(i, lp))] = True
            for p in range(self.npages):
                if not used[p] and len(self.lives[seq]) < need:
                    self.lives[seq].append(Int32(p))
            if len(self.lives[seq]) >= need:
                break
            var victim = -1
            for i in range(len(self.items)):
                if i == idx or self.items[i].pinned or not self.items[i].valid:
                    continue
                if victim < 0 or self._evict_before(i, victim):
                    victim = i
            if victim < 0:
                raise Error("BARO_KV_SHARE: " + String(self.npages) + " KV pages cannot hold " + String(need) + " for this request")
            self.items[victim].valid = False

    def grow(
        mut self, ctx: DeviceContext, seq: Int, need: Int, kvtab_h: HostBuffer[DType.int32],
        mut kvtab_d: DeviceBuffer[DType.int32], tpages: Int,
    ) raises:
        # BARO_SEQS > 2: a decoding sequence takes its next page when it reaches it, not
        # prompt + n pages at admission. The host table is rewritten only after the stream
        # drains, so no queued copy reads it mid-write.
        if len(self.lives[seq]) >= need:
            return
        self._alloc(seq, need, -1)
        ctx.synchronize()
        for lp in range(tpages):
            kvtab_h[seq * tpages + lp] = self.lives[seq][lp] if lp < need else self.lives[seq][need - 1]
        ctx.enqueue_copy(dst_buf=kvtab_d, src_buf=kvtab_h)

    def free_pages(self) -> Int:
        # Pages no live sequence and no valid or pending checkpoint holds.
        var used = List[Bool](length=self.npages, fill=False)
        for lv in self.lives:
            for p in lv:
                used[Int(p)] = True
        for i in range(len(self.items)):
            if self.items[i].valid or self.items[i].pending:
                for lp in range((self.items[i].pos + KVPAGE - 1) // KVPAGE):
                    used[Int(self.page_of(i, lp))] = True
        var n = 0
        for p in range(self.npages):
            if not used[p]:
                n += 1
        return n

    def map_pages(
        mut self, ctx: DeviceContext, idx: Int, pos: Int, need: Int, kc_d: DeviceBuffer[KVT], vc_d: DeviceBuffer[KVT],
        kvtab_h: HostBuffer[DType.int32], mut kvtab_d: DeviceBuffer[DType.int32], tpages: Int, seq: Int = 0,
    ) raises:
        # BARO_KV_SHARE: the request restored at checkpoint `idx` (pos tokens) reads that
        # checkpoint's full pages in place, gets a copy of the page holding pos, and fresh pages
        # for the rest of its `need` logical pages. Every other checkpoint keeps its pages, so a
        # second conversation no longer replays over the first one's KV (invalidate_above).
        comptime PAGE = N_ATT * NKVH * KVHSTR
        var shared = pos // KVPAGE if idx >= 0 else 0
        var cow = idx >= 0 and pos % KVPAGE != 0
        self.lives[seq] = List[Int32]()
        for lp in range(shared):
            self.lives[seq].append(self.page_of(idx, lp))
        self._alloc(seq, need, idx)
        if cow:
            var src = Int(self.page_of(idx, shared)) * PAGE
            var dst = Int(self.lives[seq][shared]) * PAGE
            ctx.enqueue_copy(dst_buf=DeviceBuffer[KVT](ctx, kc_d.unsafe_ptr().unsafe_offset(dst), PAGE, owning=False), src_buf=DeviceBuffer[KVT](ctx, kc_d.unsafe_ptr().unsafe_offset(src), PAGE, owning=False))
            ctx.enqueue_copy(dst_buf=DeviceBuffer[KVT](ctx, vc_d.unsafe_ptr().unsafe_offset(dst), PAGE, owning=False), src_buf=DeviceBuffer[KVT](ctx, vc_d.unsafe_ptr().unsafe_offset(src), PAGE, owning=False))
        for lp in range(tpages):
            kvtab_h[seq * tpages + lp] = self.lives[seq][lp] if lp < need else self.lives[seq][need - 1]
        ctx.enqueue_copy(dst_buf=kvtab_d, src_buf=kvtab_h)

    def count_valid(self) -> Int:
        var n = 0
        for i in range(len(self.items)):
            if self.items[i].valid:
                n += 1
        return n
