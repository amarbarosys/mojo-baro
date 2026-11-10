"""Persistent pthread pool for the CPU tier (Mojo 1.2 std has no parallelize, and max is not in the import graph).

`Pool[body](n).run(ctx, rows, chunk)` calls `body(ctx, lo, hi)` over [0, rows) in chunks taken from one atomic counter,
on n threads (the caller is one of them). Workers spin on a generation word between jobs, so a job costs no
thread start. A row is computed by one thread in its serial order, so results do not depend on n.
"""
from std.atomic import Atomic
from std.ffi import external_call
from std.memory import Layout, alloc

comptime A = Atomic[Int64]


struct Shared:
    var gen: A
    var next: A
    var done: A
    var ctx: Int
    var rows: Int
    var chunk: Int
    var quit: Bool

    def __init__(out self):
        self.gen = A(0)
        self.next = A(0)
        self.done = A(0)
        self.ctx = 0
        self.rows = 0
        self.chunk = 1
        self.quit = False


def take[body: def(Int, Int, Int) thin -> None](s: MutPointer[Shared, MutUntrackedOrigin]):
    while True:
        var lo = Int(s[].next.fetch_add(Int64(s[].chunk)))
        if lo >= s[].rows:
            return
        body(s[].ctx, lo, min(lo + s[].chunk, s[].rows))


def worker[body: def(Int, Int, Int) thin -> None](arg: OpaquePointer[MutAnyOrigin]) abi("C") -> OpaquePointer[MutAnyOrigin]:
    var s = MutPointer[Shared, MutUntrackedOrigin](unsafe_from_address=Int(arg))
    var seen = Int64(0)
    while True:
        var g = s[].gen.load()
        if g == seen:
            _ = external_call["sched_yield", Int32]()
            continue
        seen = g
        if s[].quit:
            return arg
        take[body](s)
        _ = s[].done.fetch_add(1)


struct Pool[body: def(Int, Int, Int) thin -> None](Movable):
    var s: MutPointer[Shared, MutUntrackedOrigin]
    var n: Int
    var tids: List[UInt64]

    def __init__(out self, n: Int):
        self.s = alloc(Layout[Shared](count=1)).unsafe_leak()
        self.s.unsafe_write(Shared())
        self.n = max(n, 1)
        self.tids = List[UInt64](length=self.n, fill=0)
        for i in range(1, self.n):
            _ = external_call["pthread_create", Int32](self.tids.unsafe_ptr().unsafe_offset(i), Int(0), worker[Self.body], OpaquePointer[MutAnyOrigin](unsafe_from_address=Int(self.s)))

    def run(self, ctx: Int, rows: Int, chunk: Int):
        if self.n == 1 or rows <= chunk:
            Self.body(ctx, 0, rows)
            return
        self.s[].ctx = ctx
        self.s[].rows = rows
        self.s[].chunk = chunk
        self.s[].next.store(0)
        self.s[].done.store(0)
        _ = self.s[].gen.fetch_add(1)
        take[Self.body](self.s)
        while self.s[].done.load() < Int64(self.n - 1):
            continue

    def __deinit__(deinit self):
        self.s[].quit = True
        _ = self.s[].gen.fetch_add(1)
        for i in range(1, self.n):
            _ = external_call["pthread_join", Int32](self.tids[i], Int(0))
