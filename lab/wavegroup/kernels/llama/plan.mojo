# Arena planner: every intermediate gets an offset in one buffer by liveness. Greedy interval colouring, largest first:
# place each buffer at the lowest 256-byte-aligned offset that overlaps no already placed buffer whose steps overlap its own.
from std.collections import Dict, List

@fieldwise_init
struct Buf(Copyable, Movable):
    var name: String
    var bytes: Int
    var first: Int # first step that writes it
    var last: Int  # last step that reads it
    var off: Int

def from_steps(sizes: Dict[String, Int], steps: List[List[String]]) raises -> List[Buf]:
    """Lifetimes from the step list (first and last step naming each buffer), never typed by hand."""
    var out = List[Buf]()
    for e in sizes.items():
        var first = -1
        var last = -1
        for s in range(len(steps)):
            if e.key in steps[s]:
                last = s
                if first < 0:
                    first = s
        if first < 0:
            raise Error("plan: " + e.key + " is in no step")
        out.append(Buf(e.key, e.value, first, last, 0))
    return out^

def plan(mut bufs: List[Buf]) -> Int:
    """Fills each off; returns the arena size (peak bytes)."""
    var keys = List[Int]() # -bytes * 4096 + index: ascending = largest first
    for i in range(len(bufs)):
        keys.append(-bufs[i].bytes * 4096 + i)
    sort(keys)
    var peak = 0
    var placed = List[Int]()
    for k in keys:
        var i = k % 4096
        var off = 0
        var moved = True
        while moved: # bump past every live-overlapping neighbour until none overlaps
            moved = False
            for j in placed:
                var live = bufs[i].first <= bufs[j].last and bufs[j].first <= bufs[i].last
                if live and off < bufs[j].off + bufs[j].bytes and bufs[j].off < off + bufs[i].bytes:
                    off = (bufs[j].off + bufs[j].bytes + 255) // 256 * 256
                    moved = True
        bufs[i].off = off
        placed.append(i)
        peak = max(peak, off + bufs[i].bytes)
    return peak
