# VENDORED COPY of amarbaro/mojo-imkit (Apache-2.0, see imkit/LICENSE), tag v0.9.0 (eb91210),
# ~/Projects/mojo/mojo-imkit, file src/imkit/anim.mojo. This repo keeps a real file
# rather than an -I path outside the tree, because a clone must build without anything
# outside it. Sync by hand if the upstream changes; tools/ci-checks.sh compares the two
# against the tag when the upstream is present and says so when they drift.
#
"""Spring table for widget animation: a widget id maps to (current value, velocity). Each frame a widget asks
`get(id, target)` and draws the value it gets back; an id first seen starts at its target (no animation on first sight).
`begin(dt)` evicts every id no widget asked for last frame, so a closed menu or a scrolled-away row leaves no state.
Semi-implicit Euler; dt is clamped to 1/20 s so a stall never makes the spring blow up.
"""
comptime STIFF = Float32(260)  # spring constant (1 / s^2)
comptime DAMP = Float32(26)  # about 0.8 of critical damping (2 * sqrt(STIFF) = 32)
comptime DT_MAX = Float32(0.05)


struct Anim(Movable):
    var id: List[Int]
    var cur: List[Float32]
    var vel: List[Float32]
    var seen: List[Int]  # frame number of the last get()
    var frame: Int
    var dt: Float32

    def __init__(out self):
        self.id = List[Int]()
        self.cur = List[Float32]()
        self.vel = List[Float32]()
        self.seen = List[Int]()
        self.frame, self.dt = 0, 0

    def begin(mut self, dt: Float32):
        """Start a frame of `dt` seconds: drop the ids the last frame did not touch."""
        var k = 0
        while k < len(self.id):
            if self.seen[k] != self.frame:
                var last = len(self.id) - 1
                self.id[k], self.cur[k], self.vel[k], self.seen[k] = self.id[last], self.cur[last], self.vel[last], self.seen[last]
                _ = self.id.pop()
                _ = self.cur.pop()
                _ = self.vel.pop()
                _ = self.seen.pop()
            else:
                k += 1
        self.frame += 1
        self.dt = min(max(dt, Float32(0)), DT_MAX)

    def get(mut self, id: Int, target: Float32) -> Float32:
        """Step id's spring one frame toward target and return its value; at rest (within 1e-3, speed under 1e-2) it snaps."""
        var k = 0
        while k < len(self.id) and self.id[k] != id:
            k += 1
        if k == len(self.id):
            self.id.append(id)
            self.cur.append(target)
            self.vel.append(0)
            self.seen.append(self.frame)
            return target
        self.seen[k] = self.frame
        var v = self.vel[k] + (STIFF * (target - self.cur[k]) - DAMP * self.vel[k]) * self.dt
        var x = self.cur[k] + v * self.dt
        if abs(target - x) < 1e-3 and abs(v) < 1e-2:
            x, v = target, 0
        self.cur[k], self.vel[k] = x, v
        return x
