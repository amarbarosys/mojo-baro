# VENDORED COPY of amarbaro/mojo-imkit (Apache-2.0, see imkit/LICENSE), tag v0.9.0 (eb91210),
# ~/Projects/mojo/mojo-imkit, file src/imkit/atlas.mojo. This repo keeps a real file
# rather than an -I path outside the tree, because a clone must build without anything
# outside it. Sync by hand if the upstream changes; tools/ci-checks.sh compares the two
# against the tag when the upstream is present and says so when they drift.
#
"""Skyline packer for the glyph atlas (font.mojo): `place(w, h)` puts a box at the lowest, then leftmost, spot of an AW x AH
sheet and returns (x, y), or (-1, -1) when it fits nowhere. The skyline is a list of (x, width, height) steps, left to right."""


struct Skyline(Movable):
    var w: Int
    var h: Int
    var xs: List[Int]
    var ws: List[Int]
    var ys: List[Int]

    def __init__(out self, w: Int, h: Int):
        self.w, self.h = w, h
        self.xs = [0]
        self.ws = [w]
        self.ys = [0]

    def place(mut self, w: Int, h: Int) -> Tuple[Int, Int]:
        var bi, bx, by = -1, 0, self.h + 1
        for i in range(len(self.xs)):
            if self.xs[i] + w > self.w:
                break
            var y, j, rem = 0, i, w  # the steps this box would stand on
            while rem > 0:
                y = max(y, self.ys[j])
                rem -= self.ws[j]
                j += 1
            if y + h <= self.h and y < by:
                bi, bx, by = i, self.xs[i], y
        if bi < 0:
            return (-1, -1)
        self.xs.insert(bi, bx)
        self.ws.insert(bi, w)
        self.ys.insert(bi, by + h)
        var k = bi + 1
        while k < len(self.xs) and self.xs[k] < bx + w:  # trim or drop the steps the new one covers
            var cut = bx + w - self.xs[k]
            if cut < self.ws[k]:
                self.xs[k] += cut
                self.ws[k] -= cut
                break
            _ = self.xs.pop(k)
            _ = self.ws.pop(k)
            _ = self.ys.pop(k)
        return (bx, by)
