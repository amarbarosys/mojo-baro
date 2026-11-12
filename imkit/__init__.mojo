# VENDORED COPY of amarbaro/mojo-imkit (Apache-2.0, see imkit/LICENSE), tag v0.9.0 (eb91210),
# ~/Projects/mojo/mojo-imkit, file src/imkit/__init__.mojo. This repo keeps a real file
# rather than an -I path outside the tree, because a clone must build without anything
# outside it. Sync by hand if the upstream changes; tools/ci-checks.sh compares the two
# against the tag when the upstream is present and says so when they drift.
#
"""imkit: an immediate-mode GUI for Mojo apps, composed on the GPU (docs/design.md).

    window   SDL3 window, input, present (system libSDL3 through FFI)
    font     glyph atlas from a TrueType font (system libfreetype through FFI)
    draw     the draw list, tile binning and the compose kernel -> an RGBA layer over the app's frame
    theme    colors by role: glass, neon, flat
    ui       widgets that append to the draw list
"""
