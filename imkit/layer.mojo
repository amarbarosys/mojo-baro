# VENDORED COPY of amarbaro/mojo-imkit (Apache-2.0, see imkit/LICENSE), tag v0.9.0 (eb91210),
# ~/Projects/mojo/mojo-imkit, file src/imkit/layer.mojo. This repo keeps a real file
# rather than an -I path outside the tree, because a clone must build without anything
# outside it. Sync by hand if the upstream changes; tools/ci-checks.sh compares the two
# against the tag when the upstream is present and says so when they drift.
#
"""GPU side of a Ui (docs/design.md, Layer): device buffers, the compose launch and the pinned RGBA the window presents,
all allocated once on the app's own DeviceContext. render_into(ui, buf) after ui.finish() uploads only what the frame used
(n commands, its tile entries, the tile offsets, styles) and composes into a device buffer; render(ui) is that into the layer's
own buffer plus the 3.7 MB copy to `pixels` (readback()), which only a host consumer such as SDL's texture upload needs.

    var layer = Layer(ctx, ui)                 # once: buffers, the font atlas from ui
    layer.render(ui)                           # every frame, after ui.finish()
    win.present(frame, Int(layer.pixels.unsafe_ptr()))
"""
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from layout import TileTensor, row_major
from .assets import Assets
from .font import AH, AW
from .draw import CW, ISRC, NBIN, NCMD, NSTY, QH, QW, TX, TY, UH, UW, blur_axis, blur_down, compose
from .ui import Ui


def image_fits(n: Int, w: Int, h: Int, off: Int = 0) -> Bool:
    """Can a w x h picture held in n pixels go to the ISRC pool at pixel offset `off` (Ui.image takes sides up to 0xFFFF)?"""
    return 0 < w <= 0xFFFF and 0 < h <= 0xFFFF and w * h <= ISRC and n >= w * h and 0 <= off <= ISRC - w * h


struct Layer(Movable):
    var ctx: DeviceContext[]
    var img: DeviceBuffer[DType.uint32]
    var cmd: DeviceBuffer[DType.uint32]
    var bins: DeviceBuffer[DType.uint16]
    var off: DeviceBuffer[DType.uint32]
    var atlas: DeviceBuffer[DType.uint8]
    var sty: DeviceBuffer[DType.uint32]
    var scene: DeviceBuffer[DType.uint32]  # the app's frame for set_scene (empty until the first set_scene)
    var q: List[DeviceBuffer[DType.uint32]]  # the 1 / BQ scene: down, after the horizontal pass blurred (q[2], sampled by GLASS)
    var src: DeviceBuffer[DType.uint32]  # image sources (Ui.image): the app writes its RGBA here
    var pixels: HostBuffer[DType.uint32]  # the composed layer, valid after render() or readback() returns (empty before the first)
    var asset_px: Int  # pool pixels [0, asset_px) hold set_assets' atlas
    var img_lo: Int  # the lowest pool offset a set_image picture was put at (ISRC: none)

    def __init__(out self, ctx: DeviceContext[], mut ui: Ui) raises:
        comptime assert UW > 0 and UH > 0 and UW % 16 == 0 and UH % 16 == 0, "UI_W, UI_H: positive multiples of 16"
        if len(ui.d.font.atlas) > AW * AH:
            raise Error("Layer: font atlas " + String(len(ui.d.font.atlas)) + " B over the compose capacity " + String(AW * AH))
        self.ctx = ctx
        self.img = ctx.enqueue_create_buffer[DType.uint32](UW * UH)
        self.cmd = ctx.enqueue_create_buffer[DType.uint32](NCMD * CW)
        self.bins = ctx.enqueue_create_buffer[DType.uint16](NBIN)
        self.off = ctx.enqueue_create_buffer[DType.uint32](TX * TY + 1)
        self.atlas = ctx.enqueue_create_buffer[DType.uint8](AW * AH)
        self.sty = ctx.enqueue_create_buffer[DType.uint32](NSTY * 8)
        self.scene = ctx.enqueue_create_buffer[DType.uint32](0)  # allocated by the first set_scene
        self.q = [ctx.enqueue_create_buffer[DType.uint32](QW * QH) for _ in range(3)]
        self.q[2].enqueue_fill(0)  # no backdrop until a scene is blurred
        self.src = ctx.enqueue_create_buffer[DType.uint32](ISRC)
        self.pixels = ctx.enqueue_create_host_buffer[DType.uint32](0)  # pinned UW * UH, allocated by the first readback()
        self.src.enqueue_fill(0)
        self.atlas.enqueue_fill(0)  # the host copy holds only the rows in use; the rest of the card's atlas stays zero
        self.asset_px, self.img_lo = 0, ISRC
        ctx.enqueue_copy(self.atlas.create_sub_buffer[DType.uint8](0, len(ui.d.font.atlas)), ui.d.font.atlas.unsafe_ptr())
        ctx.synchronize()  # the atlas is the caller's
        ui.d.font.clean()

    def set_image(mut self, px: Span[UInt32, _], w: Int, h: Int, off: Int = 0) raises:
        """Copy a w x h picture (packed r | g << 8 | b << 16 | a << 24, row-major) to pool pixel offset `off` and wait; draw it
        with Ui.image(.., off, w, h). ISRC is the pool size in pixels, set by -DUI_IMG=; a picture past it raises, so does one
        that would overwrite the atlas set_assets loaded (use off >= len(assets.px)); the pool is untouched."""
        if not image_fits(len(px), w, h, off):
            raise Error("Layer.set_image: " + String(w) + " x " + String(h) + " at " + String(off) + " (" + String(len(px)) + " px given) does not fit ISRC " + String(ISRC))
        if off < self.asset_px:
            raise Error("Layer.set_image: offset " + String(off) + " is inside the asset atlas (" + String(self.asset_px) + " px from set_assets); pass off >= " + String(self.asset_px))
        self.ctx.enqueue_copy(self.src.create_sub_buffer[DType.uint32](off, w * h), px.unsafe_ptr())
        self.ctx.synchronize()  # px is the caller's
        self.img_lo = min(self.img_lo, off)

    def blur_from(self, scene: DeviceBuffer[DType.uint32]) raises:
        """Blur a UW * UH device scene (opaque RGBA) for GLASS commands: 1 / 4 size, two triangle passes (enqueue only)."""
        var n = (QW * QH + 255) // 256
        self.ctx.enqueue_function(self.ctx.compile_function[blur_down](), TileTensor(self.q[0], row_major[QW * QH]()), TileTensor(scene, row_major[UW * UH]()), grid_dim=n, block_dim=256)
        self.ctx.enqueue_function(self.ctx.compile_function[blur_axis[1, 0]](), TileTensor(self.q[1], row_major[QW * QH]()), TileTensor(self.q[0], row_major[QW * QH]()), grid_dim=n, block_dim=256)
        self.ctx.enqueue_function(self.ctx.compile_function[blur_axis[0, 1]](), TileTensor(self.q[2], row_major[QW * QH]()), TileTensor(self.q[1], row_major[QW * QH]()), grid_dim=n, block_dim=256)

    def set_scene(mut self, px: Span[UInt32, _]) raises:
        """Copy the app's UW * UH frame (host RGBA) and blur it for GLASS; waits (px is the caller's)."""
        if len(px) != UW * UH:
            raise Error("Layer.set_scene: " + String(len(px)) + " px given, the layer is " + String(UW * UH))
        if len(self.scene) == 0:
            self.scene = self.ctx.enqueue_create_buffer[DType.uint32](UW * UH)
        self.ctx.enqueue_copy(self.scene, px.unsafe_ptr())
        self.blur_from(self.scene)
        self.ctx.synchronize()

    def set_assets(mut self, a: Assets) raises:
        """Copy an asset atlas's pixels to the image pool from offset 0 and wait; Ui.icon_img draws them. A picture set_image
        put below len(a.px) would be overwritten, so that raises: give set_image an offset past the atlas."""
        if len(a.px) > ISRC:
            raise Error("Layer.set_assets: " + String(len(a.px)) + " px over ISRC " + String(ISRC) + " (-DUI_IMG=)")
        if len(a.px) > self.img_lo:
            raise Error("Layer.set_assets: " + String(len(a.px)) + " px would overwrite a set_image picture at offset " + String(self.img_lo) + "; load the atlas first or put the picture past it")
        if len(a.px) > 0:
            self.ctx.enqueue_copy(self.src.create_sub_buffer[DType.uint32](0, len(a.px)), a.px.unsafe_ptr())
        self.ctx.synchronize()
        self.asset_px = len(a.px)

    def launch(self) raises:
        """Compose what was last uploaded into the layer's own buffer (enqueue only)."""
        self.launch_to(self.img)

    def launch_to(self, buf: DeviceBuffer[DType.uint32]) raises:
        """Compose what was last uploaded into `buf` (UW * UH u32, enqueue only)."""
        self.ctx.enqueue_function(self.ctx.compile_function[compose](), TileTensor(buf, row_major[UW * UH]()),
            TileTensor(self.cmd, row_major[NCMD * CW]()), TileTensor(self.bins, row_major[NBIN]()),
            TileTensor(self.off, row_major[TX * TY + 1]()), TileTensor(self.atlas, row_major[AW * AH]()),
            TileTensor(self.src, row_major[ISRC]()), TileTensor(self.sty, row_major[NSTY * 8]()), TileTensor(self.q[2], row_major[QW * QH]()), grid_dim=(UW * UH + 255) // 256, block_dim=256)

    def render_into(self, mut ui: Ui, buf: DeviceBuffer[DType.uint32]) raises:
        """Upload the finished frame and compose into the caller's device buffer: no layer readback, so a GPU consumer
        (a texture, a blend kernel on the same DeviceContext) reads it in stream order. Returns when ui's lists are free to reuse."""
        if ui.d.binned != ui.d.n:
            raise Error("Layer.render: call Ui.finish() first")
        if len(buf) != UW * UH:
            raise Error("Layer.render_into: buf holds " + String(len(buf)) + " u32, the layer is " + String(UW * UH))
        if ui.d.n:
            self.ctx.enqueue_copy(self.cmd.create_sub_buffer[DType.uint32](0, ui.d.n * CW), ui.d.cmd.unsafe_ptr())
        var used = Int(ui.d.off[TX * TY])
        if used:
            self.ctx.enqueue_copy(self.bins.create_sub_buffer[DType.uint16](0, used), ui.d.bins.unsafe_ptr())
        self.ctx.enqueue_copy(self.off, ui.d.off.unsafe_ptr())
        if ui.d.font.dirty1 > ui.d.font.dirty0:  # glyphs rasterized since the last frame: their rows only
            var r0, n = ui.d.font.dirty0 * AW, (ui.d.font.dirty1 - ui.d.font.dirty0) * AW
            self.ctx.enqueue_copy(self.atlas.create_sub_buffer[DType.uint8](r0, n), Span(ui.d.font.atlas)[r0 : r0 + n].unsafe_ptr())
            ui.d.font.clean()
        if ui.d.ns > 1:
            self.ctx.enqueue_copy(self.sty.create_sub_buffer[DType.uint32](0, ui.d.ns * 8), ui.d.sty.unsafe_ptr())
        self.launch_to(buf)
        self.ctx.synchronize()  # ui's lists are read by the copies above

    def readback(mut self) raises:
        """Copy the layer's own buffer to `pixels` (pinned, allocated on the first call) and wait (after render_into(ui, layer.img))."""
        if len(self.pixels) == 0:
            self.pixels = self.ctx.enqueue_create_host_buffer[DType.uint32](UW * UH)
        self.ctx.enqueue_copy(self.pixels, self.img)
        self.ctx.synchronize()

    def render(mut self, mut ui: Ui) raises:
        """Upload the finished frame, compose, copy to pixels; returns when pixels is ready (render_into + readback)."""
        self.render_into(ui, self.img)
        self.readback()
