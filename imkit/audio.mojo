# VENDORED COPY of amarbaro/mojo-imkit (Apache-2.0, see imkit/LICENSE), tag v0.9.0 (eb91210),
# ~/Projects/mojo/mojo-imkit, file src/imkit/audio.mojo. This repo keeps a real file
# rather than an -I path outside the tree, because a clone must build without anything
# outside it. Sync by hand if the upstream changes; tools/ci-checks.sh compares the two
# against the tag when the upstream is present and says so when they drift.
#
"""SDL3 audio out for an app's sound: one mono float32 playback stream on the default device, filled by the app
(no callback). Keep a little queued every frame:

    var au = Audio(48000)
    while au.queued() < 4800:          # samples: 0.1 s ahead
        synth(buf, 1024); au.put(buf.unsafe_ptr(), 1024)
"""
from std.ffi import OwnedDLHandle


struct Audio(Movable):
    var sdl: OwnedDLHandle
    var stream: Int

    def __init__(out self, rate: Int) raises:
        self.sdl = OwnedDLHandle("libSDL3.so.0")
        if not self.sdl.get_function[Bool]("SDL_InitSubSystem")(UInt32(0x10)):  # SDL_INIT_AUDIO
            raise Error("SDL audio init failed")
        var spec = SIMD[DType.int32, 4](0x8120, 1, Int32(rate), 0)  # SDL_AudioSpec: SDL_AUDIO_F32LE, mono, rate
        self.stream = self.sdl.get_function[Int]("SDL_OpenAudioDeviceStream")(UInt32(0xFFFFFFFF), UnsafePointer(to=spec), Int(0), Int(0))
        if self.stream == 0:
            raise Error("SDL_OpenAudioDeviceStream failed")
        _ = self.sdl.get_function[Bool]("SDL_ResumeAudioStreamDevice")(self.stream)  # opens paused

    def put(self, samples: UnsafePointer[Float32, _], n: Int) raises:
        _ = self.sdl.get_function[Bool]("SDL_PutAudioStreamData")(self.stream, samples, Int32(4 * n))

    def queued(self) raises -> Int:
        """Samples still waiting to play."""
        return Int(self.sdl.get_function[Int32]("SDL_GetAudioStreamQueued")(self.stream)) // 4

    def __deinit__(deinit self):
        try:
            self.sdl.get_function[NoneType]("SDL_DestroyAudioStream")(self.stream)
        except:
            pass
