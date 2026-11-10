# Llama 3.2 1B shape for mojo-baro's kernels: this file shadows mojo-baro serve/model.mojo on the include path.
# Rope is NOT mojo-baro's (GGUF Llama rotates adjacent pairs, scaled by rope_freqs): the YARN fields stay neutral.
comptime H = 2048
comptime VOCAB = 128256
comptime FFN = 8192
comptime HD = 64
comptime NQH = 32
comptime NKVH = 8
comptime N_LAYERS = 16
comptime N_ATT = N_LAYERS
comptime N_SSM = 0
comptime QF = NQH * HD
comptime KV = NKVH * HD
comptime NROT = HD
comptime YARN_LOW = Float32(0.0)
comptime YARN_HIGH = Float32(0.0)
comptime FREQ_BASE = Float32(500000.0)
comptime FREQ_SCALE = Float32(1.0)
comptime MSCALE = Float32(1.0)
comptime IS_MOE = False
comptime MEGA_ALLOWED = False
