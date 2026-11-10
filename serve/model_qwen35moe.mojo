from std.sys import get_defined_int

# Shape from the GGUF (tools/model-id.py emits these -D flags); defaults are RegesCore-1.0-35B.
comptime H = get_defined_int["BARO_H", 2048]()
comptime VOCAB = get_defined_int["BARO_VOCAB", 248320]()
comptime FFN = get_defined_int["BARO_E_FFN", 512]()
comptime N_EXP = get_defined_int["BARO_N_EXP", 256]()
comptime TOPK = get_defined_int["BARO_TOPK", 8]()
comptime SH_FFN = get_defined_int["BARO_SH_FFN", 512]()
comptime HD = get_defined_int["BARO_HD", 256]()
comptime NQH = get_defined_int["BARO_NQH", 16]()
comptime NKVH = get_defined_int["BARO_NKVH", 2]()
comptime N_LAYERS = get_defined_int["BARO_N_LAYERS", 40]()
comptime N_ATT = N_LAYERS // get_defined_int["BARO_ATT_EVERY", 4]()
comptime N_SSM = N_LAYERS - N_ATT
comptime QF = 2 * NQH * HD
comptime KV = NKVH * HD
comptime NROT = get_defined_int["BARO_NROT", 64]()
comptime YARN_LOW = Float32(14.0)
comptime YARN_HIGH = Float32(22.0)
comptime FREQ_BASE = Float32(1e7)
# RegesCore's GGUF declares NO rope scaling: no rope.scaling.type, no
# rope.scaling.factor, only rope.freq_base 1e7 and rope.dimension_count 64.
# llama.cpp therefore runs plain rope for this model (freq_scale 1.0,
# ext_factor 0, mscale 1.0) and so must we. The 0.25 here was copied from
# the qwen35 dense profile, whose GGUF really does declare yarn factor 4.0;
# it made us interpolate where the reference extrapolates.
comptime FREQ_SCALE = Float32(1.0)
comptime MSCALE = Float32(1.0)
comptime MEGA_ALLOWED = False
