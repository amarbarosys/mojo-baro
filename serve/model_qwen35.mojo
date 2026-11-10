from std.sys import get_defined_int

# Shape from the GGUF (tools/model-id.py emits these -D flags); defaults are Qwythos-9B.
comptime H = get_defined_int["BARO_H", 4096]()
comptime VOCAB = get_defined_int["BARO_VOCAB", 248320]()
comptime FFN = get_defined_int["BARO_FFN", 12288]()
comptime HD = get_defined_int["BARO_HD", 256]()
comptime NQH = get_defined_int["BARO_NQH", 16]()
comptime NKVH = get_defined_int["BARO_NKVH", 4]()
# Trunk depth only: the MTP draft block (nextn_predict_layers) rides as one extra blk.N.
comptime N_LAYERS = get_defined_int["BARO_N_LAYERS", 32]()
comptime N_ATT = N_LAYERS // get_defined_int["BARO_ATT_EVERY", 4]()
comptime N_SSM = N_LAYERS - N_ATT
comptime QF = 2 * NQH * HD
comptime KV = NKVH * HD
comptime NROT = get_defined_int["BARO_NROT", 64]()
comptime YARN_LOW = Float32(14.0)
comptime YARN_HIGH = Float32(22.0)
comptime FREQ_BASE = Float32(1e7)
# Qwythos declares yarn factor 4.0 (1M context); a GGUF with no rope.scaling
# (OrcaSAQ-27B) runs plain rope, as llama.cpp does: BARO_YARN=0.
comptime YARN = get_defined_int["BARO_YARN", 1]() == 1
comptime FREQ_SCALE = Float32(0.25) if YARN else Float32(1.0)
comptime MSCALE = Float32(1.1386294361119891) if YARN else Float32(1.0)
comptime MEGA_ALLOWED = True
