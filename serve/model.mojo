from std.sys import get_defined_string
import model_qwen35
import model_qwen35moe

comptime MODEL = get_defined_string["BARO_MODEL", "qwen35"]()
comptime IS_MOE = MODEL == "qwen35moe"

comptime H = model_qwen35moe.H if IS_MOE else model_qwen35.H
comptime VOCAB = model_qwen35moe.VOCAB if IS_MOE else model_qwen35.VOCAB
comptime FFN = model_qwen35moe.FFN if IS_MOE else model_qwen35.FFN
comptime HD = model_qwen35moe.HD if IS_MOE else model_qwen35.HD
comptime NQH = model_qwen35moe.NQH if IS_MOE else model_qwen35.NQH
comptime NKVH = model_qwen35moe.NKVH if IS_MOE else model_qwen35.NKVH
comptime N_LAYERS = model_qwen35moe.N_LAYERS if IS_MOE else model_qwen35.N_LAYERS
comptime N_SSM = model_qwen35moe.N_SSM if IS_MOE else model_qwen35.N_SSM
comptime N_ATT = model_qwen35moe.N_ATT if IS_MOE else model_qwen35.N_ATT
comptime QF = model_qwen35moe.QF if IS_MOE else model_qwen35.QF
comptime KV = model_qwen35moe.KV if IS_MOE else model_qwen35.KV
comptime NROT = model_qwen35moe.NROT if IS_MOE else model_qwen35.NROT
comptime YARN_LOW = model_qwen35moe.YARN_LOW if IS_MOE else model_qwen35.YARN_LOW
comptime YARN_HIGH = model_qwen35moe.YARN_HIGH if IS_MOE else model_qwen35.YARN_HIGH
comptime FREQ_BASE = model_qwen35moe.FREQ_BASE if IS_MOE else model_qwen35.FREQ_BASE
comptime FREQ_SCALE = model_qwen35moe.FREQ_SCALE if IS_MOE else model_qwen35.FREQ_SCALE
comptime MSCALE = model_qwen35moe.MSCALE if IS_MOE else model_qwen35.MSCALE
comptime MEGA_ALLOWED = model_qwen35moe.MEGA_ALLOWED if IS_MOE else model_qwen35.MEGA_ALLOWED
