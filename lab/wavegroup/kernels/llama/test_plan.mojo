# M3 planner gate: Llama decode intermediates (f32, one layer's steps; layers reuse them) planned into one arena.
# Prints peak vs the sum; fails if two buffers live at the same step share bytes.
from plan import Buf, plan
from model import H, FFN, VOCAB, QF, KV

def main() raises:
    comptime F = 4
    var b: List[Buf] = [
        Buf("x", H * F, 0, 12, 0),             # residual, lives through the step
        Buf("xn", H * F, 1, 2, 0),             # attn rmsnorm
        Buf("qkv", (QF + 2 * KV) * F, 2, 4, 0),
        Buf("att", H * F, 4, 5, 0),            # o gemv adds into x
        Buf("xn2", H * F, 6, 7, 0),            # ffn rmsnorm
        Buf("gu", 2 * FFN * F, 7, 8, 0),       # gate+up, one gemv
        Buf("act", FFN * F, 8, 9, 0),          # swiglu; down gemv adds into x
        Buf("xf", H * F, 10, 11, 0),           # final norm
        Buf("logits", VOCAB * F, 11, 12, 0),   # lm head, argmax reads it
    ]
    var total = 0
    for x in b:
        total += x.bytes
    var peak = plan(b)
    for i in range(len(b)):
        print(b[i].name, "off", b[i].off, "bytes", b[i].bytes)
        for j in range(i):
            var live = b[i].first <= b[j].last and b[j].first <= b[i].last
            if live and b[i].off < b[j].off + b[j].bytes and b[j].off < b[i].off + b[i].bytes:
                raise Error("FAIL plan: " + b[i].name + " overlaps " + b[j].name)
    print("arena_peak", peak, "sum", total, "saved", 1.0 - Float64(peak) / Float64(total))
    print("PASS plan")
