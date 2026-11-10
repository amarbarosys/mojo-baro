"""MTP receipt of a gguf: tools/mtp-receipt MODEL.gguf prints one JSON line, or nothing when the file has no nextn tensors.
Names the MTP layer, its tensors with type and bytes, and an FNV-1a 64 of their bytes. bake.sh stores the line as baro.mtp."""
from std.sys import argv
from gguf_reader import GGUFModel


def main() raises:
    var a = argv()
    if len(a) != 2:
        raise Error("FAIL usage: mtp-receipt MODEL.gguf")
    var path = String(a[1])
    var m = GGUFModel(path)
    var layer = String("")
    for n in m.tensor_names:
        if n.find(".nextn.") >= 0:
            layer = String(n.split(".")[1])
    if layer == "":
        return
    var fsize = 0
    with open(path, "r") as f:
        fsize = Int(f.seek(0, 2))
    var ends = List[Int]()
    for n in m.tensor_names:
        ends.append(m.tensor_off[n])
    var pre = "blk." + layer + "."
    var s = String('{"layer":') + layer + ',"tensors":['
    var total = 0
    var count = 0
    var h = UInt64(0xcbf29ce484222325)
    var f = open(path, "r")
    for n in m.tensor_names:
        if not n.startswith(pre):
            continue
        var off = m.tensor_off[n]
        var end = fsize - m.data_start
        for e in ends:
            if e > off and e < end:
                end = e
        _ = f.seek(m.data_start + off)
        var b = f.read_bytes(end - off)
        for i in range(len(b)):
            h = (h ^ UInt64(b[i])) * 0x100000001b3
        s += ("," if count > 0 else "") + '["' + n + '",' + String(m.tensor_type[n]) + "," + String(end - off) + "]"
        total += end - off
        count += 1
    f.close()
    print(s + '],"count":' + String(count) + ',"bytes":' + String(total) + ',"fnv1a64":"' + hex(h) + '"}')
