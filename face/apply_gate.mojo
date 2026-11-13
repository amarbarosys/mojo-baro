"""Apply on a real engine: the face's own apply_action with the face's own restart_cmd and stop script.
usage: .work/apply_gate MODEL.gguf PORT TMAX   (a front started by face/apply_gate.sh owns .work/gui-front.pid)
Prints the apply_action message; exit 1 unless it starts with "front: ok"."""
from std.collections import Dict
from std.sys import argv
from apply import restart_cmd, apply_action


def main() raises:
    var a = argv()
    var model = String(a[1])
    var port = Int(String(a[2]))
    var staged = Dict[String, String]()
    staged["BARO_TMAX"] = String(a[3])
    var rk = List[String]()
    rk.append("BARO_TMAX")
    var start = "setsid " + restart_cmd(model, port, staged, rk) + " > .work/gui-front.log 2>&1 & echo $! > .work/gui-front.pid"
    print("start:", start)
    var msg = apply_action("bash face/stop-engine.sh", start, "127.0.0.1", port, 180, 0)
    print("apply:", msg)
    if not msg.startswith("front: ok"):
        raise Error("apply did not reach a healthy front")
