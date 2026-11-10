# Audit row C16: each illegal stage-selector combination fails by name; the legal ones pass.
from stagecheck import check_stage_args


def fails(want: String, arm: String, steps: Int, rule: Float64, every: Int, fix: Int, plan: List[Int]) -> Bool:
    try:
        check_stage_args(arm, steps, rule, every, fix, plan)
    except e:
        return want in String(e)
    return False


def main() raises:
    var none = List[Int]()
    var two_four = List[Int]()
    two_four.append(2)
    two_four.append(4)
    var six = List[Int]()
    six.append(6)
    var bad = 0
    if not fails("rule needs every>0", "latent", 10, 0.5, 0, -1, none):
        print("FAIL rule without every")
        bad += 1
    if not fails("fix and rule are exclusive", "latent", 10, 0.5, 5, 1, none):
        print("FAIL fix with rule")
        bad += 1
    if not fails("rule needs the latent arm", "text", 10, 0.5, 5, -1, none):
        print("FAIL rule on a non-latent arm")
        bad += 1
    if not fails("sums to 6 steps but the steps argument is 10", "latent", 10, 0.0, 0, -1, two_four):
        print("FAIL stages= against steps")
        bad += 1
    check_stage_args("latent", 10, 0.5, 5, -1, none)  # legal: rule with every
    check_stage_args("latent", 6, 0.0, 0, -1, two_four)  # legal: stages= matches steps
    check_stage_args("text", 6, 0.0, 0, 3, six)  # legal: fix on any arm, plan sums to steps
    if bad > 0:
        raise Error("test_stagecheck: " + String(bad) + " failed")
    print("test_stagecheck: PASS (C16)")
