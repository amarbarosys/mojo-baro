# The legal combinations of run.mojo's stage selector (stages=, fix=, every=, rule=), checked once before the loop.
# They used to fall out of five integers: rule without every held stage 0 for the whole run, fix plus rule left fix dead,
# rule on a non-latent arm forced stage 0, and stages= replaced the steps argument silently.


def check_stage_args(arm: String, steps: Int, rule: Float64, every: Int, fix: Int, plan: List[Int]) raises:
    if rule > 0.0 and every == 0:
        raise Error("FAIL arg: rule needs every>0 (stage 0 would be held for the whole run)")
    if rule > 0.0 and fix >= 0:
        raise Error("FAIL arg: fix and rule are exclusive (rule would win and fix is dead)")
    if rule > 0.0 and arm != "latent":
        raise Error("FAIL arg: rule needs the latent arm (it forces stage 0 on " + arm + ")")
    if len(plan) > 0:
        var planned = 0
        for x in plan:
            planned += x
        if planned != steps:
            raise Error("FAIL arg: stages= sums to " + String(planned) + " steps but the steps argument is " + String(steps))
