import latentos.router as router


def make_input(
    payload: Float64,
    recipe: Float64,
    text: Float64,
    oracle: UInt8,
    local_valid: Bool = False,
) -> router.RouteInput:
    var input = router.RouteInput()
    input.local_valid = local_valid
    input.payload_valid = True
    input.recipe_valid = True
    input.text_valid = True
    input.oracle_route = oracle
    input.terms.payload_transfer_s = payload
    input.terms.payload_restore_s = 0.1
    input.terms.recipe_transfer_s = 0.01
    input.terms.recipe_prefill_s = recipe
    input.terms.text_transfer_s = 0.02
    input.terms.text_prefill_s = text
    input.terms.suffix_prefill_s = 0.01
    input.terms.first_decode_s = 0.01
    return input^


def main() raises:
    var cases = List[router.RouteInput]()
    cases.append(make_input(1.0, 4.0, 8.0, router.ROUTE_PAYLOAD))
    cases.append(make_input(5.0, 1.0, 8.0, router.ROUTE_RECIPE))
    cases.append(make_input(9.0, 8.0, 1.0, router.ROUTE_TEXT))
    cases.append(make_input(2.0, 2.05, 2.1, router.ROUTE_PAYLOAD))
    cases.append(make_input(2.0, 2.05, 2.1, router.ROUTE_RECIPE))
    var wrong = 0
    var regret = 0.0
    for i in range(len(cases)):
        var decision = router.choose_route(cases[i])
        print("route=" + router.route_name(decision.route) + " best=" + router.route_name(decision.best_route) + " margin=" + String(decision.margin) + " wrong=" + String(decision.wrong))
        if decision.wrong:
            wrong += 1
            regret += decision.regret_s
    if wrong != 0:
        raise Error("router decisive wrong choices: " + String(wrong))
    var local = make_input(100.0, 200.0, 300.0, router.ROUTE_LOCAL, True)
    var local_decision = router.choose_route(local)
    if local_decision.route != router.ROUTE_LOCAL or local_decision.margin != 1.0:
        raise Error("local handle preference")
    var unavailable = router.RouteInput()
    var unavailable_decision = router.choose_route(unavailable)
    if unavailable_decision.route != router.ROUTE_NONE or unavailable_decision.decisive:
        raise Error("router unavailable route")
    print("router cases=" + String(len(cases)) + " wrong=" + String(wrong) + " regret=" + String(regret))
    print("ROUTER_FIXTURE_PASS")
