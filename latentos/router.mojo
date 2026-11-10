import latentos.contract as contract


comptime ROUTE_NONE: UInt8 = 0
comptime ROUTE_LOCAL: UInt8 = contract.TRANSFER_LOCAL_HANDLE
comptime ROUTE_PAYLOAD: UInt8 = contract.TRANSFER_PAYLOAD
comptime ROUTE_RECIPE: UInt8 = contract.TRANSFER_RECIPE
comptime ROUTE_TEXT: UInt8 = contract.TRANSFER_TEXT
comptime DECISIVE_MARGIN: Float64 = 0.20


struct RouteTerms(Copyable, Movable):
    var authorization_s: Float64
    var queue_s: Float64
    var local_handle_s: Float64
    var payload_transfer_s: Float64
    var payload_restore_s: Float64
    var recipe_transfer_s: Float64
    var recipe_prefill_s: Float64
    var text_transfer_s: Float64
    var text_prefill_s: Float64
    var suffix_prefill_s: Float64
    var first_decode_s: Float64

    def __init__(out self):
        self.authorization_s = 0.0
        self.queue_s = 0.0
        self.local_handle_s = 0.0
        self.payload_transfer_s = 0.0
        self.payload_restore_s = 0.0
        self.recipe_transfer_s = 0.0
        self.recipe_prefill_s = 0.0
        self.text_transfer_s = 0.0
        self.text_prefill_s = 0.0
        self.suffix_prefill_s = 0.0
        self.first_decode_s = 0.0

    def cost(self, route: UInt8) -> Float64:
        if route == ROUTE_LOCAL:
            return self.authorization_s + self.queue_s + self.local_handle_s + self.suffix_prefill_s + self.first_decode_s
        if route == ROUTE_PAYLOAD:
            return self.authorization_s + self.queue_s + self.payload_transfer_s + self.payload_restore_s + self.suffix_prefill_s + self.first_decode_s
        if route == ROUTE_RECIPE:
            return self.authorization_s + self.queue_s + self.recipe_transfer_s + self.recipe_prefill_s + self.suffix_prefill_s + self.first_decode_s
        if route == ROUTE_TEXT:
            return self.authorization_s + self.queue_s + self.text_transfer_s + self.text_prefill_s + self.suffix_prefill_s + self.first_decode_s
        return Float64(1e30)


struct RouteInput(Copyable, Movable):
    var terms: RouteTerms
    var local_valid: Bool
    var payload_valid: Bool
    var recipe_valid: Bool
    var text_valid: Bool
    var oracle_route: UInt8

    def __init__(out self):
        self.terms = RouteTerms()
        self.local_valid = False
        self.payload_valid = False
        self.recipe_valid = False
        self.text_valid = False
        self.oracle_route = ROUTE_NONE


struct RouteDecision(Copyable, Movable):
    var route: UInt8
    var best_route: UInt8
    var margin: Float64
    var decisive: Bool
    var safe_fallback: Bool
    var wrong: Bool
    var regret_s: Float64

    def __init__(out self):
        self.route = ROUTE_NONE
        self.best_route = ROUTE_NONE
        self.margin = 0.0
        self.decisive = False
        self.safe_fallback = False
        self.wrong = False
        self.regret_s = 0.0


def route_name(route: UInt8) -> String:
    if route == ROUTE_LOCAL:
        return "local_handle"
    if route == ROUTE_PAYLOAD:
        return "payload"
    if route == ROUTE_RECIPE:
        return "recipe"
    if route == ROUTE_TEXT:
        return "text"
    return "none"


def choose_route(input: RouteInput) -> RouteDecision:
    var out = RouteDecision()
    if input.local_valid:
        out.route = ROUTE_LOCAL
        out.best_route = ROUTE_LOCAL
        out.margin = 1.0
        out.decisive = True
        out.wrong = input.oracle_route != ROUTE_NONE and input.oracle_route != ROUTE_LOCAL
        if out.wrong:
            out.regret_s = input.terms.cost(ROUTE_LOCAL) - input.terms.cost(input.oracle_route)
        return out^

    var best = ROUTE_NONE
    var second = ROUTE_NONE
    var best_cost = Float64(1e30)
    var second_cost = Float64(1e30)
    if input.payload_valid:
        best = ROUTE_PAYLOAD
        best_cost = input.terms.cost(ROUTE_PAYLOAD)
    if input.recipe_valid:
        var recipe_cost = input.terms.cost(ROUTE_RECIPE)
        if best == ROUTE_NONE:
            best = ROUTE_RECIPE
            best_cost = recipe_cost
        elif recipe_cost < best_cost:
            second = best
            second_cost = best_cost
            best = ROUTE_RECIPE
            best_cost = recipe_cost
        elif second == ROUTE_NONE or recipe_cost < second_cost:
            second = ROUTE_RECIPE
            second_cost = recipe_cost
    if input.text_valid:
        var text_cost = input.terms.cost(ROUTE_TEXT)
        if best == ROUTE_NONE:
            best = ROUTE_TEXT
            best_cost = text_cost
        elif text_cost < best_cost:
            second = best
            second_cost = best_cost
            best = ROUTE_TEXT
            best_cost = text_cost
        elif second == ROUTE_NONE or text_cost < second_cost:
            second = ROUTE_TEXT
            second_cost = text_cost
    if best == ROUTE_NONE:
        out.route = ROUTE_NONE
        out.best_route = ROUTE_NONE
        out.margin = 0.0
        out.decisive = False
        return out^
    if second == ROUTE_NONE:
        out.margin = 1.0
    else:
        out.margin = (second_cost - best_cost) / second_cost
    out.best_route = best
    out.decisive = out.margin >= DECISIVE_MARGIN
    if out.decisive:
        out.route = best
    else:
        out.safe_fallback = True
        if input.recipe_valid:
            out.route = ROUTE_RECIPE
        elif input.text_valid:
            out.route = ROUTE_TEXT
        else:
            out.route = best
    out.wrong = out.decisive and input.oracle_route != ROUTE_NONE and out.route != input.oracle_route
    if out.wrong:
        out.regret_s = input.terms.cost(out.route) - input.terms.cost(input.oracle_route)
    return out^
