comptime TransByte: Int = 0
comptime TransCall: Int = 1


struct Bitset(Copyable, Movable):
    var words: List[UInt64]
    var n: Int

    def __init__(out self, n: Int):
        self.n = n
        self.words = []
        var nw = (n + 63) // 64
        for _ in range(nw):
            self.words.append(UInt64(0))

    def set_bit(mut self, i: Int):
        self.words[i >> 6] |= (UInt64(1) << UInt64(i & 63))

    def get_bit(self, i: Int) -> Bool:
        return ((self.words[i >> 6] >> UInt64(i & 63)) & UInt64(1)) == UInt64(1)

    def clear_all(mut self):
        for i in range(len(self.words)):
            self.words[i] = UInt64(0)

    def count(self) -> Int:
        var c = 0
        for i in range(self.n):
            if self.get_bit(i):
                c += 1
        return c


@fieldwise_init
struct Transition(Copyable, Movable):
    var kind: Int
    var lo: UInt8
    var hi: UInt8
    var target: Int32
    var call_rule: Int32


struct AutoState(Copyable, Movable):
    var trans: List[Transition]
    var is_accept: Bool

    def __init__(out self):
        self.trans = []
        self.is_accept = False


struct Rule(Copyable, Movable):
    var states: List[AutoState]
    var start: Int32
    var name: String

    def __init__(out self, name: String):
        self.states = [AutoState()]
        self.start = 0
        self.name = name

    def new_state(mut self) -> Int32:
        self.states.append(AutoState())
        return Int32(len(self.states) - 1)

    def add_byte_trans(mut self, src: Int32, lo: UInt8, hi: UInt8, target: Int32):
        self.states[Int(src)].trans.append(Transition(TransByte, lo, hi, target, -1))

    def add_call_trans(mut self, src: Int32, call_rule: Int32, return_state: Int32):
        self.states[Int(src)].trans.append(Transition(TransCall, 0, 0, return_state, call_rule))

    def set_accept(mut self, s: Int32):
        self.states[Int(s)].is_accept = True


struct Automaton(Copyable, Movable):
    var rules: List[Rule]
    var start_rule: Int32
    var rule_names: List[String]

    def __init__(out self):
        self.rules = []
        self.start_rule = 0
        self.rule_names = []

    def add_rule(mut self, name: String) -> Int32:
        self.rules.append(Rule(name))
        self.rule_names.append(name)
        return Int32(len(self.rules) - 1)


@fieldwise_init
struct Frame(ImplicitlyCopyable, Movable):
    var rule_id: Int32
    var state_id: Int32


struct MatcherState(Copyable, Movable):
    var cur_rule: Int32
    var cur_state: Int32
    var call_stack: List[Frame]
    var terminated: Bool

    def __init__(out self, start_rule: Int32):
        self.cur_rule = start_rule
        self.cur_state = 0
        self.call_stack = []
        self.terminated = False


def resolve_epsilon(automaton: Automaton, mut ms: MatcherState):
    while True:
        ref st = automaton.rules[Int(ms.cur_rule)].states[Int(ms.cur_state)]
        if len(st.trans) == 1 and st.trans[0].kind == TransCall:
            ms.call_stack.append(Frame(ms.cur_rule, st.trans[0].target))
            ms.cur_rule = st.trans[0].call_rule
            ms.cur_state = automaton.rules[Int(ms.cur_rule)].start
            continue
        break
    ms.terminated = check_terminated(automaton, ms)


def check_terminated(automaton: Automaton, ms: MatcherState) -> Bool:
    var rule = ms.cur_rule
    var state = ms.cur_state
    var depth = len(ms.call_stack)
    while True:
        ref st = automaton.rules[Int(rule)].states[Int(state)]
        if not st.is_accept:
            return False
        if depth == 0:
            return True
        var f = ms.call_stack[depth - 1]
        rule = f.rule_id
        state = f.state_id
        depth -= 1


def _try_byte(automaton: Automaton, mut ms: MatcherState, b: UInt8) -> Bool:
    ref st = automaton.rules[Int(ms.cur_rule)].states[Int(ms.cur_state)]
    for i in range(len(st.trans)):
        ref t = st.trans[i]
        if t.kind == TransByte and b >= t.lo and b <= t.hi:
            ms.cur_state = t.target
            return True
    return False


def step_byte(automaton: Automaton, mut ms: MatcherState, b: UInt8) -> Bool:
    if _try_byte(automaton, ms, b):
        return True
    var saved_rule = ms.cur_rule
    var saved_state = ms.cur_state
    var popped: List[Frame] = []
    while True:
        ref st = automaton.rules[Int(ms.cur_rule)].states[Int(ms.cur_state)]
        if st.is_accept and len(ms.call_stack) > 0:
            var f = ms.call_stack.pop()
            popped.append(f)
            ms.cur_rule = f.rule_id
            ms.cur_state = f.state_id
            if _try_byte(automaton, ms, b):
                return True
            continue
        break
    ms.cur_rule = saved_rule
    ms.cur_state = saved_state
    var k = len(popped)
    for i in range(k):
        ms.call_stack.append(popped[k - 1 - i])
    return False


def step_byte_tracked(automaton: Automaton, mut ms: MatcherState, b: UInt8, mut popped_out: List[Frame]) -> Bool:
    if _try_byte(automaton, ms, b):
        return True
    var saved_rule = ms.cur_rule
    var saved_state = ms.cur_state
    var local_start = len(popped_out)
    while True:
        ref st = automaton.rules[Int(ms.cur_rule)].states[Int(ms.cur_state)]
        if st.is_accept and len(ms.call_stack) > 0:
            var f = ms.call_stack.pop()
            popped_out.append(f)
            ms.cur_rule = f.rule_id
            ms.cur_state = f.state_id
            if _try_byte(automaton, ms, b):
                return True
            continue
        break
    ms.cur_rule = saved_rule
    ms.cur_state = saved_state
    var k = len(popped_out)
    for j in range(k - 1, local_start - 1, -1):
        ms.call_stack.append(popped_out[j])
    while len(popped_out) > local_start:
        _ = popped_out.pop()
    return False
