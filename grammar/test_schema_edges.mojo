from grammar.automaton import Automaton, MatcherState, resolve_epsilon, step_byte
from grammar.json_value import parse_json_bytes
from grammar.json_schema import compile_root_schema, str_bytes


def compile(schema: String) raises -> Automaton:
    var doc = parse_json_bytes(str_bytes(schema))
    var a = Automaton()
    _ = compile_root_schema(a, doc)
    return a^


def accepts(a: Automaton, text: String) -> Bool:
    var ms = MatcherState(a.start_rule)
    resolve_epsilon(a, ms)
    var b = text.as_bytes()
    for i in range(len(b)):
        if not step_byte(a, ms, b[i]):
            return False
        resolve_epsilon(a, ms)
    return ms.terminated


def check(name: String, ok: Bool, mut fails: Int):
    if not ok:
        print("FAIL", name)
        fails += 1


def case_surrogate(mut fails: Int) raises:
    # a surrogate pair escape is one non-BMP character (Python json.dumps writes it so)
    var a = compile('{"enum":["\\ud83d\\ude00"]}')
    check("enum surrogate pair escape accepted as the character", accepts(a, '"\U0001F600"'), fails)


def case_control(mut fails: Int) raises:
    # a control byte in an enum literal comes out escaped, never raw
    var a = compile('{"enum":["a\\u0001b","c\\bd"]}')
    check("enum control char accepted escaped", accepts(a, '"a\\u0001b"'), fails)
    check("enum control char rejected raw", not accepts(a, String('"a') + chr(1) + 'b"'), fails)
    check("enum backspace accepted escaped", accepts(a, '"c\\bd"'), fails)


def case_min_items(mut fails: Int) raises:
    # minItems above the default cap of 10 compiles and enforces the minimum
    var a = compile('{"type":"array","items":{"type":"boolean"},"minItems":12}')
    var twelve = String("[") + String(",").join(List[String](length=12, fill="true")) + "]"
    check("minItems 12 without maxItems", accepts(a, twelve), fails)


def main() raises:
    var fails = 0
    for i in range(3):
        try:
            if i == 0:
                case_surrogate(fails)
            elif i == 1:
                case_control(fails)
            else:
                case_min_items(fails)
        except e:
            print("FAIL case", i, "raised:", e)
            fails += 1
    if fails > 0:
        raise Error("FAIL test_schema_edges: " + String(fails) + " checks")
    print("PASS")
