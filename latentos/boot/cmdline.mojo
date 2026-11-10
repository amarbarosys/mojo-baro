# cmdline.mojo: /proc/cmdline tokenizer for the S0 boot shadow.
# Read-only: one open("r") on /proc/cmdline, nothing else.

def read_cmdline() raises -> String:
    with open("/proc/cmdline", "r") as f:
        return String(String(f.read()).strip())


def split_ws(s: String) -> List[String]:
    var out = List[String]()
    var b = s.as_bytes()
    var n = len(b)
    var start = -1
    for i in range(n):
        if b[i] == 32:
            if start >= 0:
                out.append(String(s[byte=start:i]))
                start = -1
        elif start < 0:
            start = i
    if start >= 0:
        out.append(String(s[byte=start:n]))
    return out^


def cmdline_get(cmdline: String, key: String) -> String:
    """Returns the value of `key=value` in the cmdline, "1" for a bare flag
    that matches `key` exactly, or "" if `key` is absent."""
    var prefix = key + "="
    for tok in split_ws(cmdline):
        if tok == key:
            return "1"
        if tok.byte_length() > prefix.byte_length() and String(tok[byte=0:prefix.byte_length()]) == prefix:
            return String(tok[byte=prefix.byte_length():tok.byte_length()])
    return ""
