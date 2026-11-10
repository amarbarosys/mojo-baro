# Copyright 2026 amarbaro.org. Licensed under the Apache License, Version 2.0.
from std.sys import argv, exit
from minja import render_chat


def main() raises:
    var args = argv()
    if len(args) < 3:
        print("usage: minja-cli TEMPLATE.jinja CASE.json [bos eos pad]   (CASE = {messages, ...kwargs})")
        return
    var tpl: String
    with open(String(args[1]), "r") as f:
        tpl = f.read()
    var case_json: String
    with open(String(args[2]), "r") as f:
        case_json = f.read()
    var bos = String(args[3]) if len(args) > 3 else String("")
    var eos = String(args[4]) if len(args) > 4 else String("")
    var pad = String(args[5]) if len(args) > 5 else String("")
    try:
        print(render_chat(tpl, case_json, bos, eos, pad), end="")
    except e:
        print("MINJA_ERROR: " + String(e))
        exit(2)
