#!/usr/bin/env python3
"""Print the source files a BARO gguf must embed for the split engine layout:
serve/window.mojo, serve/registry.mojo and the transitive closure of the kernel
modules they import. serve/engine.mojo (pack load, stopwatch, prints) is NOT
embedded -- tools/gguf-closure.sh and tools/loop-gate.sh take it from git at the
gguf's own commit (exchange/scorer-integrity-report.md, P-A, 2026-09-08).

    tools/gguf-embed.py SRC.gguf DST.gguf $(tools/embed-files.py)
    tools/embed-files.py --arch spark   # serve/spark.mojo closure (spark.mojo itself is the
                                        # harness, taken from git); uregex/minja/latentos come
                                        # in whole, as a package glob (vendored at repo root
                                        # since M1, briefs/2026-09-15-wiring-lane.md, and
                                        # briefs/2026-09-15-vendor-latentos.md for latentos --
                                        # no longer external, but still not a single
                                        # kernels/X.mojo or serve/X.mojo candidate, so EXT is
                                        # still how the closure walker pulls a whole package
                                        # directory in)
"""
import re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXT = {"uregex": ROOT / "uregex", "minja": ROOT / "minja", "latentos": ROOT / "latentos", "grammar": ROOT / "grammar"}
ARCH = {"qwythos": (["serve/window.mojo", "serve/registry.mojo"], "serve/engine.mojo"),
        # tools/gguf_reader.mojo: serve/spark.mojo imports it (07fb0dd) and the walk only follows kernels/ and serve/;
        # without it a spark bake served (the pack tool's closure carried it) but never verified (2026-10-12)
        "spark": (["serve/spark.mojo", "tools/gguf_reader.mojo"], "serve/spark.mojo"),
        # The W2 kernel-parity closure, kept for the expert-kernel gate.
        "qwen35moe-kernels": (["kernels/test_moe_block.mojo"], "kernels/test_moe_block.mojo"),
        # The real engine closure for the MoE profile: same roots as qwythos,
        # so the walk picks up kernels/moe.mojo, the three serve/model*.mojo
        # profiles AND kernels/mega.mojo. The MoE profile does not run the
        # megakernel (MEGA_ALLOWED is False for qwen35moe), but registry.mojo
        # instantiates it unconditionally, so it is part of the closure and the
        # file does not compile without it.
        # qwen4exp (p8): serve/q4x.mojo holds main, so like spark it is the
        # harness, taken from git; gguf_reader lives in tools/, outside the walk.
        "qwen4exp": (["serve/q4x.mojo", "tools/gguf_reader.mojo"], "serve/q4x.mojo"),
        "qwen35moe": (["serve/window.mojo", "serve/registry.mojo"], "serve/engine.mojo"),
        # NCM sudoku (next-concept model): kernels and main share ncm/ncm.mojo and the gate is token identity, not timing,
        # so the whole file is embedded (no harness from git); gguf_reader lives in tools/, outside the walk.
        "ncm": (["ncm/ncm.mojo", "tools/gguf_reader.mojo"], "")}



def closure(roots):
    seen, todo = [], list(roots)
    while todo:
        f = todo.pop(0)
        if f in seen:
            continue
        seen.append(f)
        text = (ROOT / f).read_text() if isinstance(f, str) else f.read_text()
        for m in re.finditer(r"^from (\w+)[.\w]* import|^import (\w+)", text, re.M):
            mod = m.group(1) or m.group(2)
            for cand in (f"kernels/{mod}.mojo", f"serve/{mod}.mojo"):
                if (ROOT / cand).exists() and cand not in seen:
                    todo.append(cand)
            if mod in EXT:
                for p in sorted(EXT[mod].glob("*.mojo")):
                    if p not in seen and p not in todo:
                        todo.append(p)
    return sorted(x for x in seen if isinstance(x, str)) + sorted(str(x) for x in seen if not isinstance(x, str))


def harness_ext_files(harness):
    """EXT packages (uregex, minja, latentos) the harness itself imports
    directly, e.g. serve/engine.mojo's `from latentos.ipc import ...`. The
    harness's own single-file imports (harness.mojo, prefix.mojo, ...) are
    NOT walked here -- tools/gguf-closure.sh fetches those fresh from git at
    the gguf's commit when it rebuilds; only a whole vendored package, which
    that rebuild has no other way to reconstruct, is pulled in at bake time."""
    if not harness:  # arch with no separate harness (ncm): nothing extra
        return []
    text = (ROOT / harness).read_text()
    out = []
    for m in re.finditer(r"^from (\w+)[.\w]* import|^import (\w+)", text, re.M):
        mod = m.group(1) or m.group(2)
        if mod in EXT:
            for p in sorted(EXT[mod].glob("*.mojo")):
                if str(p) not in out:
                    out.append(str(p))
    return out


if __name__ == "__main__":
    arch = sys.argv[sys.argv.index("--arch") + 1] if "--arch" in sys.argv else "qwythos"
    roots, harness = ARCH[arch]
    files = [f for f in closure(roots) if f != harness]
    for f in harness_ext_files(harness):
        if f not in files:
            files.append(f)
    assert "serve/engine.mojo" not in files
    # serve/spark.mojo does `from profile import ...`; the closure walk above
    # never finds it because it is generated per-model (tools/gen-profile.mojo)
    # and does not exist as a repo file. --profile PATH adds the one this bake
    # needs; gguf-embed.py's src_key() keys it "profile.mojo" regardless of
    # where PATH lives.
    if "--profile" in sys.argv:
        files.append(sys.argv[sys.argv.index("--profile") + 1])
    print("\n".join(files) if "-1" in sys.argv else " ".join(files))
