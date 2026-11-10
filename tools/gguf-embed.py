#!/usr/bin/env python3
"""Embed baro kernel sources into a GGUF's metadata (self-describing model).

Writes a NEW file — never touches the source GGUF. Adds string KV pairs:
  baro.kernel.arch, baro.kernel.commit, baro.kernel.files (comma list),
  baro.kernel.src.<basename> = full source text
GGUF readers ignore unknown keys; tensor offsets are relative to the data
region so only the header padding changes.

Usage: tools/gguf-embed.py SRC.gguf DST.gguf $(tools/embed-files.py)
       (split layout, 2026-09-08: window.mojo + registry.mojo + kernel closure; never serve/engine.mojo)

Optional, for B1 (`tools/baro`), all in the separate baro.run.* namespace so
the closure walk above stays harness-free:
  --run-harness=serve/engine.mojo  the RUN-mode harness plus its sha256
  --run-prompt=FILE                token ids the closure decodes
  --run-ref=FILE                   the ids it must produce (GENERATED: line or bare)
  --run-env='BARO_MEGA=0 ...'      the env this model needs to run at all
  --run-pack-flags='--q4 ...'      how tools/engine-pack.py builds this model's pack
  --run-pack-tool=tools/engine-pack.mojo  the pack builder itself, so run mode needs no checkout
       tools/gguf-embed.py SRC.gguf DST.gguf --gen-only=OUT   (bench/family-gate.sh OUT after a PASS)
  writes only baro.gen.* (recipe log, profile) and baro.tune.* (flags, table, arch, spark commit), replacing
  earlier ones; every other key, baro.kernel.* / baro.run.* included, is copied unchanged.
"""
import fcntl
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

ALIGN = 32
BLOCK = 4096
FICLONERANGE = 0x4020940D
PAD_KEY = "baro.pad"


def w_str(out, s):
    b = s.encode("utf-8")
    out.write(struct.pack("<Q", len(b)))
    out.write(b)


def src_key(k):
    """Kernel/engine sources are keyed by basename (engine imports them flat);
    anything else keeps its repo-relative path so the closure can rebuild it."""
    root = Path(__file__).resolve().parent.parent
    if k.name == "profile.mojo":
        # serve/spark.mojo does `from profile import ...` flatly. A profile is
        # generated per-model (tools/gen-profile.mojo) and lives outside the
        # repo by construction (it is a function of the gguf, not source), so
        # it never has a kernels/ or serve/ prefix to key by; key it as itself.
        return "profile.mojo"
    if not k.resolve().is_relative_to(root):
        return f"{k.resolve().parent.name}/{k.name}"  # external package: <pkg>/<file>
    rel = k.resolve().relative_to(root).as_posix()
    return k.name if rel.startswith(("kernels/", "serve/")) else rel



def mojo_closure(entry):
    """Repo modules a Mojo tool imports, transitively (serve/, tools/, kernels/); std/max/layout are not files here."""
    root = Path(__file__).resolve().parent.parent
    seen, todo, out = {entry.resolve()}, [entry], []
    while todo:
        for line in todo.pop().read_text().splitlines():
            m = re.match(r"\s*(?:from|import)\s+(\w+)", line)
            if not m:
                continue
            for d in ("serve", "tools", "kernels"):
                c = (root / d / (m.group(1) + ".mojo")).resolve()
                if c.exists() and c not in seen:
                    seen.add(c)
                    todo.append(c)
                    out.append(c)
                    break
    return out

def rewrite(src, dst, new_kv, drop=("baro.kernel.", "baro.hw.", "baro.run.", "baro.data.", PAD_KEY)):
    """Write dst = src with every KV whose key starts with one of `drop`
    replaced by `new_kv` (a list of (key, string value) pairs), tensor infos
    and tensor data copied byte for byte.

    Factored out of main() so tools/gguf-receipt.py can extend one KV without
    a second copy of the container format. Returns (added, dropped).

    Aligned bake (2026-09-16): a `baro.pad` KV is sized so dst's data region
    starts at the same offset mod 4096 as src's, and the tensor data is then
    cloned with FICLONERANGE instead of copied. On btrfs the bake shares every
    tensor extent with its source and costs only its header on disk; before
    this, every bake landed at a different in-block offset and no two copies
    could share a block. Any filesystem without clone support falls back to a
    plain copy with the same bytes.
    """
    f = open(src, "rb")
    magic, version = struct.unpack("<4sI", f.read(8))
    assert magic == b"GGUF" and version == 3
    n_tensors, n_kv = struct.unpack("<QQ", f.read(16))

    kv_start = f.tell()
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location("ge", Path(__file__).parent / "gguf-extract.py")
    ge = module_from_spec(spec)
    spec.loader.exec_module(ge)
    _, infos, data_start, kv = ge.parse(src)
    # baro.data.sha256: the tensor data region, which the closure gate never
    # reads. Same bytes in src and dst (only the header moves), so hashing src
    # here is hashing dst. Rewritten only when the caller drops the old one.
    if "baro.data." in drop:
        with open(src, "rb") as h:
            h.seek(data_start)
            new_kv = list(new_kv) + [("baro.data.sha256", hashlib.file_digest(h, "sha256").hexdigest())]
    f.seek(kv_start)
    keep = []
    for _ in range(n_kv):
        a = f.tell()
        key = ge.read_str(f)
        key = key.decode("utf-8") if isinstance(key, bytes) else key
        (vtype,) = struct.unpack("<I", f.read(4))
        ge.read_value(f, vtype, want=False)
        if not any(key.startswith(p) for p in drop):
            keep.append((a, f.tell()))
    kv_end = f.tell()
    for _ in range(n_tensors):
        ge.read_str(f)
        (nd,) = struct.unpack("<I", f.read(4))
        f.seek(8 * nd + 4 + 8, 1)
    info_end = f.tell()

    out = open(dst, "wb")
    out.write(struct.pack("<4sI", b"GGUF", 3))
    out.write(struct.pack("<QQ", n_tensors, len(keep) + len(new_kv) + 1))
    for a, b in keep:
        f.seek(a)
        out.write(f.read(b - a))
    for key, val in new_kv:
        w_str(out, key)
        out.write(struct.pack("<I", 8))
        w_str(out, val)
    f.seek(kv_end)
    infos_blob = f.read(info_end - kv_end)
    fixed = 8 + len(PAD_KEY) + 4 + 8
    n_pad = (data_start - (out.tell() + fixed + len(infos_blob))) % BLOCK
    w_str(out, PAD_KEY)
    out.write(struct.pack("<I", 8))
    w_str(out, " " * n_pad)
    out.write(infos_blob)
    assert out.tell() % ALIGN == 0 and out.tell() % BLOCK == data_start % BLOCK

    size = os.fstat(f.fileno()).st_size
    head = min((BLOCK - data_start % BLOCK) % BLOCK, size - data_start)
    f.seek(data_start)
    out.write(f.read(head))
    out.flush()
    s_off, d_off = data_start + head, out.tell()
    try:
        fcntl.ioctl(out.fileno(), FICLONERANGE, struct.pack("<qQQQ", f.fileno(), s_off, 0, d_off))
        out.seek(0, 2)
    except OSError:
        f.seek(s_off)
        while chunk := f.read(1 << 24):
            out.write(chunk)
    out.close()
    return len(new_kv), n_kv - len(keep)


def main():
    extra = []
    args = []
    run_harness = None
    run_prompt = None
    run_ref = None
    run_pack_flags = None
    run_pack_tool = None
    run_env = None
    for a in sys.argv[1:]:
        if a.startswith("--kv="):
            k, v = a[5:].split("=", 1)
            assert k.startswith("baro.") and (not k.startswith("baro.kernel.") or k == "baro.kernel.model"), f"--kv key must be baro.<ns>.<name> (or baro.kernel.model): {k}"
            extra.append((k, v))
        elif a.startswith("--run-harness="):
            run_harness = Path(a.split("=", 1)[1])
        elif a.startswith("--run-prompt="):
            run_prompt = Path(a.split("=", 1)[1])
        elif a.startswith("--run-ref="):
            run_ref = Path(a.split("=", 1)[1])
        elif a.startswith("--run-pack-flags="):
            run_pack_flags = a.split("=", 1)[1]
        elif a.startswith("--run-env="):
            run_env = a.split("=", 1)[1]
        elif a.startswith("--run-pack-tool="):
            run_pack_tool = Path(a.split("=", 1)[1])
        else:
            args.append(a)
    src, dst = Path(args[0]), Path(args[1])
    gen = next((Path(a.split("=", 1)[1]) for a in args[2:] if a.startswith("--gen-only=")), None)
    if gen is not None:
        assert src.exists() and not dst.exists(), "dst must not exist"
        assert "PASS family-gate" in (gen / "SUMMARY.txt").read_text(), f"{gen}/SUMMARY.txt has no PASS family-gate"
        root = Path(__file__).resolve().parent.parent
        sc = subprocess.run(["git", "-C", str(root), "log", "-1", "--format=%h", "--", "serve/spark.mojo", "kernels"],
                            capture_output=True, text=True).stdout.strip()
        kv = [("baro.gen.recipe", (gen / "recipe.log").read_text()), ("baro.gen.profile", (gen / "prof/profile.mojo").read_text()),
              ("baro.tune.flags", (gen / "tune/flags").read_text().strip()), ("baro.tune.table", (gen / "tune/summary.txt").read_text()),
              ("baro.tune.arch", "gfx1100"), ("baro.tune.commit", sc), ("baro.gen.gate", (gen / "SUMMARY.txt").read_text())]
        added, dropped = rewrite(src, dst, kv, drop=("baro.gen.", "baro.tune.", PAD_KEY))
        print(f"wrote {dst} (+{added} kv, dropped {dropped} earlier baro.gen.*/baro.tune.* kv)")
        return
    kfiles = [Path(p) for p in args[2:]]
    assert src.exists() and not dst.exists(), "dst must not exist"

    commit = subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parent.parent),
         "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True).stdout.strip()
    new_kv = [("baro.kernel.arch", "gfx1100"),
              ("baro.kernel.commit", commit),
              ("baro.kernel.files", ",".join(src_key(k) for k in kfiles))]
    for pkg in sorted({k.resolve().parent for k in kfiles if k.name != "profile.mojo" and not k.resolve().is_relative_to(Path(__file__).resolve().parent.parent)}):
        c = subprocess.run(["git", "-C", str(pkg), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(pkg), "status", "--short"], capture_output=True, text=True).stdout.strip()
        assert not dirty, f"{pkg} dirty"
        new_kv.append((f"baro.kernel.ext.{pkg.name}.commit", c))
    if os.environ.get("BARO_KERNEL_PARENT"):
        new_kv.append(("baro.kernel.parent", os.environ["BARO_KERNEL_PARENT"]))
    # --kv=baro.hw.<name>=<value> (repeatable): the receipts a verifier on another
    # card compares against (card, driver, power cap, the 20-prompt number and the
    # protocol that produced it). Earlier baro.hw.* keys are replaced like the
    # kernel set, so a re-bake never carries two scoreboards.
    new_kv.extend(extra)
    for k in kfiles:
        new_kv.append((f"baro.kernel.src.{src_key(k)}", k.read_text()))

    # --run-harness=serve/engine.mojo: the RUN-mode harness (B1). The closure
    # walk stays harness-free -- baro.kernel.src.* never carries engine.mojo,
    # because a file that carries its own stopwatch cannot be timed by it
    # (exchange/scorer-integrity-report.md P-A). This is a separate namespace
    # for a separate job: `baro run` serves the model with no checkout and
    # labels every number it prints self-reported, while `baro verify` takes
    # the harness from git and first checks this copy is byte-identical to it,
    # which is how a tampered clock is caught.
    if run_harness is not None:
        text = run_harness.read_text()
        new_kv.append((f"baro.run.src.{run_harness.name}", text))
        new_kv.append(("baro.run.harness.sha", hashlib.sha256(text.encode()).hexdigest()))
        new_kv.append(("baro.run.harness.path", "serve/" + run_harness.name))
        # The harness's own serve-module closure (prefix, harness, moe_pack,
        # latent, ...), minus whatever the kernel closure already carries.
        # tools/gguf-closure.sh pulls these from git at the gguf's commit, which
        # verify mode can do and run mode cannot: run mode has no checkout, and
        # without them the build stops at "unable to locate module 'prefix'".
        serve_dir = run_harness.resolve().parent
        have = {src_key(k) for k in kfiles}
        seen, todo = set(), [run_harness]
        while todo:
            f = todo.pop(0)
            for m in re.findall(r"^(?:from|import)\s+([a-z_][a-z0-9_]*)", f.read_text(), re.M):
                # serve/ first, then kernels/: engine.mojo imports kernels/state.mojo flat, and a bake
                # without it stopped at "unable to locate module 'state'" (2026-10-12)
                cand = next((d / f"{m}.mojo" for d in (serve_dir, serve_dir.parent / "kernels") if (d / f"{m}.mojo").exists()),
                            serve_dir / f"{m}.mojo")
                if m in seen or not cand.exists() or cand.name in have:
                    continue
                seen.add(m)
                todo.append(cand)
                new_kv.append((f"baro.run.src.{cand.name}", cand.read_text()))
        if seen:
            print(f"run closure: {len(seen)} extra serve module(s): {' '.join(sorted(seen))}")

    # The closure used to reach for .work/ paths that the file never carried
    # (the qwen35moe branch defaulted BARO_PROMPT to .work/moe-w3/one.tokens,
    # which no longer exists on this box, so tools/gguf-verify.sh exited 1 on
    # the MoE bake). A file that verifies itself carries its own prompt and its
    # own reference ids.
    if run_prompt is not None:
        new_kv.append(("baro.run.prompt.tokens", " ".join(run_prompt.read_text().split())))
    if run_ref is not None:
        ids = run_ref.read_text().replace("GENERATED:", " ").split()
        new_kv.append(("baro.run.ref.tokens", " ".join(ids)))
    if run_env is not None:
        # How this model must be run, in the file: the MoE profile aborts on
        # BARO_MEGA=1, which is the engine's own default, so a file that does
        # not carry its env cannot serve itself.
        new_kv.append(("baro.run.env", run_env))
    if run_pack_flags is not None:
        new_kv.append(("baro.run.pack.flags", run_pack_flags))
    if run_pack_tool is not None:
        # The pack builder too, so run mode needs no checkout to turn the file's
        # own weights into the engine's pack. It loads gguf-extract.py from its
        # own directory, so that rides along: without it the pack step dies on
        # FileNotFoundError for a sibling that only exists in a checkout.
        # A Mojo pack tool (tools/spark-pack.mojo) carries its local import closure instead, named in
        # baro.run.pack.deps so tools/pack-run.sh can build it with -I next to its sources.
        if run_pack_tool.suffix == ".mojo":
            deps = mojo_closure(run_pack_tool)
            new_kv.append(("baro.run.pack.deps", ",".join(d.name for d in deps)))
            tools = [run_pack_tool] + deps
        else:
            tools = [run_pack_tool, run_pack_tool.parent / "gguf-extract.py"]
        for tool in tools:
            new_kv.append((f"baro.run.src.{tool.name}", tool.read_text()))
        # Named separately from baro.run.src.<name> so a reader (tools/gguf-closure.sh's
        # spark branch) knows which embedded file is the pack builder without
        # guessing between engine-pack.py (llama/qwen2/granite dense path) and
        # spark-pack.py (spark2_5's per-head attention gate needs the other).
        new_kv.append(("baro.run.pack.tool", run_pack_tool.name))

    # Existing KVs are copied except any earlier baro.kernel.*/baro.hw.*/baro.run.*
    # set: re-embedding from a BARO file replaces its sources and its scoreboard,
    # it never appends a second one.
    added, dropped = rewrite(src, dst, new_kv)
    print(f"wrote {write_challenge(dst)}")
    print(f"wrote {dst} (+{added} kv, dropped {dropped} earlier baro.kernel.*/baro.hw.*/baro.run.* kv)")


def write_challenge(dst):
    """The reference lives OUTSIDE the file too: a file carrying its own ref can
    never fail its own gate. tools/gguf-check.py compares the two."""
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location("ge", Path(__file__).parent / "gguf-extract.py")
    ge = module_from_spec(spec)
    spec.loader.exec_module(ge)
    _, _, _, kv = ge.parse(str(dst))
    ch = {k: kv[k] for k in ("baro.kernel.commit", "baro.data.sha256", "baro.run.harness.sha",
                             "baro.run.prompt.tokens", "baro.run.ref.tokens") if k in kv}
    out = Path(str(dst) + ".challenge.json")
    out.write_text(json.dumps(ch, indent=1) + "\n")
    return out


if __name__ == "__main__":
    main()
