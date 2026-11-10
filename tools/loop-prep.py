#!/usr/bin/env python3
"""Lay out a loop iteration from git instead of a gguf: .work/loop/<ITER>/{src/,FILES,meta.json}, the same
files and keys tools/gguf-embed.py would embed for --arch qwythos at COMMIT. tools/loop-gate.sh then runs on it
unchanged. For proposers that are not the served model (the positive control, bench/loop-protocol.md
iteration 008), so no 6 GB bake is needed per kernel commit.

Usage: tools/loop-prep.py ITER [COMMIT]   (default HEAD; the sources are read from git, never the work tree)
"""
import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def key(path):
    return path.split("/")[-1] if path.startswith(("kernels/", "serve/")) else path


def main():
    if len(sys.argv) not in (2, 3):
        raise SystemExit(__doc__)
    it = sys.argv[1]
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", sys.argv[2] if len(sys.argv) == 3 else "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    files = subprocess.run([sys.executable, str(ROOT / "tools/embed-files.py"), "--arch", "qwythos"],
                           capture_output=True, text=True, check=True).stdout.split()
    rel = [Path(f).resolve().relative_to(ROOT).as_posix() for f in files]
    out = ROOT / ".work/loop" / it
    if (out / "src").exists():
        raise SystemExit(f"FAIL loop-prep: {out}/src exists; pick a new ITER")
    keys = []
    for r in rel:
        k = key(r)
        dst = out / "src" / k
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{r}"],
                                      capture_output=True, text=True, check=True).stdout)
        keys.append(k)
    (out / "FILES").write_text("\n".join(keys) + "\n")
    (out / "meta.json").write_text(json.dumps({"baro.kernel.commit": commit, "baro.kernel.arch": "qwythos",
                                               "baro.kernel.files": ",".join(keys)}, indent=1))
    print(f"loop-prep: {out} commit {commit}, {len(keys)} files")


if __name__ == "__main__":
    main()
