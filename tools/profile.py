#!/usr/bin/env python3
"""Named run profiles: profiles/<name>.toml, checked against docs/settings-template.toml.

A profile:
  [profile]  name, model (qwen35 | qwen35moe | spark), description,
             source ("file" = engine from the GGUF's embedded harness, default;
             "checkout" = engine from this repo), optional pack (path), port,
             speed (one line) and receipt (where the number comes from)
  [env]      BARO_* runtime settings (kind env or server-env in the template)
  [build]    BARO_* -D flags (kind build)

usage:
  tools/profile.py list                 JSON array of every profile (for the UI)
  tools/profile.py check                validate every profile; exit 1 on any problem
  tools/profile.py env NAME             KEY=VALUE words for `env`
  tools/profile.py build NAME           -D KEY=VALUE words for `mojo build`
  tools/profile.py get NAME FIELD       one [profile] field (source, pack, port, model, ...)
"""
import json, sys, tomllib
from pathlib import Path

root = Path(__file__).resolve().parent.parent
pdir = root / "profiles"
template = tomllib.loads((root / "docs/settings-template.toml").read_text())["settings"]
MODELS = ("qwen35", "qwen35moe", "spark")


def load(name):
    p = pdir / f"{name}.toml"
    if not p.exists():
        sys.exit(f"FAIL profile: no {p} (have: {', '.join(sorted(q.stem for q in pdir.glob('*.toml')))})")
    return p, tomllib.loads(p.read_text())


def problems(name, d):
    out = []
    meta = d.get("profile", {})
    if meta.get("name") != name:
        out.append(f"[profile] name '{meta.get('name')}' must equal the file name '{name}'")
    model = meta.get("model")
    if model not in MODELS:
        out.append(f"[profile] model '{model}' not one of {MODELS}")
    if meta.get("source", "file") not in ("file", "checkout"):
        out.append("[profile] source must be file or checkout")
    if not str(meta.get("description", "")).strip():
        out.append("[profile] description is empty")
    for section, kinds in (("env", ("env", "server-env")), ("build", ("build",))):
        for k in d.get(section, {}):
            t = template.get(k)
            if t is None:
                out.append(f"[{section}] {k}: not in docs/settings-template.toml")
            elif t["kind"] not in kinds:
                out.append(f"[{section}] {k}: template kind is '{t['kind']}', belongs under "
                           f"[{'build' if t['kind'] == 'build' else 'env'}]")
            elif not ({"all", model} & set(t["models"])):
                out.append(f"[{section}] {k}: applies to {t['models']}, not {model}")
            elif k == "BARO_STATE_HMAC_KEY":
                out.append(f"[{section}] {k}: a secret, never in a profile")
    return out


cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
if cmd in ("list", "check"):
    rows, bad = [], []
    for p in sorted(pdir.glob("*.toml")):
        d = tomllib.loads(p.read_text())
        errs = problems(p.stem, d)
        bad += [f"{p.name}: {e}" for e in errs]
        rows.append({**d.get("profile", {}), "env": d.get("env", {}), "build": d.get("build", {}),
                     "file": str(p.relative_to(root)), "problems": errs})
    if cmd == "list":
        print(json.dumps(rows, indent=1))
        sys.exit(0)
    for b in bad:
        print("  " + b)
    print(f"{'FAIL' if bad else 'PASS'} profiles: {len(rows)} checked, {len(bad)} problem(s)")
    sys.exit(1 if bad else 0)

if len(sys.argv) < 3:
    sys.exit(__doc__)
name = sys.argv[2]
p, d = load(name)
errs = problems(name, d)
if errs:
    sys.exit(f"FAIL profile {name}: " + "; ".join(errs))
if cmd == "env":
    print(" ".join(f"{k}={v}" for k, v in d.get("env", {}).items()))
elif cmd == "build":
    print(" ".join(f"-D {k}={v}" for k, v in d.get("build", {}).items()))
elif cmd == "get":
    v = d["profile"].get(sys.argv[3], "file" if sys.argv[3] == "source" else "")
    print(v)
else:
    sys.exit(__doc__)
