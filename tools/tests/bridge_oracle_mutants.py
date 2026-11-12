"""Mutant gate for tools/bridge_oracle.py: the tie snap must not hide real errors.
Usage: python -P tools/tests/bridge_oracle_mutants.py CASE DATA  (CASE = a passing tiny dump, e.g. .work/bridge/case-tiny-A)
Exit 0 = the clean case passes and every mutant fails."""
import numpy as np, pathlib, shutil, subprocess, sys

case, data = pathlib.Path(sys.argv[1]), sys.argv[2]
work = case.parent / "mutants"
oracle = pathlib.Path(__file__).resolve().parents[1] / "bridge_oracle.py"
rc = lambda c: subprocess.run([sys.executable, "-P", str(oracle), "check", str(c), data, "1e-4"], capture_output=True).returncode
bad = [] if rc(case) == 0 else ["clean case fails"]
rng = np.random.default_rng(1)
for name, f, scale in (("grad-3e-3", "grad.f32", 3e-3), ("grad-3e-4", "grad.f32", 3e-4), ("act-dkk-3e-3", "act-dkk.f32", 3e-3)):
    d = work / name
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(case, d)
    a = np.fromfile(d / f, np.float32)
    (a * (1 + scale * rng.standard_normal(a.size))).astype(np.float32).tofile(d / f)
    if rc(d) == 0:
        bad.append(f"mutant {name} passes")
print("FAIL bridge-oracle-mutants: " + "; ".join(bad) if bad else "PASS bridge-oracle-mutants")
sys.exit(1 if bad else 0)
