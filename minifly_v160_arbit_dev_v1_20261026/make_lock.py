"""Write LOCK.json: the runtime closure of the V160 scripts (sys.modules after importing them, which includes the V155
and V153 modules they use), native assets, the design, the settings and the launch wrapper.
Rebuild and commit after any code change (each new lock is a new design in designs_tried_k).
python make_lock.py   (MiniFly environment)
"""
import hashlib
import importlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
for name in ("v160_lib", "v160_testbed", "v160_technical", "v160_dev"):
    if (HERE / f"{name}.py").exists():
        importlib.import_module(name)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


mods = {Path(m.__file__).resolve() for m in list(sys.modules.values())
        if getattr(m, "__file__", None) and Path(m.__file__).is_absolute() and Path(m.__file__).is_file()
        and str(Path(m.__file__).resolve()).startswith(str(ROOT))}
mods.discard(Path(__file__).resolve())
extra = [ROOT / "minifly_a_v3/package/FULL151_CANONICAL_B.npz",
         ROOT / "minifly_a_v3/package/FULL151_BIRTH_FINGERPRINT.json",
         ROOT / "minifly_cls_dev_20261017/OUTPUT_CODEBOOK.json",
         ROOT / "minifly_v160_arbit_design_20261026/DESIGN.md",
         HERE / "launch.py", HERE / "SETTINGS.json"] + [p for p in sorted(HERE.glob("*.py")) if p.name != "make_lock.py"]
closure = {str(p.resolve().relative_to(ROOT)): sha(p) for p in sorted(set(mods) | {q.resolve() for q in extra})}
lock = {"design": "minifly_v160_arbit_design_20261026/DESIGN.md",
        "settings": json.loads((HERE / "SETTINGS.json").read_text()), "python": sys.version.split()[0],
        "packages": {m: sys.modules[m].__version__ for m in ("numpy", "scipy") if m in sys.modules},
        "closure_sha256": closure}
(HERE / "LOCK.json").write_text(json.dumps(lock, indent=1) + "\n")
print(len(closure), "files locked")
