"""Write LOCK.json: the runtime closure of the MiniFly-side scripts (sys.modules after importing them), the torch-side
reference script, native assets, the six pinned texts, the design, the settings and the launch wrapper.
Rebuild and commit after any code change (each new lock is a new design in designs_tried_k).
python make_lock.py   (MiniFly environment)
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import cache  # noqa: E402,F401
import score  # noqa: E402,F401

SETTINGS = json.loads((HERE / "SETTINGS.json").read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


mods = {Path(m.__file__).resolve() for m in list(sys.modules.values())
        if getattr(m, "__file__", None) and str(Path(m.__file__).resolve()).startswith(str(ROOT))}
mods.discard(Path(__file__).resolve())
extra = [ROOT / "minifly_a_v3/package/FULL151_CANONICAL_B.npz",
         ROOT / "minifly_a_v3/package/FULL151_BIRTH_FINGERPRINT.json",
         ROOT / "minifly_cls_dev_20261017/OUTPUT_CODEBOOK.json",
         ROOT / "minifly_v155_keys_design_20261026/DESIGN.md",
         HERE / "refs.py", HERE / "runtime.py", HERE / "launch.py", HERE / "SETTINGS.json"] + \
        sorted(HERE.glob("*.py")) + [p for p, _, _ in score.corpus.TEXTS.values()]
extra = [p for p in extra if p.resolve() != Path(__file__).resolve()]
closure = {str(p.resolve().relative_to(ROOT)): sha(p) for p in sorted(set(mods) | {q.resolve() for q in extra})}
lock = {"design": "minifly_v155_keys_design_20261026/DESIGN.md", "pairs": score.corpus.PAIRS,
        "settings": SETTINGS, "python": sys.version.split()[0],
        "packages": {m: sys.modules[m].__version__ for m in ("numpy", "scipy") if m in sys.modules},
        "torch_env": "/tmp/venv_t (torch 2.14.1, refs.py only)",
        "closure_sha256": closure}
(HERE / "LOCK.json").write_text(json.dumps(lock, indent=1) + "\n")
print(len(closure), "files locked")
