"""Write LOCK.json: runtime closure (sys.modules after importing the runner), native assets, corpus files, the design,
the scorer and settings. Rebuild and commit after any code change (each new lock is a new design in designs_tried_k).
python make_lock.py
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import run  # noqa: E402,F401

SETTINGS = json.loads((HERE / "SETTINGS.json").read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


mods = {Path(m.__file__).resolve() for m in list(sys.modules.values())
        if getattr(m, "__file__", None) and str(Path(m.__file__).resolve()).startswith(str(ROOT))}
mods.discard(Path(__file__).resolve())
extra = [ROOT / "minifly_a_v3/package/FULL151_CANONICAL_B.npz",
         ROOT / "minifly_a_v3/package/FULL151_BIRTH_FINGERPRINT.json",
         ROOT / "minifly_cls_dev_20261017/OUTPUT_CODEBOOK.json",
         ROOT / "minifly_v153_bits_design_v1_20261025/DESIGN.md",
         HERE / "launch.py", HERE / "summarize.py", HERE / "SETTINGS.json"] + \
        [p for p, _, _ in run.corpus.TEXTS.values()]
closure = {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(mods) | set(extra))}
lock = {"design": "minifly_v153_bits_design_v1_20261025/DESIGN.md", "main_text": "persuasion",
        "settings": SETTINGS, "python": sys.version.split()[0],
        "packages": {m: sys.modules[m].__version__ for m in ("numpy", "scipy") if m in sys.modules},
        "closure_sha256": closure}
(HERE / "LOCK.json").write_text(json.dumps(lock, indent=1) + "\n")
print(len(closure), "files locked")
