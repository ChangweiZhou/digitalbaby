"""Write LOCK.json: SHA-256 of every V172 script, the settings, the design, the launch wrapper, the vendored Codex
sources and the native modules they read. Rebuild and commit after any code change.  python make_lock.py"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CLS = ROOT / "minifly_cls_dev_20261017"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


files = [ROOT / "minifly_v172_native_e1_design_20261027/DESIGN.md", CLS / "fastmem.py", CLS / "bootstrap.py",
         CLS / "OUTPUT_CODEBOOK.json", ROOT / "minifly_a_v3/src/stores.py", HERE / "vendor/models152.py",
         HERE / "vendor/support152.py", HERE / "launch.py", HERE / "SETTINGS.json"] + \
        [p for p in sorted(HERE.glob("*.py")) if p.name != "make_lock.py"]
np_v = subprocess.run(["/tmp/venv_n/bin/python", "-c", "import numpy, numba; print(numpy.__version__, numba.__version__)"],
                      capture_output=True, text=True).stdout.split()
lock = {"design": "minifly_v172_native_e1_design_20261027/DESIGN.md", "settings": json.loads((HERE / "SETTINGS.json").read_text()),
        "python": sys.version.split()[0], "packages": {"numpy": np_v[0], "numba": np_v[1]},
        "vendored_from": "ec38e8be63e0a448d387e4698fd619c7368ddf12 (minifly_v152_dev_20261025/models152.py, support152.py)",
        "closure_sha256": {str(p.resolve().relative_to(ROOT)): sha(p) for p in files}}
(HERE / "LOCK.json").write_text(json.dumps(lock, indent=1) + "\n")
print(len(files), "files locked")
