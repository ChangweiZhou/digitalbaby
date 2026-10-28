"""Write LOCK.json: SHA-256 of every V174 script, the settings, the design, the launch wrapper, V172's library and settings,
and the native modules they read. Rebuild and commit after any code change.  python make_lock.py"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CLS = ROOT / "minifly_cls_dev_20261017"
V172 = ROOT / "minifly_v172_native_e1_dev_v1_20261027"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


files = [ROOT / "minifly_v174_replay_design_20261028/DESIGN.md", V172 / "v172_lib.py", V172 / "SETTINGS.json",
         CLS / "fastmem.py", CLS / "bootstrap.py", CLS / "OUTPUT_CODEBOOK.json", ROOT / "minifly_a_v3/src/stores.py",
         HERE / "launch.py", HERE / "SETTINGS.json"] + [p for p in sorted(HERE.glob("*.py")) if p.name != "make_lock.py"]
np_v = subprocess.run(["/tmp/venv_n/bin/python", "-c", "import numpy, numba; print(numpy.__version__, numba.__version__)"],
                      capture_output=True, text=True).stdout.split()
lock = {"design": "minifly_v174_replay_design_20261028/DESIGN.md", "settings": json.loads((HERE / "SETTINGS.json").read_text()),
        "python": sys.version.split()[0], "packages": {"numpy": np_v[0], "numba": np_v[1]},
        "closure_sha256": {str(p.resolve().relative_to(ROOT)): sha(p) for p in files}}
(HERE / "LOCK.json").write_text(json.dumps(lock, indent=1) + "\n")
print(len(files), "files locked")
