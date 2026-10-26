"""Write LOCK.json: SHA-256 of every V166 script, the settings, the design and the launch wrapper. V166 imports no
other delivery's code or data. Rebuild and commit after any code change and before each launch.
python make_lock.py (any environment; torch version read from the torch env)."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


files = [ROOT / "minifly_v166_memloop_design_20261026/DESIGN.md", HERE / "launch.py", HERE / "SETTINGS.json",
         HERE / "AUTHORIZATION.json"] + [p for p in sorted(HERE.glob("*.py")) if p.name not in ("make_lock.py", "launch.py")]
torch_v = subprocess.run(["/tmp/venv_t/bin/python", "-c", "import torch; print(torch.__version__)"],
                         capture_output=True, text=True).stdout.strip()
lock = {"design": "minifly_v166_memloop_design_20261026/DESIGN.md", "settings": json.loads((HERE / "SETTINGS.json").read_text()),
        "python": sys.version.split()[0], "packages": {"torch": torch_v},
        "closure_sha256": {str(p.resolve().relative_to(ROOT)): sha(p) for p in files}}
(HERE / "LOCK.json").write_text(json.dumps(lock, indent=1) + "\n")
print(len(files), "files locked")
