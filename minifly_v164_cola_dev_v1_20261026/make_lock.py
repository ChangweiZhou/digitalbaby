"""Write LOCK.json: SHA-256 of every V164 script, the settings, the design, the launch wrapper, the V163 modules imported
(v163_lib and the V149 readouts it loads), V163's settings and the data read (V163's code cache and C0 results).
Rebuild and commit after any code change. python make_lock.py (any environment; torch version read from the torch env)."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V163 = ROOT / "minifly_v163_sslcola_dev_v1_20261026"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


files = [ROOT / "minifly_v164_cola_design_20261026/DESIGN.md", V163 / "v163_lib.py", V163 / "SETTINGS.json",
         V163 / "cache" / "codes.npz", V163 / "results" / "testbed" / "TESTBED.json",
         ROOT / "minifly_v149_input_ceiling_dev_20261025/readouts.py", HERE / "launch.py", HERE / "SETTINGS.json"] + \
        [p for p in sorted(HERE.glob("*.py")) if p.name != "make_lock.py"]
torch_v = subprocess.run(["/tmp/venv_t/bin/python", "-c", "import torch; print(torch.__version__)"],
                         capture_output=True, text=True).stdout.strip()
lock = {"design": "minifly_v164_cola_design_20261026/DESIGN.md", "settings": json.loads((HERE / "SETTINGS.json").read_text()),
        "python": sys.version.split()[0], "packages": {"torch": torch_v},
        "closure_sha256": {str(p.resolve().relative_to(ROOT)): sha(p) for p in files}}
(HERE / "LOCK.json").write_text(json.dumps(lock, indent=1) + "\n")
print(len(files), "files locked")
