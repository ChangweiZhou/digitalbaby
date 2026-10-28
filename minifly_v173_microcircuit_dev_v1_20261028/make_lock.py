"""Write LOCK.json: SHA-256 of every V173 script, the settings, the design, the launch wrapper, and the V164/V163 modules
and references they read. Rebuild and commit after any code change.  python make_lock.py"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V164 = ROOT / "minifly_v164_cola_dev_v1_20261026"
V163 = ROOT / "minifly_v163_sslcola_dev_v1_20261026"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


files = [ROOT / "minifly_v173_microcircuit_design_20261028/DESIGN.md", V164 / "v164_lib.py", V164 / "SETTINGS.json",
         V164 / "results/testbed/TESTBED.json", V163 / "v163_lib.py", V163 / "SETTINGS.json", HERE / "launch.py",
         HERE / "SETTINGS.json"] + [p for p in sorted(HERE.glob("*.py")) if p.name != "make_lock.py"]
torch_v = subprocess.run(["/tmp/venv_t/bin/python", "-c", "import torch; print(torch.__version__)"],
                         capture_output=True, text=True).stdout.strip()
lock = {"design": "minifly_v173_microcircuit_design_20261028/DESIGN.md", "settings": json.loads((HERE / "SETTINGS.json").read_text()),
        "python": sys.version.split()[0], "packages": {"torch": torch_v},
        "closure_sha256": {str(p.resolve().relative_to(ROOT)): sha(p) for p in files}}
(HERE / "LOCK.json").write_text(json.dumps(lock, indent=1) + "\n")
print(len(files), "files locked")
