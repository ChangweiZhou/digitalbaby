"""Write LOCK.json: SHA-256 of every V163 script, the settings, the design, the launch wrapper, the repository modules the
MiniFly-side export imports (its runtime closure), the V149 readouts module the torch side loads, and the data files read.
Rebuild and commit after any code change (each new lock is a new design in designs_tried_k).
python make_lock.py   (MiniFly environment; the torch version is read from the torch environment)
"""
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
importlib.import_module("v163_export")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


mods = {Path(m.__file__).resolve() for m in list(sys.modules.values())
        if getattr(m, "__file__", None) and Path(m.__file__).is_absolute() and Path(m.__file__).is_file()
        and str(Path(m.__file__).resolve()).startswith(str(ROOT))}
mods.discard(Path(__file__).resolve())
hc = ROOT / "minifly_v155_headroom_check_20261026" / "results" / "main"
extra = [ROOT / "minifly_v163_sslcola_design_20261026/DESIGN.md",
         ROOT / "minifly_v149_input_ceiling_dev_20261025/readouts.py",
         ROOT / "minifly_v155_keys_dev_v1_20261026/cache/birth.npz",
         ROOT / "minifly_v155_keys_dev_v1_20261026/cache/thursday_A.npz",
         ROOT / "minifly_v162_text_dev_v1_20261026/cache/edgeworth_B.npz",
         hc / "states.npz", hc / "REFERENCE.json", hc / "x_train.npz", hc / "x_calib.npz", hc / "x_held.npz",
         HERE / "launch.py", HERE / "SETTINGS.json"] + [p for p in sorted(HERE.glob("*.py")) if p.name != "make_lock.py"]
closure = {str(p.resolve().relative_to(ROOT)): sha(p) for p in sorted(set(mods) | {q.resolve() for q in extra})}
torch_v = subprocess.run(["/tmp/venv_t/bin/python", "-c", "import torch; print(torch.__version__)"],
                         capture_output=True, text=True).stdout.strip()
lock = {"design": "minifly_v163_sslcola_design_20261026/DESIGN.md",
        "settings": json.loads((HERE / "SETTINGS.json").read_text()), "python": sys.version.split()[0],
        "packages": {**{m: sys.modules[m].__version__ for m in ("numpy", "scipy") if m in sys.modules}, "torch": torch_v},
        "closure_sha256": closure}
(HERE / "LOCK.json").write_text(json.dumps(lock, indent=1) + "\n")
print(len(closure), "files locked")
