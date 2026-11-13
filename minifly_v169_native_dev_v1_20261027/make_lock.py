"""Write LOCK.json: SHA-256 of every V169 script, the settings, the design (revision 2), the launch wrapper, and the
modules and data they read. Rebuild and commit after any code change.  python make_lock.py"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V167 = ROOT / "minifly_v167_local_dev_v1_20261026"
V163 = ROOT / "minifly_v163_sslcola_dev_v1_20261026"
V165 = ROOT / "minifly_v165_dfa_dev_v1_20261026"
V155 = ROOT / "minifly_v155_keys_dev_v1_20261026"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


files = [ROOT / "minifly_v169_native_design_20261027/DESIGN.md", ROOT / "minifly_memo_credit_horizon_20261027/MEMO.md",
         V167 / "v167_local.py", V167 / "v167_common.py", V167 / "SETTINGS.json", V167 / "cache" / "codes_rehearsal.npz",
         V167 / "results" / "testbed" / "TESTBED.json", V167 / "results" / "testbed" / "EXPORT.json", V167 / "LOCK.json",
         V163 / "v163_lib.py", V163 / "v163_testbed_r2.py", V163 / "SETTINGS.json", V165 / "v165_dfa.py", V165 / "v165_common.py",
         V165 / "SETTINGS.json", V155 / "corpus.py", V155 / "data" / "austen-emma.txt", V155 / "data" / "chesterton-ball.txt",
         ROOT / "minifly_v148_pretrain_probe_dev_20261025" / "ladder.py",
         ROOT / "minifly_v149_input_ceiling_dev_20261025" / "readouts.py",
         HERE / "launch.py", HERE / "SETTINGS.json"] + [p for p in sorted(HERE.glob("*.py")) if p.name != "make_lock.py"]
torch_v = subprocess.run(["/tmp/venv_t/bin/python", "-c", "import torch; print(torch.__version__)"],
                         capture_output=True, text=True).stdout.strip()
lock = {"design": "minifly_v169_native_design_20261027/DESIGN.md (revision 2)",
        "settings": json.loads((HERE / "SETTINGS.json").read_text()), "python": sys.version.split()[0],
        "packages": {"torch": torch_v}, "closure_sha256": {str(p.resolve().relative_to(ROOT)): sha(p) for p in files}}
(HERE / "LOCK.json").write_text(json.dumps(lock, indent=1) + "\n")
print(len(files), "files locked")
