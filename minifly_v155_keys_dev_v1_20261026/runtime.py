"""Write runlog/RUNTIME_CLOSURE_<attempt>.json: every repository file imported by the running process, with SHA-256."""
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def write_closure():
    attempt = os.environ.get("MINIFLY_ATTEMPT_ID", "unlaunched")
    mods = sorted({str(Path(m.__file__).resolve()) for m in list(sys.modules.values())
                   if getattr(m, "__file__", None) and str(Path(m.__file__).resolve()).startswith(str(ROOT))})
    clo = {str(Path(p).relative_to(ROOT)): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in mods}
    (HERE / "runlog").mkdir(exist_ok=True)
    path = HERE / "runlog" / f"RUNTIME_CLOSURE_{attempt}.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps({**old, **clo}, indent=1))
