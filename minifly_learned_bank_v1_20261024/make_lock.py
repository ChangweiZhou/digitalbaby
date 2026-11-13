"""Write LOCK.json: code hashes, settings, worlds, decision rule, environment. python make_lock.py"""
import hashlib
import json
import platform
from pathlib import Path

import numba
import numpy
import scipy

HERE = Path(__file__).resolve().parent
CODE = ["lb1.py", "data.py", "run.py", "summarize.py", "main.sh", "../minifly_cls_dev_20261017/fastmem.py"]
lock = {
    "locked": "2026-10-24",
    "design": "DESIGN.md sections 1-10 with Amendment A1 (section 11)",
    "code_sha256": {c: hashlib.sha256((HERE / c).read_bytes()).hexdigest() for c in CODE},
    "settings": {"hidden": 128, "eta0": 0.03, "lam": 0.001, "eta_ramp": 1024, "centre_tau": 0.0,
                 "passes": 150, "probe_every": 25, "final_rest_s": 3600, "native_warmup": "1/16 -> 1 over 1024 records"},
    "main_worlds": ["M1", "M2", "M3", "M4"],
    "development_worlds": ["D1", "D2", "D3"],
    "runs": [{"task": "PAR", "arms": ["LEARN", "FROZEN", "SHUF", "ORACLE", "BASE"]},
             {"task": "COPY", "arms": ["LEARN", "FROZEN", "BASE"]}],
    "command": "sh main.sh",
    "decision_rule": {
        "metric": "balanced held accuracy of the learned bank, T-PAR, final checkpoint (after 1 h rest)",
        "per_world": ["LEARN >= FROZEN + 0.20", "LEARN >= SHUF + 0.20",
                      "LEARN held transfer gap final > birth", "LEARN kernel-target alignment final > birth"],
        "gates": ["ORACLE >= 0.90", "FROZEN learned-bank codes unchanged", "every T-COPY arm system held >= 0.30"],
        "overall": "supported if 4/4 worlds pass; partial if 3/4; not supported otherwise",
    },
    "environment": {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__,
                    "numba": numba.__version__},
}
(HERE / "LOCK.json").write_text(json.dumps(lock, indent=1) + "\n")
print(json.dumps(lock["code_sha256"], indent=1))
