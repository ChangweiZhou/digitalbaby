"""SOURCE_LOCK.json for Package A V3-CLAUDE: one write-once lock over the complete executed closure.

Covers every package file (by the scaffold MANIFEST), all src/ and tests/ modules, SPEC_LOCK.md, ARM_ROSTER.json,
RESOURCE_BUDGET.json, the R calibration file, both input archives, the final technical receipts and the
environment. ``python src/lock.py create`` writes it (refuses if present); ``verify`` recomputes everything.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

ROOT = paths.ROOT
LOCK = ROOT / "SOURCE_LOCK.json"
LOCKED_TOP = ("SPEC_LOCK.md", "ARM_ROSTER.json", "RESOURCE_BUDGET.json", "results/calibration/R_CALIBRATION.json",
              "input/MINIFLY_MUSE_A_V3_CAUSAL_GATE_20260929.zip", "input/CLAUDE_PACKAGE_A_MUSE_INTAKE_20260929.zip")
TECH_FINAL = "results/technical_final"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def package_files() -> dict:
    manifest = json.loads((paths.PKG / "MANIFEST.json").read_text())
    out = {"package/" + rel: _sha(paths.PKG / rel) for rel in sorted(manifest["files"])}
    out["package/MANIFEST.json"] = _sha(paths.PKG / "MANIFEST.json")
    bad = [rel for rel, meta in manifest["files"].items()
           if out["package/" + rel] != (meta["sha256"] if isinstance(meta, dict) else meta)]
    if bad:
        raise AssertionError(f"package file differs from its scaffold MANIFEST: {bad[:5]}")
    return out


def code_files() -> dict:
    out = {}
    for sub in ("src", "tests"):
        for p in sorted((ROOT / sub).glob("*.py")):
            out[p.relative_to(ROOT).as_posix()] = _sha(p)
    return out


def _files() -> dict:
    out = {**package_files(), **code_files()}
    for name in LOCKED_TOP:
        out[name] = _sha(ROOT / name)
    for p in sorted((ROOT / TECH_FINAL).rglob("*.json.gz")):
        out[p.relative_to(ROOT).as_posix()] = _sha(p)
    return out


def _env() -> dict:
    import numba
    import numpy
    import pandas
    import scipy
    return {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__,
            "numba": numba.__version__, "pandas": pandas.__version__, "machine": platform.machine()}


def _digest(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def create() -> dict:
    if LOCK.exists():
        raise FileExistsError("SOURCE_LOCK.json already exists; it is write-once")
    roster = json.loads((ROOT / "ARM_ROSTER.json").read_text())
    body = {"schema": "MINIFLY-A3-CLAUDE-SOURCE-LOCK-v1",
            "package": "MINIFLY_A_V3_CLAUDE (new implementation; not Muse V2)",
            "arms": roster["science_arms"], "worlds": roster["science_worlds"],
            "family_size_m": 26, "files": _files(), "environment": _env()}
    body["lock_digest"] = _digest({k: v for k, v in body.items()})
    LOCK.write_text(json.dumps(body, indent=1, sort_keys=True) + "\n")
    return body


def verify_lock() -> dict:
    lock = json.loads(LOCK.read_text())
    body = {k: v for k, v in lock.items() if k != "lock_digest"}
    if _digest(body) != lock["lock_digest"]:
        raise AssertionError("lock digest does not match its body")
    now = _files()
    bad = sorted(set(lock["files"]) ^ set(now)) + sorted(k for k in lock["files"] if now.get(k) != lock["files"][k])
    if bad:
        raise AssertionError(f"locked files changed/missing/added: {bad[:8]}")
    if _env() != lock["environment"]:
        raise AssertionError(f"environment differs from lock: {_env()}")
    return lock


if __name__ == "__main__":
    if sys.argv[1:] == ["create"]:
        print(create()["lock_digest"])
    elif sys.argv[1:] == ["verify"]:
        print(verify_lock()["lock_digest"])
    else:
        raise SystemExit("usage: lock.py create|verify")
