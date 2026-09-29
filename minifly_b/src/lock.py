"""SOURCE_LOCK.json: write-once lock over every scientific file, anchor and budget.

``python src/lock.py create`` writes it (refuses if present); science and analysis
call ``verify_lock()`` which recomputes every hash and the environment.
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
LOCKED_TOP = ("SPEC_LOCK.md", "GRAPH_MANIFEST.json", "RESOURCE_BUDGET.json", "ARM_ROSTER.json")
ANCHOR = ("package/portable_birth.py", "package/FULL151_CANONICAL_B.npz",
          "package/FULL151_BIRTH_FINGERPRINT.json", "package/BIRTH_PORTABILITY_ADDENDUM.md")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _files() -> dict:
    out = {}
    manifest = json.loads((paths.PKG / "MANIFEST.json").read_text())
    for rel in sorted(manifest["files"]):
        out["package/" + rel] = _sha(paths.PKG / rel)
    out["package/MANIFEST.json"] = _sha(paths.PKG / "MANIFEST.json")
    for sub in ("src", "tests"):
        for p in sorted((ROOT / sub).glob("*.py")):
            out[p.relative_to(ROOT).as_posix()] = _sha(p)
    for name in LOCKED_TOP:
        out[name] = _sha(ROOT / name)
    return out


def _env() -> dict:
    import numba
    import numpy
    import scipy
    return {"python": platform.python_version(), "numpy": numpy.__version__,
            "scipy": scipy.__version__, "numba": numba.__version__,
            "machine": platform.machine()}


def _digest(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def create() -> dict:
    if LOCK.exists():
        raise SystemExit("SOURCE_LOCK.json already exists (write-once)")
    manifest = json.loads((paths.PKG / "MANIFEST.json").read_text())
    for rel, h in manifest["files"].items():
        if _sha(paths.PKG / rel) != h:
            raise SystemExit(f"package file differs from its manifest: {rel}")
    roster = json.loads((ROOT / "ARM_ROSTER.json").read_text())
    contract = json.loads((paths.PKG / "CONTRACT.json").read_text())
    body = {"schema": "MINIFLY-B-SOURCE-LOCK-v1",
            "files": _files(),
            "anchor_files": {a: _sha(ROOT / a) for a in ANCHOR},
            "parent_source_lock_digest": contract["parent_source_lock_digest"],
            "fixture_sha256": contract["fixture_sha256"],
            "science_worlds": contract["science_worlds"],
            "technical_world": contract["technical_world"],
            "arms": roster["arms"], "family_size_m": 26,
            "environment": _env()}
    body["lock_digest"] = _digest(body)
    LOCK.write_text(json.dumps(body, indent=1, sort_keys=True) + "\n")
    return body


def verify_lock() -> dict:
    body = json.loads(LOCK.read_text())
    claimed = body.pop("lock_digest")
    if _digest(body) != claimed:
        raise RuntimeError("SOURCE_LOCK.json digest mismatch")
    now = _files()
    bad = sorted(k for k in set(now) | set(body["files"]) if now.get(k) != body["files"].get(k))
    if bad:
        raise RuntimeError(f"locked files changed: {bad[:10]}")
    if _env() != body["environment"]:
        raise RuntimeError(f"environment differs from lock: {_env()} vs {body['environment']}")
    body["lock_digest"] = claimed
    return body


if __name__ == "__main__":
    if sys.argv[1:] == ["create"]:
        print(create()["lock_digest"])
    elif sys.argv[1:] == ["verify"]:
        print(verify_lock()["lock_digest"])
    else:
        raise SystemExit("usage: lock.py create|verify")
