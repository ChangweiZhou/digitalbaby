#!/usr/bin/env python3
"""Read-only-ish handoff integrity and import smoke check; no trajectory."""
from __future__ import annotations

import hashlib
import json
import sys
import ast
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest = json.loads((HERE / "MANIFEST.json").read_text())
    expected = manifest["files"]
    actual = {p.relative_to(HERE).as_posix() for p in HERE.rglob("*") if p.is_file()}
    # No executable/results can be slipped in alongside the frozen snapshot.
    if actual != set(expected) | {"MANIFEST.json"}:
        raise AssertionError(f"file-set mismatch: missing={set(expected)-actual}, extra={actual-set(expected)-{'MANIFEST.json'}}")
    for relative, digest in expected.items():
        if sha(HERE / relative) != digest:
            raise AssertionError(f"sha256 mismatch: {relative}")
    contract = json.loads((HERE / "CONTRACT.json").read_text())
    if contract["fixture_sha256"] != sha(HERE / "REFERENCE_SOURCE" / "MINIFLY_THREE_MECHANISM_ROUND_20260928/fixture.py"):
        raise AssertionError("fixture digest mismatch")
    if len(contract["science_worlds"]) != 64 or len(contract["variants"]) not in (6, 7):
        raise AssertionError("contract roster mismatch")
    for path in (HERE / "REFERENCE_SOURCE").rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    from causal_branch_gate import self_test  # noqa: PLC0415
    branch_gate = self_test()
    if "--integrity-only" in sys.argv:
        print(json.dumps({"package": contract["package"], "verified_files": len(expected),
                          "variants": contract["variants"], "source_syntax": "verified",
                          "native_import": "not_checked", "causal_branch_gate": branch_gate,
                          "science_ready": False, "muse_runtime_source": "not_in_bundle",
                          "science_trajectories_run": 0}, sort_keys=True))
        return
    src = HERE / "REFERENCE_SOURCE" / "MINIFLY_THREE_MECHANISM_ROUND_20260928"
    sys.path.insert(0, str(src))
    from portable_birth import canonical_fresh_native, EXPECTED_B  # noqa: PLC0415
    from fixture import make_world  # noqa: PLC0415
    model, raw_b_digest = canonical_fresh_native()
    if model.brain_t != 0.0 or model.fly.m.elapsed != 0.0:
        raise AssertionError("fresh model clock mismatch")
    hashes = __import__("brain_byte").bc.object_hashes(model.fly)
    if hashes != {"B": EXPECTED_B,
                  "Q": "1aa98dbc6c7424b3f0ddc8cb2af6e02a4d405f718d2de0d829bde05f3575f202"}:
        raise AssertionError("Full151 B/Q source object mismatch")
    world = make_world(contract["technical_world"])
    if len(world["records"]) != 600 or len(world["old_relation"]["heldout"]) != 6:
        raise AssertionError("technical fixture mismatch")
    print(json.dumps({"package": contract["package"], "verified_files": len(expected),
                      "variants": contract["variants"], "fixture_world": world["world"],
                      "records": len(world["records"]), "native_B_Q": "verified",
                      "causal_branch_gate": branch_gate,
                      "science_ready": False, "muse_runtime_source": "not_in_bundle",
                      "raw_B_match": raw_b_digest == EXPECTED_B,
                      "raw_B_sha256": raw_b_digest,
                      "science_trajectories_run": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
