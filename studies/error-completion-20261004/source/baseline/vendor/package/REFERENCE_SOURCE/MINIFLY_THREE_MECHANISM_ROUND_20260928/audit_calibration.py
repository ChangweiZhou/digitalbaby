"""Independent read-only R0 technical calibration audit; no model execution."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

from fixture import make_world

HERE = Path(__file__).resolve().parent
WORLDS = tuple(range(170000, 170008))


def audit_calibration(doc: dict) -> dict:
    if (doc.get("schema") != "MINIFLY-R0-CALIBRATION-v1" or
            doc.get("worlds") != list(WORLDS) or len(doc.get("receipts", [])) != len(WORLDS)):
        raise AssertionError("calibration world roster")
    for name, digest in doc.get("source_sha256", {}).items():
        if name not in ("fixture.py", "common_platform.py", "r2_model.py", "calibrate_r2.py"):
            raise AssertionError("unexpected calibration source")
        if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != digest:
            raise AssertionError("calibration source drift")
    if set(doc["source_sha256"]) != {"fixture.py", "common_platform.py", "r2_model.py", "calibrate_r2.py"}:
        raise AssertionError("calibration source list")
    squares = {"shared": [], "private": []}
    for world, receipt in zip(WORLDS, doc["receipts"], strict=True):
        fixture = make_world(world)
        expected_cues = (fixture["old_fact"]["cues_hex"] +
                         [r["cue_hex"] for r in fixture["old_relation"]["taught"]])
        if (receipt["world"] != world or receipt["fixture_digest"] != fixture["digest"] or
                [r["cue_hex"] for r in receipt["samples"]] != expected_cues or
                len(receipt["end_digests"]) != 8):
            raise AssertionError("calibration prompt/state roster")
        for sample in receipt["samples"]:
            for bank in squares:
                values = sample[bank]
                if len(values) != 4 or any(not math.isfinite(v) for v in values):
                    raise AssertionError("calibration finite values")
                center = sum(values) / 4
                squares[bank].extend((v - center) ** 2 for v in values)
    for bank, values in squares.items():
        expected = math.sqrt(sum(values) / len(values))
        actual = doc["scales"][bank]
        if actual < 1e-8 or abs(expected - actual) > 1e-10:
            raise AssertionError("calibration scale equation")
    return {"pass": True, "worlds": len(WORLDS), "prompts": 28 * len(WORLDS),
            "scales": doc["scales"]}
