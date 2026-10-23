"""R0-only, label-blind dual-bank output scaling on reserved technical worlds."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

from common_platform import DT, RECORD_SECONDS, clone_model, fresh_native, output_before_teacher
from fixture import make_world
from r2_model import DualStore

HERE = Path(__file__).resolve().parent
OUT = HERE / "technical" / "R0_CALIBRATION.json"
WORLDS = tuple(range(170000, 170008))


def technical_world(world: int, base) -> dict:
    doc = make_world(world)
    actor = DualStore.from_native(clone_model(base), arm="R0", scales=(1.0, 1.0))
    for index, row in enumerate(doc["records"][:336]):
        if row["stage"] != "old":
            raise AssertionError("unexpected calibration phase")
        cue = bytes.fromhex(row["cue_hex"])
        begin = index * RECORD_SECONDS
        actor.set_record_context(world, "W", index, "old", row["domain"])
        _, _, at, domain = output_before_teacher(actor, cue, begin)
        if domain != row["domain"]:
            raise AssertionError("calibration route mismatch")
        actor.teach(row["answer"], at, domain, write=True)
        actor.feed(10, begin + 13 * DT)
        if actor.flush(begin + RECORD_SECONDS) >= 1e-6:
            raise AssertionError("calibration clock")
    samples = []
    for group in (doc["old_fact"]["cues_hex"],
                  [r["cue_hex"] for r in doc["old_relation"]["taught"]]):
        for cue_hex in group:
            scratch = actor.clone()
            cue = bytes.fromhex(cue_hex)
            for j, byte in enumerate(cue):
                scratch.feed(byte, doc["old_end_s"] + j * DT)
            s, p = scratch.raw_values(doc["old_end_s"] + 12 * DT)
            samples.append({"cue_hex": cue_hex, "shared": s, "private": p})
    if len(samples) != 28:
        raise AssertionError("R0 calibration roster")
    return {"world": world, "fixture_digest": doc["digest"], "samples": samples,
            "end_digests": actor.digests()}


def scales(receipts: list[dict]) -> tuple[float, float]:
    if len(receipts) != len(WORLDS):
        raise AssertionError("R0 calibration worlds missing")
    out = []
    for name in ("shared", "private"):
        sq = []
        for receipt in receipts:
            for sample in receipt["samples"]:
                values = sample[name]
                if len(values) != 4 or any(not math.isfinite(v) for v in values):
                    raise AssertionError("nonfinite calibration value")
                mean = sum(values) / 4
                sq.extend((v - mean) ** 2 for v in values)
        val = math.sqrt(sum(sq) / len(sq))
        if not math.isfinite(val) or val < 1e-8:
            raise AssertionError("degenerate R0 calibration")
        out.append(val)
    return tuple(out)


def main():
    if OUT.exists():
        raise RuntimeError("R0 technical calibration already exists")
    birth = fresh_native()
    receipts = []
    for world in WORLDS:
        receipts.append(technical_world(world, birth))
        print(json.dumps({"technical_R0_world": world, "completed": len(receipts)}), flush=True)
    s = scales(receipts)
    doc = {"schema": "MINIFLY-R0-CALIBRATION-v1", "worlds": list(WORLDS),
           "source_sha256": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                             for name in ("fixture.py", "common_platform.py", "r2_model.py", "calibrate_r2.py")},
           "scales": {"shared": s[0], "private": s[1]}, "receipts": receipts}
    OUT.parent.mkdir(exist_ok=True)
    pending = OUT.with_suffix(".json.pending")
    with pending.open("w") as f:
        json.dump(doc, f, sort_keys=True, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(pending, OUT)
    print(json.dumps({"calibration": "complete", "scales": doc["scales"]}), flush=True)


if __name__ == "__main__":
    main()
