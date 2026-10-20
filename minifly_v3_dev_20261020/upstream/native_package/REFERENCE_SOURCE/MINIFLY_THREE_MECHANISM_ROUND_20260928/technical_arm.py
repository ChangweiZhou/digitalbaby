"""Run one reserved full-life technical arm, with atomic audited receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path

from audit_round import audit_life
from common_platform import FourStore, clone_model, fresh_native, run_fourstore_life
from fixture import make_world
from r2_model import DualStore, random_gate_schedule

HERE = Path(__file__).resolve().parent
OUT = HERE / "technical"
WORLD = 170000
ARMS = ("R0", "R2", "Rrand", "Z0", "Z1", "Z1_BUDGET_RANDOM", "T0_2", "T2")


def atomic_json(path: Path, doc: dict):
    path.parent.mkdir(exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("w") as f:
        json.dump(doc, f, sort_keys=True, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(pending, path)


def construct(arm: str, world: int, *, pair_cache=None, r2_receipt=None):
    base = fresh_native()
    graph_receipt = None
    if arm in ("R0", "R2", "Rrand"):
        calibration = json.loads((OUT / "R0_CALIBRATION.json").read_text())
        scales = (calibration["scales"]["shared"], calibration["scales"]["private"])
        gates = None
        if arm == "Rrand":
            saved = (r2_receipt if r2_receipt is not None else
                     json.loads((OUT / f"R2_{world}.json").read_text()))
            audit_life(saved, "R2")
            doc = make_world(world)
            gates = {branch: random_gate_schedule(doc, branch, saved["mechanism_events"][branch])
                     for branch in saved["branches"]}
        actor = DualStore.from_native(base, arm=arm, scales=scales, rand_gates=gates)
    elif arm in ("Z0", "Z1", "Z1_BUDGET_RANDOM"):
        from z1_model import from_native
        selected = from_native(base, arm=arm, random_key=world)
        actor = FourStore([clone_model(selected) for _ in range(4)])
    elif arm in ("T0_2", "T2"):
        from t2_graph import build_pair
        from t2_model import GraphBrain
        pair = pair_cache if pair_cache is not None else build_pair(base.fly.m.B, base.fly.m.pn_type_index, world)
        graph_receipt = pair.receipt
        selected = GraphBrain.from_base(base, pair, arm)
        actor = FourStore([clone_model(selected) for _ in range(4)])
    else:
        raise ValueError(arm)
    return actor, graph_receipt


def run(arm: str):
    if arm not in ARMS:
        raise ValueError(arm)
    path = OUT / f"{arm}_{WORLD}.json"
    if path.exists():
        raise RuntimeError(f"technical receipt already exists: {path}")
    start = time.monotonic()
    actor, graph_receipt = construct(arm, WORLD)
    doc = run_fourstore_life(WORLD, actor, technical=True)
    doc["arm"] = arm
    doc["elapsed_wall_s"] = time.monotonic() - start
    doc["max_rss_raw"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    doc["graph_receipt"] = graph_receipt
    doc["source_sha256"] = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                             for name in ("fixture.py", "common_platform.py", "r2_model.py",
                                          "z1_model.py", "t2_graph.py", "t2_model.py",
                                          "technical_arm.py", "audit_round.py")}
    doc["audit"] = audit_life(doc, arm)
    atomic_json(path, doc)
    print(json.dumps({"technical_arm": arm, "world": WORLD, "elapsed_wall_s": doc["elapsed_wall_s"],
                      "max_rss_raw": doc["max_rss_raw"], "bytes": path.stat().st_size,
                      "audit_pass": doc["audit"]["pass"]}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("arm", choices=ARMS)
    run(p.parse_args().arm)
