"""Label-blind reserved-roster graph qualification; no learner trajectories."""
from __future__ import annotations

import json
import os
from pathlib import Path

from common_platform import fresh_native
from t2_graph import build_pair, qualify_technical_roster

HERE = Path(__file__).resolve().parent
OUT = HERE / "technical" / "T2_GRAPH_ROSTER.json"
WORLDS = tuple(range(170000, 170008))


def main():
    if OUT.exists():
        raise RuntimeError("T2 graph roster already qualified")
    base = fresh_native()
    receipts = []
    for world in WORLDS:
        pair = build_pair(base.fly.m.B, base.fly.m.pn_type_index, world)
        receipts.append(pair.receipt)
        print(json.dumps({"technical_graph_world": world,
                          "completed": len(receipts),
                          "lower_tail_gain": pair.receipt["audit"]["lower_tail_gain"]}), flush=True)
    outcome = qualify_technical_roster(receipts, WORLDS)
    doc = {"schema": "MINIFLY-T2-TECHNICAL-ROSTER-v1", "worlds": list(WORLDS),
           "qualification": outcome, "graphs": receipts}
    OUT.parent.mkdir(exist_ok=True)
    pending = OUT.with_suffix(".json.pending")
    with pending.open("w") as f:
        json.dump(doc, f, sort_keys=True, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(pending, OUT)
    print(json.dumps({"graph_roster_qualified": True,
                      "median_gain": outcome["median_effective_type_lower_tail_gain"]}), flush=True)


if __name__ == "__main__":
    main()
