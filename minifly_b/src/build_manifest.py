"""Pre-science GRAPH_MANIFEST.json: every T pair and S support for the locked roster.

Graph-only: reads the canonical birth B, pn_type_index and world IDs; never a
fixture, label or score.  Parallel over worlds; output is sorted and deterministic.
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

WORLDS = [190000] + list(range(190001, 190065))


def one(world: int) -> tuple[int, dict]:
    import portable_birth as pb
    import tfam_graph as tf
    import topo_model as tm
    base, _ = pb.canonical_fresh_native()
    B, types = base.fly.m.B, base.fly.m.pn_type_index
    out = {"T": {}, "S": {}}
    for fam in ("T1", "T3"):
        t0 = time.monotonic()
        pair = tf.build_pair(B, types, fam, world)
        gen = time.monotonic() - t0
        t1 = time.monotonic()
        aud = tf.structural_audit(pair, B, types)
        ctrl = f"T0_{fam[1]}"
        r = pair.receipt
        out["T"][fam] = {"graph_digest": r["graph_digest"],
                         "topo_digest": {fam: tm.graph_bytes_digest(pair.candidate),
                                         ctrl: tm.graph_bytes_digest(pair.control)},
                         "accepted_switches": r["optimization"]["accepted_switches"],
                         "levels": {k: {str(a): b for a, b in v.items()} for k, v in r["levels"].items()},
                         "modularity_q": r["modularity_q"],
                         "type_symmetric_difference_edges": r["type_symmetric_difference_edges"],
                         "support_symdiff_candidate_control": aud["support_symdiff_candidate_control"],
                         "support_symdiff_candidate_native": aud["support_symdiff_candidate_native"],
                         "generation_s": round(gen, 2), "audit_s": round(time.monotonic() - t1, 2)}
    sup = tm.Support(B, world)
    out["S"] = {"support_digest": sup.digest, "slots": sup.n,
                "birth_topo_digest": tm.graph_bytes_digest(B)}
    return world, out


def main() -> None:
    dest = paths.ROOT / "GRAPH_MANIFEST.json"
    if dest.exists():
        raise SystemExit("GRAPH_MANIFEST.json already exists; it is write-once")
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    doc = {"schema": "MINIFLY-B-GRAPH-MANIFEST-v1", "worlds": WORLDS, "T": {"T1": {}, "T3": {}}, "S": {}}
    with ProcessPoolExecutor(workers) as ex:
        for world, out in ex.map(one, WORLDS):
            for fam in ("T1", "T3"):
                doc["T"][fam][str(world)] = out["T"][fam]
            doc["S"][str(world)] = out["S"]
            print(world, flush=True)
    dest.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
