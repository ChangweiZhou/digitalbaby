"""Package-B world-arm runner: one full common life -> one atomic, never-overwritten receipt.

Usage:  python src/runner.py ARM WORLD [--technical]

Science mode (no --technical) refuses to start unless SOURCE_LOCK.json verifies,
the world is in the locked roster and the receipt does not already exist.
Srand_x requires its committed same-world candidate receipt (yoked diagnostic).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
import numpy as np  # noqa: E402

import portable_birth as pb  # noqa: E402
from common_platform import BRANCHES, run_fourstore_life  # noqa: E402
import topo_model as tm  # noqa: E402

ROOT = paths.ROOT
RESULTS = ROOT / "results"
TECH_WORLD = 190000
SCIENCE_WORLDS = tuple(range(190001, 190065))
RECEIPT_SCHEMA = "MINIFLY-B-TOPO-RECEIPT-v1"
CANDIDATE_OF = {f"Srand_{r[1]}": r for r in tm.S_RULES}


def receipt_path(arm: str, world: int, technical: bool) -> Path:
    return RESULTS / ("technical" if technical else "science") / arm / f"{world}.json.gz"


def load_receipt(path: Path) -> dict:
    with gzip.open(path, "rt") as f:
        return json.load(f)


def atomic_write_new(path: Path, body: bytes) -> None:
    """Write-once: O_EXCL on the final name; never replaces a committed receipt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".pending-{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(body)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(tmp, path)          # fails if path exists: no overwrite
    finally:
        os.unlink(tmp)
    dfd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def replay_schedule(candidate: dict) -> dict:
    """Realised partner-change count per (branch, store, event n) of the candidate."""
    sched = {}
    for branch, rows in candidate["mechanism_events"].items():
        for ev in rows:
            if "n_changes" in ev:
                n = ev["n_changes"]
            else:
                n = sum(1 for c in ev["changes"] if c[3] != "accept")
            sched[(branch, ev["store"], ev["n"])] = n
    return sched


def construct(arm: str, world: int, *, technical: bool):
    info = {}
    t0 = time.monotonic()
    if arm in tm.T_ARMS:
        import tfam_graph as tf
        base, raw = pb.canonical_fresh_native()
        family = "T1" if arm in ("T1", "T0_1") else "T3"
        pair = tf.build_pair(base.fly.m.B, base.fly.m.pn_type_index, family, world)
        info["graph_generation_s"] = time.monotonic() - t0
        t1 = time.monotonic()
        info["graph_structural_audit"] = tf.structural_audit(pair, base.fly.m.B, base.fly.m.pn_type_index)
        info["graph_audit_s"] = time.monotonic() - t1
        info["graph_receipt"] = pair.receipt
        graph = pair.candidate if arm == family else pair.control
        brain = tm.TopoBrain.newborn(arm, world, static_graph=graph)
    else:
        base, raw = pb.canonical_fresh_native()
        support = tm.Support(base.fly.m.B, world)
        info["support_digest"] = support.digest
        info["support_slots"] = support.n
        info["graph_generation_s"] = time.monotonic() - t0
        replay = None
        if arm.startswith("Srand"):
            cand = CANDIDATE_OF[arm]
            cpath = receipt_path(cand, world, technical)
            candidate = load_receipt(cpath)
            if candidate["arm"] != cand or candidate["world"] != world:
                raise RuntimeError("Srand candidate receipt identity mismatch")
            info["yoked_candidate_receipt_sha256"] = hashlib.sha256(cpath.read_bytes()).hexdigest()
            replay = replay_schedule(candidate)
        brain = tm.TopoBrain.newborn(arm, world, support=support, replay=replay)
    stores = [brain.clone_round() for _ in range(4)]
    actor = tm.TopoFourStore(stores).mark_base()
    info["birth"] = [{"store": m.store, "raw_B_sha256": m.raw_b_digest,
                      "canonical_B_sha256": m.canonical_b_digest,
                      "installed_graph_digest": m._gdig} for m in actor.stores]
    if len({(b["raw_B_sha256"], b["canonical_B_sha256"], b["installed_graph_digest"])
            for b in info["birth"]}) != 1 or info["birth"][0]["canonical_B_sha256"] != pb.EXPECTED_B:
        raise RuntimeError("four stores do not share the canonical birth and installed graph")
    return actor, info


def compact_events(doc: dict, technical: bool) -> dict:
    """Science receipts keep full change lists for W; other branches keep counts+digest."""
    out = {}
    for branch, rows in doc["mechanism_events"].items():
        keep = []
        for ev in rows:
            row = {"context": ev["context"], "store": ev["store"], "n": ev["n"], "t": ev["t"],
                   "rule": ev["rule"], "graph": ev["graph"]}
            if technical or branch == "W":
                row["changes"] = ev["changes"]
            else:
                row["n_changes"] = sum(1 for c in ev["changes"] if c[3] != "accept")
                row["n_accept"] = sum(1 for c in ev["changes"] if c[3] == "accept")
            keep.append(row)
        out[branch] = keep
    return out


def source_hashes() -> dict:
    files = sorted(p for p in (ROOT / "src").glob("*.py"))
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def run(arm: str, world: int, technical: bool) -> dict:
    if arm not in tm.ALL_ARMS:
        raise ValueError(arm)
    lock = None
    if technical:
        if world != TECH_WORLD:
            raise ValueError("technical runs use world 190000 only")
    else:
        import lock as lk
        lock = lk.verify_lock()
        if world not in SCIENCE_WORLDS:
            raise ValueError("world outside the locked science roster")
    path = receipt_path(arm, world, technical)
    if path.exists():
        raise RuntimeError(f"receipt already committed: {path}")
    start = time.monotonic()
    actor, info = construct(arm, world, technical=technical)
    t_life = time.monotonic()
    doc = run_fourstore_life(world, actor, technical=technical)
    life_s = time.monotonic() - t_life
    doc["schema_parent"] = doc.pop("schema")
    doc["schema"] = RECEIPT_SCHEMA
    doc["arm"] = arm
    doc["mechanism_events"] = compact_events(doc, technical)
    doc["construct"] = info
    if len(actor.branch_children) != len(BRANCHES):
        raise RuntimeError("branch clone registry mismatch")
    doc["end_graph_digests"] = {b: [m._gdig for m in sysm.stores]
                                for b, sysm in zip(BRANCHES, actor.branch_children)}
    doc["end_state_digests_check"] = {b: sysm.digests()
                                      for b, sysm in zip(BRANCHES, actor.branch_children)}
    if doc["end_state_digests_check"] != doc["end_state_digests"]:
        raise RuntimeError("branch registry does not match the life's end states")
    del doc["end_state_digests_check"]
    doc["lock_digest"] = None if lock is None else lock["lock_digest"]
    doc["source_sha256"] = source_hashes()
    doc["environment"] = {"python": platform.python_version(), "numpy": np.__version__,
                          "machine": platform.machine(), "processor": platform.processor(),
                          "cpu_count": os.cpu_count()}
    doc["resources"] = {"learner_life_s": life_s, "total_wall_s": time.monotonic() - start,
                        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    body = gzip.compress(json.dumps(doc, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode(), mtime=0)
    doc["resources"]["receipt_bytes"] = len(body)
    body = gzip.compress(json.dumps(doc, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode(), mtime=0)
    atomic_write_new(path, body)
    return {"arm": arm, "world": world, "technical": technical, "life_s": round(life_s, 1),
            "peak_rss_bytes": doc["resources"]["peak_rss_bytes"], "receipt_bytes": len(body)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=tm.ALL_ARMS)
    ap.add_argument("world", type=int)
    ap.add_argument("--technical", action="store_true")
    a = ap.parse_args()
    print(json.dumps(run(a.arm, a.world, a.technical)), flush=True)
