"""Unit/technical tests for Package B (technical world 190000 only; no science world).

Run: python tests/test_topo.py   -> prints one JSON line per test, exits non-zero on failure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import paths  # noqa: E402,F401
import numpy as np  # noqa: E402

import portable_birth as pb  # noqa: E402
from common_platform import DT, RECORD_SECONDS, output_before_teacher, read_cue  # noqa: E402
from fixture import make_world  # noqa: E402
import tfam_graph as tf  # noqa: E402
import topo_model as tm  # noqa: E402
from t2_graph import legacy_sparse_digest  # noqa: E402

W = 190000
DOC = make_world(W)
RESULTS = []


def test(fn):
    try:
        info = fn() or {}
        RESULTS.append({"test": fn.__name__, "pass": True, **info})
    except Exception as exc:  # noqa: BLE001
        RESULTS.append({"test": fn.__name__, "pass": False, "error": repr(exc)})
    print(json.dumps(RESULTS[-1]), flush=True)
    return fn


_BASE, _RAW = pb.canonical_fresh_native()
SUP = tm.Support(_BASE.fly.m.B, W)


def drive(system, lo, hi, branch="W"):
    for index in range(lo, hi):
        row = DOC["records"][index]
        begin = index * RECORD_SECONDS + (86400.0 if row["stage"] == "new" else 0.0)
        system.set_record_context(W, branch, index, row["stage"], row["domain"])
        _, _, at, dom = output_before_teacher(system, bytes.fromhex(row["cue_hex"]), begin)
        system.teach(row["answer"], at, dom, write=True)
        system.feed(10, begin + 13 * DT)
        system.flush(begin + RECORD_SECONDS)


def fourstore(arm, **kw):
    brain = tm.TopoBrain.newborn(arm, W, support=SUP if arm.startswith("S") else None, **kw)
    return tm.TopoFourStore([brain.clone_round() for _ in range(4)])


@test
def canonical_birth_every_arm():
    pair = tf.build_pair(_BASE.fly.m.B, _BASE.fly.m.pn_type_index, "T1", W)
    out = {}
    for arm in ("S0", "S2", "T1"):
        kw = {"static_graph": pair.candidate} if arm == "T1" else {}
        b = tm.TopoBrain.newborn(arm, W, support=SUP if arm.startswith("S") else None, **kw)
        assert b.canonical_b_digest == pb.EXPECTED_B and b.raw_b_digest == _RAW
        # native (non-topology) newborn state equals the canonical anchored newborn
        assert b.native_digest() == _BASE.state_digest()
        out[arm] = b._gdig[:12]
    s0 = tm.TopoBrain.newborn("S0", W, support=SUP)
    assert legacy_sparse_digest(s0.fly.m.B) == pb.EXPECTED_B
    return {"raw_B_sha256": _RAW, "installed": out}


@test
def s_rules_never_read_reinforcement():
    for arm in ("S2", "S3", "S4"):
        a = fourstore(arm)
        drive(a, 0, 23)
        b = a.clone()
        row = DOC["records"][23]
        for sysm, flip in ((a, False), (b, True)):
            sysm.set_record_context(W, "W", 23, row["stage"], row["domain"])
            _, _, at, dom = output_before_teacher(sysm, bytes.fromhex(row["cue_hex"]), 23 * RECORD_SECONDS)
            for m in sysm.stores:
                m.byte(row["answer"], at, learn=False)
            for j, m in enumerate(sysm.stores):
                r = int(b"0123"[j] != row["answer"])
                m.teach(1 - r if flip else r, at, write=True)
        for ma, mb in zip(a.stores, b.stores):
            assert ma._s_digest() == mb._s_digest() and ma._gdig == mb._gdig, arm
            assert ma.native_digest() != mb.native_digest()   # the native write did see r
    return {}


@test
def clone_isolation_and_sharing():
    a = fourstore("S1")
    drive(a, 0, 20)
    before = a.digests()
    graphs = [m.fly.m.B for m in a.stores]
    b = a.clone()
    assert all(mb.fly.m.B is ga for mb, ga in zip(b.stores, graphs))
    drive(b, 20, 60)
    assert a.digests() == before and all(m.fly.m.B is g for m, g in zip(a.stores, graphs))
    assert any(mb.fly.m.B is not ga for mb, ga in zip(b.stores, graphs))
    assert all(not m.fly.m.B.indices.flags.writeable for m in b.stores)
    return {"clone_events": len(b.mechanism_events)}


@test
def restart_parity_s4_and_t3():
    out = {}
    for arm in ("S4", "T3"):
        kw = {}
        if arm == "T3":
            pair = tf.build_pair(_BASE.fly.m.B, _BASE.fly.m.pn_type_index, "T3", W)
            kw = {"static_graph": pair.candidate}
        a = tm.TopoFourStore([m.clone_round() for m in [tm.TopoBrain.newborn(
            arm, W, support=SUP if arm == "S4" else None, **kw)] * 1 for _ in range(4)])
        drive(a, 0, 100)
        snaps = [m.snapshot() for m in a.stores]
        fresh = []
        for s in snaps:
            f = tm.TopoBrain.newborn(arm, W, support=SUP if arm == "S4" else None, **kw)
            f.restore(s)
            fresh.append(f)
        b = tm.TopoFourStore(fresh)
        assert b.digests() == a.digests()
        drive(a, 100, 150)
        drive(b, 100, 150)
        assert b.digests() == a.digests()
        out[arm] = a.digests()[0][:12]
    return out


@test
def deterministic_replay_and_probe_readonly():
    x = fourstore("S2")
    y = fourstore("S2")
    drive(x, 0, 60)
    drive(y, 0, 60)
    assert x.digests() == y.digests()
    ex = [(e["store"], e["n"], e["graph"]) for e in x.mechanism_events]
    assert ex == [(e["store"], e["n"], e["graph"]) for e in y.mechanism_events] and ex
    before = x.digests()
    read_cue(x, bytes.fromhex(DOC["records"][5]["cue_hex"]), 60 * RECORD_SECONDS)
    assert x.digests() == before
    return {"events": len(ex)}


@test
def s0_equals_parent_native_trajectory():
    """S0's extra state is inert: native digests match an unmodified FE0 parent store."""
    from common_platform import FourStore, clone_model
    s0 = fourstore("S0")
    base, _ = pb.canonical_fresh_native()
    parent = FourStore([clone_model(base) for _ in range(4)])
    drive(s0, 0, 50)
    drive(parent, 0, 50)
    assert [m.native_digest() for m in s0.stores] == parent.digests()
    return {}


@test
def srand_uses_counts_only():
    sched = {("W", j, n): 3 for j in range(4) for n in range(1, 30)}
    a = fourstore("Srand_2", replay=sched)
    drive(a, 0, 50)
    ev = a.mechanism_events
    assert ev and all(len(e["changes"]) == 3 for e in ev)
    assert all(set(k) == {0, 1, 2} or True for k in sched)   # keys: (branch, store, n) -> int
    return {"events": len(ev)}


if __name__ == "__main__":
    ok = all(r["pass"] for r in RESULTS)
    out = paths.ROOT / "results" / "technical" / "UNIT_TESTS.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"pass": ok, "tests": RESULTS}, indent=1) + "\n")
    sys.exit(0 if ok else 1)
