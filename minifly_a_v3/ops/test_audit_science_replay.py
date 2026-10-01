"""Recovery audit plumbing tests; no experiment or replay model is run."""
import gzip
import json
import sys
import types

import pytest

import audit_science_replay as replay


def test_sample_is_the_predeclared_52():
    assert replay.WORLDS == (190001, 190002, 190003, 190004)
    assert len(replay.ARMS) == len(set(replay.ARMS)) == 13
    assert len(replay.ARMS) * len(replay.WORLDS) == 52


def test_input_identity_and_source_are_checked(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "ROOT", tmp_path)
    path = replay.receipt_path("R0", 190001)
    path.parent.mkdir(parents=True)
    ctx = {"lock_digest": "L", "receipt_source_sha256": {"source": "H"}}
    receipt = {"schema": "MINIFLY-A3-CLAUDE-RECEIPT-v1", "arm": "R0", "world": 190001,
               "kind": "science", "lock_digest": "L", "source_sha256": {"source": "H"}}
    path.write_bytes(gzip.compress(json.dumps(receipt).encode()))
    assert replay.load("R0", 190001, ctx)[0] == receipt
    receipt["kind"] = "technical_final"
    path.write_bytes(gzip.compress(json.dumps(receipt).encode()))
    with pytest.raises(AssertionError, match="identity or source"):
        replay.load("R0", 190001, ctx)


def test_key_changes_with_receipt_yoke_source_or_runtime():
    base = {"receipt_sha256": {"Z2_rand/190001": "a", "Z2/190001": "b"},
            "wrapper_sha256": "c", "runtime": "d"}
    for changed in ({**base, "receipt_sha256": {"Z2_rand/190001": "e", "Z2/190001": "b"}},
                    {**base, "receipt_sha256": {"Z2_rand/190001": "a", "Z2/190001": "e"}},
                    {**base, "wrapper_sha256": "e"}, {**base, "runtime": "e"}):
        assert replay.digest(base) != replay.digest(changed)


def test_reports_are_write_once(tmp_path):
    path = tmp_path / "record.json"
    replay.write_once(path, {"original": True})
    with pytest.raises(FileExistsError):
        replay.write_once(path, {"replacement": True})
    assert json.loads(path.read_text()) == {"original": True}
    assert not list(tmp_path.glob("*.tmp"))


def test_cache_requires_identity_and_complete_scope(tmp_path):
    path = tmp_path / "cache.json"
    identity = {"arm": "R3", "world": 190001}
    report = {"schema": replay.SCHEMA, "key": "k", "identity": identity, "pass": True,
              "resources": {}, "error": None,
              "checks": {"independent_log_audit": {"pass": True, **identity},
                         "independent_replay": {"arm": "R3", "novelty_checked": 600,
                                                "shared_values": {"records_checked": 600}}}}
    path.write_text(json.dumps(report))
    assert replay.checked_cache(path, identity, "k") == report
    with pytest.raises(AssertionError, match="invalid cached"):
        replay.checked_cache(path, {**identity, "world": 190002}, "k")
    report["checks"]["independent_replay"]["shared_values"]["records_checked"] = 1
    path.write_text(json.dumps(report))
    with pytest.raises(AssertionError, match="incomplete cached audit scope"):
        replay.checked_cache(path, identity, "k")


@pytest.mark.parametrize("arm", replay.ARMS)
def test_frozen_replay_dispatch_and_full_scope(monkeypatch, arm):
    calls = []
    auditor = types.SimpleNamespace(Z_ARMS=("Z0_resource", "Z2", "Z2_rand"),
                                    P_ARMS=("P0", "P1", "P2", "P4"))
    def log(receipt, **kwargs):
        calls.append(("log", receipt["arm"], kwargs))
        return {"pass": True, "arm": receipt["arm"], "world": receipt["world"]}
    auditor.audit_receipt = log
    def z(receipt, paired):
        calls.append(("z", paired))
        return {"arm": arm, "z_events_checked": 1, "branch": "W", "store": 0}
    def p(receipt, **kwargs):
        calls.append(("p", kwargs))
        return {"arm": arm, "p_updates_checked": 600, "pd_checkpoints_checked": 2}
    def novelty(receipt):
        calls.append(("novelty",))
        return {"arm": arm, "novelty_checked": 600}
    def shared(receipt, scale, **kwargs):
        calls.append(("shared", scale, kwargs))
        return {"arm": arm, "records_checked": 600}
    frozen = types.SimpleNamespace(replay_z=z, replay_p=p, replay_r_novelty=novelty, replay_r_shared=shared)
    monkeypatch.setitem(sys.modules, "audit_a", auditor)
    monkeypatch.setitem(sys.modules, "audit_replay", frozen)
    receipt = {"arm": arm, "world": 190001, "mechanism_events": {"W": [["Z", 0, 0]]}}
    dep = {"arm": replay.DEPENDENCIES[arm], "world": 190001} if arm in replay.DEPENDENCIES else None
    cal = {"theta": 0.3, "scales": {"shared": 2, "private": 3}}
    result = replay.replay(receipt, dep, cal)
    assert result["independent_log_audit"]["pass"]
    if arm == "Z2_rand":
        assert ("z", dep) in calls
        assert calls[1][2]["z2_receipt"] == dep
    if arm == "R1_rand":
        assert calls[1][2]["r1_receipt"] == dep
    if arm in replay.SIGNED:
        assert ("shared", 2, {"limit": 600}) in calls
    if arm.startswith("P"):
        assert ("p", {"require_pd": True}) in calls


def test_incomplete_replay_is_rejected(monkeypatch):
    auditor = types.SimpleNamespace(Z_ARMS=(), P_ARMS=(), audit_receipt=lambda *a, **k: {"pass": True})
    frozen = types.SimpleNamespace(replay_r_novelty=lambda r: {"novelty_checked": 1})
    monkeypatch.setitem(sys.modules, "audit_a", auditor)
    monkeypatch.setitem(sys.modules, "audit_replay", frozen)
    with pytest.raises(AssertionError, match="incomplete R novelty"):
        replay.replay({"arm": "R0"}, None, {"theta": 0, "scales": {"shared": 1, "private": 1}})
