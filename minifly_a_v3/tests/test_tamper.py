"""Deliberate receipt corruptions that the independent auditor must reject (run on committed technical receipts)."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import paths  # noqa: E402
import audit_a  # noqa: E402
from fixture import make_world  # noqa: E402
import gzip  # noqa: E402

RES = paths.ROOT / "results" / "technical"


def load(arm):
    return json.loads(gzip.decompress((RES / arm / "190000.json.gz").read_bytes()))


def cal():
    c = json.loads((paths.ROOT / "results" / "calibration" / "R_CALIBRATION.json").read_text())
    return c["theta"], (c["scales"]["shared"], c["scales"]["private"])


def kwargs_for(arm):
    theta, scales = cal()
    kw = {"theta": theta, "scales": scales}
    if arm == "R1_rand":
        kw["r1_receipt"] = load("R1")
    return kw


def rows(r, branch, kind, bank=None):
    return [x for x in r["mechanism_events"][branch] if x[0] == kind and (bank is None or x[2] == bank)]


def mutations(arm, base):
    fx = make_world(190000)
    old_rel = [s["index"] for s in fx["records"] if s["stage"] == "old" and s["domain"] == "relation"]
    old_fact = [s["index"] for s in fx["records"] if s["stage"] == "old" and s["domain"] == "fact"]
    out = {}

    def m_restore_288(r):   # V2 defect: N_old_rel performs every old-relation write
        for x in r["mechanism_events"]["N_old_rel"]:
            if x[0] in ("w", "s") and x[1] in old_rel and x[3] < 2:
                x[4] = 1
        r["writes"]["N_old_rel"]["old_relation"] = 288
    out["restore_288_old_relation_writes"] = m_restore_288

    def m_restore_288_ledger_only(r):
        for x in r["mechanism_events"]["N_old_rel"]:
            if x[0] in ("w", "s") and x[1] in old_rel and x[3] < 2 and x[2] == 0:
                x[4] = 1
    out["restore_288_ledger_only"] = m_restore_288_ledger_only

    def m_compensating_pair(r):  # aggregate unchanged: one wrong write, one missing write, same domain
        led = {(x[1], x[2], x[3]): x for x in r["mechanism_events"]["W"] if x[0] in ("w", "s")}
        led[(old_rel[0], 0, 0)][4] = 0      # missing permitted write
        led[(old_rel[1], 0, 2)][4] = 1      # wrong write on an inactive relation store
    out["compensating_write_pair"] = m_compensating_pair

    def m_fact_pair(r):
        led = {(x[1], x[2], x[3]): x for x in r["mechanism_events"]["N_old_fact"] if x[0] in ("w", "s")}
        led[(old_fact[0], 0, 1)][4] = 1
        nf = [s["index"] for s in fx["records"] if s["stage"] == "new" and s["domain"] == "fact"]
        led[(nf[0], 0, 1)][4] = 0
    out["cross_domain_pair"] = m_fact_pair

    out["teacher_time"] = lambda r: r["first"][5].__setitem__("teacher_at", r["first"][5]["teacher_at"] + 1.0)
    out["drop_ledger_row"] = lambda r: r["mechanism_events"]["W"].remove(rows(r, "W", "w")[10])
    out["duplicate_ledger_row"] = lambda r: r["mechanism_events"]["W"].append(list(rows(r, "W", "w")[3]))
    out["wrong_birth"] = lambda r: r["births"][1].__setitem__("canonical_B_sha256", "0" * 64)
    out["shared_birth"] = lambda r: r["births"].pop()
    out["world_id"] = lambda r: r.__setitem__("world", 190001)
    out["probe_correct_flag"] = lambda r: r["probes"][3].__setitem__("correct", 1 - r["probes"][3]["correct"])
    out["aggregate_count"] = lambda r: r["writes"]["W"].__setitem__("new_fact", 767)
    out["branch_label"] = lambda r: r["mechanism_events"].__setitem__(
        "N_old_rel", r["mechanism_events"]["W"])
    if arm in audit_a.R_ARMS:
        def m_private_gate(r):
            x = next(x for x in rows(r, "W", "R") if x[1] in old_fact)
            x[3] = 1 - x[3]
        out["private_gate_flip"] = m_private_gate
        out["novelty_branch_mismatch"] = lambda r: rows(r, "N_new_fact", "R")[7].__setitem__(2, 0.123)
        if arm in audit_a.SIGNED:
            def m_coeff(r):
                x = next(x for x in rows(r, "W", "s", 1) if x[4] == 1)
                x[7] = x[7] + 0.01
            out["signed_coefficient"] = m_coeff
        else:
            def m_private_write(r):
                x = next(x for x in rows(r, "W", "w", 1) if x[4] == 1)
                x[4] = 0
            out["private_write_flip"] = m_private_write
    if arm in audit_a.Z_ARMS:
        def m_z(r):
            x = rows(r, "W", "Z")[4]
            x[5] = x[4] * 1.5 + 1.0
        out["z_gate_exceeds_native"] = m_z
    if arm in audit_a.P_ARMS:
        def m_p(r):
            x = rows(r, "N_new_rel", "P")[9]
            x[4] = x[4] + (1.0 if arm == "P0" else 0.5)
        out["p_update_branch_or_install"] = m_p
    return out


def run(arms=None):
    results = {}
    for arm in arms or sorted(p.name for p in RES.iterdir() if p.is_dir()):
        base = load(arm)
        kw = kwargs_for(arm)
        audit_a.audit_receipt(base, **kw)
        res = {}
        for name, fn in mutations(arm, base).items():
            r = copy.deepcopy(base)
            fn(r)
            try:
                audit_a.audit_receipt(r, **kw)
                res[name] = "ACCEPTED"
            except (AssertionError, KeyError, IndexError, TypeError, StopIteration) as exc:
                res[name] = "rejected: " + str(exc)[:70]
        results[arm] = res
        bad = [k for k, v in res.items() if v == "ACCEPTED"]
        print(arm, f"{len(res) - len(bad)}/{len(res)} rejected", ("ACCEPTED: " + ",".join(bad)) if bad else "", flush=True)
    return results


if __name__ == "__main__":
    out = run(sys.argv[1:] or None)
    if any(v == "ACCEPTED" for r in out.values() for v in r.values()):
        raise SystemExit("a tampered receipt was accepted")
