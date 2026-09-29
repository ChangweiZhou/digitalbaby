"""Independent read-only audit of Package-B receipts (no learner is instantiated).

Fixture exposure, first-answer timing, write ledger, probe roster and scoring are
re-derived from the sealed fixture (the historical independent auditor's roster
reconstruction is reused for the probe order).  Family-specific structure is
checked against the pre-science GRAPH_MANIFEST and, for S arms, by rebuilding the
W-branch graph chain from the recorded partner changes.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

import paths  # noqa: F401
from audit_fixture import audit as audit_fixture
from audit_round import _expected_probes, _near, _values
from common_platform import BRANCHES, DT, RECORD_SECONDS, branch_allows, mode_from_bytes
from fixture import make_world
import topo_model as tm

HEX = re.compile(r"[0-9a-f]{64}\Z")
STAGE_COUNTS = {"old_end": 34, "old_day": 34, "new_end": 56, "final": 152}
CANONICAL_B = "32a3726cb25a8255c1aef5696ff40aa303138c5d061a598004d2087d4da5f964"


class AuditError(AssertionError):
    pass


def _need(cond, msg):
    if not cond:
        raise AuditError(msg)


def store_write_records(fixture: dict, branch: str, store: int) -> list[int]:
    """Record indices at which one store performs a write-enabled teach."""
    out = []
    for spec in fixture["records"]:
        if branch_allows(branch, spec["stage"], spec["domain"]) and (
                spec["domain"] == "fact" or store < 2):
            out.append(spec["index"])
    return out


def audit_common(r: dict, arm: str, world: int) -> dict:
    _need(r.get("schema") == "MINIFLY-B-TOPO-RECEIPT-v1", "receipt schema")
    _need(r.get("arm") == arm and r.get("world") == world and type(world) is int, "world/arm identity")
    fixture = make_world(world)
    audit_fixture(fixture)
    _need(r.get("fixture_digest") == fixture["digest"], "fixture digest")
    _need(r.get("branches") == list(BRANCHES), "branch roster")
    _need(math.isfinite(r.get("max_clock_error", float("nan"))) and 0 <= r["max_clock_error"] < 1e-6,
          "clock error")
    first = r.get("first")
    _need(isinstance(first, list) and len(first) == 600 * len(BRANCHES), "first-answer roster")
    for spec in fixture["records"]:
        i = spec["index"]
        at = i * RECORD_SECONDS + (86400.0 if spec["stage"] == "new" else 0.0) + 12 * DT
        dom = mode_from_bytes(bytes.fromhex(spec["cue_hex"]))
        _need(dom == spec["domain"], "byte-only mode")
        for j, b in enumerate(BRANCHES):
            g = first[i * len(BRANCHES) + j]
            _need(g.get("record") == i and g.get("branch") == b and
                  _near(g.get("teacher_at"), at, tolerance=1e-12) and
                  g.get("emitted") == _values(g.get("values"), 4 if dom == "fact" else 2),
                  "first-answer timing/readout")
    writes = {}
    for b in BRANCHES:
        c = Counter()
        for spec in fixture["records"]:
            if branch_allows(b, spec["stage"], spec["domain"]):
                c[f'{spec["stage"]}_{spec["domain"]}'] += 4 if spec["domain"] == "fact" else 2
        writes[b] = {k: c[k] for k in ("old_fact", "old_relation", "new_fact", "new_relation")}
    _need(r.get("writes") == writes, "branch write ledger")
    probes = r.get("probes")
    expected = list(_expected_probes(fixture))
    _need(isinstance(probes, list) and len(probes) == len(expected), "probe roster count")
    stage = Counter()
    for got, spec in zip(probes, expected, strict=True):
        for key, value in spec.items():
            _need(got.get(key) == value, f"probe roster/target field {key}")
        stage[(spec["stage"], spec["branch"])] += 1
        if spec["set"] == "old_relation_heldout":
            ov, sc = got.get("option_values"), got.get("option_scores")
            _need(isinstance(ov, list) and len(ov) == 2 and isinstance(sc, list) and len(sc) == 2,
                  "heldout values")
            exp = [v[1] - v[0] for v in ov]
            _need(all(_near(a, b, tolerance=1e-12) for a, b in zip(sc, exp)) and
                  got.get("emitted") == (ord("L") if sc[0] >= sc[1] else ord("R")), "heldout readout")
        else:
            n = 2 if "relation" in spec["set"] else 4
            _need(got.get("emitted") == _values(got.get("values"), n), "probe readout")
        _need(got.get("correct") == int(got["emitted"] == spec["target"]), "correctness flag")
    _need(all(stage[(s, b)] == n for s, n in STAGE_COUNTS.items() for b in BRANCHES), "stage completeness")
    d = r.get("end_state_digests")
    _need(isinstance(d, dict) and set(d) == set(BRANCHES) and
          all(len(v) == 4 and all(HEX.fullmatch(x) for x in v) for v in d.values()), "end digests")
    birth = r["construct"]["birth"]
    _need(len(birth) == 4 and all(b["canonical_B_sha256"] == CANONICAL_B for b in birth) and
          len({b["installed_graph_digest"] for b in birth}) == 1 and
          len({b["raw_B_sha256"] for b in birth}) == 1, "four-store canonical birth")
    return fixture


def audit_receipt(r: dict, arm: str, world: int, *, manifest: dict, lock: dict | None = None,
                  candidate: dict | None = None, support: "tm.Support | None" = None) -> dict:
    fixture = audit_common(r, arm, world)
    if lock is not None:
        _need(r.get("lock_digest") == lock["lock_digest"], "lock digest")
        for name, h in r["source_sha256"].items():
            _need(lock["files"].get(name) == h, f"source hash {name}")
    birth_graph = r["construct"]["birth"][0]["installed_graph_digest"]
    endg = r["end_graph_digests"]
    _need(set(endg) == set(BRANCHES) and all(len(v) == 4 for v in endg.values()), "end graph roster")
    ev = r["mechanism_events"]
    _need(set(ev) == set(BRANCHES), "event branches")
    wkey = str(world)
    if arm in tm.T_ARMS:
        fam = "T1" if arm in ("T1", "T0_1") else "T3"
        m = manifest["T"][fam][wkey]
        _need(r["construct"]["graph_receipt"]["graph_digest"][arm] == m["graph_digest"][arm],
              "T graph digest vs pre-science manifest")
        _need(birth_graph == m["topo_digest"][arm], "installed T graph bytes")
        _need(all(g == birth_graph for v in endg.values() for g in v), "T graph changed during life")
        _need(all(rows == [] for rows in ev.values()), "T arm carried structural events")
        return {"pass": True, "arm": arm, "world": world}
    # ---------------- S family
    _need(r["construct"]["support_digest"] == manifest["S"][wkey]["support_digest"], "S support digest")
    _need(birth_graph == manifest["S"][wkey]["birth_topo_digest"], "S birth graph")
    sched = {}
    for b in BRANCHES:
        rows = ev[b]
        by_store = {j: [e for e in rows if e["store"] == j] for j in range(4)}
        for j in range(4):
            recs = store_write_records(fixture, b, j)
            n_ev = len(recs) // tm.K_EVENT
            got = by_store[j]
            _need(len(got) == n_ev, f"S event count {b}/{j}")
            for n, e in enumerate(got, start=1):
                idx = recs[n * tm.K_EVENT - 1]
                spec = fixture["records"][idx]
                at = idx * RECORD_SECONDS + (86400.0 if spec["stage"] == "new" else 0.0) + 12 * DT
                _need(e["n"] == n and e["context"][1] == b and e["context"][2] == idx and
                      _near(e["t"], at, tolerance=1e-12), f"S event clock {b}/{j}/{n}")
                rule = "Srand" if arm.startswith("Srand") else arm
                _need(e["rule"] == rule, "S rule label")
                if "changes" in e:
                    ch = e["changes"]
                    nc = sum(1 for c in ch if c[3] != "accept")
                    _audit_change_semantics(arm, ch)
                else:
                    nc = e["n_changes"]
                _need(0 <= nc <= 2 * tm.R_MAX and (arm in ("S4", "Srand_4") or nc <= tm.R_MAX),
                      "S per-event budget")
                if arm == "S0":
                    _need(nc == 0 and e["graph"] == birth_graph, "S0 changed its graph")
                sched[(b, j, n)] = nc
            last = got[-1]["graph"] if got else birth_graph
            _need(endg[b][j] == last, f"S end graph vs last event {b}/{j}")
    if arm.startswith("Srand"):
        _need(candidate is not None, "Srand requires its candidate receipt")
        csched = {}
        for b, rows in candidate["mechanism_events"].items():
            for e in rows:
                csched[(b, e["store"], e["n"])] = (e["n_changes"] if "n_changes" in e else
                                                   sum(1 for c in e["changes"] if c[3] != "accept"))
        _need(csched == sched, "Srand realised count/timing != candidate")
    if support is not None:
        _rebuild_chain(support, ev["W"], birth_graph)
    if arm not in ("S0",):
        moved = sum(sched.values())
        _need(moved > 0 and any(g != birth_graph for g in endg["W"]), "S arm never changed indices")
    return {"pass": True, "arm": arm, "world": world, "partner_changes": sum(sched.values())}


def _audit_change_semantics(arm: str, ch: list) -> None:
    for kc, p_old, p_new, kind, eo, en in ch:
        _need(p_old != p_new and all(math.isfinite(v) for v in (eo, en)), "change row")
        if arm == "S1":
            _need(kind == "set" and eo == en, "S1 weight carried")
        elif arm == "S2":
            _need(kind == "coact" and en > eo, "S2 evidence ordering")
        elif arm == "S3":
            _need(kind == "homeo" and en != eo, "S3 homeostatic direction")
        elif arm == "S4":
            _need(kind in ("tentative", "accept", "rollback"), "S4 kind")
            if kind == "accept":
                _need(en > eo, "S4 accept rule")
            if kind == "rollback":
                _need(not (eo > en), "S4 rollback rule")
        elif arm.startswith("Srand"):
            _need(kind == "random" and eo == en, "Srand kind")


def _rebuild_chain(sup, rows, birth_graph) -> None:
    """Replay recorded W partner changes on the birth slots; each digest must match."""
    for j in range(4):
        on, epos, sw = sup.init_on.copy(), sup.init_epos.copy(), sup.init_w.copy()
        g = birth_graph
        for e in (x for x in rows if x["store"] == j):
            for kc, p_old, p_new, kind, _, _ in e["changes"]:
                if kind == "accept":
                    continue
                lo, hi = int(sup.kc_ptr[kc]), int(sup.kc_ptr[kc + 1])
                slots = {int(sup.pn[s]): s for s in range(lo, hi)}
                _need(p_old in slots and p_new in slots, "change partner outside the KC's birth support")
                so, sn = slots[p_old], slots[p_new]
                _need(on[so] and not on[sn], "replayed change illegal")
                on[so], on[sn] = False, True
                epos[sn], epos[so] = epos[so], -1
                sw[sn], sw[so] = sw[so], 0.0
            if e["changes"]:
                g = tm.graph_bytes_digest(tm.build_csr(sup, on, epos, sw))
            _need(g == e["graph"], "replayed W graph digest mismatch")
