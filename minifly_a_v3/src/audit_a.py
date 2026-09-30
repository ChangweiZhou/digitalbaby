"""Independent auditor for Package A V3-CLAUDE receipts.

Never imports runner.py, systems.py, stores.py or the platform's branch_allows. Expected write decisions come
from the sealed fixture and the literal branch matrix below; aggregates are also passed through the V3
causal_branch_gate. R1_rand placement and the R3_randtarget derangement are re-derived here from their SHA256
definitions in SPEC_DRAFT.md.
"""
from __future__ import annotations

import hashlib
import math
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
sys.path.insert(0, str(paths.PKG))
import causal_branch_gate as gate  # noqa: E402
from audit_round import _expected_probes  # noqa: E402  (historical roster reconstruction; no model import)
from fixture import DT, RECORD_SECONDS, make_world  # noqa: E402

BRANCHES = ("W", "N_old_fact", "N_old_rel", "N_new_fact", "N_new_rel")
BLOCKED = {"W": None, "N_old_fact": ("old", "fact"), "N_old_rel": ("old", "relation"),
           "N_new_fact": ("new", "fact"), "N_new_rel": ("new", "relation")}
CANONICAL_B = "32a3726cb25a8255c1aef5696ff40aa303138c5d061a598004d2087d4da5f964"
R_ARMS = ("R0", "R1", "R1_rand", "R0_signed", "R3", "R3_randtarget")
SIGNED = ("R0_signed", "R3", "R3_randtarget")
Z_ARMS = ("Z0_resource", "Z2", "Z2_rand")
P_ARMS = ("P0", "P1", "P2", "P4")
STAGE_COUNTS = {"old_end": 34, "old_day": 34, "new_end": 56, "final": 152}
CH = b"0123"


class AuditError(AssertionError):
    pass


def need(cond, msg):
    if not cond:
        raise AuditError(msg)


def expected_write(branch, stage, domain, store) -> bool:
    return BLOCKED[branch] != (stage, domain) and (domain == "fact" or store < 2)


def teacher_time(i, stage):
    return i * RECORD_SECONDS + (86400.0 if stage == "new" else 0.0) + 12 * DT


def derangement(n, key):
    order = sorted(range(n), key=lambda j: hashlib.sha256(f"{key}|{j}".encode()).digest())
    perm = [0] * n
    for i in range(n):
        perm[order[i]] = order[(i + 1) % n]
    return perm


def r1_rand_gates(fixture, branch, r1_rows):
    rows = {r[1]: r for r in r1_rows if r[0] == "R"}
    eligible, counts = {}, Counter()
    for rec in fixture["records"]:
        i = rec["index"]
        if rows[i][4]:
            key = (rec["stage"], rec["domain"], rec["answer"])
            eligible.setdefault(key, []).append(i)
            counts[key] += rows[i][3]
    chosen = set()
    for key, idx in eligible.items():
        stage, domain, answer = key
        chosen.update(sorted(idx, key=lambda i: hashlib.sha256(
            f"A3-R1RAND-v1|{fixture['world']}|{branch}|{stage}|{domain}|{answer}|{i}".encode()).digest())
            [:counts[key]])
    return chosen


def audit_receipt(r: dict, *, r1_receipt: dict | None = None, theta: float | None = None,
                  scales: tuple | None = None, z2_receipt: dict | None = None) -> dict:
    arm, world = r.get("arm"), r.get("world")
    need(r.get("schema") == "MINIFLY-A3-CLAUDE-RECEIPT-v1", "schema")
    need(arm in R_ARMS + Z_ARMS + P_ARMS and type(world) is int, "arm/world identity")
    fixture = make_world(world)
    need(r.get("fixture_digest") == fixture["digest"], "fixture digest")
    need(r.get("branches") == list(BRANCHES), "branch roster")
    gate.check_receipt(r)                                   # V3 literal aggregate gate
    nbanks = 2 if arm in R_ARMS else 1
    births = r.get("births", [])
    need(len(births) == 4 * nbanks and all(b["canonical_B_sha256"] == CANONICAL_B for b in births) and
         len({b["raw_B_sha256"] for b in births}) == 1 and
         len({(b["bank"], b["store"]) for b in births}) == 4 * nbanks, "separate canonical births")
    need(math.isfinite(r.get("max_clock_error", float("nan"))) and r["max_clock_error"] < 1e-6, "clock")
    # first responses: identical exposure/timing in every branch
    first = r["first"]
    need(len(first) == 3000, "first-response roster")
    for row in first:
        spec = fixture["records"][row["record"]]
        need(abs(row["teacher_at"] - teacher_time(row["record"], spec["stage"])) < 1e-9, "teacher time")
    # probes
    probes, expected = r["probes"], list(_expected_probes(fixture))
    need(len(probes) == len(expected), "probe count")
    stage = Counter()
    for got, spec in zip(probes, expected, strict=True):
        for k, v in spec.items():
            need(got.get(k) == v, f"probe field {k}")
        need(got["correct"] == int(got["emitted"] == spec["target"]), "probe correctness flag")
        stage[(spec["stage"], spec["branch"])] += 1
    need(all(stage[(s, b)] == n for s, n in STAGE_COUNTS.items() for b in BRANCHES), "stage completeness")
    ev = r["mechanism_events"]
    need(set(ev) == set(BRANCHES), "event branches")
    rrows_by_branch, prow_by_branch = {}, {}
    for b in BRANCHES:
        rows = ev[b]
        ledger = {}
        for row in rows:
            if row[0] in ("w", "s"):
                key = (row[1], row[2], row[3])
                need(key not in ledger, f"duplicate ledger row {b}/{key}")
                ledger[key] = row
        need(len(ledger) == 600 * 4 * nbanks, f"ledger incomplete in {b}")
        counts = Counter()
        rrows = {row[1]: row for row in rows if row[0] == "R"}
        rrows_by_branch[b] = rrows
        if arm in R_ARMS:
            need(sorted(rrows) == list(range(600)), f"R rows incomplete in {b}")
        rand = r1_rand_gates(fixture, b, r1_receipt["mechanism_events"][b]) if arm == "R1_rand" else None
        for rec in fixture["records"]:
            i, st, dom, ans = rec["index"], rec["stage"], rec["domain"], rec["answer"]
            n = 4 if dom == "fact" else 2
            for j in range(4):
                exp = expected_write(b, st, dom, j)
                row = ledger.get((i, 0, j))
                need(row is not None and row[0] == "w" and row[4] == int(exp), f"shared/native write {b}/{i}/{j}")
                need(row[4] == 1 or row[5] == 0.0, f"write-free store changed alpha {b}/{i}/{j}")
                if exp:
                    counts[f"{st}_{dom}"] += 1
            if arm not in R_ARMS:
                continue
            rr = rrows[i]
            need(rr[4] == int(BLOCKED[b] != (st, dom)), f"R permission flag {b}/{i}")
            if arm == "R1":
                need(theta is not None and rr[3] == int(rr[2] > theta), f"R1 gate {b}/{i}")
            elif arm == "R1_rand":
                need(rr[3] == int(i in rand), f"R1_rand gate {b}/{i}")
            else:
                need(rr[3] == 1, f"ungated R arm gate {b}/{i}")
            coeff = None
            if arm in SIGNED:
                v = [x / scales[0] for x in rr[5][:n]]
                mx = max(v)
                e = [math.exp(x - mx) for x in v]
                p = [x / sum(e) for x in e]
                y = [1.0 if CH[k] == ans else 0.0 for k in range(n)]
                if arm == "R0_signed":
                    coeff = [1.0] + [1.0 - yk for yk in y]
                else:
                    s = [p[k] - y[k] for k in range(n)]
                    if arm == "R3_randtarget":
                        perm = derangement(n, f"A3-R3RT-v1|{world}|{b}|{i}")
                        s = [s[perm[k]] for k in range(n)]
                    coeff = [0.0] + s
                need(rr[6] is not None and len(rr[6]) == n + 1 and
                     all(abs(a - c) <= 1e-12 for a, c in zip(rr[6], coeff)), f"signed coefficients {b}/{i}")
            for j in range(4):
                exp = expected_write(b, st, dom, j)
                row = ledger.get((i, 1, j))
                need(row is not None, f"private row {b}/{i}/{j}")
                if arm in SIGNED:
                    need(row[0] == "s" and row[4] == int(exp), f"private signed write {b}/{i}/{j}")
                    sj = coeff[1 + j] if j < n else 0.0
                    need(row[6] == coeff[0] and abs(row[7] - sj) <= 1e-12, f"private coefficients {b}/{i}/{j}")
                    need(row[4] == 1 or row[5] == 0.0, f"private alpha without write {b}/{i}/{j}")
                    need(not (coeff[0] == 0.0 and sj == 0.0) or row[5] == 0.0, f"zero residual wrote {b}/{i}/{j}")
                else:
                    need(row[0] == "w" and row[4] == int(exp and rr[3] == 1), f"private write {b}/{i}/{j}")
                    need(row[4] == 1 or row[5] == 0.0, f"private alpha without write {b}/{i}/{j}")
        need(dict(counts) == {k: v for k, v in r["writes"][b].items() if v} and
             all(counts.get(k, 0) == v for k, v in r["writes"][b].items()), f"aggregate vs ledger {b}")
        if arm in Z_ARMS:
            zrows = [row for row in rows if row[0] == "Z"]
            permitted = sum(1 for (i, _, j), row in ledger.items() if row[4] == 1)
            need(0 < len(zrows) <= permitted, f"Z event rows {b}")
            zkeys = {(z[1], z[2]) for z in zrows}
            need(len(zkeys) == len(zrows), f"duplicate Z rows {b}")
            ref = None
            if arm == "Z2_rand":
                # EXTERNAL reference: the paired Z2 receipt's realised per-bucket dose, same world/branch/record/store
                need(z2_receipt is not None and z2_receipt.get("arm") == "Z2" and z2_receipt["world"] == world,
                     "Z2_rand needs its paired Z2 receipt")
                ref = {(z[1], z[2]): z for z in z2_receipt["mechanism_events"][b] if z[0] == "Z"}
                need(set(ref) == zkeys, f"Z2_rand event set differs from paired Z2 in {b}")
            for z in zrows:
                _, i, j, conflicts, native, gated, target, buckets = z
                tol = 1e-9 * max(1.0, native)
                need(ledger[(i, 0, j)][4] == 1 and 0 <= gated <= native * (1 + 1e-12) + 1e-15, f"Z gate bounds {b}/{i}/{j}")
                need(isinstance(buckets, dict) and set(buckets) == {"0:-1", "0:+1", "1:-1", "1:+1"} and
                     all(0 <= v[2] <= v[1] * (1 + 1e-12) + 1e-15 for v in buckets.values()) and
                     abs(sum(v[1] for v in buckets.values()) - native) <= tol and
                     abs(sum(v[2] for v in buckets.values()) - gated) <= tol, f"Z bucket accounting {b}/{i}/{j}")
                if arm == "Z0_resource":
                    need(abs(gated - native) <= 1e-12 * max(1.0, native), f"Z0 changed the split {b}/{i}/{j}")
                elif arm == "Z2":
                    need(abs(gated - target) <= tol, f"Z2 gate L1 {b}/{i}/{j}")
                else:
                    zb = ref[(i, j)][7]
                    need(all(abs(buckets[k][2] - zb[k][2]) <= 1e-9 * max(1.0, zb[k][2]) for k in zb) and
                         abs(target - sum(v[2] for v in zb.values())) <= tol,
                         f"Z2_rand dose differs from paired Z2 {b}/{i}/{j}")
        if arm in P_ARMS:
            prow = [row[1:] for row in rows if row[0] == "P"]
            need(len(prow) == 600 * 4, f"P rows {b}")
            need(all(math.isfinite(x[4]) and x[5] > 0 for x in prow), f"P weights bounded/positive {b}")
            if arm == "P0":
                need(all(x[3] == 0.0 for x in prow), f"P0 installed a weight change {b}")
            else:
                need(sum(x[3] for x in prow) > 0, f"{arm} made no weight change {b}")
            prow_by_branch[b] = prow
    if arm in R_ARMS:
        nov = {b: [rrows_by_branch[b][i][2] for i in range(600)] for b in BRANCHES}
        need(all(nov[b] == nov["W"] for b in BRANCHES), "novelty differs across branches")
    if arm in P_ARMS:
        need(all(prow_by_branch[b] == prow_by_branch["W"] for b in BRANCHES), "P updates differ across branches")
    d = r["end_state_digests"]
    need(set(d) == set(BRANCHES) and len({tuple(v) for v in d.values()}) == len(BRANCHES), "end digests")
    return {"pass": True, "arm": arm, "world": world, "ledger_rows": 600 * 4 * nbanks * 5}
