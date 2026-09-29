"""Technical qualification on world 190000 -> results/technical/TECHNICAL_AUDIT.json."""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

import audit_topo as au  # noqa: E402
import topo_model as tm  # noqa: E402

TECH = paths.ROOT / "results" / "technical"
W = 190000
ROSTER = ("T1", "T0_1", "T3", "T0_3", "S0", "S1", "S2", "S3", "S4",
          "Srand_1", "Srand_2", "Srand_3", "Srand_4")


def load(arm):
    raw = (TECH / arm / f"{W}.json.gz").read_bytes()
    return json.loads(gzip.decompress(raw)), hashlib.sha256(raw).hexdigest()


def replay(arm: str) -> dict:
    import runner
    from common_platform import run_fourstore_life
    actor, _ = runner.construct(arm, W, technical=True)
    doc = run_fourstore_life(W, actor, technical=True)
    ref, _ = load(arm)
    same = (doc["end_state_digests"] == ref["end_state_digests"] and doc["probes"] == ref["probes"]
            and doc["first"] == ref["first"])
    return {"arm": arm, "bit_identical_replay": same}


def expect_reject(fn, label):
    try:
        fn()
    except AssertionError as exc:
        return {"tamper": label, "rejected": True, "error": str(exc)[:120]}
    return {"tamper": label, "rejected": False}


def main() -> None:
    manifest = json.loads((paths.ROOT / "GRAPH_MANIFEST.json").read_text())
    import portable_birth as pb
    base, raw = pb.canonical_fresh_native()
    sup = tm.Support(base.fly.m.B, W)
    receipts, hashes, audits = {}, {}, {}
    for arm in ROSTER:
        r, h = load(arm)
        receipts[arm], hashes[arm] = r, h
    for arm in ROSTER:
        cand = receipts[f"S{arm[-1]}"] if arm.startswith("Srand") else None
        audits[arm] = au.audit_receipt(receipts[arm], arm, W, manifest=manifest, candidate=cand,
                                       support=sup if arm in tm.S_ARMS else None)
    # ---------------- deliberate corruptions
    tam = []
    t1, s2, sr = receipts["T1"], receipts["S2"], receipts["Srand_2"]

    def mut(r, f):
        x = copy.deepcopy(r)
        f(x)
        return x
    A = lambda r, arm, **kw: au.audit_receipt(r, arm, W, manifest=manifest, **kw)  # noqa: E731
    tam.append(expect_reject(lambda: A(mut(t1, lambda x: x["probes"][100].__setitem__(
        "target", x["probes"][100]["target"] ^ 1)), "T1"), "answer/target"))
    tam.append(expect_reject(lambda: A(mut(t1, lambda x: x["first"][7].__setitem__("branch", "W")), "T1"),
                             "branch label"))
    tam.append(expect_reject(lambda: au.audit_receipt(mut(t1, lambda x: x.__setitem__("world", 190001)),
                                                      "T1", 190001, manifest=manifest), "world ID"))
    tam.append(expect_reject(lambda: A(mut(t1, lambda x: x["probes"].pop(500)), "T1"), "dropped probe"))
    tam.append(expect_reject(lambda: A(mut(t1, lambda x: x["probes"][1300].__setitem__(
        "correct", 1 - x["probes"][1300]["correct"])), "T1"), "score flag"))
    tam.append(expect_reject(lambda: A(mut(t1, lambda x: x["end_graph_digests"]["N_old_rel"].__setitem__(
        2, "0" * 64)), "T1"), "wrong T graph (changed in life)"))
    tam.append(expect_reject(lambda: A(mut(t1, lambda x: x["construct"]["graph_receipt"]["graph_digest"]
                                           .__setitem__("T1", "f" * 64)), "T1"), "wrong T graph (manifest)"))

    def swap_partner(x):
        e = next(e for e in x["mechanism_events"]["W"] if e["changes"])
        e["changes"][0][2] = (e["changes"][0][2] + 1) % 302
    tam.append(expect_reject(lambda: A(mut(s2, swap_partner), "S2", support=sup), "wrong S partner"))
    tam.append(expect_reject(lambda: A(mut(s2, lambda x: x["mechanism_events"]["N_new_fact"].pop()),
                                       "S2", support=sup), "dropped S event"))
    tam.append(expect_reject(lambda: A(mut(sr, lambda x: x["mechanism_events"]["W"][0]["changes"].pop()),
                                       "Srand_2", candidate=s2), "Srand count != candidate"))
    fake_lock = {"lock_digest": "x", "files": dict(t1["source_sha256"])}
    t1l = mut(t1, lambda x: x.__setitem__("lock_digest", "x"))
    first = sorted(fake_lock["files"])[0]
    bad_lock = {"lock_digest": "x", "files": {**fake_lock["files"], first: "0" * 64}}
    tam.append(expect_reject(lambda: A(t1l, "T1", lock=bad_lock), "wrong source hash"))
    ok_lock = A(t1l, "T1", lock=fake_lock)["pass"]
    # ---------------- full-life replay parity
    with ProcessPoolExecutor(2) as ex:
        reps = list(ex.map(replay, ("S4", "T1")))
    # ---------------- structure summaries
    s_summary = {}
    for arm in ROSTER:
        r = receipts[arm]
        if arm in tm.S_ARMS:
            nch = [len([c for c in e["changes"] if c[3] != "accept"])
                   for rows in r["mechanism_events"].values() for e in rows]
            s_summary[arm] = {"structural_events": len(nch), "partner_changes": sum(nch),
                              "W_end_graph_changed_stores": sum(
                                  g != r["construct"]["birth"][0]["installed_graph_digest"]
                                  for g in r["end_graph_digests"]["W"])}
    res = {arm: {"learner_life_s": receipts[arm]["resources"]["learner_life_s"],
                 "total_wall_s": receipts[arm]["resources"]["total_wall_s"],
                 "graph_generation_s": receipts[arm]["construct"].get("graph_generation_s"),
                 "graph_audit_s": receipts[arm]["construct"].get("graph_audit_s"),
                 "peak_rss_bytes": receipts[arm]["resources"]["peak_rss_bytes"],
                 "receipt_bytes_technical": receipts[arm]["resources"]["receipt_bytes"]}
           for arm in ROSTER}
    verify = json.loads((paths.ROOT / "audit" / "verify_bundle_v2.json").read_text())
    unit = json.loads((TECH / "UNIT_TESTS.json").read_text())
    doc = {"schema": "MINIFLY-B-TECHNICAL-AUDIT-v1", "world": W,
           "verify_bundle_v2": verify, "raw_B_sha256_host": raw, "canonical_B_sha256": pb.EXPECTED_B,
           "unit_tests": unit, "receipt_audits": audits, "receipt_sha256": hashes,
           "tamper_tests": tam, "true_lock_accepted": ok_lock, "full_life_replay": reps,
           "s_structure": s_summary, "resources": res,
           "technical_score_used_for_tuning": False}
    doc["pass"] = (unit["pass"] and all(a["pass"] for a in audits.values()) and
                   all(t["rejected"] for t in tam) and ok_lock and all(r["bit_identical_replay"] for r in reps)
                   and all(v["W_end_graph_changed_stores"] > 0 for k, v in s_summary.items() if k != "S0"))
    (TECH / "TECHNICAL_AUDIT.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"pass": doc["pass"], "tamper": [(t["tamper"], t["rejected"]) for t in tam],
                      "replay": reps, "s_structure": s_summary}, indent=1))


if __name__ == "__main__":
    main()
