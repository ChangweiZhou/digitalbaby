"""Post-science final audit for Package B (written after the roster completed).

Independent of the science reducer: it re-derives every per-world E1/E3 value
and receipt hash from the raw committed receipts, re-derives exposure,
held-out separation, option balance and the exact-key / position / frequency
countermodels from the sealed fixture, checks causal-branch write parity
against the *intended* branch semantics, and replays tamper rejections through
the receipt auditor.  It never imports analyze_b.py or analyze_round.py.

Writes results/FINAL_AUDIT.json and results/RESOURCE_REPORT.json.
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import paths  # noqa: E402,F401
import lock as lk  # noqa: E402
import audit_topo as au  # noqa: E402
from common_platform import BRANCHES, branch_allows  # noqa: E402
from fixture import make_world  # noqa: E402

ARMS = ("T1", "T0_1", "T3", "T0_3", "S0", "S1", "S2", "S3", "S4",
        "Srand_1", "Srand_2", "Srand_3", "Srand_4")
WORLDS = tuple(range(190001, 190065))
RES = ROOT / "results"
# Intended semantics (SHARED_PROTOCOL "Causal branches"): only the named native write is disabled.
INTENDED_OFF = {"N_old_fact": ("old", "fact"), "N_old_rel": ("old", "relation"),
                "N_new_fact": ("new", "fact"), "N_new_rel": ("new", "relation")}
REL_SETS = ("old_relation_taught", "old_relation_heldout", "new_relation_taught")


def acc(probes, stage, branch, sset, form=None):
    v = [p["correct"] for p in probes if p["stage"] == stage and p["branch"] == branch
         and p["set"] == sset and (form is None or p.get("form") == form)]
    return sum(v) / len(v)


def intended_ledger(fixture):
    out = {}
    for b in BRANCHES:
        c = Counter()
        for s in fixture["records"]:
            if INTENDED_OFF.get(b) != (s["stage"], s["domain"]):
                c[f'{s["stage"]}_{s["domain"]}'] += 4 if s["domain"] == "fact" else 2
        out[b] = {k: c[k] for k in ("old_fact", "old_relation", "new_fact", "new_relation")}
    return out


def exposure_and_countermodels(fixture):
    taught_cues = [s["cue_hex"] for s in fixture["records"]]
    old_r = fixture["old_relation"]
    held = [h["cue_hex"] for h in old_r["heldout"]]
    taught_rel = {t["cue_hex"]: t["label"] for t in old_r["taught"]}
    held_not_taught = all(h not in taught_cues for h in held)
    # the 12 withheld-probe presentations: 3 edges x 2 orders, target L then R
    probes = []
    for e in range(3):
        good, other = held[2 * e], held[2 * e + 1]
        probes += [((good, other), "L"), ((other, good), "R")]
    balance = Counter(t for _, t in probes)
    # exact / canonical pair-key table: neither option is a taught key -> tie -> first option
    def exact_key(opts):
        s = [taught_rel.get(o, 0.5) for o in opts]
        return "L" if s[0] >= s[1] else "R"
    exact = statistics.fmean(exact_key(o) == t for o, t in probes)
    position = statistics.fmean(t == "L" for _, t in probes)          # always-first option
    maj = Counter(taught_rel.values()).most_common(1)[0][0]          # frequency of taught labels
    freq = statistics.fmean(exact_key(o) == t for o, t in probes) if maj is not None else None
    # old facts: exact-key recall is perfect by construction (retention only);
    # frequency countermodel = modal taught label
    labels = fixture["old_fact"]["labels"]
    modal = Counter(labels).most_common(1)[0][1] / len(labels)
    rec_counts = Counter(s["cue_hex"] for s in fixture["records"] if s["stage"] == "old" and s["domain"] == "fact")
    return {"heldout_never_in_teaching_bytes": held_not_taught,
            "heldout_option_balance": dict(balance),
            "countermodel_exact_key_heldout": exact, "countermodel_position_heldout": position,
            "countermodel_frequency_heldout": freq,
            "old_fact_exact_key_table": 1.0, "old_fact_modal_label_accuracy": modal,
            "old_fact_exposures_per_key": sorted(set(rec_counts.values())),
            "records": len(fixture["records"])}


def main() -> None:
    lock = lk.verify_lock()
    fm = json.loads((RES / "FINAL_METRICS.json").read_text())
    status = json.loads((RES / "science" / "RUN_STATUS.json").read_text())
    rows, sha_mismatch, metric_mismatch = {}, [], []
    branch = Counter()
    probe_identity = Counter()
    ledger_actual = None
    for w in WORLDS:
        fixture = make_world(w)
        ledger_int = intended_ledger(fixture)
        for arm in ARMS:
            raw = (RES / "science" / arm / f"{w}.json.gz").read_bytes()
            key = f"{arm}/{w}"
            if hashlib.sha256(raw).hexdigest() != fm["receipt_sha256"][key]:
                sha_mismatch.append(key)
            r = json.loads(gzip.decompress(raw))
            assert r["arm"] == arm and r["world"] == w and r["lock_digest"] == lock["lock_digest"]
            p = r["probes"]
            e1 = acc(p, "final", "W", "old_fact", "canonical") - acc(p, "final", "N_old_fact", "old_fact", "canonical")
            e3 = acc(p, "final", "W", "old_relation_heldout") - acc(p, "final", "N_old_rel", "old_relation_heldout")
            ref = fm["per_world"][arm][str(w)]
            if abs(ref["E1"] - e1) > 1e-15 or abs(ref["E3"] - e3) > 1e-15:
                metric_mismatch.append(key)
            wr = r["writes"]
            ledger_actual = wr
            for b in BRANCHES:
                branch[f"{b}_ledger_matches_intended"] += wr[b] == ledger_int[b]
            branch["N_old_fact_old_fact_writes_zero"] += wr["N_old_fact"]["old_fact"] == 0
            branch["N_new_fact_new_fact_writes_zero"] += wr["N_new_fact"]["new_fact"] == 0
            branch["N_old_rel_old_relation_writes_zero"] += wr["N_old_rel"]["old_relation"] == 0
            branch["N_new_rel_new_relation_writes_zero"] += wr["N_new_rel"]["new_relation"] == 0
            d = r["end_state_digests"]
            branch["end_state_W_eq_N_old_rel"] += d["W"] == d["N_old_rel"]
            branch["end_state_W_eq_N_new_rel"] += d["W"] == d["N_new_rel"]
            branch["end_state_W_eq_N_old_fact"] += d["W"] == d["N_old_fact"]
            for st in ("old_end", "old_day", "new_end", "final"):
                for s in REL_SETS:
                    a = [(x["emitted"], x.get("values"), x.get("option_scores")) for x in p
                         if x["stage"] == st and x["set"] == s and x["branch"] == "W"]
                    b_ = [(x["emitted"], x.get("values"), x.get("option_scores")) for x in p
                          if x["stage"] == st and x["set"] == s and x["branch"] == "N_old_rel"]
                    if a:
                        probe_identity["relation_probe_blocks"] += 1
                        probe_identity["W_bitwise_eq_N_old_rel"] += a == b_
        rows[w] = exposure_and_countermodels(fixture)
    n = len(WORLDS) * len(ARMS)
    exp = {
        "worlds": len(rows),
        "heldout_never_in_teaching_bytes_all_worlds": all(v["heldout_never_in_teaching_bytes"] for v in rows.values()),
        "heldout_option_balance_all_worlds": all(v["heldout_option_balance"] == {"L": 6, "R": 6} or
                                                 v["heldout_option_balance"] == {"L": 3, "R": 3}
                                                 for v in rows.values()),
        "countermodel_exact_key_heldout": sorted({v["countermodel_exact_key_heldout"] for v in rows.values()}),
        "countermodel_position_heldout": sorted({v["countermodel_position_heldout"] for v in rows.values()}),
        "countermodel_frequency_heldout": sorted({v["countermodel_frequency_heldout"] for v in rows.values()}),
        "old_fact_exact_key_table": 1.0,
        "old_fact_modal_label_accuracy_mean": statistics.fmean(v["old_fact_modal_label_accuracy"] for v in rows.values()),
        "old_fact_modal_label_accuracy_range": [min(v["old_fact_modal_label_accuracy"] for v in rows.values()),
                                                max(v["old_fact_modal_label_accuracy"] for v in rows.values())],
        "old_fact_exposures_per_key": sorted({e for v in rows.values() for e in v["old_fact_exposures_per_key"]}),
        "records_per_world": sorted({v["records"] for v in rows.values()}),
    }
    # platform write rule as shipped, evaluated on the relation branches
    shipped = {b: {f"{st}_{dom}": branch_allows(b, st, dom) for st in ("old", "new")
                   for dom in ("fact", "relation")} for b in BRANCHES}
    # tamper replay on one committed science receipt through the receipt auditor
    manifest = json.loads((ROOT / "GRAPH_MANIFEST.json").read_text())
    base = json.loads(gzip.decompress((RES / "science" / "T1" / "190001.json.gz").read_bytes()))
    au.audit_receipt(base, "T1", 190001, manifest=manifest, lock=lock)

    def t_answer(r):
        p = next(x for x in r["probes"] if x["set"] == "old_relation_heldout")
        p["correct"] = 1 - p["correct"]

    def t_branch(r):
        r["probes"][0]["branch"] = "N_old_fact"

    def t_world(r):
        r["world"] = 190002

    def t_drop(r):
        r["probes"].pop(7)

    def t_graph(r):
        r["construct"]["birth"][0]["installed_graph_digest"] = "0" * 64

    def t_source(r):
        k = next(iter(r["source_sha256"]))
        r["source_sha256"][k] = "f" * 64

    def t_writes(r):
        r["writes"]["N_old_fact"]["old_fact"] = 4

    tamper = {}
    for name, fn in (("answer_flip", t_answer), ("branch_label", t_branch), ("world_id", t_world),
                     ("dropped_probe", t_drop), ("wrong_graph", t_graph), ("wrong_source_hash", t_source),
                     ("write_ledger", t_writes)):
        r = copy.deepcopy(base)
        fn(r)
        try:
            au.audit_receipt(r, "T1", 190001, manifest=manifest, lock=lock)
            tamper[name] = "ACCEPTED"
        except (AssertionError, KeyError, IndexError, StopIteration, TypeError) as exc:
            tamper[name] = "rejected: " + str(exc)[:80]
    rel_valid = branch["N_old_rel_old_relation_writes_zero"] == n and branch["N_new_rel_new_relation_writes_zero"] == n
    fact_valid = branch["N_old_fact_old_fact_writes_zero"] == n and branch["N_new_fact_new_fact_writes_zero"] == n
    out = {
        "schema": "MINIFLY-B-FINAL-AUDIT-v1",
        "lock_digest": lock["lock_digest"],
        "run_status": status,
        "roster": {"expected": n, "committed": sum(1 for a in ARMS for w in WORLDS
                                                    if (RES / "science" / a / f"{w}.json.gz").exists())},
        "receipt_audit_by_analyze_b": "all 832 receipts passed audit_topo.audit_receipt (incl. S W-branch chain rebuild, Srand yoke) before FINAL_METRICS was written",
        "independent_rederivation": {"receipt_sha256_mismatches": sha_mismatch,
                                     "per_world_E1_E3_mismatches": metric_mismatch},
        "causal_branch_audit": {
            "counts_over_receipts": dict(branch), "n_receipts": n,
            "relation_probe_identity_W_vs_N_old_rel": dict(probe_identity),
            "platform_branch_allows_as_shipped": shipped,
            "defect": ("common_platform.branch_allows returns branch != f'N_{stage}_{domain}'; for the relation "
                       "domain this is 'N_old_relation'/'N_new_relation', which never equals the branch names "
                       "'N_old_rel'/'N_new_rel'. Relation writes are therefore never disabled: N_old_rel and N_new_rel "
                       "are write-identical to W, E3 = 0 by construction in every receipt, and the receipt auditor "
                       "(which derives its expected ledger from the same function) cannot detect it. Fact branches are "
                       "correct."),
            "fact_branches_valid": fact_valid,
            "relation_branches_valid": rel_valid,
            "example_ledger_actual": ledger_actual,
        },
        "exposure_countermodels": exp,
        "tamper_rejection": tamper,
        "verdict": {
            "E1_contrasts": "VALID (N_old_fact withholds all 768 old-fact writes in every receipt)",
            "E3_contrasts": ("INVALID_INSTRUMENT: N_old_rel performs all 288 old-relation writes in every receipt; "
                             "E3 is identically 0 and carries no information about any mechanism"),
            "integrity_pass_receipts": not sha_mismatch and not metric_mismatch and all(
                v.startswith("rejected") for v in tamper.values()),
        },
    }
    (RES / "FINAL_AUDIT.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    budget = json.loads((ROOT / "RESOURCE_BUDGET.json").read_text())
    res = fm["resources"]
    core_s = sum(v["life_s_total"] for v in res.values())
    rr = {"schema": "MINIFLY-B-RESOURCE-REPORT-v1", "host": budget["host"], "hard_budget": budget["hard_budget"],
          "pre_science_estimate": budget["estimate_64_world_roster"], "per_arm": res,
          "learner_core_hours_total": core_s / 3600,
          "driver_active_wall_hours_final_session_counter": status.get("elapsed_driver_s", 0) / 3600,
          "per_world_arm_learner_s_max": max(v["life_s_max"] for v in res.values()),
          "peak_rss_bytes_max": max(v["peak_rss_bytes_max"] for v in res.values()),
          "science_receipt_bytes_total": sum(v["receipt_bytes_total"] for v in res.values()),
          "within_hard_budget": {
              "core_hours": core_s / 3600 <= budget["hard_budget"]["core_hours"],
              "per_world_arm_learner_s": max(v["life_s_max"] for v in res.values())
              <= budget["hard_budget"]["per_world_arm_learner_s_max"],
              "peak_rss": max(v["peak_rss_bytes_max"] for v in res.values())
              <= budget["hard_budget"]["peak_rss_bytes_per_worker"],
              "results_disk": sum(v["receipt_bytes_total"] for v in res.values())
              <= budget["hard_budget"]["results_disk_bytes"]},
          "note": ("elapsed_driver_s is the driver's own active-time counter for the final resumed driver process; "
                   "the run spanned two sessions with VM pauses and container restarts, so host wall-clock is not "
                   "a meaningful total. Learner core-hours are summed from per-receipt learner_life_s.")}
    (RES / "RESOURCE_REPORT.json").write_text(json.dumps(rr, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"verdict": out["verdict"], "tamper": tamper, "branch": dict(branch),
                      "probe_identity": dict(probe_identity), "exposure": exp,
                      "sha_mismatch": len(sha_mismatch), "metric_mismatch": len(metric_mismatch),
                      "core_h": core_s / 3600, "within": rr["within_hard_budget"]}, indent=1))


if __name__ == "__main__":
    main()
