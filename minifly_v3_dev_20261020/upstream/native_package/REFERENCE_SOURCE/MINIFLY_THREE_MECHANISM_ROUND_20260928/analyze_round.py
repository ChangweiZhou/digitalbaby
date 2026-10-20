"""Final paired-world analysis for three distinct mechanism families."""
from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from scipy.stats import t as student_t

from audit_round import audit_life, audit_r_family, audit_t_pair
from run_science import ARMS, RESULTS, SCIENCE_WORLDS, atomic_json, require_lock, _graph_manifest

FAMILIES = {"R2": "R0", "Z1": "Z0", "T2": "T0_2"}
DOSE_DIAGNOSTICS = {"R2": "Rrand", "Z1": "Z1_BUDGET_RANDOM"}
PRIMARY = ("old_fact_final_W", "heldout_final_oldwrite_effect")


def accuracy(rows: list[dict], *, stage: str, branch: str, target_set: str,
             form: str | None = None) -> float:
    selected = [r for r in rows if r["stage"] == stage and r["branch"] == branch
                and r["set"] == target_set and (form is None or r.get("form") == form)]
    if not selected:
        raise AssertionError("empty prespecified endpoint")
    return statistics.mean(r["correct"] for r in selected)


def world_metrics(receipt: dict) -> dict:
    rows = receipt["probes"]
    return {
        "old_fact_final_W": accuracy(rows, stage="final", branch="W", target_set="old_fact", form="canonical"),
        "old_fact_final_N": accuracy(rows, stage="final", branch="N_old_fact", target_set="old_fact", form="canonical"),
        "old_fact_old_end_W": accuracy(rows, stage="old_end", branch="W", target_set="old_fact", form="canonical"),
        "new_fact_final_W": accuracy(rows, stage="final", branch="W", target_set="new_fact", form="canonical"),
        "old_relation_taught_final_W": accuracy(rows, stage="final", branch="W", target_set="old_relation_taught"),
        "heldout_old_end_W": accuracy(rows, stage="old_end", branch="W", target_set="old_relation_heldout"),
        "heldout_final_W": accuracy(rows, stage="final", branch="W", target_set="old_relation_heldout"),
        "heldout_final_N_old_rel": accuracy(rows, stage="final", branch="N_old_rel", target_set="old_relation_heldout"),
        "old_fact_spacing_final_W": accuracy(rows, stage="final", branch="W", target_set="old_fact", form="spacing"),
        "old_fact_prefix_final_W": accuracy(rows, stage="final", branch="W", target_set="old_fact", form="prefix"),
        "old_fact_marker_final_W": accuracy(rows, stage="final", branch="W", target_set="old_fact", form="inner_marker"),
    }


def interval(values: list[float], *, simultaneous_m: int = 1) -> dict:
    n = len(values)
    if n < 2 or not all(math.isfinite(x) for x in values):
        raise AssertionError("world interval requires finite replicated worlds")
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    crit = float(student_t.ppf(1 - 0.05 / (2 * simultaneous_m), df=n - 1))
    half = crit * sd / math.sqrt(n)
    return {"n_worlds": n, "mean": mean, "sd_world": sd,
            "lower": mean - half, "upper": mean + half,
            "interval": "two-sided t; Bonferroni 95% familywise" if simultaneous_m > 1
                        else "two-sided t 95%"}


def paired_effects(by_world: dict, candidate: str, control: str, worlds) -> tuple[list[float], list[float], list[float]]:
    """World-paired learning effects; W-only difference is descriptive."""
    e1 = [(by_world[w][candidate]["old_fact_final_W"] -
           by_world[w][candidate]["old_fact_final_N"]) -
          (by_world[w][control]["old_fact_final_W"] -
           by_world[w][control]["old_fact_final_N"])
          for w in worlds]
    e3 = [(by_world[w][candidate]["heldout_final_W"] -
           by_world[w][candidate]["heldout_final_N_old_rel"]) -
          (by_world[w][control]["heldout_final_W"] -
           by_world[w][control]["heldout_final_N_old_rel"])
          for w in worlds]
    w_only = [by_world[w][candidate]["old_fact_final_W"] -
              by_world[w][control]["old_fact_final_W"] for w in worlds]
    return e1, e3, w_only


def mechanism_budget(receipt: dict) -> dict:
    arm = receipt["arm"]
    rows = receipt["mechanism_events"]["W"]
    if arm in ("R0", "R2", "Rrand"):
        per_stratum = defaultdict(lambda: {"eligible": 0, "gated": 0})
        for row in rows:
            if row["allowed"]:
                key = f"{row['stage']}:{row['domain']}:{row['answer']}"
                per_stratum[key]["eligible"] += 1
                per_stratum[key]["gated"] += int(row["gate"])
        return {"kind": "R_private_record_gate", "by_stage_domain_answer": dict(per_stratum),
                "eligible_records": sum(x["eligible"] for x in per_stratum.values()),
                "gated_records": sum(x["gated"] for x in per_stratum.values())}
    if arm in ("Z0", "Z1", "Z1_BUDGET_RANDOM"):
        active = [row for row in rows if row["write"]]
        return {"kind": "Z_native_write_split", "write_events": len(active),
                "raw_alpha_l1_total": sum(row["raw_alpha_l1"] for row in active),
                "raw_fast_l1_total": sum(row["raw_fast_l1"] for row in active),
                "raw_slow_l1_total": sum(row["raw_slow_l1"] for row in active)}
    if arm in ("T0_2", "T2"):
        graph = receipt["graph_receipt"]
        return {"kind": "T_fixed_graph", "graph_digest": graph["graph_digest"][arm],
                "edges": graph["edges"],
                "effective_type_lower_tail_gain": graph["audit"]["lower_tail_gain"],
                "type_symmetric_difference_edges": graph["audit"]["type_symmetric_difference_edges"],
                "type_pair_overlap_J": graph["audit"]["type_pair_overlap"][arm]["J"]}
    raise AssertionError("unknown mechanism arm")


def analyze() -> dict:
    lock = require_lock()
    graphs = _graph_manifest(lock)
    if not (RESULTS / "RUN_COMPLETE.json").is_file():
        raise RuntimeError("science run not complete")
    if (RESULTS / "FAILURE.json").exists():
        raise RuntimeError("scientific failure present; no final analysis")
    by_world = {}
    resources = []
    budgets = {arm: {} for arm in ARMS}
    per_item_counts = defaultdict(lambda: [0, 0])
    for world in SCIENCE_WORLDS:
        by_world[world] = {}
        receipts = {}
        for arm in ARMS:
            path = RESULTS / "worlds" / str(world) / f"{arm}.json"
            receipt = json.loads(path.read_text())
            if receipt.get("source_lock_digest") != lock["digest"]:
                raise AssertionError("world source mismatch")
            receipts[arm] = receipt
        for arm in ARMS:
            path = RESULTS / "worlds" / str(world) / f"{arm}.json"
            receipt = receipts[arm]
            audit_life(receipt, arm, expected_world=world,
                       r2_receipt=receipts["R2"] if arm == "Rrand" else None,
                       graph_expected=graphs[str(world)] if arm in ("T0_2", "T2") else None)
            by_world[world][arm] = world_metrics(receipt)
            budgets[arm][world] = mechanism_budget(receipt)
            for probe in receipt["probes"]:
                if probe["stage"] != "final" or probe["branch"] != "W":
                    continue
                key = (arm, probe["set"], probe.get("form", "canonical"),
                       probe["item"], probe.get("order", -1))
                per_item_counts[key][0] += int(probe["correct"])
                per_item_counts[key][1] += 1
            resources.append({"world": world, "arm": arm,
                              "elapsed_wall_s": receipt["elapsed_wall_s"],
                              "receipt_bytes": path.stat().st_size,
                              "max_rss_raw": receipt["max_rss_raw"]})
        audit_r_family(receipts["R0"], receipts["R2"], receipts["Rrand"],
                       expected_world=world)
        audit_t_pair(receipts["T0_2"], receipts["T2"], expected_world=world,
                     graph_expected=graphs[str(world)])
    absolute = {}
    for arm in ARMS:
        absolute[arm] = {}
        for key in next(iter(by_world.values()))[arm]:
            vals = [by_world[w][arm][key] for w in SCIENCE_WORLDS]
            absolute[arm][key] = interval(vals)
        effects = [by_world[w][arm]["heldout_final_W"] -
                   by_world[w][arm]["heldout_final_N_old_rel"] for w in SCIENCE_WORLDS]
        absolute[arm]["heldout_final_oldwrite_effect"] = interval(effects)
        fact_effects = [by_world[w][arm]["old_fact_final_W"] -
                        by_world[w][arm]["old_fact_final_N"] for w in SCIENCE_WORLDS]
        absolute[arm]["old_fact_final_oldwrite_effect"] = interval(fact_effects)
    comparisons = {}
    for candidate, control in FAMILIES.items():
        e1, e3, e1_w_only = paired_effects(by_world, candidate, control, SCIENCE_WORLDS)
        comparisons[candidate] = {"control": control,
                                   "old_fact_final_oldwrite_effect_diff": interval(e1, simultaneous_m=6),
                                   "old_fact_final_W_diff_secondary": interval(e1_w_only),
                                   "heldout_oldwrite_effect_diff": interval(e3, simultaneous_m=6)}
    diagnostics = {}
    for candidate, dose_control in DOSE_DIAGNOSTICS.items():
        e1, e3, w_only = paired_effects(by_world, candidate, dose_control, SCIENCE_WORLDS)
        diagnostics[candidate] = {"dose_control": dose_control,
                                  "old_fact_final_oldwrite_effect_diff": interval(e1),
                                  "heldout_oldwrite_effect_diff": interval(e3),
                                  "old_fact_final_W_diff": interval(w_only),
                                  "status": "descriptive dose/location diagnostic; not a primary family contrast"}
    decisions = {}
    for candidate, row in comparisons.items():
        e1_confirmed = row["old_fact_final_oldwrite_effect_diff"]["lower"] > 0
        e3_confirmed = row["heldout_oldwrite_effect_diff"]["lower"] > 0
        own_e1 = absolute[candidate]["old_fact_final_oldwrite_effect"]["mean"] > 0
        own_e3 = absolute[candidate]["heldout_final_oldwrite_effect"]["mean"] > 0
        decisions[candidate] = {"old_fact_axis_confirmed": e1_confirmed,
                                "heldout_relation_axis_confirmed": e3_confirmed,
                                "own_learning_effects_positive": own_e1 and own_e3,
                                "integrated_improvement": e1_confirmed and e3_confirmed and own_e1 and own_e3}
    return {"schema": "MINIFLY-THREE-MECHANISM-ANALYSIS-v1",
            "source_lock_digest": lock["digest"], "science_worlds": list(SCIENCE_WORLDS),
            "families": FAMILIES, "absolute": absolute, "paired_comparisons": comparisons,
            "dose_diagnostics": diagnostics,
            "prespecified_decisions": decisions,
            "mechanism_budgets": budgets,
            "per_item_final_W": {"|".join(map(str, key)): {"correct": n[0], "total": n[1]}
                                 for key, n in sorted(per_item_counts.items())},
            "resources": resources}


def render_report(doc: dict) -> str:
    lines = ["# Three mechanism families: final paired-world report", "",
             "FE1 was excluded from every candidate, control and trajectory in this round.", "",
             f"Science worlds: {len(doc['science_worlds'])}; each world contains the same byte task and continuing lifetime.",
             "Primary comparisons use each candidate's own matched family control.", "",
             "| Candidate vs control | Delayed old-fact learning effect difference (pp) | Final withheld relation old-write effect difference (pp) |",
             "|---|---:|---:|"]
    for candidate, row in doc["paired_comparisons"].items():
        e1, e3 = row["old_fact_final_oldwrite_effect_diff"], row["heldout_oldwrite_effect_diff"]
        def fmt(x):
            return f"{100*x['mean']:+.1f} [{100*x['lower']:+.1f}, {100*x['upper']:+.1f}]"
        lines.append(f"| {candidate} vs {row['control']} | {fmt(e1)} | {fmt(e3)} |")
    lines += ["", "Predeclared integrated-improvement decision: " +
              ", ".join(f"{name}={'YES' if result['integrated_improvement'] else 'NO'}"
                        for name, result in doc["prespecified_decisions"].items()) + "."]
    lines += ["", "Intervals are simultaneous 95% Bonferroni t intervals across the six primary family-by-axis contrasts; worlds are the sampling units.",
              "The delayed old-fact contrast is (candidate W−N_old_fact)−(matched control W−N_old_fact); W-only scores and their differences are descriptive secondary results in SUMMARY.json.",
              "R2 versus yoked Rrand and Z1 versus budget-matched random eligibility are descriptive dose/location diagnostics in SUMMARY.json. Rrand's schedule uses R2's completed within-stratum gate counts; it is not an online autonomous control.",
              "The held-out relation is a limited synthetic E3 assay: a one-bit symbol-class rule could solve it. Exact taught facts are E1; unchanged-key formatting variants are narrow E2.",
              "A candidate-versus-control difference alone is not proof of E3. Inspect the candidate's own W minus N_old_rel effect and the absolute W/N accuracies in SUMMARY.json.",
              "Per-item final output, mechanism write/graph budgets, and all absolute scores are retained in SUMMARY.json.",
              "This round did not test autonomous learning without teacher bytes, open-ended generation, or learned V9.2 predictor representations.", ""]
    return "\n".join(lines)


def main():
    doc = analyze()
    atomic_json(RESULTS / "SUMMARY.json", doc)
    (RESULTS / "REPORT.md").write_text(render_report(doc))
    print(json.dumps({"analysis": "COMPLETE", "worlds": len(SCIENCE_WORLDS),
                      "report": str(RESULTS / "REPORT.md")}), flush=True)


if __name__ == "__main__":
    main()
