#!/usr/bin/env python3
"""Descriptive V43 full-roster supplement; never changes qualification.

Run after all 64 physical cells finish:
    python3 v43_build/all_cells_summary.py v43_final_fb8ebb_results

All 16 seeds contribute at each load, including cells failing acquisition.
SD describes between-seed dispersion. No confidence interval, p-value, or
selectively valid-subset estimate is produced here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile

POLICIES = ["SUPER", "READ-HYBRID", "SILENT-HARD", "ANCHOR-ONLY", "FIXED-50",
            "GLOBAL-BAYES", "LOCAL-BAYES-0", "LOCAL-BAYES-E2"]
SEEDS = list(range(43101, 43117))
LOADS = [64, 128, 256, 512]
CHECKPOINTS = ["after_A", "after_B", "after_C", "after_D"]
METRICS = ["whole_bpb", "G_bpb", "A_after_D_heldout", "physical_acquisition_minimum"]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else canonical(value)).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for data in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def acquisition(record):
    rows = record.get("audits", {}).get("physical_acquisition_current")
    if not isinstance(rows, list) or len(rows) != 4:
        return {"valid": False, "minimum": None, "by_checkpoint": {}, "complete": False}
    by = {row.get("checkpoint"): row for row in rows}
    complete = set(by) == set(CHECKPOINTS) and all(
        by[cp].get("population_n", 0) > 0 and number(by[cp].get("population_accuracy"))
        for cp in CHECKPOINTS)
    values = {cp: float(by[cp]["population_accuracy"]) if cp in by and number(by[cp].get("population_accuracy")) else None
              for cp in CHECKPOINTS}
    minimum = min(values.values()) if complete else None
    return {"valid": bool(complete and minimum >= 0.9), "minimum": minimum,
            "by_checkpoint": values, "complete": complete}


def metric(record, policy, name):
    if name == "physical_acquisition_minimum":
        return acquisition(record)["minimum"]
    value = record.get("metrics_by_arm", {}).get(policy, {}).get(name)
    return float(value) if number(value) else None


def describe(values):
    """Require a complete 16-seed metric; never silently average a subset."""
    import numpy as np
    observed = [float(value) for value in values if number(value)]
    complete = len(values) == len(SEEDS) and len(observed) == len(values)
    result = {"n_seeds": len(values), "n_observed": len(observed), "full_roster_metric_complete": complete,
              "mean": None, "sd": None, "minimum": None, "maximum": None}
    if complete:
        result.update(mean=float(np.mean(observed)), sd=float(np.std(observed, ddof=1)),
                      minimum=min(observed), maximum=max(observed))
    return result


def read_all(root):
    root = Path(root)
    plan = json.loads((root / "PLAN.json").read_bytes())
    expected = {(seed, load) for seed in SEEDS for load in LOADS}
    jobs = plan.get("jobs", [])
    if (plan.get("mode") != "science" or plan.get("seeds") != SEEDS or plan.get("loads") != LOADS or
            plan.get("policies") != POLICIES or plan.get("expected_trajectories") != 64 or
            len(jobs) != 64 or {(job.get("seed"), job.get("load")) for job in jobs} != expected):
        raise ValueError("Supplement requires the exact frozen V43 16-seed × 4-load science roster")
    plan_hash = sha(plan)
    records, evidence = [], []
    expected_names = {f"seed_{seed}_load_{load}.json" for seed, load in expected}
    present_names = {path.name for path in (root / "completed").glob("*.json")}
    if present_names != expected_names:
        raise ValueError(f"Supplement requires exactly 64 completed cells; found {len(present_names)}; "
                         f"missing={sorted(expected_names-present_names)}, unexpected={sorted(present_names-expected_names)}")
    for job in jobs:
        path = root / "completed" / f"seed_{job['seed']}_load_{job['load']}.json"
        raw = path.read_bytes()
        envelope = json.loads(raw)
        if envelope.get("identity") != {"plan_sha256": plan_hash, "job": job}:
            raise ValueError(f"Completed plan/cell identity mismatch: {path}")
        record = envelope["data"]
        if envelope.get("sha256") != sha(record):
            raise ValueError(f"Completed payload checksum mismatch: {path}")
        if (record.get("seed"), record.get("load")) != (job["seed"], job["load"]) or record.get("physical_trajectories") != 1:
            raise ValueError(f"Completed physical-cell identity mismatch: {path}")
        checked_events = False
        if "authority_events_file" in record:
            event_path = (root / record["authority_events_file"]).resolve()
            if not event_path.is_relative_to(root.resolve()):
                raise ValueError("Authority-events path escapes result directory")
            if not event_path.is_file() or file_sha(event_path) != record.get("authority_events_sha256"):
                raise ValueError(f"Authority-event checksum mismatch: {event_path}")
            checked_events = True
        records.append(record)
        evidence.append({"file": str(path.relative_to(root)), "file_sha256": sha(raw),
                         "payload_sha256": envelope["sha256"], "authority_event_checksum_verified": checked_events,
                         "seed": job["seed"], "load": job["load"]})
    return plan, records, evidence


def summarize(records):
    roster = {(record["seed"], record["load"]) for record in records}
    if len(records) != 64 or roster != {(seed, load) for seed in SEEDS for load in LOADS}:
        raise ValueError("Exactly all 64 unique science cells must be present")
    table, cells = [], []
    validity = {}
    for load in LOADS:
        group = sorted((record for record in records if record["load"] == load), key=lambda record: record["seed"])
        valid_count = sum(acquisition(record)["valid"] for record in group)
        validity[str(load)] = {"n_seeds": 16, "acquisition_valid_cells": valid_count,
                               "acquisition_invalid_cells": 16 - valid_count,
                               "acquisition_rule": "unchanged SLOTS accuracy >=0.9 at all four current-wave checkpoints"}
        for policy in POLICIES:
            for name in METRICS:
                values = [metric(record, policy, name) for record in group]
                table.append({"load": load, "policy": policy, "metric": name,
                              "unit": "bits/byte" if name.endswith("bpb") else "accuracy_fraction",
                              "acquisition_valid_cells": valid_count,
                              "selection": "all_16_seeds_including_acquisition_invalid",
                              **describe(values)})
        for checkpoint in CHECKPOINTS:
            values = [acquisition(record)["by_checkpoint"].get(checkpoint) for record in group]
            table.append({"load": load, "policy": "SHARED-PHYSICAL", "metric": "physical_acquisition_" + checkpoint,
                          "unit": "accuracy_fraction", "acquisition_valid_cells": valid_count,
                          "selection": "all_16_seeds_including_acquisition_invalid", **describe(values)})
        for record in group:
            cells.append({"seed": record["seed"], "load": load, "acquisition": acquisition(record),
                          "policy_metrics": {policy: {name: metric(record, policy, name) for name in METRICS[:3]}
                                             for policy in POLICIES}})
    return {"status": "complete_full_roster_descriptive", "physical_cells": 64,
            "independent_seeds": SEEDS, "loads": LOADS, "policies": POLICIES,
            "validity_by_load": validity, "summary_rows": table, "all_cell_values": cells,
            "qualification_changed": False, "training_updates": 0,
            "interpretation": ["All 16 seeds at each load are included, including failed acquisition cells.",
                               "Means, sample SD, minima and maxima are descriptive; no p-values or inferential intervals are produced.",
                               "A missing metric makes its complete-roster summary unavailable; no selective-subset mean is substituted.",
                               "Acquisition-valid counts refer only to the cell-level acquisition requirement of Q0, not all engineering/qualification conditions.",
                               "The physical acquisition minimum is the minimum accuracy across after_A/B/C/D within each seed, then summarized across seeds.",
                               "Physical acquisition is shared by all read policies; repeated acquisition rows do not represent additional experiments.",
                               "Policy performance in invalid cells is descriptive evidence of substrate behavior and cannot establish arbitration qualification."]}


def plot(root, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    palette = ["#727272", "#0072B2", "#56B4E9", "#CC79A7", "#E69F00", "#009E73", "#D55E00", "#111111"]
    markers = ["o", "s", "^", "v", "D", "P", "X", "*"]
    styles = ["--", "-", ":", "-.", "--", "-", ":", "-"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    rows = {(row["load"], row["policy"], row["metric"]): row for row in report["summary_rows"]}
    names = [("whole_bpb", "Whole-stream loss", "Bits per byte"),
             ("G_bpb", "Supported similarity (G) loss", "Bits per G event"),
             ("A_after_D_heldout", "Durable old held-out recall", "Accuracy (%)")]
    for axis, (name, title, ylabel) in zip(axes, names):
        factor = 100 if name.endswith("heldout") else 1
        for j, policy in enumerate(POLICIES):
            values = [rows[load, policy, name]["mean"] for load in LOADS]
            values = [np.nan if value is None else value * factor for value in values]
            axis.plot(LOADS, values, label=policy, color=palette[j], marker=markers[j], linestyle=styles[j],
                      linewidth=2.2 if policy == "LOCAL-BAYES-E2" else 1.45, markersize=7 if policy == "LOCAL-BAYES-E2" else 4.8,
                      alpha=0.96, zorder=5 if policy == "LOCAL-BAYES-E2" else 3)
        axis.set_xscale("log", base=2)
        axis.set_xticks(LOADS, [str(load) for load in LOADS])
        axis.set_xlabel("Identity load")
        axis.set_title(title, pad=10, fontweight="semibold")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.23)
        if name.endswith("heldout"):
            axis.set_ylim(-2, 102)
    labels = ";  ".join(f"{load}: {report['validity_by_load'][str(load)]['acquisition_valid_cells']}/16" for load in LOADS)
    fig.suptitle("V43 | all 64 completed cells", x=0.065, ha="left", y=0.98, fontsize=17, fontweight="semibold")
    fig.text(0.065, 0.916, "Full-roster means: 16 seeds per load, including acquisition failures. No qualification filtering.", fontsize=10.5)
    handles, legends = axes[0].get_legend_handles_labels()
    fig.legend(handles, legends, ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.08),
               columnspacing=2.0, handlelength=2.7)
    fig.text(0.065, 0.039, "Acquisition-valid cells by load: " + labels + ".  Counts cover the acquisition part of Q0 only.", fontsize=9.5)
    fig.subplots_adjust(left=0.065, right=0.99, top=0.82, bottom=0.31, wspace=0.27)
    target = Path(root) / "V43_all_cells.png"
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=170)
    plt.close(fig)
    atomic(target, buffer.getvalue())


def run(root):
    root = Path(root)
    plan, records, evidence = read_all(root)
    report = summarize(records)
    report.update(plan_sha256=sha(plan), source_sha256=plan.get("source_sha256"),
                  supplement_source_sha256=file_sha(__file__), provenance=evidence,
                  completed_envelopes_verified=64,
                  authority_event_files_verified=sum(item["authority_event_checksum_verified"] for item in evidence))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(report["summary_rows"][0]))
    writer.writeheader()
    writer.writerows(report["summary_rows"])
    atomic(root / "ALL_CELLS_SUMMARY.csv", buffer.getvalue().encode())
    atomic(root / "ALL_CELLS_SUMMARY.json", json.dumps(report, sort_keys=True, indent=2, allow_nan=False).encode())
    plot(root, report)
    return report


def self_test():
    records = []
    for seed in SEEDS:
        for load in LOADS:
            offset = (seed - SEEDS[0]) / 100
            accuracy = 0.95 if seed % 2 else 0.85
            records.append({"seed": seed, "load": load, "physical_trajectories": 1,
                            "metrics_by_arm": {policy: {"whole_bpb": 2 + offset, "G_bpb": 3 + offset,
                                                               "A_after_D_heldout": 0.6 + offset} for policy in POLICIES},
                            "audits": {"physical_acquisition_current": [{"checkpoint": cp, "population_n": 10,
                                                                           "population_accuracy": accuracy} for cp in CHECKPOINTS]}})
    report = summarize(records)
    assert all(item["acquisition_valid_cells"] == 8 for item in report["validity_by_load"].values())
    row = next(row for row in report["summary_rows"] if row["load"] == 64 and row["policy"] == "SUPER" and row["metric"] == "whole_bpb")
    assert row["n_seeds"] == 16 and abs(row["mean"] - 2.075) < 1e-12
    assert describe([1.0] * 15 + [None])["mean"] is None
    try:
        summarize(records[:-1])
    except ValueError:
        pass
    else:
        raise AssertionError("Incomplete roster was accepted")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "completed").mkdir()
        jobs = [{"seed": record["seed"], "load": record["load"], "config": {}} for record in records]
        plan = {"mode": "science", "seeds": SEEDS, "loads": LOADS, "jobs": jobs,
                "expected_trajectories": 64, "policies": POLICIES, "source_sha256": "synthetic_only"}
        (root / "PLAN.json").write_bytes(canonical(plan))
        for job, record in zip(jobs, records):
            envelope = {"identity": {"plan_sha256": sha(plan), "job": job}, "sha256": sha(record), "data": record}
            (root / "completed" / f"seed_{job['seed']}_load_{job['load']}.json").write_bytes(canonical(envelope))
        saved = run(root)
        assert saved["completed_envelopes_verified"] == 64
        assert (root / "V43_all_cells.png").stat().st_size > 10000
        first = next((root / "completed").glob("*.json"))
        envelope = json.loads(first.read_bytes())
        envelope["data"]["metrics_by_arm"]["SUPER"]["whole_bpb"] = 100
        first.write_bytes(canonical(envelope))
        try:
            read_all(root)
        except ValueError:
            pass
        else:
            raise AssertionError("Corrupted envelope was accepted")
    return {"passed": True, "checks": ["all 64 cells including invalid acquisition are summarized", "exact 16-seed means",
                                        "missing metric produces no subset mean", "incomplete roster rejected",
                                        "all envelope and plan checks pass on synthetic full roster", "CSV/JSON/PNG generation",
                                        "corrupt completed payload rejected"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default="v43_final_fb8ebb_results")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    result = self_test() if args.self_test else run(args.root)
    print(json.dumps(result if args.self_test else {"status": result["status"], "physical_cells": result["physical_cells"],
                                                   "validity_by_load": result["validity_by_load"]}, indent=2))
