"""Package A V3-CLAUDE final analysis: audits every science receipt, then the 12 available Package-A
contrasts of the shared m=26 family (P3's two are unavailable). Runs only when the full roster is committed.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
import audit_a  # noqa: E402
from scipy.stats import t as student_t  # noqa: E402

M_FAMILY = 26
ALPHA = 0.05
PRIMARY = {"R1": "R0", "R3": "R0_signed", "Z2": "Z0_resource", "P1": "P0", "P2": "P0", "P4": "P0"}
DIAGNOSTIC = {"R1": "R1_rand", "R3": "R3_randtarget", "Z2": "Z2_rand"}
UNAVAILABLE = {"P3": "NOT_INSTANTIATED (technically unqualified/unresolved; no replacement)"}
ROSTER = ("R0", "R1", "R1_rand", "R0_signed", "R3", "R3_randtarget",
          "Z0_resource", "Z2", "Z2_rand", "P0", "P1", "P2", "P4")
WORLDS = tuple(range(190001, 190065))
RES = paths.ROOT / "results" / "science"


def acc(rows, stage, branch, sset, form=None):
    sel = [r["correct"] for r in rows if r["stage"] == stage and r["branch"] == branch and
           r["set"] == sset and (form is None or r.get("form") == form)]
    if not sel:
        raise AssertionError("empty endpoint")
    return sum(sel) / len(sel)


def metrics(r: dict) -> dict:
    p = r["probes"]
    out = {
        "old_fact_final_W": acc(p, "final", "W", "old_fact", "canonical"),
        "old_fact_final_N": acc(p, "final", "N_old_fact", "old_fact", "canonical"),
        "heldout_final_W": acc(p, "final", "W", "old_relation_heldout"),
        "heldout_final_N_old_rel": acc(p, "final", "N_old_rel", "old_relation_heldout"),
        "old_fact_old_end_W": acc(p, "old_end", "W", "old_fact", "canonical"),
        "new_fact_final_W": acc(p, "final", "W", "new_fact", "canonical"),
        "new_fact_final_N_new_fact": acc(p, "final", "N_new_fact", "new_fact", "canonical"),
        "new_relation_taught_final_W": acc(p, "final", "W", "new_relation_taught"),
        "new_relation_taught_final_N_new_rel": acc(p, "final", "N_new_rel", "new_relation_taught"),
        "old_relation_taught_final_W": acc(p, "final", "W", "old_relation_taught"),
        "old_relation_taught_final_N_old_rel": acc(p, "final", "N_old_rel", "old_relation_taught"),
        "heldout_old_end_W": acc(p, "old_end", "W", "old_relation_heldout"),
        "old_fact_spacing_final_W": acc(p, "final", "W", "old_fact", "spacing"),
        "old_fact_prefix_final_W": acc(p, "final", "W", "old_fact", "prefix"),
        "old_fact_marker_final_W": acc(p, "final", "W", "old_fact", "inner_marker"),
    }
    out["E1"] = out["old_fact_final_W"] - out["old_fact_final_N"]
    out["E3"] = out["heldout_final_W"] - out["heldout_final_N_old_rel"]
    return out


def interval(values, m=M_FAMILY):
    n = len(values)
    mean = statistics.fmean(values)
    sd = statistics.stdev(values)
    a = ALPHA / m
    if sd == 0.0:
        half = 4.0 * math.sqrt(math.log(2.0 / a) / (2.0 * n))   # Hoeffding on [-2, 2]
        kind = "Hoeffding bounded (zero paired variance), Bonferroni m=26"
    else:
        half = float(student_t.ppf(1 - a / 2, df=n - 1)) * sd / math.sqrt(n)
        kind = "two-sided Student-t, Bonferroni m=26"
    return {"n_worlds": n, "mean": mean, "sd_world": sd, "lower": mean - half, "upper": mean + half,
            "interval": kind}


def describe(values):
    n = len(values)
    mean = statistics.fmean(values)
    half = float(student_t.ppf(0.975, df=n - 1)) * statistics.stdev(values) / math.sqrt(n)
    return {"mean": mean, "lower95": mean - half, "upper95": mean + half}


def main() -> None:
    import lock as lk
    lock = lk.verify_lock()
    cal = json.loads((paths.ROOT / "results" / "calibration" / "R_CALIBRATION.json").read_text())
    kw = {"theta": cal["theta"], "scales": (cal["scales"]["shared"], cal["scales"]["private"])}
    receipts_sha, per = {}, defaultdict(dict)
    loaded = {}
    for arm in ROSTER:
        for w in WORLDS:
            path = RES / arm / f"{w}.json.gz"
            if not path.exists():
                raise SystemExit(f"roster incomplete: {arm}/{w}")
            raw = path.read_bytes()
            receipts_sha[f"{arm}/{w}"] = hashlib.sha256(raw).hexdigest()
            loaded[(arm, w)] = json.loads(gzip.decompress(raw))
    for arm in ROSTER:
        for w in WORLDS:
            r = loaded[(arm, w)]
            if r.get("lock_digest") != lock["lock_digest"]:
                raise AssertionError(f"lock digest {arm}/{w}")
            extra = {}
            if arm == "R1_rand":
                extra["r1_receipt"] = loaded[("R1", w)]
            if arm == "Z2_rand":
                extra["z2_receipt"] = loaded[("Z2", w)]
            audit_a.audit_receipt(r, **kw, **extra)
            per[arm][w] = {**metrics(r), "life_s": r["resources"]["life_s"],
                           "peak_rss_bytes": r["resources"]["peak_rss_bytes"]}
    contrasts = {}
    for c, b in PRIMARY.items():
        d1 = [per[c][w]["E1"] - per[b][w]["E1"] for w in WORLDS]
        d3 = [per[c][w]["E3"] - per[b][w]["E3"] for w in WORLDS]
        i1, i3 = interval(d1), interval(d3)
        contrasts[c] = {"control": b, "dE1": i1, "dE3": i3, "dE1_improvement_claim": i1["lower"] > 0,
                        "dE3_improvement_claim": i3["lower"] > 0, "per_world_dE1": d1, "per_world_dE3": d3}
    diagnostics = {c: {"control": b, "dE1": describe([per[c][w]["E1"] - per[b][w]["E1"] for w in WORLDS]),
                       "dE3": describe([per[c][w]["E3"] - per[b][w]["E3"] for w in WORLDS])}
                   for c, b in DIAGNOSTIC.items()}
    absolute = {arm: {k: describe([per[arm][w][k] for w in WORLDS])
                      for k in per[arm][WORLDS[0]] if k not in ("life_s", "peak_rss_bytes")} for arm in ROSTER}
    out = {"schema": "MINIFLY-A3-CLAUDE-FINAL-METRICS-v1", "lock_digest": lock["lock_digest"],
           "family_size_m": M_FAMILY, "package_contrasts_available": 2 * len(PRIMARY),
           "package_contrasts_unavailable": {k: {"dE1": v, "dE3": v} for k, v in UNAVAILABLE.items()},
           "t_critical": float(student_t.ppf(1 - ALPHA / (2 * M_FAMILY), df=len(WORLDS) - 1)),
           "contrasts": contrasts, "diagnostics": diagnostics, "absolute": absolute,
           "per_world": {arm: {str(w): v for w, v in per[arm].items()} for arm in ROSTER},
           "receipt_sha256": receipts_sha}
    (paths.ROOT / "results" / "FINAL_METRICS.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps({c: {k: [round(v[k][z], 4) for z in ("mean", "lower", "upper")] for k in ("dE1", "dE3")}
                      for c, v in contrasts.items()}, indent=1))


if __name__ == "__main__":
    main()
