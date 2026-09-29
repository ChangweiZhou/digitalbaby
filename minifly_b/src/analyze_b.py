"""Package-B final analysis: audits every science receipt, then the 12 Package-B
contrasts of the shared m=26 family.  Runs only when the full roster is committed.
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
from scipy.stats import t as student_t  # noqa: E402

import analyze_round as hist  # historical reducer: used only as a cross-check  # noqa: E402
import audit_topo as au  # noqa: E402
import topo_model as tm  # noqa: E402

M_FAMILY = 26
ALPHA = 0.05
PRIMARY = {"T1": "T0_1", "T3": "T0_3", "S1": "S0", "S2": "S0", "S3": "S0", "S4": "S0"}
DIAGNOSTIC = {"S1": "Srand_1", "S2": "Srand_2", "S3": "Srand_3", "S4": "Srand_4"}
ROSTER = ("T1", "T0_1", "T3", "T0_3", "S0", "S1", "S2", "S3", "S4",
          "Srand_1", "Srand_2", "Srand_3", "Srand_4")
WORLDS = tuple(range(190001, 190065))


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
        "old_relation_taught_final_W": acc(p, "final", "W", "old_relation_taught"),
        "old_relation_taught_final_N_old_rel": acc(p, "final", "N_old_rel", "old_relation_taught"),
        "heldout_old_end_W": acc(p, "old_end", "W", "old_relation_heldout"),
        "old_fact_spacing_final_W": acc(p, "final", "W", "old_fact", "spacing"),
        "old_fact_prefix_final_W": acc(p, "final", "W", "old_fact", "prefix"),
        "old_fact_marker_final_W": acc(p, "final", "W", "old_fact", "inner_marker"),
    }
    out["E1"] = out["old_fact_final_W"] - out["old_fact_final_N"]
    out["E3"] = out["heldout_final_W"] - out["heldout_final_N_old_rel"]
    ref = hist.world_metrics(r)
    for k, v in ref.items():
        if abs(v - out[k]) > 1e-15:
            raise AssertionError(f"metric cross-check failed: {k}")
    return out


def interval(values, m=M_FAMILY):
    n = len(values)
    mean = statistics.fmean(values)
    sd = statistics.stdev(values)
    a = ALPHA / m
    if sd == 0.0:
        half = 4.0 * math.sqrt(math.log(2.0 / a) / (2.0 * n))   # Hoeffding, range [-2,2]
        kind = "Hoeffding bounded (zero paired variance), Bonferroni m=26"
    else:
        half = float(student_t.ppf(1 - a / 2, df=n - 1)) * sd / math.sqrt(n)
        kind = "two-sided Student-t, Bonferroni m=26"
    return {"n_worlds": n, "mean": mean, "sd_world": sd, "lower": mean - half,
            "upper": mean + half, "interval": kind}


def describe(values):
    n = len(values)
    mean = statistics.fmean(values)
    sd = statistics.stdev(values)
    half = float(student_t.ppf(0.975, df=n - 1)) * sd / math.sqrt(n)
    return {"mean": mean, "lower95": mean - half, "upper95": mean + half}


def main() -> None:
    import lock as lk
    lock = lk.verify_lock()
    manifest = json.loads((paths.ROOT / "GRAPH_MANIFEST.json").read_text())
    res = paths.ROOT / "results" / "science"
    receipts_sha = {}
    per = defaultdict(dict)
    supports = {}
    for arm in ROSTER:
        for w in WORLDS:
            path = res / arm / f"{w}.json.gz"
            if not path.exists():
                raise SystemExit(f"roster incomplete: {arm}/{w}")
            raw = path.read_bytes()
            receipts_sha[f"{arm}/{w}"] = hashlib.sha256(raw).hexdigest()
            r = json.loads(gzip.decompress(raw))
            cand = None
            if arm.startswith("Srand"):
                cand = json.loads(gzip.decompress((res / f"S{arm[-1]}" / f"{w}.json.gz").read_bytes()))
            sup = None
            if arm in tm.S_ARMS:
                if w not in supports:
                    import portable_birth as pb
                    base, _ = pb.canonical_fresh_native()
                    supports[w] = tm.Support(base.fly.m.B, w)
                sup = supports[w]
            au.audit_receipt(r, arm, w, manifest=manifest, lock=lock, candidate=cand, support=sup)
            per[arm][w] = {**metrics(r), "life_s": r["resources"]["learner_life_s"],
                           "peak_rss_bytes": r["resources"]["peak_rss_bytes"],
                           "receipt_bytes": len(raw)}
    contrasts = {}
    for c, b in PRIMARY.items():
        d1 = [per[c][w]["E1"] - per[b][w]["E1"] for w in WORLDS]
        d3 = [per[c][w]["E3"] - per[b][w]["E3"] for w in WORLDS]
        contrasts[c] = {"control": b, "dE1": interval(d1), "dE3": interval(d3),
                        "dE1_improvement_claim": interval(d1)["lower"] > 0,
                        "dE3_improvement_claim": interval(d3)["lower"] > 0,
                        "per_world_dE1": d1, "per_world_dE3": d3}
    diagnostics = {}
    for c, b in DIAGNOSTIC.items():
        diagnostics[c] = {"control": b, "non_autonomous_yoked": True,
                          "dE1": describe([per[c][w]["E1"] - per[b][w]["E1"] for w in WORLDS]),
                          "dE3": describe([per[c][w]["E3"] - per[b][w]["E3"] for w in WORLDS])}
    absolute = {arm: {k: describe([per[arm][w][k] for w in WORLDS])
                      for k in per[arm][WORLDS[0]] if k not in ("life_s", "peak_rss_bytes", "receipt_bytes")}
                for arm in ROSTER}
    resources = {arm: {"life_s_total": sum(per[arm][w]["life_s"] for w in WORLDS),
                       "life_s_max": max(per[arm][w]["life_s"] for w in WORLDS),
                       "peak_rss_bytes_max": max(per[arm][w]["peak_rss_bytes"] for w in WORLDS),
                       "receipt_bytes_total": sum(per[arm][w]["receipt_bytes"] for w in WORLDS)}
                 for arm in ROSTER}
    out = {"schema": "MINIFLY-B-FINAL-METRICS-v1", "lock_digest": lock["lock_digest"],
           "family_size_m": M_FAMILY, "package_contrasts": len(PRIMARY) * 2,
           "t_critical": float(student_t.ppf(1 - ALPHA / (2 * M_FAMILY), df=len(WORLDS) - 1)),
           "contrasts": contrasts, "diagnostics": diagnostics, "absolute": absolute,
           "per_world": {arm: {str(w): v for w, v in per[arm].items()} for arm in ROSTER},
           "resources": resources, "receipt_sha256": receipts_sha}
    dest = paths.ROOT / "results" / "FINAL_METRICS.json"
    dest.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps({c: {"dE1": [round(v["dE1"][k], 4) for k in ("mean", "lower", "upper")],
                          "dE3": [round(v["dE3"][k], 4) for k in ("mean", "lower", "upper")]}
                      for c, v in contrasts.items()}, indent=1))


if __name__ == "__main__":
    main()
