"""Reduce LEARNED-BANK V1 history JSONs: per-history table, then the decision rule of DESIGN.md section 7.

python summarize.py results/TAG [--out results/TAG/SUMMARY.json]
"""
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path


def final(cp):
    return cp[-1]


def main():
    d = Path(sys.argv[1])
    rows = []
    for f in sorted(glob.glob(str(d / "*_*_*.json"))):
        if f.endswith("SUMMARY.json"):
            continue
        r = json.load(open(f))
        cps = r["checkpoints"]
        b, fin = cps[0], final(cps)
        best_pass = max(cps, key=lambda c: c["acc"]["held"]["L"] if c["acc"]["held"]["L"] is not None
                        else c["acc"]["held"]["sys"])
        mL0, mL1 = b["meters"].get("L", {}), fin["meters"].get("L", {})
        rows.append({
            "task": r["task"], "arm": r["arm"], "world": r["world"], "records": r["train_records"],
            "held_sys": fin["acc"]["held"]["sys"], "held_L": fin["acc"]["held"]["L"], "held_S": fin["acc"]["held"]["S"],
            "taught_sys": fin["acc"]["taught"]["sys"], "taught_L": fin["acc"]["taught"]["L"],
            "held_L_best_checkpoint": best_pass["acc"]["held"]["L"], "best_label": best_pass["label"],
            "kta_L_birth": mL0.get("kta_all"), "kta_L_final": mL1.get("kta_all"),
            "gap_L_birth": mL0.get("held_transfer_gap"), "gap_L_final": mL1.get("held_transfer_gap"),
            "min_drift_L": min([c["meters"].get("L", {}).get("train_code_jaccard_vs_prev", 1.0) for c in cps[1:]]
                               or [1.0]),
            "cpu_per_record": r["train_cpu_per_record_s"], "total_cpu_s": r["total_cpu_s"]})
    print(f"{'task':5} {'arm':7} {'world':6} {'rec':>5} {'heldSys':>7} {'heldL':>6} {'tchSys':>6} {'tchL':>6} "
          f"{'ktaL0':>7} {'ktaL1':>7} {'gapL0':>7} {'gapL1':>7} {'drift':>6}")
    fmt = lambda v: "   -  " if v is None else f"{v:6.3f}"
    for x in rows:
        print(f"{x['task']:5} {x['arm']:7} {x['world']:6} {x['records']:5d} {fmt(x['held_sys'])} {fmt(x['held_L'])} "
              f"{fmt(x['taught_sys'])} {fmt(x['taught_L'])} {fmt(x['kta_L_birth'])} {fmt(x['kta_L_final'])} "
              f"{fmt(x['gap_L_birth'])} {fmt(x['gap_L_final'])} {fmt(x['min_drift_L'])}")
    # decision rule (T-MOD): in every world LEARN held_L > FROZEN held_L and > SHUF held_L,
    # and LEARN's held transfer gap rises above its birth value while FROZEN's does not.
    by = defaultdict(dict)
    for x in rows:
        by[(x["task"], x["world"])][x["arm"]] = x
    verdict = {}
    for (task, w), arms in sorted(by.items()):
        if task != "MOD" or not {"LEARN", "FROZEN", "SHUF"} <= set(arms):
            continue
        L, F, S = arms["LEARN"], arms["FROZEN"], arms["SHUF"]
        verdict[w] = {
            "learn_gt_frozen": L["held_L"] > F["held_L"],
            "learn_gt_shuf": L["held_L"] > S["held_L"],
            "learn_gap_rises": L["gap_L_final"] > L["gap_L_birth"],
            "frozen_gap_flat": not (F["gap_L_final"] > F["gap_L_birth"] + 1e-9),
            "oracle_ok": arms.get("ORACLE", {}).get("held_L", None),
        }
        verdict[w]["supports_condition_1"] = all(verdict[w][k] for k in
                                                 ("learn_gt_frozen", "learn_gt_shuf", "learn_gap_rises",
                                                  "frozen_gap_flat"))
    if verdict:
        print(json.dumps(verdict, indent=1))
    out = {"rows": rows, "verdict_by_world": verdict,
           "supports_condition_1_all_worlds": bool(verdict) and all(v["supports_condition_1"]
                                                                    for v in verdict.values())}
    (d / "SUMMARY.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
