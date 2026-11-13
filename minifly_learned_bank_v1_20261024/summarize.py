"""Reduce LEARNED-BANK V1 history JSONs: per-history table, then the decision rule of DESIGN.md section 7.

python summarize.py results/TAG [--out results/TAG/SUMMARY.json]
"""
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import data


PRIMARY = "PAR"
MARGIN = 0.20


def final(cp):
    return cp[-1]


def held_balanced(r, cp, k):
    """Balanced held accuracy (mean per-answer recall) of bank k (0 sys, 1 S, 2 L) from stored outputs."""
    world = data.make_world(r["world"])
    hit = defaultdict(list)
    for a, b in world["held_pairs"]:
        out = cp["outputs"][f"{a}{b}"][k]
        if out is None:
            return None
        t = data.answer(r["task"], a, b)
        hit[t].append(out == t)
    return round(sum(sum(v) / len(v) for v in hit.values()) / len(hit), 4)


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
            "held_bal_sys": held_balanced(r, fin, 0), "held_bal_S": held_balanced(r, fin, 1),
            "held_bal_L": held_balanced(r, fin, 2),
            "taught_sys": fin["acc"]["taught"]["sys"], "taught_L": fin["acc"]["taught"]["L"],
            "held_L_best_checkpoint": best_pass["acc"]["held"]["L"], "best_label": best_pass["label"],
            "kta_L_birth": mL0.get("kta_all"), "kta_L_final": mL1.get("kta_all"),
            "gap_L_birth": mL0.get("held_transfer_gap"), "gap_L_final": mL1.get("held_transfer_gap"),
            "min_drift_L": min([c["meters"].get("L", {}).get("train_code_jaccard_vs_prev", 1.0) for c in cps[1:]]
                               or [1.0]),
            "cpu_per_record": r["train_cpu_per_record_s"], "total_cpu_s": r["total_cpu_s"]})
    print(f"{'task':5} {'arm':7} {'world':6} {'rec':>5} {'heldSys':>7} {'heldL':>6} {'balL':>6} {'tchSys':>6} {'tchL':>6} "
          f"{'ktaL0':>7} {'ktaL1':>7} {'gapL0':>7} {'gapL1':>7} {'drift':>6}")
    fmt = lambda v: "   -  " if v is None else f"{v:6.3f}"
    for x in rows:
        print(f"{x['task']:5} {x['arm']:7} {x['world']:6} {x['records']:5d} {fmt(x['held_sys'])} {fmt(x['held_L'])} "
              f"{fmt(x['held_bal_L'])} "
              f"{fmt(x['taught_sys'])} {fmt(x['taught_L'])} {fmt(x['kta_L_birth'])} {fmt(x['kta_L_final'])} "
              f"{fmt(x['gap_L_birth'])} {fmt(x['gap_L_final'])} {fmt(x['min_drift_L'])}")
    # decision rule (LOCK.json; DESIGN.md amendment A1): primary task PAR, metric = balanced held accuracy of the
    # learned bank at the final checkpoint. In each world: LEARN >= FROZEN + 0.20 and >= SHUF + 0.20, LEARN's held
    # transfer gap and kernel-target alignment both above birth; gates: ORACLE >= 0.90, FROZEN codes unchanged, and every
    # T-COPY arm's system held accuracy >= 0.30 (measurement chain).
    by = defaultdict(dict)
    for x in rows:
        by[(x["task"], x["world"])][x["arm"]] = x
    verdict = {}
    for (task, w), arms in sorted(by.items()):
        if task != PRIMARY or not {"LEARN", "FROZEN", "SHUF"} <= set(arms):
            continue
        L, F, S = arms["LEARN"], arms["FROZEN"], arms["SHUF"]
        o = arms.get("ORACLE", {}).get("held_bal_L")
        verdict[w] = {
            "learn_bal": L["held_bal_L"], "frozen_bal": F["held_bal_L"], "shuf_bal": S["held_bal_L"], "oracle_bal": o,
            "learn_ge_frozen_plus_margin": L["held_bal_L"] >= F["held_bal_L"] + MARGIN,
            "learn_ge_shuf_plus_margin": L["held_bal_L"] >= S["held_bal_L"] + MARGIN,
            "learn_gap_rises": L["gap_L_final"] > L["gap_L_birth"],
            "learn_kta_rises": L["kta_L_final"] > L["kta_L_birth"],
            "gate_frozen_codes_unchanged": F["min_drift_L"] == 1.0 and F["gap_L_final"] == F["gap_L_birth"],
            "gate_oracle": None if o is None else o >= 0.90,
        }
        copy = by.get(("COPY", w), {})
        v = verdict[w]
        v["gate_copy_sys_held_ge_0.30"] = None if not copy else all(c["held_sys"] >= 0.30 for c in copy.values())
        v["gates_ok"] = (v["gate_frozen_codes_unchanged"] and v["gate_oracle"] is not False
                         and v["gate_copy_sys_held_ge_0.30"] is not False)
        v["supports_condition_1"] = v["gates_ok"] and all(v[k] for k in (
            "learn_ge_frozen_plus_margin", "learn_ge_shuf_plus_margin", "learn_gap_rises", "learn_kta_rises"))
    if verdict:
        print(json.dumps(verdict, indent=1))
    n_ok = sum(v["supports_condition_1"] for v in verdict.values())
    overall = ("not evaluated" if not verdict else "supported" if n_ok == len(verdict)
               else "partial" if n_ok >= len(verdict) - 1 and len(verdict) >= 4 else "not supported")
    print("overall:", overall, f"({n_ok}/{len(verdict)} worlds)")
    out = {"rows": rows, "verdict_by_world": verdict, "worlds_supporting": n_ok, "overall": overall}
    (d / "SUMMARY.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
