"""Aggregate results/runs/*.json into results/SUMMARY.json and print a table."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def value(s, out):
    # Numeric reading of an emitted answer (None if it is not a well-formed number in that format).
    if out == "dec":
        return int(s) if s.isdigit() and (s == "0" or not s.startswith("0")) else None
    return ord(s) - 65 if len(s) == 1 and 0 <= ord(s) - 65 <= 18 else None


def main():
    groups = defaultdict(list)
    for p in sorted((HERE / "results" / "runs").glob("*.json")):
        r = json.loads(p.read_text())
        groups[(r["regime"], r["wd"], r["inp"], r["out"], r.get("aux", "none"), r.get("width", 64))].append(r)
    table = []
    for key, rs in sorted(groups.items()):
        errs = [value(h["got"], r["out"]) - int(value(h["want"], r["out"]))
                if value(h["got"], r["out"]) is not None else None for r in rs for h in r["held"]]
        wrong = [e for e in errs if e != 0]
        steps = [c[0] for c in rs[0]["curve"]]
        held_curve = np.mean([[c[2] for c in r["curve"]] for r in rs], 0)
        train_curve = np.mean([[c[1] for c in r["curve"]] for r in rs], 0)
        peak = int(np.argmax(held_curve))
        table.append(dict(
            regime=key[0], wd=key[1], inp=key[2], out=key[3], aux=key[4], width=key[5], runs=len(rs),
            train_final=float(np.mean([r["train_final"] for r in rs])),
            held_final=float(np.mean([r["held_final"] for r in rs])),
            held_final_min_max=[min(r["held_final"] for r in rs), max(r["held_final"] for r in rs)],
            held_best_per_run=float(np.mean([r["held_best"] for r in rs])),
            mean_curve_peak=dict(step=steps[peak], held=float(held_curve[peak]), train=float(train_curve[peak])),
            held_wrong_answers=len(wrong),
            held_wrong_off_by_one=sum(e is not None and abs(e) == 1 for e in wrong),
            held_wrong_malformed=sum(e is None for e in wrong),
            held_wrong_abs_err_mean=float(np.mean([abs(e) for e in wrong if e is not None])) if wrong else 0.0,
            probe_order_mean=float(np.mean([r["probe_order"] for r in rs])),
            cpu_s_mean=float(np.mean([r["cpu_s"] for r in rs])),
            curve=dict(step=steps, held=[float(x) for x in held_curve], train=[float(x) for x in train_curve])))
    (HERE / "results" / "SUMMARY.json").write_text(json.dumps(table, indent=1) + "\n")
    hdr = f"{'regime':7} {'wd':>3} {'inp':6} {'out':4} {'aux':10} {'w':>2} {'n':>2} {'train':>6} {'held':>6} {'[min,max]':>11} " \
          f"{'best':>5} {'peak@step':>14} {'wrong':>5} {'±1':>4} {'bad':>4} {'order':>5}"
    print(hdr)
    for t in table:
        print(f"{t['regime']:7} {t['wd']:>3g} {t['inp']:6} {t['out']:4} {t['aux']:10} {t['width']:>2} {t['runs']:>2} {t['train_final']:6.2f} "
              f"{t['held_final']:6.2f} [{t['held_final_min_max'][0]:.2f},{t['held_final_min_max'][1]:.2f}] "
              f"{t['held_best_per_run']:5.2f} {t['mean_curve_peak']['held']:5.2f}@{t['mean_curve_peak']['step']:<7} "
              f"{t['held_wrong_answers']:5} {t['held_wrong_off_by_one']:4} {t['held_wrong_malformed']:4} "
              f"{t['probe_order_mean']:5.2f}")


if __name__ == "__main__":
    main()
