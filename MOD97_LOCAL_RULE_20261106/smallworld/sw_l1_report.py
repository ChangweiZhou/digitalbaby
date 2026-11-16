"""E39 job L1: the tables of E39_REPORT.md, from the per-rung files E39_L1_<rung>.json (sw_l1.py table) and the score tables.

    python3 sw_l1_report.py <rung> [<rung> ...]   # -> exp45/e39/E39_L1_TABLES.md, E39_L1_first_visit.csv, E39_L1_score_by_age.csv
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "exp45" / "e39"
WORLDS = range(1, 9)
ARMS = ("naive", "sweep")


def f(x, d=3):
    return "–" if x is None else f"{x:.{d}f}"


def pct(x):
    return "–" if x is None else f"{100 * x:.1f}"


def main(rungs):
    md, fv_rows, sba_rows = [], [], []
    for rung in rungs:
        T = json.loads((OUT / f"E39_L1_{rung}.json").read_text())
        L = T["lives"]; g = T["gate"]
        key = {(v["arm"], v["world"]): k for k, v in L.items()}
        md += [f"## Rung {rung}: {T['settings']}", "", "### The gate (naive)", "",
               "| World | Groups met | (1) first group's two costs (taught items) | (1) | (2) median first-visit level, tasks n ≥ 3 | (2) | Both |",
               "|---|---|---|---|---|---|---|"]
        for r in g["rows"]:
            v = L[key[("naive", r["world"])]]
            md.append(f"| {r['world']} | {v['groups_met']} | {v['first_group_cost'][0]}, {v['first_group_cost'][1]} | {'yes' if r['part1'] else 'no'} | "
                      f"{f(r['median'])} | {'yes' if r['part2'] else 'no'} | {'yes' if r['both'] else 'no'} |")
        md += ["", f"Worlds passing both: **{g['worlds_passing']} of 8** (needed 6); part (1) in {g['part1_worlds']}, part (2) in {g['part2_worlds']}; "
                   f"median below 0.10 in {g['below_low']}, above 0.90 in {g['above_high']}. Forecast for 1,024 visits (largest naive life): "
                   f"{g['forecast_1024_hours_naive_max']:.2f} CPU-hours (bound 6). **Gate {'passes' if g['passed'] else 'fails'}.**", ""]
        md += ["### Per life", "",
               "| Arm | World | Mean score, 64 visits | Blind, 64 visits | Late level (visits 48–63) | Score trend | Trend of learning | First group's first-visit levels | Mean first-visit level, n ≥ 3 | CPU-s per visit | Digest = generator's |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        for arm in ARMS:
            for w in WORLDS:
                v = L[key[(arm, w)]]
                fg = [d["level"] for d in v["first_visit"] if d["n"] <= 2]
                blind = np.mean([x for x in v["blind_by_age"] if x is not None])
                md.append(f"| {arm} | {w} | {pct(v['mean_score'])} | {pct(blind)} | {pct(v['late_level'])} | {f(v['score_trend'])} | {f(v['learning_trend'])} | "
                          f"{f(fg[0])}, {f(fg[1])} | {f(v['mean_after_first_group'])} | {v['cpu_per_visit']:.2f} | {'yes' if v['digest_equal'] else 'NO'} |")
        md += ["", "Scores in percent of asked items; levels as shares; trends are slopes per unit of ln(visit + 1) and of ln(n).", ""]
        md += ["### The first group: the mean of each task's last 32 asked items at the end of each of its visits", "",
               "| Arm | World | Visits of group 0: level of task a / task b |", "|---|---|---|"]
        for arm in ARMS:
            for w in WORLDS:
                v = L[key[(arm, w)]]
                by = {}
                for e in v["first_group_end_of_visit"]:
                    by.setdefault(e["visit"], []).append(e["level"])
                md.append(f"| {arm} | {w} | " + "; ".join(f"v{vv}: {f(a[0], 2)}/{f(a[1], 2)}" for vv, a in sorted(by.items())) + " |")
        md += ["", "### Kept at return: share right in the visit's first update (before any update of the visit), pooled by absence class", "",
               "| Arm | World | Short: returns, all / wrap / no wrap | Medium | Long | Short, first 32 | Medium, first 32 | Long, first 32 |",
               "|---|---|---|---|---|---|---|---|"]
        for arm in ARMS:
            for w in WORLDS:
                kp = L[key[(arm, w)]]["kept_pooled"]
                cells = []
                for c in ("short", "medium", "long"):
                    k = kp[c]
                    cells.append(f"{k['returns']}: {pct(k['first_update']['share'])} / {pct(k['first_update_wrap']['share'])} / {pct(k['first_update_nowrap']['share'])}")
                cells += [pct(kp[c]["first32"]["share"]) for c in ("short", "medium", "long")]
                md.append(f"| {arm} | {w} | " + " | ".join(cells) + " |")
        md += ["", "### Symmetric operations, split by `mirror` (percent right, items)", "",
               "| Arm | World | Symmetric, mirror = 1 | Symmetric, mirror = 0 | Non-symmetric | Blind on mirror = 1 |", "|---|---|---|---|---|---|"]
        for arm in ARMS:
            for w in WORLDS:
                s = L[key[(arm, w)]]["symmetric"]
                md.append(f"| {arm} | {w} | {pct(s['mirror']['share'])} ({s['mirror']['n']:,}) | {pct(s['no_mirror']['share'])} ({s['no_mirror']['n']:,}) | "
                          f"{pct(s['non_symmetric']['share'])} ({s['non_symmetric']['n']:,}) | {pct(s['blind_mirror']['share'])} |")
        # loss, restoration, late tasks
        md += ["", "### Loss when the stream leaves a task, restoration at return, late tasks against the first", ""]
        md += ["| Arm | Absence class | Returns (8 worlds) | Level before (end of previous visit) | Kept, first update | Drop | Relearning reached in the visit | Median taught items to 0.8 within the visit |",
               "|---|---|---|---|---|---|---|---|"]
        for arm in ARMS:
            for c in ("short", "medium", "long"):
                rows = []
                for w in WORLDS:
                    sc = json.loads((OUT / L[key[(arm, w)]]["score_table"]).read_text())
                    rows += [(k, x) for k, x in zip(sc["kept"], sc["relearn"]) if k["cls"] == c]
                if not rows:
                    md.append(f"| {arm} | {c} | 0 | – | – | – | – | – |"); continue
                bef = [k["before"] for k, _ in rows if k["before"] is not None]
                kept_r = sum(k["first_update"]["right"] for k, _ in rows); kept_n = sum(k["first_update"]["n"] for k, _ in rows)
                kept = kept_r / kept_n if kept_n else None
                rl = [x["items"] for _, x in rows if x["items"] is not None]
                md.append(f"| {arm} | {c} | {len(rows)} | {pct(np.mean(bef))} | {pct(kept)} | {pct(np.mean(bef) - kept) if kept is not None else '–'} | "
                          f"{len(rl)} of {len(rows)} | {f(float(np.median(rl)), 0) if rl else '–'} |")
        md += ["", "| Arm | First group (n = 1, 2) | n = 3–10 | n = 11–20 | n ≥ 21 | Tasks met (8 worlds) |", "|---|---|---|---|---|---|"]
        for arm in ARMS:
            b = {k: [] for k in ("a", "b", "c", "d")}; tot = 0
            for w in WORLDS:
                for d in L[key[(arm, w)]]["first_visit"]:
                    tot += 1
                    b["a" if d["n"] <= 2 else "b" if d["n"] <= 10 else "c" if d["n"] <= 20 else "d"].append(d["level"])
            md.append(f"| {arm} | " + " | ".join(f"{f(np.mean(x)) if x else '–'} ({len(x)})" for x in b.values()) + f" | {tot} |")
        md += ["", "Mean first-visit level (count of tasks) by order of meeting, pooled over the eight worlds.", ""]
        for arm in ARMS:
            for w in WORLDS:
                v = L[key[(arm, w)]]
                for d in v["first_visit"]:
                    fv_rows.append(dict(rung=rung, arm=arm, world=w, n=d["n"], task=d["task"], symbol=d["symbol"], op=d["op"], p=d["p"],
                                        first_visit=d["visit"], level=d["level"], first_visit_cost=d["cost"]))
                for vv, (s_, b_) in enumerate(zip(v["score_by_age"], v["blind_by_age"])):
                    sba_rows.append(dict(rung=rung, arm=arm, world=w, visit=vv, score=s_, blind=b_))
    for name, rows in (("E39_L1_first_visit.csv", fv_rows), ("E39_L1_score_by_age.csv", sba_rows)):
        with open(OUT / name, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    (OUT / "E39_L1_TABLES.md").write_text("# E39 job L1: tables\n\nGenerated by `smallworld/sw_l1_report.py` from `E39_L1_<rung>.json` and the score tables. "
                                           "Every task's first-visit level and cost: `E39_L1_first_visit.csv`; score by age and the blind predictor's: "
                                           "`E39_L1_score_by_age.csv`.\n\n" + "\n".join(md) + "\n")
    print("tables written for", rungs)


if __name__ == "__main__":
    main(sys.argv[1:])
