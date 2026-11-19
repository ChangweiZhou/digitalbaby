"""Table of the runs of the reviewer's program at d81a85c7 (ratio control gain=2), from out/*.json. Standard library only.

    python3 grid_table3.py [seed ...]      # writes GRID_TABLE3.md beside this file

One row a tissue, window A2: connections a cell; median gain; cells sending a step; share neither silent nor constant at d = 1, 2, 4;
S(d, 0), S(d, 1), S(d, 2) at d = 1, 2, 4; c0 at lags 0, 1, 2; ms a byte. A score is marked ** above its level, the reviewer's fixed
levels: lag 0: 0.1103; lag 1: 0.090; lag 2: 0.076. Failed runs are listed with their log's last line.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEV = {"0": 0.1103, "1": 0.090, "2": 0.076}


def f(v, n=3):
    return "–" if v is None else f"{v:.{n}f}"


def main(seeds):
    rows = []
    for p in sorted((HERE / "out").glob("*.json")):
        r = json.loads(p.read_text())
        if seeds and int(r["args"].get("seed", 73129)) not in seeds:
            continue
        rows.append(r)
    a_ = lambda r, k, d: r["args"].get(k, d)
    rows.sort(key=lambda r: ({"fixed": 0, "first": 1, "coincidence": 2}[a_(r, "rule", "first")], -eval(a_(r, "rho", "7/8")),
                             eval(a_(r, "floor", "0")), int(a_(r, "by_use", 0)), int(a_(r, "seed", 73129))))
    L = ["| Tissue | seed | conns a cell | median gain | sending a step | share d1 / d2 / d4 | S(d,0) d1 / d2 / d4 | S(d,1) d1 / d2 / d4 | S(d,2) d1 / d2 / d4 | c0 lag 0 / 1 / 2 | ms a byte |",
         "|" + "---|" * 11]
    above = []
    for r in rows:
        a = r["A2"]; c0 = a["c0"]

        def sc(d, Lg):
            v = a["S"][d][Lg]
            return f(v, 4) + ("**" if v is not None and v > LEV[Lg] else "")
        L.append(f"| {r['label']} | {a_(r, 'seed', '73129')} | {r['conns_per_cell']:.1f} | {r['gain_median']:.2f} | {r['active_per_step']:.0f} | "
                 + " / ".join(f(a["share"][d]) for d in ("1", "2", "4")) + " | "
                 + " | ".join(" / ".join(sc(d, Lg) for d in ("1", "2", "4")) for Lg in ("0", "1", "2")) + " | "
                 + " / ".join(f(c0.get(k), 4) for k in ("0", "1", "2")) + f" | {r['ms_per_byte']:.1f} |")
        if all(a["S"][d]["0"] is not None and a["S"][d]["0"] > LEV["0"] for d in ("1", "2", "4")):
            above.append(r)
    failed = HERE / "logs" / "FAILED.txt"
    if failed.exists():
        L.append("\n**Failed runs**\n")
        for nm in failed.read_text().split():
            if nm == "FAILED":
                continue
            lg = HERE / "logs" / f"{nm}.log"
            t = lg.read_text().strip() if lg.exists() else ""
            L.append(f"- {nm}: {t.splitlines()[-1][:200] if t else ''}")
    L.append(f"\nMarked ** : above the level (lag 0: 0.1103; lag 1: 0.090; lag 2: 0.076). Tissues: {len(rows)}. "
             f"With S(d, 0) in A2 above 0.1103 at d = 1, 2 and 4: {len(above)}: " + ", ".join(r["label"] for r in above))
    (HERE / "GRID_TABLE3.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return above


if __name__ == "__main__":
    main([int(s) for s in sys.argv[1:]])
