"""The table of the round's grid on the reviewer's tissue program (drive.py), from out/*.json. Read-only; standard library.

    python3 grid_table.py [seed ...]      # writes GRID_TABLE.md beside this file (all seeds found, or those given)

One row a tissue, window A2: connections a cell; median gain; cells sending a step (drive.py's active_per_step); share neither silent nor
constant at d = 1, 2, 4; S(d, 0), S(d, 1), S(d, 2) at d = 1, 2, 4; c0 at lags 0, 1, 2; ms a byte. A score is marked ** when it is above its
level: for lag 0 the level 0.1103; for lags 1 and 2, the c0 of that lag in the same window plus 0.02. A tissue whose run failed is listed
with its log's last line.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEVEL0 = 0.1103


def f(v, n=3):
    return "–" if v is None else f"{v:.{n}f}"


def main(seeds):
    rows = []
    for p in sorted((HERE / "out").glob("*.json")):
        r = json.loads(p.read_text())
        s = int(r["args"].get("seed", 73129))
        if seeds and s not in seeds:
            continue
        rows.append(r)
    order = lambda r: (r["args"]["rule"], int(r["args"].get("seed", 73129)), eval(r["args"]["astar"]) * -1, r["args"]["floor"], r["args"]["by_use"], r["args"]["gain"])
    rows.sort(key=order)
    L = ["| Tissue | seed | conns a cell | median gain | sending a step | share d1 / d2 / d4 | S(d,0) d1 / d2 / d4 | S(d,1) d1 / d2 / d4 | S(d,2) d1 / d2 / d4 | c0 lag 0 / 1 / 2 | ms a byte |",
         "|" + "---|" * 11]
    above0 = []
    for r in rows:
        a = r["A2"]; c0 = a["c0"]
        lev = {"0": LEVEL0, "1": (c0.get("1") or 0) + 0.02, "2": (c0.get("2") or 0) + 0.02}

        def sc(d, Lg):
            v = a["S"][d][Lg]
            return f(v, 4) + ("**" if v is not None and v > lev[Lg] else "")
        cells = [" / ".join(sc(d, Lg) for d in ("1", "2", "4")) for Lg in ("0", "1", "2")]
        L.append(f"| {r['label']} | {r['args'].get('seed', '73129')} | {r['conns_per_cell']:.1f} | {r['gain_median']:.2f} | {r['active_per_step']:.0f} | "
                 + " / ".join(f(a["share"][d]) for d in ("1", "2", "4")) + " | " + " | ".join(cells) + " | "
                 + " / ".join(f(c0.get(k), 4) for k in ("0", "1", "2")) + f" | {r['ms_per_byte']:.1f} |")
        if a["S"]["2"]["0"] is not None and a["S"]["2"]["0"] > LEVEL0:
            above0.append(r)
    failed = (HERE / "logs" / "FAILED.txt")
    if failed.exists():
        L.append("\n**Failed runs**\n")
        for nm in failed.read_text().split():
            if nm == "FAILED":
                continue
            lg = HERE / "logs" / f"{nm}.log"
            last = lg.read_text().strip().splitlines()[-1] if lg.exists() and lg.read_text().strip() else ""
            L.append(f"- {nm}: {last[:200]}")
    L.append(f"\nMarked ** : above the level (lag 0: {LEVEL0}; lags 1 and 2: c0 of the lag + 0.02). Tissues: {len(rows)}. "
             f"Settings with S(2, 0) in A2 above {LEVEL0}: {len(above0)}: " + ", ".join(r["label"] for r in above0))
    (HERE / "GRID_TABLE.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return above0


if __name__ == "__main__":
    main([int(s) for s in sys.argv[1:]])
