"""Tables of the spread over births (drive7.py's out/*.json), by E52's worker. Standard library only.

    python3 -I table7.py          # writes SEEDS15.md

1. One row a tissue (window A2): connections a cell (mean / largest); largest distance; S(d, L) by the tool at d = 1, 2, 4, 8, 10 and
   L = 0, 1, 2, and near the output lines, ** above the seed's own level (L = 0: 0.1103; L = 1, 2: c0(L) + 0.02); cells / answering
   cells at d = 8, 10 and near the output lines; divergence after 512 bytes.
2. Per tissue (N0, N64, N256): on how many seeds all nine entries of step 1 (L = 0, 1, 2 at d = 1, 2, 4) are above their levels; on how
   many the present byte is above 0.1103 at d = 8, at d = 10 and near the output lines; smallest, median and largest of each entry.
3. N0 seeds whose far cells are weak (S(10, 0) not above 0.1103) beside the others: the birth's numbers.
"""
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
DS = ("1", "2", "4", "8", "10")
LG = ("0", "1", "2")


def f(v, n=3):
    return "–" if v is None else f"{v:.{n}f}"


def above(v, lev):
    return v is not None and v > lev


rows = {}
for p in sorted((HERE / "out").glob("*.json")):
    r = json.loads(p.read_text())
    tis, seed = r["label"].split("_s")
    rows.setdefault(tis, {})[int(seed)] = r

L = []
for tis in ("N0", "N64", "N256"):
    R = rows.get(tis, {})
    L += [f"\n## {tis} ({len(R)} seeds)\n",
          "| seed | conns mean / max | largest d | " + " | ".join(f"S(d,{g}) d1 / 2 / 4 / 8 / 10" for g in LG) + " | near S lag 0 / 1 / 2 | levels lag 1 / 2 | cells answering / all d8, d10, near | div 512 |",
          "|" + "---|" * 9]
    for s in sorted(R):
        r = R[s]; a = r["A2"]; lev = r["levels"]; nr = r["near"]
        cell = lambda v, g: f(v) + ("**" if above(v, lev[g]) else "")
        L.append(f"| {s} | {a['at_end']['mean']:.1f} / {a['at_end']['max']} | {r['max_distance']} | "
                 + " | ".join(" / ".join(cell(a['S'][d][g], g) for d in DS) for g in LG) + " | "
                 + " / ".join(cell(nr['S'][g], g) for g in LG) + f" | {lev['1']:.3f} / {lev['2']:.3f} | "
                 + f"{a['cells_any_output']['8']}/{a['cells']['8']}, {a['cells_any_output']['10']}/{a['cells']['10']}, {nr['answering']}/{nr['cells']} | {r['divergence']['512']:.0e} |")
    # counts
    nine = [s for s in R if all(above(R[s]["A2"]["S"][d][g], R[s]["levels"][g]) for d in ("1", "2", "4") for g in LG)]
    n8 = [s for s in R if above(R[s]["A2"]["S"]["8"]["0"], 0.1103)]
    n10 = [s for s in R if above(R[s]["A2"]["S"]["10"]["0"], 0.1103)]
    nn = [s for s in R if above(R[s]["near"]["S"]["0"], 0.1103)]
    L.append(f"\n{tis}: all nine entries of step 1 above their levels on {len(nine)} of {len(R)} seeds; present byte above 0.1103 at d = 8 on {len(n8)}, "
             f"at d = 10 on {len(n10)}, near the output lines on {len(nn)}; divergence after 512 bytes 0 on {sum(R[s]['divergence']['512'] == 0 for s in R)}.")
    L.append("\n| entry | smallest | median | largest | above its level on |\n|---|---|---|---|---|")
    for d in DS + ("near",):
        for g in LG:
            vals = [(R[s]["near"]["S"][g] if d == "near" else R[s]["A2"]["S"][d][g]) for s in R]
            ok = [v for v in vals if v is not None]
            nab = sum(above(v, R[s]["levels"][g]) for v, s in zip(vals, R))
            if ok:
                L.append(f"| S({d}, {g}) | {min(ok):.3f} | {st.median(ok):.3f} | {max(ok):.3f} | {nab} of {len(R)} |")
# part 3
R = rows.get("N0", {})
if R:
    weak = sorted(s for s in R if not above(R[s]["A2"]["S"]["10"]["0"], 0.1103))
    strong = sorted(s for s in R if s not in weak)
    L.append(f"\n## N0: seeds with S(10, 0) not above 0.1103 ({len(weak)}: {', '.join(map(str, weak))}) beside the others ({len(strong)})\n")
    keys = [("cells at d = 8 to 10", lambda r: r["cells_d8_10"]), ("largest distance", lambda r: r["max_distance"]),
            ("input lines in use", lambda r: r["lines_present"]), ("input lines in use reached by a cell", lambda r: r["lines_reached"]),
            ("cells near the output lines", lambda r: r["near"]["cells"]), ("inhibitory cells near the output lines", lambda r: r["near"]["inhibitory"]),
            ("cells at d = 10", lambda r: r["A2"]["cells"]["10"]), ("connections a cell", lambda r: r["A2"]["at_end"]["mean"]),
            ("S(10, 0)", lambda r: r["A2"]["S"]["10"]["0"]), ("S(8, 0)", lambda r: r["A2"]["S"]["8"]["0"])]
    L.append("| number | " + " | ".join(map(str, weak)) + " | others: smallest / median / largest |\n|" + "---|" * (len(weak) + 2))
    for nm, fn in keys:
        o = [fn(R[s]) for s in strong if fn(R[s]) is not None]
        L.append(f"| {nm} | " + " | ".join(str(round(fn(R[s]), 3)) if fn(R[s]) is not None else "–" for s in weak)
                 + (f" | {min(o):.3g} / {st.median(o):.3g} / {max(o):.3g} |" if o else " | – |"))
fl = HERE / "logs" / "FAILED.txt"
L.append("\nFailed runs: " + (fl.read_text().strip() if fl.exists() else "none") + ".")
(HERE / "SEEDS15.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
