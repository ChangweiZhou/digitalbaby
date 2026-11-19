"""Part A: the linear reader near the output lines with inputs x and [x, s2, s4] (drive11.py's outA/*.json), settings (3) and (7) on 15
seeds. By E52's worker. Standard library only.  python3 -I table_A.py   # writes TABLE_A.md
Per setting, group and input set, for the bytes 0 to 3 back: all values named right, end mark, equals sign (smallest / median / largest
over the seeds), on how many seeds the end mark and the equals sign are above 0.9, digits above 0.9 (smallest / median / largest)."""
import json
import math
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
SET = {"3": "long=1-1/4, no tau", "7": "no long, no tau (base)"}
R = {}
for p in sorted((HERE / "outA").glob("A*_s*.json")):
    r = json.loads(p.read_text()); n, s = r["label"][1:].split("_s"); R.setdefault(n, {})[int(s)] = r


def sml(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return "–" if not v else f"{min(v):.2f} / {st.median(v):.2f} / {max(v):.2f}"


L = []
for n in ("3", "7"):
    D = R.get(n, {})
    if not D:
        continue
    L.append(f"\n## ({n}) {SET[n]}: {len(D)} seeds\n")
    for grp in ("near output", "within 1/8 of a driver"):
        for inp in ("x", "x,s2,s4"):
            L.append(f"**{grp}, inputs {inp}**\n")
            L.append("| bytes back | all values | end mark | equals sign | end mark > 0.9 on | equals sign > 0.9 on | digits > 0.9 (of 10) | values never named |")
            L.append("|" + "---|" * 8)
            for lag in "0123":
                g = [r["groups"][grp]["inputs"][inp][lag] for r in D.values()]
                L.append(f"| {lag} | {sml([x['overall'] for x in g])} | {sml([x['end'] for x in g])} | {sml([x['eq'] for x in g])} | "
                         f"{sum(x['end'] > 0.9 for x in g if not math.isnan(x['end']))} | {sum(x['eq'] > 0.9 for x in g if not math.isnan(x['eq']))} | "
                         f"{sml([x['digits_above_09'] for x in g])} | {sml([x['never'] for x in g])} |")
            L.append("")
fl = HERE / "logs" / "FAILED_A.txt"
L.append("Build: the reviewer's tissue.c and tis.py at 97ba8349, -ffp-contract=off; 16,384 bytes. Failed runs: " + (fl.read_text().strip() if fl.exists() else "none") + ".")
(HERE / "TABLE_A.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
