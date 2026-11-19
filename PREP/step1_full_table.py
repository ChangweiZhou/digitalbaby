"""The tables of step 1 at full length: reads out/<seed>_<rule>.json of PREP/step1_full.py and writes out/STEP1_FULL_TABLES.md.

    python3 PREP/step1_full_table.py <out folder>
"""
import json
import sys
from pathlib import Path

D = Path(sys.argv[1])
SEEDS, RULES = (73129, 93683, 116089), ("first", "coincidence", "fixed")
STARTS = ("0", "86016", "172032", "253952")
md = ["# Step 1 at full length on the fast program: tables", "",
      "Written by `PREP/step1_full_table.py` from `out/<seed>_<rule>.json` (`PREP/step1_full.py`). Shares: neither silent nor constant, "
      "with the number of cells at d. S(d, L): the nearest-centroid score of `PREP/readings.py` (balanced accuracy, chance 1/46 = 0.0217). "
      "C0(L): what the present byte alone gives.", ""]
rows = []
for s in SEEDS:
    for r in RULES:
        p = D / f"{s}_{r}.json"
        if not p.exists():
            rows.append((s, r, None)); continue
        rows.append((s, r, json.loads(p.read_text())))
md += ["## Cost and connections", "", "| Seed | Tissue | complete | CPU-min | ms a byte (min–max of the 16 records) | connections a cell at 16,384 / 131,072 / 262,144 |",
       "|---|---|---|---|---|---|"]
for s, r, x in rows:
    if x is None:
        md.append(f"| {s} | {r} | missing | | | |"); continue
    ms = [q["cpu_ms_a_byte"] for q in x["records"]]
    pc = {q["byte"]: q["per_cell"] for q in x["records"]}
    md.append(f"| {s} | {r} | {x['complete']} | {x['cpu_s'] / 60:.1f} | {min(ms):.3f}–{max(ms):.3f} | "
              f"{pc.get(16384, float('nan')):.2f} / {pc.get(131072, float('nan')):.2f} / {pc.get(262144, float('nan')):.2f} |")
md += [""]
for s, r, x in rows:
    if x is None:
        continue
    md += [f"## Seed {s}, {r}", "", "| Window (first byte) | share d=1 | share d=2 | share d=4 | S(1,0) | S(1,1) | S(1,2) | S(2,0) | S(2,1) | S(2,2) | S(4,0) | S(4,1) | S(4,2) | C0(1) | C0(2) |",
           "|" + "---|" * 15]
    for w in STARTS:
        if w not in x["windows"]:
            continue
        R = x["windows"][w]["readings"]
        sh = [f"{R['shares'][d]['share']:.3f} of {R['shares'][d]['n']}" if R["shares"][d]["share"] is not None else "— of 0" for d in ("1", "2", "4")]
        sc = []
        for d in ("1", "2", "4"):
            for L in ("0", "1", "2"):
                v = R["scores"][d]["by_lag"].get(L)
                sc.append("—" if not v else f"{v['S']:.4f}")
        md.append(f"| {w} | " + " | ".join(sh + sc + [f"{R['c0']['1']['S']:.4f}", f"{R['c0']['2']['S']:.4f}"]) + " |")
    md += [""]
(D / "STEP1_FULL_TABLES.md").write_text("\n".join(md))
print("\n".join(md[:20]))
