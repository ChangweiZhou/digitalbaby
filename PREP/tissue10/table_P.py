"""Byte-by-byte readings near the output lines on 15 seeds (drive10.py's out/P*.json), by E52's worker. Standard library only.
    python3 -I table_P.py      # writes TABLE_P.md
Per setting and group (perbyte3's "near output" and "within 1/8 of a driver"), the LINEAR reader, for the bytes 0, 1, 2, 3 back:
smallest / median / largest over the seeds of the share of all values named right, of the digits (mean of the per-value shares), of
the end mark and of the equals sign; values never named (smallest / median / largest); and the counts asked: on how many seeds the end
mark and the equals sign are above 0.9, and how many of the ten digits are above 0.9 (smallest / median / largest over seeds). Beside it,
drive9's figures: connections a cell, made in the last quarter, divergence after 512 bytes, cells with a connection from an input line."""
import json
import math
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
SET = {"1": "long=1-1/4 tau=u2-256", "4": "long=8-1/2 tau=u2-256 (reference)", "3": "long=1-1/4, no tau", "2": "long=1-1/4 tau=u2-64",
       "5": "long=1-3/16 tau=u2-256", "6": "long=1-5/16 tau=u2-256", "7": "no long, no tau (base)"}
R = {}
for p in sorted((HERE / "out").glob("P*_s*.json")):
    r = json.loads(p.read_text())
    n, s = r["label"][1:].split("_s")
    R.setdefault(n, {})[int(s)] = r


def sml(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return "–" if not v else f"{min(v):.2f} / {st.median(v):.2f} / {max(v):.2f}"


L = []
for n in ("1", "4", "3", "2", "5", "6", "7"):
    D = R.get(n, {})
    if not D:
        continue
    L.append(f"\n## ({n}) {SET[n]}: {len(D)} seeds\n")
    L.append(f"connections a cell (mean) {sml([r['conns_mean'] for r in D.values()])}; largest {sml([r['conns_max'] for r in D.values()])}; "
             f"made in the last quarter {sml([r['made_last_quarter'] for r in D.values()])}; cells with a connection from an input line "
             f"{sml([r['cells_from_input_line'] for r in D.values()])}; divergence after 512 bytes 0 on {sum(r['divergence']['512'] == 0 for r in D.values())} of {len(D)}.\n")
    for grp in ("near output", "within 1/8 of a driver"):
        L.append(f"**{grp}** (cells {sml([r['groups'][grp]['cells'] for r in D.values()])}), linear reader\n")
        L.append("| bytes back | all values | digits | end mark | equals sign | values never named | end mark > 0.9 on | equals sign > 0.9 on | digits > 0.9 (of 10) |")
        L.append("|" + "---|" * 10)
        for lag in "0123":
            g = [r["groups"][grp]["lags"][lag]["linear"] for r in D.values()]
            L.append(f"| {lag} | {sml([x['overall'] for x in g])} | {sml([x['digits'] for x in g])} | {sml([x['end'] for x in g])} | {sml([x['eq'] for x in g])} | "
                     f"{sml([x['never'] for x in g])} | {sum(x['end'] > 0.9 for x in g if not math.isnan(x['end']))} | {sum(x['eq'] > 0.9 for x in g if not math.isnan(x['eq']))} | "
                     f"{sml([x['digits_above_09'] for x in g])} |")
        L.append("")
fl = HERE / "logs" / "FAILED.txt"
L.append("Build: the reviewer's tissue.c and tis.py at 97ba8349, -ffp-contract=off. Failed runs: " + (fl.read_text().strip() if fl.exists() else "none") + ".")
(HERE / "TABLE_P.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
