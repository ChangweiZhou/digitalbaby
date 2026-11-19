"""Part B: online readers at the output end naming the next byte (drive12.py's outB/*.json), setting (3) for 65,536 bytes on 15 seeds.
By E52's worker. Standard library only.  python3 -I table_B.py   # writes TABLE_B.md
Share of the last quarter's bytes whose next byte the reader names (largest unit; for Q also the share emitted right): smallest /
median / largest over the seeds, and on how many seeds above the reviewer's levels (present byte alone 0.106, two bytes 0.123, three
bytes 0.25); beside it, from the stream alone on each seed, the most frequent successor of the last 1, 2, 3 bytes (counts online)."""
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = [json.loads(p.read_text()) for p in sorted((HERE / "outB").glob("B*_s*.json"))]
LEV = (("present byte alone", 0.106), ("two bytes", 0.123), ("three bytes", 0.25))
sml = lambda v: f"{min(v):.3f} / {st.median(v):.3f} / {max(v):.3f}"
L = [f"## Setting (3), long=1-1/4 no tau, 65,536 bytes: {len(R)} seeds\n",
     "| reader | inputs | right in the last quarter | above 0.106 on | above 0.123 on | above 0.25 on |", "|---|---|---|---|---|---|"]
for rd, key in (("D", "D"), ("Q (largest unit)", "Q"), ("Q (emitted)", "Q_emitted")):
    for inp in ("x", "x,s2,s4"):
        v = [r[key][inp] for r in R]
        L.append(f"| {rd} | {inp} | {sml(v)} | " + " | ".join(str(sum(x > lv for x in v)) for _, lv in LEV) + " |")
L.append("\nFrom the stream on the same seeds (most frequent successor, counts online from the start): "
         + "; ".join(f"{k} byte{'s' if k != '1' else ''} {sml([r['context'][k] for r in R])}" for k in ("1", "2", "3")) + ".\n")
L += ["| seed | D x | D x,s2,s4 | Q x | Q x,s2,s4 | stream 1 / 2 / 3 bytes |", "|---|---|---|---|---|---|"]
for r in sorted(R, key=lambda r: int(r["label"].split("_s")[1])):
    L.append(f"| {r['label'].split('_s')[1]} | {r['D']['x']:.3f} | {r['D']['x,s2,s4']:.3f} | {r['Q']['x']:.3f} | {r['Q']['x,s2,s4']:.3f} | "
             + " / ".join(f"{r['context'][k]:.3f}" for k in ("1", "2", "3")) + " |")
fl = HERE / "logs" / "FAILED_B.txt"
L.append("\nBuild: the reviewer's tissue.c and tis.py at 97ba8349, -ffp-contract=off. Failed runs: " + (fl.read_text().strip() if fl.exists() else "none") + ".")
(HERE / "TABLE_B.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
