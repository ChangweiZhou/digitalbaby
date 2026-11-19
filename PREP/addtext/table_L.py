"""Tables of levels.py's runs (resL/*.json), by E52's worker. Standard library only.  python3 -I table_L.py [n9|n99]   # writes TABLE_LEVELS_<scale>.md
Per input (level, scrambled, onehot), cells (fixed, adapt) and number of cells: whole answer right, per cent, smallest / median / largest
over the seeds, on sums shown (no learning) and never shown; the same by answer length; the whole answer in learning blocks 1 and 6."""
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sc = sys.argv[1] if len(sys.argv) > 1 else "n9"
sml = lambda v: "–" if not v else f"{100 * min(v):.1f} / {100 * st.median(v):.1f} / {100 * max(v):.1f}"
R = {}
for p in sorted((HERE / "resL").glob(f"{sc}_*_s*.json")):
    r = json.loads(p.read_text()); a = r["args"]
    R.setdefault((a["input"], a["cells"], int(a["nc"])), []).append(r)
lens = sorted({l for v in R.values() for r in v for l in r["rows"][6][1]["by_len"]}, key=int)
L = [f"## {sc}: whole answer right, per cent, smallest / median / largest over seeds\n",
     "| input | cells | n | seeds | shown | NEVER shown | " + " | ".join(f"shown, {l} digit(s)" for l in lens) + " | " + " | ".join(f"never, {l} digit(s)" for l in lens) + " | block 1 | block 6 |",
     "|" + "---|" * (8 + 2 * len(lens))]
for inp in ("level", "scrambled", "onehot"):
    for cel in ("fixed", "adapt"):
        for n in (16, 64, 256):
            v = R.get((inp, cel, n))
            if not v:
                continue
            L.append(f"| {inp} | {cel} | {n} | {len(v)} | {sml([r['rows'][6][1]['whole'] for r in v])} | {sml([r['rows'][7][1]['whole'] for r in v])} | "
                     + " | ".join(sml([r['rows'][6][1]['by_len'][l][0] for r in v if l in r['rows'][6][1]['by_len']]) for l in lens) + " | "
                     + " | ".join(sml([r['rows'][7][1]['by_len'][l][0] for r in v if l in r['rows'][7][1]['by_len']]) for l in lens) + " | "
                     + f"{sml([r['rows'][0][1]['whole'] for r in v])} | {sml([r['rows'][5][1]['whole'] for r in v])} |")
fl = HERE / "logs" / "FAILED_L.txt"
L.append("\nFailed runs: " + (fl.read_text().strip() if fl.exists() else "none") + ".")
(HERE / f"TABLE_LEVELS_{sc}.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
