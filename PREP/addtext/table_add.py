"""Tables of addtext_m.py's runs (res/*.json), by E52's worker. Standard library only.  python3 -I table_add.py   # writes TABLE_ADD.md
For each scale (n9: numbers 0 to 9, 20 of 100 pairs never shown; n99: numbers 0 to 99, 2,000 of 10,000 never shown), over the seeds,
smallest / median / largest of: the cell's whole-answer score on sums shown (no learning) and never shown, and by answer length; the
ceilings of the most frequent next byte given the last K = 1 to 4 bytes (counts online), whole answer and by length, shown and never
shown; and the cell's answer bytes and whole answer in each learning block (to see a rise)."""
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
sml = lambda v: "–" if not v else f"{100 * min(v):.1f} / {100 * st.median(v):.1f} / {100 * max(v):.1f}"
L = []
for sc, title in (("n9", "numbers 0 to 9, 20 of 100 pairs never shown"), ("n99", "numbers 0 to 99, 2,000 of 10,000 pairs never shown")):
    R = [json.loads(p.read_text()) for p in sorted((HERE / "res").glob(f"{sc}_s*.json"))]
    if not R:
        continue
    L.append(f"\n## {title}: {len(R)} seeds (per cent; smallest / median / largest)\n")
    lens = sorted({l for r in R for e in r["extra"][-2:] for l in e["by_len"]}, key=int)
    L.append("| who | sums | whole answer | " + " | ".join(f"answer of {l} digit{'s' if l != '1' else ''}" for l in lens) + " | answer bytes |")
    L.append("|" + "---|" * (4 + len(lens)))
    for which, idx in (("shown", 6), ("NEVER shown", 7)):
        L.append(f"| cell (rule Q) | {which} | {sml([r['rows'][idx][3] for r in R])} | "
                 + " | ".join(sml([r['extra'][idx]['by_len'][l][0] for r in R if l in r['extra'][idx]['by_len']]) for l in lens)
                 + f" | {sml([r['rows'][idx][1] for r in R])} |")
        for k in ("1", "2", "3", "4"):
            L.append(f"| last {k} byte{'s' if k != '1' else ''}, most frequent next | {which} | {sml([r['extra'][idx]['ceiling'][k]['whole'] for r in R])} | "
                     + " | ".join(sml([r['extra'][idx]['ceiling'][k]['by_len'][l] for r in R if l in r['extra'][idx]['ceiling'][k]['by_len']]) for l in lens)
                     + f" | {sml([r['extra'][idx]['ceiling'][k]['answer'] for r in R])} |")
    L.append("\nThe cell in the learning blocks (answer bytes right / whole answer right):\n")
    L.append("| block | answer bytes | whole answer |\n|---|---|---|")
    for b in range(6):
        L.append(f"| {b + 1} | {sml([r['rows'][b][1] for r in R])} | {sml([r['rows'][b][3] for r in R])} |")
    L.append("\nShare of answers of each length in the test blocks (shown / never shown), first seed: "
             + "; ".join(f"{l} digit(s): {R[0]['extra'][6]['by_len'].get(l, [0, 0])[1]} / {R[0]['extra'][7]['by_len'].get(l, [0, 0])[1]}" for l in lens) + ".")
fl = HERE / "logs" / "FAILED.txt"
L.append("\nFailed runs: " + (fl.read_text().strip() if fl.exists() else "none") + ".")
(HERE / "TABLE_ADD.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
