"""Tables of warmup.py's runs (resW/*.json), by E52's worker. Standard library only.  python3 -I table_W.py   # writes TABLE_WARMUP.md
For each addition (none, a, b, c) and warm-up (W1, W2, W3; R = 1, 5, 20): whole answer right, per cent, median over the seeds (smallest -
largest in brackets), shown / never shown, at the end of the warm-up and at life lines 10,000, 30,000, 60,000; beside each, the control
(no warm-up) read at the same total count of lines."""
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = {}
for p in sorted((HERE / "resW").glob("w_*_s*.json")):
    r = json.loads(p.read_text()); a = r["args"]
    key = (a["add"], a["warm"], int(a.get("reps", 1)) if a["warm"] != "none" else 0)
    R.setdefault(key, {})[int(a["seed"])] = r


def at(r, lines):
    for x in r["reads"]:
        if x["lines"] == lines:
            return x
    return None


def cell(vals):
    v = [x for x in vals if x is not None]
    return "–" if not v else f"{100 * st.median(v):.1f} ({100 * min(v):.1f}-{100 * max(v):.1f})"


L = []
for add in ("none", "a", "b", "c"):
    if not any(k[0] == add for k in R):
        continue
    C = R.get((add, "none", 0), {})
    L.append(f"\n## addition {add}: whole answer right, per cent, median (smallest-largest) over seeds; shown / never shown; control = no warm-up at equal total lines ({len(C)} seeds)\n")
    L.append("| warm-up | R | seeds | read at | total lines | shown | never shown | control shown | control never |")
    L.append("|" + "---|" * 9)
    for w in ("W1", "W2", "W3"):
        for reps in (1, 5, 20):
            D = R.get((add, w, reps), {})
            if not D:
                continue
            for i, nm in enumerate(("end of warm-up", "life 10,000", "life 30,000", "life 60,000")):
                rows = [(s, r["reads"][i]) for s, r in D.items() if len(r["reads"]) > i]
                if not rows:
                    continue
                tl = rows[0][1]["lines"]
                cs = [at(C[s], tl) for s, _ in rows if s in C]
                L.append(f"| {w} | {reps} | {len(rows)} | {nm} | {tl} | {cell([x['shown'] for _, x in rows])} | {cell([x['never'] for _, x in rows])} | "
                         f"{cell([x['shown'] for x in cs if x])} | {cell([x['never'] for x in cs if x])} |")
fl = HERE / "logs" / "FAILED_W.txt"
L.append("\nFailed runs: " + (fl.read_text().strip() if fl.exists() else "none") + ".")
(HERE / "TABLE_WARMUP.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
