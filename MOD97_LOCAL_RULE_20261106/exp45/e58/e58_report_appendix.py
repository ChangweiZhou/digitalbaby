"""E58: the appendix tables of E58_REPORT.md, printed as Markdown from the receipts.  python3 exp45/e58/e58_report_appendix.py"""
import json
from pathlib import Path
D = Path(__file__).resolve().parent
SEEDS = (73129, 93683, 116089); LATE = (4096, 5120, 6144, 7168, 8192)
LIFE4 = ("add97", "sub97", "add73", "sub73"); FIRST2 = ("add89", "sub89")
NP = {"05": 344, "10": 689, "20": 1378}
L = lambda a, r, f, s: json.loads((D / f"e58_{a}_{r}_f{f}_{s}.json").read_text())
red = lambda x: x["scores"]["summary"]["reduction_score"]
pct = lambda d: 100.0 * d["exact"] / d["n"]
mean = lambda v: sum(v) / len(v)


def first(r, lv):
    for x in r["rows"]:
        if red(x) >= lv:
            return x["updates"]


def cell(u, f):
    return "never" if u is None else f"{u} ({u * 64 / (4 * NP[f]):.1f})"


print("### A.1 Part B: R-late of the reduction score on each run's own untaught pairs, and the taught panel at row 8,192\n")
print("| Rule | Arm | f | R-late 73129 | R-late 93683 | R-late 116089 | taught 73129 | taught 93683 | taught 116089 |\n|---|---|---|---|---|---|---|---|---|")
for r in "SB":
    for f in ("05", "10", "20"):
        for a in ("free", "free-wd0", "term"):
            runs = [L(a, r, f, s) for s in SEEDS]
            rl = [mean([red(x) for x in q["rows"] if x["updates"] in LATE]) for q in runs]
            tp = [q["rows"][-1]["scores"]["summary"]["taught_add_sub_mean"] for q in runs]
            print(f"| {r} | {a} | 0.{f} | " + " | ".join(f"{v:.1f}" for v in rl + tp) + " |")
print("\n### A.2 First row at which the reduction score is 80 or more and 90 or more (in parentheses: presentations of each taught pair by that row)\n")
print("| Rule | f | Arm | 80: 73129 | 80: 93683 | 80: 116089 | 90: 73129 | 90: 93683 | 90: 116089 |\n|---|---|---|---|---|---|---|---|---|")
for r in "SB":
    for f in ("05", "10", "20"):
        for a in ("free", "free-wd0", "term"):
            runs = [L(a, r, f, s) for s in SEEDS]
            print(f"| {r} | 0.{f} | {a} | " + " | ".join(cell(first(q, lv), f) for lv in (80, 90) for q in runs) + " |")
cnt = {}
for lv in (80, 90):
    c = dict(term_later=0, equal=0, term_earlier=0, free_never_term_reaches=0, term_never_free_reaches=0, neither=0)
    for r in "SB":
        for f in ("05", "10", "20"):
            for s in SEEDS:
                t = first(L("term", r, f, s), lv)
                for a in ("free", "free-wd0"):
                    q = first(L(a, r, f, s), lv)
                    if t is None and q is None: c["neither"] += 1
                    elif q is None: c["free_never_term_reaches"] += 1
                    elif t is None: c["term_never_free_reaches"] += 1
                    elif t > q: c["term_later"] += 1
                    elif t == q: c["equal"] += 1
                    else: c["term_earlier"] += 1
    cnt[lv] = c
print("\nComparisons of `term` with `free` and with `free-wd0` (2 rules x 3 fractions x 3 seeds x 2 arms = 36 at each level):\n")
for lv, c in cnt.items():
    print(f"- level {lv}: " + "; ".join(f"{k.replace('_', ' ')} {v}" for k, v in c.items()))
print("\n### A.3 Part A, point 5, every row of the three free arms: mean of the life's four panels / mean of the first family's two panels / norm of the tensors outside the byte table relative to the lived network / norm of their change relative to the lived norm\n")
for r in "SB":
    for a in ("free", "free-wd0", "free-anchor"):
        for s in SEEDS:
            q = L(a, r, "10", s)
            tot = lambda x, k: sum(v[k] ** 2 for v in x["norms"].values()) ** .5
            n0 = tot(q["rows"][0], "norm")
            cells = []
            for x in q["rows"]:
                e = x["scores"]["earlier_k12"]
                cells.append(f"{x['updates']}: {mean([pct(e[n]) for n in LIFE4]):.1f} / {mean([pct(e[n]) for n in FIRST2]):.1f} / {tot(x, 'norm') / n0:.3f} / {tot(x, 'diff_from_lived') / n0:.3f}")
            print(f"- **{r} {a} {s}**: " + "; ".join(cells))
