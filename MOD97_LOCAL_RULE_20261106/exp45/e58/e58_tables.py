"""E58: the tables and readings of the brief's "How the result is stated", printed from the receipts in this folder.
    python3 exp45/e58/e58_tables.py [A|B|all] [seed ...]      (from MOD97_LOCAL_RULE_20261106; standard library only)
Every figure is read from exp45/e58/e58_<arm>_<rule>_f<05|10|20>_<seed>.json. A missing or incomplete receipt is shown as '-'."""
import json
import sys
from pathlib import Path

D = Path(__file__).resolve().parent
SEEDS = (73129, 93683, 116089)
LATE = (4096, 5120, 6144, 7168, 8192)
MID = (512, 1024)
LIFE4 = ("add97", "sub97", "add73", "sub73")
FIRST2 = ("add89", "sub89")
MARGIN = 5.0
pct = lambda d: 100.0 * d["exact"] / d["n"]


def load(arm, rule, f, seed):
    p = D / f"e58_{arm}_{rule}_f{f}_{seed}.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    return r if r.get("complete") else None


def rows(r):
    return {x["updates"]: x for x in r["rows"]}


def red(x):
    return x["scores"]["summary"]["reduction_score"]


def com(x):
    return x["common"]["common_score"]


def mean(v):
    return sum(v) / len(v)


def figures(r):
    rw = rows(r)
    hi = max((u for u in rw if u >= 64), key=lambda u: red(rw[u]))
    return dict(rmid=mean([red(rw[u]) for u in MID]), rlate=mean([red(rw[u]) for u in LATE]), hi=red(rw[hi]), hi_row=hi,
                d84=red(rw[8192]) - red(rw[4096]), clate=mean([com(rw[u]) for u in LATE]), taught8192=rw[8192]["scores"]["summary"]["taught_add_sub_mean"])


def fig(arm, rule, f, seed, key="rlate"):
    r = load(arm, rule, f, seed)
    return None if r is None else figures(r)[key]


def f1(v):
    return "-" if v is None else f"{v:.1f}"


def verdict(x, y):
    """x and y: lists of three figures (one a seed). The brief's rule, with its ceiling."""
    if any(v is None for v in x + y):
        return "incomplete"
    d = [a - b for a, b in zip(x, y)]
    ceiling = sum(min(a, b) > 95 for a, b in zip(x, y))
    if ceiling >= 2:
        return "at the ceiling (" + ", ".join(f"{a:.1f}/{b:.1f}, 100-: {100 - a:.1f}/{100 - b:.1f}" for a, b in zip(x, y)) + ")"
    up = sum(v >= MARGIN for v in d); down = sum(v <= -MARGIN for v in d)
    if up >= 2 and down == 0:
        return "above"
    if down >= 2 and up == 0:
        return "below"
    return "not distinguished"


def scatter(rule, f="10"):
    """N for each seed: |R-late(free-again) - R-late(free)|; and whether the margin separates two runs of one arm under this rule."""
    n = []
    for s in SEEDS:
        a, b = fig("free-again", rule, f, s), fig("free", rule, f, s)
        n.append(None if a is None or b is None else abs(a - b))
    return n, (None if None in n else all(v < MARGIN for v in n))


def table_runs(f, arms, seeds):
    print(f"\n== f = 0.{f}: per run. R-mid (rows 512, 1024), R-late (rows 4096..8192), highest row from 64 (row), row 8192 - row 4096; R-late on the common pairs; taught panel at 8192")
    for rule in "SB":
        for arm in arms:
            for s in seeds:
                r = load(arm, rule, f, s)
                if r is None:
                    print(f"{rule} {arm:12s} {s:6d}  -"); continue
                g = figures(r)
                print(f"{rule} {arm:12s} {s:6d}  R-mid {g['rmid']:5.1f}  R-late {g['rlate']:5.1f}  highest {g['hi']:5.1f} ({g['hi_row']:4d})  8192-4096 {g['d84']:+5.1f}"
                      f"  common R-late {g['clate']:5.1f}  taught@8192 {g['taught8192']:5.1f}")


def curves(f, arms, seeds):
    print(f"\n== f = 0.{f}: reduction score at every row")
    for rule in "SB":
        for arm in arms:
            for s in seeds:
                r = load(arm, rule, f, s)
                if r is not None:
                    print(f"{rule} {arm:12s} {s:6d} " + " ".join(f"{x['updates']}:{red(x):.1f}" for x in r["rows"]))


def earlier(f, seeds):
    print(f"\n== f = 0.{f}: free arms, mean of the four panels of the life's families | the two panels of the first family | norm of the non-table tensors (sum of squares) | norm of the change; at every row")
    for rule in "SB":
        for arm in ("free", "free-wd0", "free-anchor"):
            for s in seeds:
                r = load(arm, rule, f, s)
                if r is None:
                    continue
                out = []
                for x in r["rows"]:
                    e = x["scores"]["earlier_k12"]
                    nn = sum(v["norm"] ** 2 for v in x["norms"].values()) ** .5; dd = sum(v["diff_from_lived"] ** 2 for v in x["norms"].values()) ** .5
                    out.append(f"{x['updates']}:{mean([pct(e[n]) for n in LIFE4]):.1f}|{mean([pct(e[n]) for n in FIRST2]):.1f}|{nn:.2f}|{dd:.2f}")
                print(f"{rule} {arm:12s} {s:6d} " + " ".join(out))


def compare(rule, a, b, f="10", key="rlate", fa=None, fb=None, ra=None, rb=None):
    x = [fig(a, ra or rule, fa or f, s, key) for s in SEEDS]
    y = [fig(b, rb or rule, fb or f, s, key) for s in SEEDS]
    d = ["-" if p is None or q is None else f"{p - q:+.1f}" for p, q in zip(x, y)]
    return verdict(x, y), x, y, d


def part_a(seeds):
    table_runs("10", ("free", "free-again", "free-wd0", "free-anchor", "term", "term-wd0"), seeds)
    curves("10", ("free", "free-again", "free-wd0", "free-anchor", "term", "term-wd0"), seeds)
    earlier("10", seeds)
    print("\n== Part A readings (figure R-late unless named; differences by seed 73129, 93683, 116089)")
    for rule in "SB":
        n, ok = scatter(rule)
        print(f"\n-- rule {rule}: N = " + ", ".join(f1(v) for v in n) + ("" if ok is None else ("  (below 5 on every seed: verdicts stated)" if ok else "  (5 or more on a seed: NO VERDICT under this rule)")))
        for a, b in (("term", "free"), ("free-anchor", "free"), ("term", "free-anchor"), ("term-wd0", "term"), ("free-wd0", "free-anchor"),
                     ("term-wd0", "free-wd0"), ("free-wd0", "free")):
            v, x, y, d = compare(rule, a, b)
            print(f"{a} against {b}: {v}; differences {', '.join(d)}")
        v, x, y, d = compare(rule, "term", "free", key="rmid")
        print(f"term against free on R-mid: {v}; differences {', '.join(d)}")
        print("split: L = term - free, A = free-anchor - free, P = term - free-anchor, a = A / L where L >= 5")
        for s in SEEDS:
            t, fr, an = fig("term", rule, "10", s), fig("free", rule, "10", s), fig("free-anchor", rule, "10", s)
            if None in (t, fr, an):
                print(f"  {s}: -"); continue
            L, A, P = t - fr, an - fr, t - an
            print(f"  {s}: L {L:+.1f}  A {A:+.1f}  P {P:+.1f}  a {'-' if L < MARGIN else f'{A / L:.2f}'}")
        for s in SEEDS:
            t, fr = load("term", rule, "10", s), load("free", rule, "10", s)
            if t and fr:
                gt, gf = figures(t), figures(fr)
                print(f"  {s}: R-mid -> R-late: term {gt['rmid']:.1f} -> {gt['rlate']:.1f}; free {gf['rmid']:.1f} -> {gf['rlate']:.1f}; lead {gt['rmid'] - gf['rmid']:+.1f} -> {gt['rlate'] - gf['rlate']:+.1f}")
        for s in SEEDS:
            out = []
            for arm in ("free", "free-wd0", "free-anchor"):
                r = load(arm, rule, "10", s)
                if r:
                    e = rows(r)[1024]["scores"]["earlier_k12"]
                    out.append(f"{arm} {mean([pct(e[n]) for n in LIFE4]):.1f}")
            print(f"  {s}: life's four panels at row 1024: " + "; ".join(out))
        print("  point 5: for each seed, the three free arms ordered by the mean of the life's four panels over rows 4096..8192, with R-late beside")
        agree = 0; n = 0
        for s in SEEDS:
            v = []
            for arm in ("free", "free-wd0", "free-anchor"):
                r = load(arm, rule, "10", s)
                if r:
                    rw = rows(r)
                    v.append((mean([mean([pct(rw[u]["scores"]["earlier_k12"][q]) for q in LIFE4]) for u in LATE]), figures(r)["rlate"], arm))
            if len(v) == 3:
                v.sort(reverse=True)
                top_keeps_most = v[0][1] == max(x[1] for x in v)
                agree += top_keeps_most; n += 1
                print(f"  {s}: " + "; ".join(f"{a} life4-late {l:.1f} R-late {rl:.1f}" for l, rl, a in v) + f"  | the arm keeping most has the highest R-late: {top_keeps_most}")
        print(f"  seeds on which the arm keeping most of the life's families has the highest R-late: {agree} of {n}")


def first_row(r, level):
    for x in r["rows"]:
        if red(x) >= level:
            return x["updates"]
    return None


def part_b(seeds):
    for f in ("05", "10", "20"):
        table_runs(f, ("free", "free-wd0", "term"), seeds)
    for f in ("05", "20"):
        curves(f, ("free", "free-wd0", "term"), seeds)
    print("\n== first row at which the reduction score is 80 or more / 90 or more, and presentations of each taught pair by then (64 items a batch over 4 x taught pairs)")
    npairs = {"05": 344, "10": 689, "20": 1378}
    for rule in "SB":
        for f in ("05", "10", "20"):
            for arm in ("free", "free-wd0", "term"):
                for s in seeds:
                    r = load(arm, rule, f, s)
                    if r is None:
                        continue
                    a, b = first_row(r, 80), first_row(r, 90)
                    pres = lambda u: "-" if u is None else f"{u * 64 / (4 * npairs[f]):.1f}"
                    print(f"{rule} f0.{f} {arm:9s} {s:6d}  80: row {a} ({pres(a)} each)  90: row {b} ({pres(b)} each)")
    print("\n== Part B readings")
    for rule in "SB":
        print(f"\n-- rule {rule}")
        for f in ("05", "10", "20"):
            for b in ("free", "free-wd0"):
                v, x, y, d = compare(rule, "term", b, f=f)
                print(f"f 0.{f}: term against {b}: {v}; differences {', '.join(d)}")
    for ra, rb in (("S", "S"), ("B", "B"), ("S", "B")):
        print(f"\n-- term under {ra} at f against free / free-wd0 under {rb} at g, on R-late of the common pairs")
        for f, g in (("05", "10"), ("10", "20"), ("05", "20")):
            va, xa, ya, da = compare(rb, "free", "free", key="clate", fa=g, fb=f)
            tf = [fig("term", ra, f, s, "clate") for s in SEEDS]
            best = [None if None in (fig("free", rb, g, s, "clate"), fig("free-wd0", rb, g, s, "clate")) else
                    max(fig("free", rb, g, s, "clate"), fig("free-wd0", rb, g, s, "clate")) for s in SEEDS]
            if None in tf or None in best:
                print(f"({f}, {g}): incomplete"); continue
            deficit = [b_ - t for t, b_ in zip(tf, best)]
            cond_a = va == "above"
            cond_b = sum(x <= 2 for x in deficit) >= 2 and all(x < 5 for x in deficit)
            ceiling = sum(min(t, b_) > 95 for t, b_ in zip(tf, best)) >= 2
            print(f"({f}, {g}): (a) free at g against free at f: {va}, differences {', '.join(da)}; (b) deficit of term at f below the higher free arm at g: "
                  f"{', '.join(f'{x:+.1f}' for x in deficit)} -> {cond_b}; (c) ceiling: {ceiling}")
            if not cond_a:
                print(f"   -> between {f} and {g} the level of free learning does not depend on the number of taught pairs; no statement on pairs")
            elif ceiling:
                print("   -> at the ceiling of the score; no statement")
            elif cond_b:
                print(f"   -> the gated term reaches from f = 0.{f} the level that free learning reaches from g = 0.{g}")
            else:
                print(f"   -> not stated; the deficit by seed is above")


if __name__ == "__main__":
    a = sys.argv[1:] or ["all"]
    seeds = tuple(int(s) for s in a[1:]) or SEEDS
    if a[0] in ("A", "all"):
        part_a(seeds)
    if a[0] in ("B", "all"):
        part_b(seeds)
