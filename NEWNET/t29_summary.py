"""Summary of t29_learnset2.py runs (learnset2_runs/res.jsonl): per net, every test rule's measures in each condition (new, fam, rnd,
fresh), the paired comparisons and the three-way classification fixed in LEARNSET2_PLAN.md, the history curves, probes and bits.
Usage: python3 t29_summary.py [learnset2_runs/res.jsonl]"""
import sys, json, numpy as np
from collections import defaultdict
rows = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "learnset2_runs/res.jsonl")]
NL = {"13": "net13", "12": "net12", "10": "K8 net10"}
T = defaultdict(dict)                                    # (net, cond, rule, seed) -> block
H = defaultdict(dict)                                    # (net, life) -> seed -> row
for r in rows:
    H[(r["net"], r["life"])][r["seed"]] = r
    for b in r["blocks"]:
        if b["kind"] == "T": T[(r["net"], b["cond"], b["test"])][r["seed"]] = b
def v(b, k):
    x = b[k]
    if x is None: return b["n"] + 1
    return x
def cls(a, b, seeds, rules, net):
    """three-way classification of condition a against b (err80, f100) over the (rule, seed) pairs present in both"""
    pa, pb = [], []
    for t in rules:
        for s in seeds:
            if s in T[(net, a, t)] and s in T[(net, b, t)]: pa.append(T[(net, a, t)][s]); pb.append(T[(net, b, t)][s])
    if not pa: return None
    ea, eb = np.array([x["err80"] for x in pa], float), np.array([x["err80"] for x in pb], float)
    fa, fb = np.mean([x["f100"] for x in pa]), np.mean([x["f100"] for x in pb])
    n = len(pa); lt, gt = int((ea < eb).sum()), int((ea > eb).sum()); need = int(np.ceil(0.75 * n))
    if ea.mean() <= 0.8 * eb.mean() and lt >= need and fa >= fb: c = "faster"
    elif ea.mean() >= 1.25 * eb.mean() and gt >= need: c = "slower"
    else: c = "no clear difference"
    f20 = np.mean([x["f20"] for x in pa]) - np.mean([x["f20"] for x in pb])
    return dict(n=n, ea=ea.mean(), eb=eb.mean(), ratio=ea.mean() / max(eb.mean(), 1e-9), lt=lt, gt=gt, fa=fa, fb=fb, df20=f20, c=c)
RULES = ("X1", "X2", "Y1", "Y2")
for net in ("13", "12", "10"):
    seeds = sorted({s for k in T if k[0] == net for s in T[k]})
    if not seeds: continue
    print(f"\n## {NL[net]} (seeds {seeds})\n")
    print("Per test rule and condition: err80 / L80s / f20 / f100 / f500 / last500 (seeds " + ", ".join(map(str, seeds)) + "; '-' = not reached)\n")
    print("| rule | cond | err80 | L80s | f20 | f100 | f500 | last500 | L80 (100-line) |\n|---|---|---|---|---|---|---|---|---|")
    for t in RULES:
        for c in ("fresh", "new", "fam", "rnd"):
            d = T[(net, c, t)]
            if not d: continue
            f = lambda k: " / ".join("-" if d[s][k] is None else str(d[s][k]) for s in seeds if s in d)
            print(f"| {t} | {c} | {f('err80')} | {f('L80s')} | {f('f20')} | {f('f100')} | {f('f500')} | {f('last500')} | {f('L80')} |")
    print("\nMeans over all (rule, seed) pairs, and the classification fixed in the plan:\n")
    print("| comparison | pairs | mean err80 a vs b (ratio) | a < b / a > b | mean f100 a vs b | f20 a - b | class |\n|---|---|---|---|---|---|---|")
    for a_, b_ in (("new", "fresh"), ("fam", "fresh"), ("new", "fam"), ("new", "rnd"), ("rnd", "fresh")):
        r = cls(a_, b_, seeds, RULES, net)
        if r: print(f"| {a_} vs {b_} | {r['n']} | {r['ea']:.0f} vs {r['eb']:.0f} ({r['ratio']:.2f}) | {r['lt']} / {r['gt']} | {r['fa']:.3f} vs {r['fb']:.3f} | {r['df20']:+.3f} | {r['c']} |")
    print("\nSame, without X2 (its fresh plateau is 0.75 in the pilot):\n")
    print("| comparison | pairs | mean err80 a vs b (ratio) | a < b / a > b | mean f100 a vs b | f20 a - b | class |\n|---|---|---|---|---|---|---|")
    for a_, b_ in (("new", "fresh"), ("fam", "fresh"), ("new", "fam"), ("new", "rnd")):
        r = cls(a_, b_, seeds, ("X1", "Y1", "Y2"), net)
        if r: print(f"| {a_} vs {b_} | {r['n']} | {r['ea']:.0f} vs {r['eb']:.0f} ({r['ratio']:.2f}) | {r['lt']} / {r['gt']} | {r['fa']:.3f} vs {r['fb']:.3f} | {r['df20']:+.3f} | {r['c']} |")
    print("\nPer seed, mean err80 over the 4 test rules (new / fam / rnd / fresh):\n")
    for s in seeds:
        m = lambda c: np.mean([T[(net, c, t)][s]["err80"] for t in RULES if s in T[(net, c, t)]]) if any(s in T[(net, c, t)] for t in RULES) else float("nan")
        print(f"- seed {s}: {m('new'):.0f} / {m('fam'):.0f} / {m('rnd'):.0f} / {m('fresh'):.0f}")
    print("\nHistory curves (err80 per history slot 1-8; f20 in brackets), probes and bits:\n")
    for life in ("A", "B", "RA", "RB"):
        for s in seeds:
            r = H[(net, life)].get(s)
            if not r: continue
            hb = [b for b in r["blocks"] if b["kind"] == "H"]
            hs = " ".join(f"{b['err80']}({b['f20']})" for b in hb) if hb else "random answers"
            ph = r["probe_after_history"] or {}; pe = r["probe_end"]
            phm = np.mean(list(ph.values())) if ph else float("nan")
            pet = " ".join(f"{b['test']} {pe[str(b['slot'])]}" for b in r["blocks"] if b["kind"] == "T")
            peh = np.mean([pe[str(b["slot"])] for b in hb]) if hb else float("nan")
            print(f"- life {life} seed {s}: {hs}; probe history rules after history {phm:.2f}, at end {peh:.2f}; end probes {pet};"
                  f" bits {r['bits']['total']:,} (answer {r['bits']['answer']:,}); {r['s']} s")
