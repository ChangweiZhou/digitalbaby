#!/usr/bin/env python3
"""PLAN_5.md (and PLAN_1.md, addition 2): the loss on streams read late, by diagnostic and by system.   python3 report_5.py > report_5.txt"""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); N = ("shakespeare", "book", "py", "c", "table", "xml"); TAGS = ("dev", "seed73129", "seed93683", "seed116089", "seed406364", "seed545353")
L = lambda n: json.load(open(os.path.join(HERE, "res", n + ".json"))); have = lambda n: os.path.exists(os.path.join(HERE, "res", n + ".json"))
sg = {n: L(f"life__ZA_single_{n}")["stages"][0] for n in N}; sm = float(np.mean([sg[n]["tests"][n]["going_on"]["cell2"] for n in N]))
def own(d): return {st["stream"]: st for st in d["stages"]}
def fwd(d, single=sg): o = own(d); return {n: o[n]["tests"][n]["going_on"]["cell2"] - single[n]["tests"][n]["going_on"]["cell2"] for n in d["order"][1:]}
def endmean(d, key="cell2"): return float(np.mean([d["stages"][-1]["tests"][n]["going_on"][key] for n in N]))
ARMS = [("plain life", "life__ZA_{}"), ("trusts set back (PLAN_1, addition 2)", "life__ZAr_{}"), ("short runs emptied", "mem__D_short_{}"), ("long runs emptied", "mem__D_long_{}"), ("places emptied", "mem__D_places_{}")]
print("## Diagnostics: the stream's own stage less its single, going on, second cell (positions after the first), and the mean over the six tests at the end\n")
for tag in TAGS:
    if not have(f"life__ZA_{tag}"): continue
    p = L(f"life__ZA_{tag}"); print(f"**{tag}: {', '.join(p['order'])}**\n\n| arm | " + " | ".join(p["order"][1:]) + " | mean at the end (less the singles' 0.8357) |\n|---|" + "---|" * 6)
    for name, pat in ARMS:
        if have(pat.format(tag)): d = L(pat.format(tag)); f = fwd(d); print(f"| {name} | " + " | ".join(f"{f[n]:+.3f}" for n in p["order"][1:]) + f" | {endmean(d):.4f} ({endmean(d) - sm:+.4f}) |")
    print()
rows = {name: [] for name, _ in ARMS}
for tag in TAGS[1:]:
    if not all(have(pat.format(tag)) for _, pat in ARMS): continue
    base = fwd(L(f"life__ZA_{tag}"))
    for name, pat in ARMS:
        f = fwd(L(pat.format(tag))); rows[name] += [(base[n], f[n]) for n in base if base[n] >= 0.02]
for name, r in rows.items():
    if r and name != "plain life": print(f"* {name}: over the {len(r)} later positions of the reported orders where the plain life is 0.02 or more above its single, the share of that loss removed, at the median: {np.median([1 - b / a for a, b in r]):.2f}; positions within 0.01 of the single: {sum(b <= 0.01 for a, b in r)}")
print("\n## The forms of bounded kinds on the development order (going on, second cell)\n\n| system | " + " | ".join(N) + " | mean at the end | cells at the end (of bounded kinds) |\n|---|" + "---|" * 8)
dev = L("life__ZA_dev"); o = own(dev); print("| plain: own stage, end | " + " | ".join(f"{o[n]['tests'][n]['going_on']['cell2']:.4f}, {dev['stages'][-1]['tests'][n]['going_on']['cell2']:.4f}" for n in N) + f" | {endmean(dev):.4f} | {dev['stages'][-1]['sizes']['cells']:,} |")
for f in ("ZAM32", "ZAM32x512", "ZAM32all", "ZAM8x64"):
    if have(f"mem__{f}_dev"): d = L(f"mem__{f}_dev"); o = own(d); z = d["stages"][-1]["sizes"]; print(f"| {f} {d['bounded']}: own stage, end | " + " | ".join(f"{o[n]['tests'][n]['going_on']['cell2']:.4f}, {d['stages'][-1]['tests'][n]['going_on']['cell2']:.4f}" for n in N) + f" | {endmean(d):.4f} | {z['cells']:,} ({z['cells_of_bounded_kinds']:,}) |")
for f in ("ZAM32", "ZAM32x512", "ZAM32all", "ZAM8x64"):
    if not all(have(f"mem__{f}_single_{n}") for n in N): continue
    s2 = {n: L(f"mem__{f}_single_{n}")["stages"][0] for n in N}; m2 = float(np.mean([s2[n]["tests"][n]["going_on"]["cell2"] for n in N]))
    print(f"\n## {f}: singles and reported orders\n\nSingles going on (less the plain single): " + " | ".join(f"{n} {s2[n]['tests'][n]['going_on']['cell2']:.4f} ({s2[n]['tests'][n]['going_on']['cell2'] - sg[n]['tests'][n]['going_on']['cell2']:+.4f})" for n in N) + f". Mean {m2:.4f} (plain {sm:.4f}). Nothing changing (less the plain single): " + " | ".join(f"{n} {s2[n]['tests'][n]['nothing_changing']['cell2'] - sg[n]['tests'][n]['nothing_changing']['cell2']:+.4f}" for n in N) + ". Cells: " + ", ".join(f"{s2[n]['sizes']['cells']:,} ({s2[n]['sizes']['cells_of_bounded_kinds']:,})" for n in N))
    print("\n| order | own stage less its own single, later positions | plain system, the same | mean at the end | cost of the life | plain system's cost | tests on which it is below the plain system at the end |\n|---|---|---|---|---|---|---|"); cuts = []
    for tag in TAGS[1:]:
        if not have(f"mem__{f}_{tag}"): continue
        d = L(f"mem__{f}_{tag}"); p = L(f"life__ZA_{tag}"); a = fwd(d, s2); b = fwd(p); cuts += [(b[n], a[n]) for n in b if b[n] >= 0.02]
        print(f"| {tag}: {', '.join(x[:2] for x in d['order'])} | " + " ".join(f"{a[n]:+.3f}" for n in d["order"][1:]) + " | " + " ".join(f"{b[n]:+.3f}" for n in d["order"][1:]) + f" | {endmean(d):.4f} | {endmean(d) - m2:+.4f} | {endmean(p) - sm:+.4f} | {sum(d['stages'][-1]['tests'][n]['going_on']['cell2'] < p['stages'][-1]['tests'][n]['going_on']['cell2'] for n in N)} of 6 |")
    if cuts: print(f"\n* the loss on streams read late (own stage less own single) is cut, at the median over the {len(cuts)} positions where the plain system's is 0.02 or more, by {np.median([1 - y / x for x, y in cuts]):.2f}")
