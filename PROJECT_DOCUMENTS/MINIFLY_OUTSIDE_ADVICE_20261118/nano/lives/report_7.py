#!/usr/bin/env python3
"""PLAN_7.md: the chosen system (MC: all memories and the cell's mixture speak) against M (PLAN_5.md's form) and the plain system.   python3 report_7.py > report_7.txt"""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); N = ("shakespeare", "book", "py", "c", "table", "xml"); TAGS = ("dev", "seed73129", "seed93683", "seed116089", "seed406364", "seed545353")
L = lambda n: json.load(open(os.path.join(HERE, "res", n + ".json"))); have = lambda n: os.path.exists(os.path.join(HERE, "res", n + ".json"))
S = dict(plain={n: L(f"life__ZA_single_{n}")["stages"][0]["tests"][n] for n in N}, M={n: L(f"mem__ZAM32x512_single_{n}")["stages"][0]["tests"][n] for n in N}, MC={n: L(f"cellmix__MC_single_{n}")["stages"][0]["tests"][n] for n in N})
F = dict(plain="life__ZA_{}", M="mem__ZAM32x512_{}", MC="cellmix__MC_{}"); sm = {k: float(np.mean([v[n]["going_on"]["cell2"] for n in N])) for k, v in S.items()}
for key in ("going_on", "nothing_changing"): print(f"Read alone, {key.replace('_', ' ')}: " + "; ".join(f"{k}: " + " ".join(f"{S[k][n][key]['cell2']:.4f}" for n in N) + f" (mean {np.mean([S[k][n][key]['cell2'] for n in N]):.4f})" for k in S))
print("MC less M, read alone: going on " + " ".join(f"{S['MC'][n]['going_on']['cell2'] - S['M'][n]['going_on']['cell2']:+.4f}" for n in N) + "; nothing changing " + " ".join(f"{S['MC'][n]['nothing_changing']['cell2'] - S['M'][n]['nothing_changing']['cell2']:+.4f}" for n in N))
print("\n| order | mean at the end: plain, M, MC | cost of the life: plain, M, MC | tests on which MC is below M at the end | end less own stage, streams not read last: M | MC |\n|---|---|---|---|---|---|"); smaller = total = 0; costs = []
for tag in TAGS:
    if not all(have(F[k].format(tag)) for k in F): continue
    D = {k: L(F[k].format(tag)) for k in F}; end = {k: {n: D[k]["stages"][-1]["tests"][n]["going_on"]["cell2"] for n in N} for k in D}; own = {k: {st["stream"]: st["tests"][st["stream"]]["going_on"]["cell2"] for st in D[k]["stages"]} for k in D}; o = D["MC"]["order"]
    ex = {k: {n: end[k][n] - own[k][n] for n in o[:-1]} for k in ("M", "MC")}; me = {k: float(np.mean(list(end[k].values()))) for k in D}
    print(f"| {tag}: {', '.join(x[:2] for x in o)} | " + ", ".join(f"{me[k]:.4f}" for k in D) + " | " + ", ".join(f"{me[k] - sm[k]:+.4f}" for k in D) + f" | {sum(end['MC'][n] < end['M'][n] for n in N)} of 6 | " + " ".join(f"{ex['M'][n]:+.3f}" for n in o[:-1]) + " | " + " ".join(f"{ex['MC'][n]:+.3f}" for n in o[:-1]) + " |")
    if tag != "dev": smaller += sum(ex["MC"][n] < ex["M"][n] for n in o[:-1]); total += len(o) - 1; costs.append((me["M"] - sm["M"], me["MC"] - sm["MC"]))
if total: print(f"\nReported orders: MC's going-on excess at the end over the own stage is smaller than M's in {smaller} of {total} cases; its cost of a life is lower than M's in {sum(b < a for a, b in costs)} of {len(costs)} orders")
