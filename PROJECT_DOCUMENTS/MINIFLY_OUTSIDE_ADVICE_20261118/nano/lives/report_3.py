#!/usr/bin/env python3
"""PLAN_3.md: remembered trusts on the reported orders, against the plain answer of the same run.   python3 report_3.py > report_3.txt"""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); N = ("shakespeare", "book", "py", "c", "table", "xml"); TAGS = ("dev_N65536", "seed73129", "seed93683", "seed116089", "seed406364", "seed545353")
L = lambda n: json.load(open(os.path.join(HERE, "res", n + ".json"))); have = lambda n: os.path.exists(os.path.join(HERE, "res", n + ".json"))
sg = {n: L(f"life__ZA_single_{n}")["stages"][0]["tests"][n] for n in N}; sm = float(np.mean([sg[n]["going_on"]["cell2"] for n in N]))
print(f"Pace of bookmarks 65,536 (chosen on the development order: mean of the mixture at the end 0.8811, against 0.8837 at 262,144). Plain singles' mean going on {sm:.4f}.\n")
print("| order | " + " | ".join(N) + " | mean at the end: plain, mixture | the life costs: plain, mixture | bookmarks |\n|---|" + "---|" * 9); cases = []; worst = 0.0; fwd = []; costs = []
for tag in TAGS:
    if not have(f"books__B_{tag}"): continue
    d = L(f"books__B_{tag}"); own = {st["stream"]: (i, st) for i, st in enumerate(d["stages"])}; end = d["stages"][-1]["tests"]; mp = float(np.mean([end[n]["going_on"]["cell2"] for n in N])); mm = float(np.mean([end[n]["going_on"]["mix"] for n in N]))
    print(f"| {tag}: {', '.join(x[:2] for x in d['order'])} | " + " | ".join(f"{own[n][1]['tests'][n]['going_on']['mix']:.3f}; {end[n]['going_on']['cell2']:.3f}, {end[n]['going_on']['mix']:.3f}" for n in N) + f" | {mp:.4f}, {mm:.4f} | {mp - sm:+.4f}, {mm - sm:+.4f} | {d['stages'][-1]['sizes']['bookmarks']} |")
    worst = max(worst, max(t[k]["mix"] - t[k]["cell2"] for st in d["stages"] for t in st["tests"].values() for k in ("going_on", "nothing_changing")))
    if tag.startswith("dev"): continue
    costs.append((mp - sm, mm - sm))
    for n in d["order"][:-1]:
        i, st = own[n]; o = st["tests"][n]; e = end[n]; body = e["going_on"].get("largest_share_living_then_by_body"); tot = sum(body) if body else 0
        cases.append(dict(tag=tag, n=n, gp=e["going_on"]["cell2"] - o["going_on"]["cell2"], gm=e["going_on"]["mix"] - o["going_on"]["mix"], fp=e["nothing_changing"]["cell2"] - o["nothing_changing"]["cell2"], fm=e["nothing_changing"]["mix"] - o["nothing_changing"]["mix"], own_share=(body[i + 1] / tot) if body else None))
    for n in d["order"][1:]: i, st = own[n]; fwd.append(abs(st["tests"][n]["going_on"]["mix"] - st["tests"][n]["going_on"]["cell2"]))
print("\n(each cell, going on: the mixture at the stream's own stage; at the end of the life the plain answer, the mixture)")
if cases:
    big = [c for c in cases if c["gp"] > 0.02]; bigf = [c for c in cases if c["fp"] > 0.02]
    print(f"\nReported orders, {len(cases)} cases (a stream not read last, an order):")
    print(f"* B1: the largest (mixture less plain) over every test, stage, measure and order (development included): {worst:+.5f}")
    print(f"* B2: going on, the share of the excess (end less own stage) removed, at the median over the {len(big)} cases with a plain excess above 0.02: {np.median([1 - c['gm'] / c['gp'] for c in big]):.2f} (smallest {min(1 - c['gm'] / c['gp'] for c in big):.2f}, largest {max(1 - c['gm'] / c['gp'] for c in big):.2f}); cases still above 0.02 with the mixture: {sum(c['gm'] > 0.02 for c in cases)} of {len(cases)}")
    print(f"* B3: the cost of the life, plain and with the mixture, by order: " + "; ".join(f"{a:+.4f} -> {b:+.4f} ({100 * (1 - b / a):.0f}% less)" for a, b in costs))
    print(f"* B4: nothing changing, the share of the excess removed at the median over the {len(bigf)} cases with a plain excess above 0.02: {np.median([1 - c['fm'] / c['fp'] for c in bigf]):.2f}")
    print(f"* B5: at its own stage, going on, the mixture is within 0.01 of the plain answer in {sum(v <= 0.01 for v in fwd)} of the {len(fwd)} later positions (largest difference {max(fwd):.4f})")
    sh = [c["own_share"] for c in cases if c["own_share"] is not None]
    if sh: print(f"* B7: at the end, on the test of a stream read earlier, the largest share was on a bookmark taken while that stream was read for more than half of the test's bytes in {sum(v > 0.5 for v in sh)} of {len(sh)} cases (median share of the bytes {np.median(sh):.2f})")
if all(have(f"books__B_single_{n}") for n in N):
    b = {n: L(f"books__B_single_{n}")["stages"][0]["tests"][n] for n in N}
    print("\nRead alone (B6), mixture less plain: going on " + " | ".join(f"{n} {b[n]['going_on']['mix'] - b[n]['going_on']['cell2']:+.4f}" for n in N) + "; nothing changing " + " | ".join(f"{n} {b[n]['nothing_changing']['mix'] - b[n]['nothing_changing']['cell2']:+.4f}" for n in N))
