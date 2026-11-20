#!/usr/bin/env python3
"""PLAN_1.md: the tables of the outcome, from res/.   python3 report_1.py [ZA ZA0 ZPA] > report_1.txt"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); N = ("shakespeare", "book", "py", "c", "table", "xml"); TAGS = ("dev", "seed73129", "seed93683", "seed116089", "seed406364", "seed545353")
L = lambda n: json.load(open(os.path.join(HERE, "res", n + ".json"))); have = lambda n: os.path.exists(os.path.join(HERE, "res", n + ".json"))
def cases(sysn, tags=TAGS[1:]):
    rows = []
    for tag in tags:
        if not have(f"life__{sysn}_{tag}"): continue
        d = L(f"life__{sysn}_{tag}"); end = d["stages"][-1]["tests"]; own = {st["stream"]: st for st in d["stages"]}
        for n in d["order"][:-1]:
            o = own[n]["tests"][n]; e = end[n]
            rows.append(dict(tag=tag, n=n, own=o["going_on"]["cell2"], end=e["going_on"]["cell2"], g=e["going_on"]["cell2"] - o["going_on"]["cell2"], f=e["nothing_changing"]["cell2"] - o["nothing_changing"]["cell2"], f1=e["nothing_changing"]["cell1"] - o["nothing_changing"]["cell1"],
                             d1=d["d1"][n]["cell2"] - o["nothing_changing"]["cell2"], d3a=d["d3"][n]["trusts_and_sums_of_the_own_stage"]["cell2"] - o["going_on"]["cell2"], d3b=d["d3"][n]["sums_of_the_own_stage_only"]["cell2"] - o["going_on"]["cell2"],
                             g4=e["going_on"]["first"]["4000"][1] - o["going_on"]["first"]["4000"][1]))
    return rows
for sysn in sys.argv[1:] or ["ZA", "ZA0", "ZPA"]:
    if not all(have(f"life__{sysn}_single_{n}") for n in N): continue
    sg = {n: L(f"life__{sysn}_single_{n}")["stages"][0] for n in N}; sm = float(np.mean([sg[n]["tests"][n]["going_on"]["cell2"] for n in N]))
    print(f"\n## {sysn}: {L(f'life__{sysn}_single_shakespeare')['args']}\n\nSingles (a system for each stream), second cell, nothing changing / going on: " + " | ".join(f"{n} {sg[n]['tests'][n]['nothing_changing']['cell2']:.4f} / {sg[n]['tests'][n]['going_on']['cell2']:.4f}" for n in N) + f". Mean going on {sm:.4f}. Cells: " + ", ".join(f"{sg[n]['sizes']['cells']:,}" for n in N) + f" (sum {sum(sg[n]['sizes']['cells'] for n in N):,}).")
    print("\n| order of the life | " + " | ".join(N) + " | mean at the end | the life costs | cells at the end |\n|---|" + "---|" * 9)
    for tag in TAGS:
        if not have(f"life__{sysn}_{tag}"): continue
        d = L(f"life__{sysn}_{tag}"); end = d["stages"][-1]["tests"]; own = {st["stream"]: st for st in d["stages"]}; mean = float(np.mean([end[n]["going_on"]["cell2"] for n in N]))
        print(f"| {tag}: {', '.join(d['order'])} | " + " | ".join(f"{sg[n]['tests'][n]['going_on']['cell2']:.3f}, {own[n]['tests'][n]['going_on']['cell2']:.3f}, {end[n]['going_on']['cell2']:.3f}" for n in N) + f" | {mean:.4f} | {mean - sm:+.4f} | {d['stages'][-1]['sizes']['cells']:,} |")
    print("\n(each cell: going on, second answering cell: the single, the life at the stream's own stage, the life at its end)")
    rows = cases(sysn)
    if not rows: continue
    g = np.array([r["g"] for r in rows]); f = np.array([r["f"] for r in rows]); print(f"\nReported orders, {len(rows)} cases (a stream not read last, an order):")
    print(f"* going on, end less own stage: above 0.02 in {(g > 0.02).sum()} cases, largest {g.max():+.3f}, median {np.median(g):+.3f}; streams above 0.02 by order: " + "; ".join(f"{t}: {', '.join(r['n'] for r in rows if r['tag'] == t and r['g'] > 0.02) or 'none'}" for t in TAGS[1:] if any(r['tag'] == t for r in rows)))
    print(f"* nothing changing, end less own stage: above 0.05 in {(f > 0.05).sum()} cases, median {np.median(f):+.3f}, largest {f.max():+.3f}; the first cell's excess is the smaller in {sum(r['f1'] < r['f'] for r in rows)} cases")
    big = [r for r in rows if r["f"] > 0.02]; bg = [r for r in rows if r["g"] > 0.02]; b5 = [r for r in rows if r["g"] > 0.005]
    print(f"* D1 (the end's counts, the own stage's trusts, nothing changing) removes, at the median over the {len(big)} cases with an excess above 0.02: {np.median([1 - r['d1'] / r['f'] for r in big]):.2f} of the excess")
    print(f"* D3 (going on from the end): (a) trusts, parts and sums of the own stage put back removes {np.median([1 - r['d3a'] / r['g'] for r in bg]):.2f} of the going-on excess at the median over the {len(bg)} cases above 0.02; (b) only the sums: {np.median([1 - r['d3b'] / r['g'] for r in bg]):.2f}")
    print(f"* the excess over the first 4,000 bytes of the test is {np.median([r['g4'] / r['g'] for r in b5]):.2f} times the excess over the whole test (median over the {len(b5)} cases above 0.005)")
    print("* the stream's own stage less its single, going on, by order: " + "; ".join(f"{t}: " + ", ".join(f"{st['stream']} {st['tests'][st['stream']]['going_on']['cell2'] - sg[st['stream']]['tests'][st['stream']]['going_on']['cell2']:+.3f}" for st in L(f"life__{sysn}_{t}")["stages"]) for t in TAGS[1:] if have(f"life__{sysn}_{t}")))
    print("* while its body was read, the life less the single (whole body; last tenth of the body): " + "; ".join(f"{t}: " + ", ".join(f"{st['stream']} {st['reading']['cell2'] - sg[st['stream']]['reading']['cell2']:+.3f} ({st['reading']['last_tenth'][1] - sg[st['stream']]['reading']['last_tenth'][1]:+.3f})" for st in L(f"life__{sysn}_{t}")["stages"][1:]) for t in TAGS[1:] if have(f"life__{sysn}_{t}")))
if all(have(f"life__ZA0_{t}") and have(f"life__ZA_{t}") for t in TAGS[1:]):
    a = cases("ZA"); b = cases("ZA0"); print(f"\n## The constant step against the rule (reported orders, {len(a)} cases)\n\n* the constant step's going-on excess is the smaller in {sum(y['g'] < x['g'] for x, y in zip(a, b))} cases; its loss at the end of the life is the lower in {sum(y['end'] < x['end'] for x, y in zip(a, b))} cases")
if have("mapped__shakespeare_ZA_guard0.0"):
    print("\n## What the guard changes (nano/suite's mapped streams, a system for each; second cell, nothing changing / going on)\n")
    for sysn in ("ZA", "ZPA"):
        for n in N:
            if have(f"mapped__{n}_{sysn}_guard0.0") and have(f"mapped__{n}_{sysn}_guard1e-20"): a = L(f"mapped__{n}_{sysn}_guard0.0"); b = L(f"mapped__{n}_{sysn}_guard1e-20"); print(f"* {sysn} {n}: guard at zero {a['nothing_changing']['cell2']:.4f} / {a['going_on']['cell2']:.4f}; with the guard {b['nothing_changing']['cell2']:.4f} / {b['going_on']['cell2']:.4f}")
