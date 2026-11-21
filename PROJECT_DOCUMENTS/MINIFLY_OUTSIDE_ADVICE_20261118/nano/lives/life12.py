#!/usr/bin/env python3
"""nano/lives, PLAN_9.md: a life over twelve streams (the six of nano/suite and the six of PLAN_6.md that are not text). Nothing in a system is changed.
    python3 life12.py single sys=MC stream=elf_c            a stream of PLAN_6.md read alone (ZA and M singles of those streams are PLAN_6.md's)
    python3 life12.py life sys=ZA|M|MC seed=73129           a life over the twelve in the order numpy.random.default_rng(seed).permutation(12)
    python3 life12.py report"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import lives as LV, kinds6 as K6, mem as MM, cellmix as CM
TEXT = tuple(LV.NAMES); OTHER = K6.NAMES[:6]; NAMES12 = TEXT + OTHER
def prepare():
    for n in OTHER: b, cut = K6.load(n); LV._RAW[n] = (np.frombuffer(b, np.uint8).astype(np.int32), cut)
    LV.NAMES = NAMES12
def runner(sysname): return (lambda kw: MM.run_life(dict(kw, sys={"ZA": "ZA", "M": "ZAM32x512"}[sysname]))) if sysname in ("ZA", "M") else (lambda kw: CM.run_life(dict(kw, sys=sysname)))
PRE = dict(ZA="mem__", M="mem__", MC="cellmix__")
def single_file(sysname, n):
    if n in TEXT: return dict(ZA=f"life__ZA_single_{n}", M=f"mem__ZAM32x512_single_{n}", MC=f"cellmix__MC_single_{n}")[sysname]
    return dict(ZA=f"mem__K6_ZA_single_{n}", M=f"mem__K6_ZAM32x512_single_{n}", MC=f"cellmix__L12_MC_single_{n}")[sysname]

def report():
    L = lambda n: json.load(open(os.path.join(HERE, "res", n + ".json"))); have = lambda n: os.path.exists(os.path.join(HERE, "res", n + ".json")); seeds = (73129, 93683, 116089)
    sg = {s: {n: L(single_file(s, n))["stages"][0]["tests"][n]["going_on"]["cell2"] for n in NAMES12} for s in ("ZA", "M", "MC") if all(have(single_file(s, n)) for n in NAMES12)}
    for s, d in sg.items(): print(f"singles going on, {s}: " + " | ".join(f"{n} {d[n]:.4f}" for n in NAMES12) + f" | mean {np.mean(list(d.values())):.4f}")
    R = {}
    for s in sg:
        for seed in seeds:
            f = f"{PRE[s]}L12_{s}_seed{seed}"
            if not have(f): continue
            d = L(f); own = {st["stream"]: st["tests"][st["stream"]]["going_on"]["cell2"] for st in d["stages"]}; end = {n: d["stages"][-1]["tests"][n]["going_on"]["cell2"] for n in NAMES12}; R[s, seed] = dict(order=d["order"], own=own, end=end, cells=d["stages"][-1]["sizes"]["cells"])
    print("\n| system | order | mean at the end | mean of its singles | the life costs | cells at the end |\n|---|---|---|---|---|---|")
    for (s, seed), r in R.items(): m1 = np.mean(list(r["end"].values())); m0 = np.mean(list(sg[s].values())); print(f"| {s} | {seed}: {', '.join(r['order'])} | {m1:.4f} | {m0:.4f} | {m1 - m0:+.4f} ({100 * (m1 / m0 - 1):+.1f}%) | {r['cells']:,} |")
    for seed in seeds:
        if ("ZA", seed) not in R: continue
        p = R["ZA", seed]; o = p["order"]; fw = {n: p["own"][n] - sg["ZA"][n] for n in o[1:]}; bw = {n: p["end"][n] - p["own"][n] for n in o[:-1]}
        print(f"\norder {seed}: plain system: own stage less single, later positions: " + " ".join(f"{n} {fw[n]:+.3f}" for n in o[1:]) + f"\n   end less own stage: " + " ".join(f"{n} {bw[n]:+.3f}" for n in o[:-1]))
        print(f"   mean end less own stage: text {np.mean([bw[n] for n in bw if n in TEXT]):+.4f}, not text {np.mean([bw[n] for n in bw if n in OTHER]):+.4f}; mean own stage less single: text {np.mean([fw[n] for n in fw if n in TEXT]):+.4f}, not text {np.mean([fw[n] for n in fw if n in OTHER]):+.4f}")
        for s in ("M", "MC"):
            if (s, seed) not in R: continue
            q = R[s, seed]; f2 = {n: q["own"][n] - sg[s][n] for n in o[1:]}; big = [n for n in fw if fw[n] >= 0.02]
            print(f"   {s}: own stage less own single: " + " ".join(f"{n} {f2[n]:+.3f}" for n in o[1:]) + (f" | cut at the median over the {len(big)} positions where the plain loss is 0.02 or more: {np.median([1 - f2[n] / fw[n] for n in big]):.2f}" if big else "") + f" | below the plain system at the end on {sum(q['end'][n] < p['end'][n] for n in NAMES12)} of 12" + (f"; below M on {sum(q['end'][n] < R['M', seed]['end'][n] for n in NAMES12)} of 12" if s == "MC" and ("M", seed) in R else ""))

if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "res"), exist_ok=True); cmd = sys.argv[1]; kw = dict(a.split("=", 1) for a in sys.argv[2:])
    if cmd == "report": report(); sys.exit(0)
    prepare(); s = kw["sys"]
    if cmd == "single":
        tag = f"L12_{s}_single_{kw['stream']}"
        if not os.path.exists(os.path.join(HERE, "res", f"{PRE[s]}{tag}.json")): out = runner(s)(dict(order=kw["stream"], tag=tag)); t = out["stages"][0]["tests"][kw["stream"]]; print(tag, "| nothing changing", round(t["nothing_changing"]["cell2"], 4), "| going on", round(t["going_on"]["cell2"], 4), flush=True)
    elif cmd == "life":
        tag = f"L12_{s}_seed{kw['seed']}"
        if not os.path.exists(os.path.join(HERE, "res", f"{PRE[s]}{tag}.json")):
            out = runner(s)(dict(order=f"seed:{kw['seed']}", tag=tag)); e = out["stages"][-1]["tests"]; print(tag, "order", ",".join(out["order"]), "| mean over the twelve tests at the end, going on:", round(float(np.mean([t["going_on"]["cell2"] for t in e.values()])), 4), "| cells", out["stages"][-1]["sizes"]["cells"], "| wall", out["wall"], flush=True)
