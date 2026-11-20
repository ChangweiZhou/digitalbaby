#!/usr/bin/env python3
"""Runs many settings on 4 cores (PROG=sparse2.py in the environment for the second-layer program): python3 grid.py <prefix> <common args...> -- key=v1,v2 key=v1,v2 ...   Prints a table sorted by the last-three mean."""
import itertools, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); a = sys.argv[1:]; i = a.index("--"); pre, common, var = a[0], a[1:i], [x.split("=") for x in a[i + 1:]]
jobs = []
for vals in itertools.product(*[v.split(",") for _, v in var]):
    kv = [f"{k}={v}" for (k, _), v in zip(var, vals)]; jobs.append((pre + "_" + "_".join(kv).replace("=", "").replace("/", "o"), common + kv))
def go(j):
    f = os.path.join(HERE, "res", f"s_{j[0]}.json")
    if not os.path.exists(f): subprocess.run([sys.executable, os.path.join(HERE, os.environ.get("PROG", "sparse.py")), j[0]] + j[1], stdout=subprocess.DEVNULL, check=True)
    return j[0], json.load(open(f))
with ThreadPoolExecutor(4) as ex: R = list(ex.map(go, jobs))
ph = max(r["phase"] for _, o in R for r in o["rows"])
for lab, o in sorted(R, key=lambda r: -r[1]["last3_add_phase0"]):
    last = [r for r in o["rows"] if r["phase"] == 0][-1]; s = f"{lab:60s} add last3 {o['last3_add_phase0']:5.1f} last {last['add']:5.1f} h,t,u {last['add_place']} forced {last['add_forced']:5.1f}"
    if ph: e = o["rows"][-1]; s += f" | after: add last3 {o[f'last3_add_phase{ph}']:5.1f} last {e['add']:5.1f} {e['add_place']} sub {e['sub']:5.1f}"
    print(s + f" | cells {o['cells']} cpu {o['cpu']}")
