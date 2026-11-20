#!/usr/bin/env python3
"""PLAN_23.md, predictions V5 and V6: the room that cells of each kind and level have learned to leave for followers they have not met.
    python3 room_look.py prose | sums <seed>"""
import itertools, json, os, sys
import numpy as np
import gate, score
from gate import G, Sys2
HERE = os.path.dirname(os.path.abspath(__file__)); NL = 40
def share(Rn, k, lo, hi):
    """Over the levels whose count-level lies in [lo, hi): spoke, met a new follower, the share."""
    sp = Rn[k, 4 * lo:4 * hi, 0].sum(); nv = Rn[k, 4 * lo:4 * hi, 1].sum(); return int(sp), int(nv), (round(float(nv / sp), 4) if sp else None)
Sys2.GATE = 0; Sys2.RULE = 2; Sys2.ALPHA = 0.03; Sys2.ROOM = 1; G.Sys = Sys2
if sys.argv[1] == "prose":
    x, cut, _ = score.load(); s = G.prose_sys(x, dict(mode="none")); s.learn(G.ctx_of(x, s.D, cut, s.D), x[s.D:cut]); out = {}
    for k in range(s.K1): out[f"run of {k} bytes"] = dict(counted_once=share(s.Rn, k, 0, 1), counted_2_to_7=share(s.Rn, k, 1, 3), counted_8_to_63=share(s.Rn, k, 3, 6), counted_64_or_more=share(s.Rn, k, 6, 10))
else:
    seed = int(sys.argv[2]); o, s = G.run_sums(dict(seed=str(seed), lines="40000", mode="none")); out = {}; tot = {}
    for k in range(s.K1):
        for name, lo, hi in (("counted_once", 0, 1), ("counted_2_to_7", 1, 3), ("counted_8_to_63", 3, 6), ("counted_64_or_more", 6, 10)):
            sp, nv, _ = share(s.Rn, k, lo, hi); a = tot.setdefault((len(s.kinds[k]), name), [0, 0]); a[0] += sp; a[1] += nv
    for (nd, name), (sp, nv) in sorted(tot.items()): out[f"kinds of {nd} delays, {name}"] = (sp, nv, round(nv / sp, 4) if sp else None)
json.dump(out, open(os.path.join(HERE, "res", "room_look_" + "_".join(sys.argv[1:]) + ".json"), "w"))
for k, v in out.items(): print(k, v)
