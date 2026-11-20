#!/usr/bin/env python3
"""PLAN_5.md (line C, step C4): one sign for each cell in between (the `dale=1` of loop_small_case/cells2.py, copied from commit fe397ca), with and
without the right of either end to withdraw from a connection (arm E of joint2.py, PLAN_2.md).
    python3 joint4.py <label> seed=1 [dale=1] [who=N|E] [T=720000] [theta=0.1] [lines=400000] [every=50000] [fast=1]
    dale=1: a strength from a line is never below zero (born as the size of the plain system's birth value; the threshold born at -2.5, the mean
    drive); each cell in between is excitatory or inhibitory (a quarter inhibitory, drawn as in cells2.py) and its strengths onto the answering
    cell, and its return strengths, keep that sign: after every byte a strength of the other sign is set to zero.
    who=E: every T bytes either end withdraws for good from its connections below theta times the largest it holds (joint2.py's rule, unchanged).
    fast=1: the inner loop runs compiled (kerns.c through fasts.py); the numpy loop here is the reference (check_joint4.py)."""
import json, os, sys, time
import numpy as np
import joint, joint2
from joint import HERE, S, D, OFF, text, sg

class Sys4(joint2.Sys2):
    def __init__(s, seed, dale=0, **kw):
        super().__init__(seed, **kw); s.dale = dale; s.sign = np.ones(s.H)
        if dale:
            g2 = np.random.default_rng([seed, 20261121]); s.W1 = np.abs(s.W1); s.b1 = np.full(s.H, -D * .25); s.sign = np.where(g2.random(s.H) < .25, -1.0, 1.0)
    def learn(s, y):
        act = OFF + np.asarray(s.hist); super().learn(y)
        if s.dale:
            s.W1[:, act] = np.maximum(s.W1[:, act], 0); s.A = np.where(s.A * s.sign > 0, s.A, 0.0); s.R = np.where(s.R * s.sign[:, None] > 0, s.R, 0.0)

def run(kw):
    t0 = time.process_time(); seed = int(kw.get("seed", 1)); rng = np.random.default_rng([seed, 99]); held, shown = {}, {}
    for op in "+-":
        o = rng.permutation(10000); P = [(int(q) // 100, int(q) % 100) for q in o]; held[op] = P[:500]; shown[op] = P[2000:]
    s = Sys4(seed, dale=int(kw.get("dale", 0)), H=int(kw.get("H", 256))); who = kw.get("who", "N"); T = int(kw.get("T", 720000)); theta = float(kw.get("theta", .1))
    n = int(kw.get("lines", 400000)); every = int(kw.get("every", 50000)); fast = int(kw.get("fast", 0)); out = dict(label=kw.get("label"), args=kw, rows=[])
    TAB = np.array([text("+", a, b) for a, b in shown["+"]], np.int32); it = 0; nb = 0; total = n * TAB.shape[1]
    def go(xs, ys):
        if fast: import fasts; fasts.run(s, xs, ys)
        else: s.steps(xs, ys)
    while it < n:
        nxt = min((it // every + 1) * every, n) - 1; X = TAB[rng.integers(8000, size=nxt - it + 1)]
        xs = X.ravel(); ys = np.concatenate([X[:, 1:], np.full((len(X), 1), -1, np.int32)], 1).ravel(); it = nxt + 1; p = 0
        while who != "N" and p < len(xs):
            e = min(len(xs), p + T - nb % T); go(xs[p:e], ys[p:e]); nb += e - p; p = e
            if nb % T == 0 and nb < total:
                z = int(((s.K > 0) & (s.A == 0)).sum()); s.withdraw(who, theta); s.log[-1]["living_at_zero_before"] = z
        if who == "N": go(xs, ys)
        if (nxt + 1) % every == 0: out["rows"].append(dict(lines=nxt + 1, add=s.score(held["+"]), living=int(s.K.sum())))
    out["last3"] = float(np.mean([r["add"] for r in out["rows"][-3:]])) if out["rows"] else None
    out["withdrawn"] = [e["n"] for e in s.log]; out["living_after"] = [e["living"] for e in s.log]; out["living_at_zero_before"] = [e["living_at_zero_before"] for e in s.log]
    out["m"] = joint2.measures(s, held["+"]); live = s.K > 0
    out["m"].update(share_of_living_at_zero=float((s.A[live] == 0).mean()), share_of_all_at_zero=float((s.A == 0).mean()), inhibitory_cells=int((s.sign < 0).sum()),
                    living_from_inhibitory=int(live[:, s.sign < 0].sum()), living_from_excitatory=int(live[:, s.sign > 0].sum()), share_of_line_strengths_at_zero=float((s.W1 == 0).mean()))
    out["cpu"] = round(time.process_time() - t0, 1); run.last = s
    if not kw.get("nowrite"):
        os.makedirs(os.path.join(HERE, "res"), exist_ok=True); json.dump(out, open(os.path.join(HERE, "res", f"g_{kw['label']}.json"), "w"))
    return out

if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[2:]); kw["label"] = sys.argv[1]; o = run(kw); m = o["m"]
    print(o["label"], "last3", round(o["last3"], 1), "| " + " ".join(f"{r['lines']//1000}k {r['add']:.0f}/{r['living']}" for r in o["rows"]), f"| end {m['end']:.1f} living {m['living']} (from inhibitory cells {m['living_from_inhibitory']}, "
          f"excitatory {m['living_from_excitatory']}) cells with none {m['cells_with_none']} | at zero: {100*m['share_of_all_at_zero']:.1f}% of all, {100*m['share_of_living_at_zero']:.1f}% of living | withdrawn {o['withdrawn']} at zero before {o['living_at_zero_before']}", o["cpu"], "s")
