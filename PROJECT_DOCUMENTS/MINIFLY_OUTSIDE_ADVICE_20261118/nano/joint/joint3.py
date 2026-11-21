#!/usr/bin/env python3
"""PLAN_3.md (line C, step C2b): the plain system (form 1 of joint.py, which reproduces loop_small_case/cells2.py) in which a cell in between may
drop out for good: from then on it sends nothing (its output is zero) and all its connections, from the lines and to and from the answering cell,
are zero and do not change.
    python3 joint3.py <label> seed=1 [who=N|L|D|X] [T=720000] [c=0.5] [sched=<label of a D run>] [H=256] [lines=400000] [every=50000] [fast=1]
    who=N: nobody drops out.  L: nobody drops out, but the cells' own quantity is recorded at every moment (a look; the same system as N).
    D: every T bytes a living cell drops out if the sum of the sizes of the return strengths it holds, rho_j = sum over v of |R[j, v]|, is below c.
    X (a control): at each moment as many living cells, drawn by chance, as the run named by sched= dropped at that moment.
    H=n: the system born with n cells.   The rule of dropping out is here, in Python, for the numpy loop (the reference) and the compiled one
    (fast=1: kernd.c through fastd.py); check_joint3.py compares the two."""
import json, os, sys, time
import numpy as np
import joint
from joint import HERE, S, D, OFF, text, sg

class Sys3(joint.Sys):
    def __init__(s, seed, **kw):
        super().__init__(seed, form=1, **kw); s.L = np.ones(s.H); s.chance = np.random.default_rng([seed, 20261122]); s.log = []
    def learn(s, y):
        act = OFF + np.asarray(s.hist); h = sg(s.W1[:, act].sum(1) + s.b1) * s.L; a = s.A @ h + s.b2; m = -a; m[y] += 1; g = s.eta2 / (h @ h + 1)
        sig = (s.R @ m) * h * (1 - h); d2 = np.outer(g * m, h); d = s.eta1 * sig
        s.R += d2.T; s.W1[:, act] += np.repeat(d[:, None], D, 1); s.b1 += d; s.A += d2
        s.b2 += g * m; s.A *= 1 - s.lam; s.R *= 1 - s.lam
    def score(s, pairs, W=None):
        W = s.A if W is None else W; Ls = np.array([text("+", a, b) for a, b in pairs]); ok = np.ones(len(pairs), bool); W1T = s.W1.T
        for t in (5, 6, 7):
            hist = np.full((len(pairs), D), joint.END); hist[:, :t + 1] = Ls[:, t::-1]
            ok &= ((sg(W1T[OFF + hist].sum(1) + s.b1) * s.L) @ W.T + s.b2).argmax(1) == Ls[:, t + 1]
        return float(100 * ok.sum() / len(pairs))
    def moment(s, who, c, count=None):
        live = s.L > 0; rho = np.abs(s.R).sum(1); out = np.zeros(s.H, bool)
        if who == "D": out = live & (rho < c)
        if who == "X": out[s.chance.choice(np.flatnonzero(live), min(count, int(live.sum())), replace=False)] = True
        q = np.quantile(rho[live], [0, .1, .25, .5, .75, .9, 1]) if live.any() else np.zeros(7)
        s.L[out] = 0.0; s.A[:, out] = 0.0; s.R[out] = 0.0; s.W1[out] = 0.0; s.b1[out] = 0.0
        s.log.append(dict(n=int(out.sum()), living=int(s.L.sum()), rho_quantiles_0_10_25_50_75_90_100=[float(x) for x in q]))

def run(kw):
    t0 = time.process_time(); seed = int(kw.get("seed", 1)); rng = np.random.default_rng([seed, 99]); held, shown = {}, {}
    for op in "+-":
        o = rng.permutation(10000); P = [(int(q) // 100, int(q) % 100) for q in o]; held[op] = P[:500]; shown[op] = P[2000:]
    s = Sys3(seed, H=int(kw.get("H", 256))); who = kw.get("who", "N"); T = int(kw.get("T", 720000)); c = float(kw.get("c", .5))
    n = int(kw.get("lines", 400000)); every = int(kw.get("every", 50000)); fast = int(kw.get("fast", 0)); out = dict(label=kw.get("label"), args={k: v for k, v in kw.items() if k != "sched_list"}, rows=[])
    sched = kw.get("sched_list") or (json.load(open(os.path.join(HERE, "res", f"c_{kw['sched']}.json")))["dropped"] if who == "X" else None)
    TAB = np.array([text("+", a, b) for a, b in shown["+"]], np.int32); it = 0; nb = 0; total = n * TAB.shape[1]
    def go(xs, ys):
        if fast: import fastd; fastd.run(s, xs, ys)
        else: s.steps(xs, ys)
    while it < n:
        nxt = min((it // every + 1) * every, n) - 1; X = TAB[rng.integers(8000, size=nxt - it + 1)]
        xs = X.ravel(); ys = np.concatenate([X[:, 1:], np.full((len(X), 1), -1, np.int32)], 1).ravel(); it = nxt + 1; p = 0
        while who != "N" and p < len(xs):               # up to the next moment
            e = min(len(xs), p + T - nb % T); go(xs[p:e], ys[p:e]); nb += e - p; p = e
            if nb % T == 0 and nb < total: s.moment(who, c, sched[len(s.log)] if who == "X" else None)
        if who == "N": go(xs, ys)                        # one call for each block, as in joint.py
        if (nxt + 1) % every == 0: out["rows"].append(dict(lines=nxt + 1, add=s.score(held["+"]), living=int(s.L.sum())))
    out["last3"] = float(np.mean([r["add"] for r in out["rows"][-3:]])) if out["rows"] else None
    out["dropped"] = [e["n"] for e in s.log]; out["living_after"] = [e["living"] for e in s.log]; out["moments"] = s.log
    out["living"] = int(s.L.sum()); out["end"] = s.score(held["+"]); out["cpu"] = round(time.process_time() - t0, 1); run.last = s
    if not kw.get("nowrite"):
        os.makedirs(os.path.join(HERE, "res"), exist_ok=True); json.dump(out, open(os.path.join(HERE, "res", f"c_{kw['label']}.json"), "w"))
    return out

if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[2:]); kw["label"] = sys.argv[1]; o = run(kw)
    print(o["label"], "last3", round(o["last3"], 1), "| " + " ".join(f"{r['lines']//1000}k {r['add']:.0f}/{r['living']}" for r in o["rows"]), f"| end {o['end']:.1f} living {o['living']} dropped {o['dropped']}", o["cpu"], "s")
