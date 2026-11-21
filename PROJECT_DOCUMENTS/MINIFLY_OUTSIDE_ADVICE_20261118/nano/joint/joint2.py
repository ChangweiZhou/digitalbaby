#!/usr/bin/env python3
"""PLAN_2.md (line C, step C2): the plain system (form 1 of joint.py, which reproduces loop_small_case/cells2.py) in which an end of a connection
from a cell in between to a branch of the answering cell may withdraw from it for good.
    python3 joint2.py <label> seed=1 [who=N|R|S|E|X] [T=180000] [theta=0.1] [sched=<label of an E run>] [lines=400000] [every=50000] [fast=1]
    who=N: nobody withdraws.  R: the receiving end (the answering cell, branch by branch): every T bytes it withdraws from each living connection
    whose |A| is below theta times the largest |A| among the living connections of that branch.  S: the sending end (the cell in between): the same
    with its own return strengths |R|, among its own living connections.  E: either.  X (a control): at each moment as many living connections,
    drawn by chance, as the run named by sched= withdrew at that moment.
    A withdrawn connection has forward and return strength zero and never changes again.  The rule of withdrawal is here, in Python, for both the
    numpy loop (the reference) and the compiled one (fast=1: kernw.c through fastw.py); check_joint2.py compares the two."""
import json, os, sys, time
import numpy as np
import joint
from joint import HERE, S, D, OFF, text, sg

class Sys2(joint.Sys):
    def __init__(s, seed, **kw):
        super().__init__(seed, form=1, **kw); s.K = np.ones((S, s.H)); s.chance = np.random.default_rng([seed, 20261120]); s.log = []
    def learn(s, y):
        act = OFF + np.asarray(s.hist); h = sg(s.W1[:, act].sum(1) + s.b1); a = s.A @ h + s.b2; m = -a; m[y] += 1; g = s.eta2 / (h @ h + 1)
        sig = (s.R @ m) * h * (1 - h); d2 = np.outer(g * m, h) * s.K; d = s.eta1 * sig
        s.R += d2.T; s.W1[:, act] += np.repeat(d[:, None], D, 1); s.b1 += d; s.A += d2
        s.b2 += g * m; s.A *= 1 - s.lam; s.R *= 1 - s.lam
    def withdraw(s, who, theta, count=None):
        """One moment of withdrawal. Each end uses only the strengths it holds, as they are before anything is withdrawn at this moment."""
        live = s.K > 0; aA = np.where(live, np.abs(s.A), 0.0); aR = np.where(live, np.abs(s.R.T), 0.0)
        r = live & (aA < theta * aA.max(1, keepdims=True)) if who in "RE" else np.zeros_like(live)
        q = live & (aR < theta * aR.max(0, keepdims=True)) if who in "SE" else np.zeros_like(live)
        if who == "X":
            idx = np.flatnonzero(live); r = np.zeros(live.size, bool); r[s.chance.choice(idx, min(count, len(idx)), replace=False)] = True; r = r.reshape(live.shape)
        w = r | q; s.K[w] = 0.0; s.A[w] = 0.0; s.R.T[w] = 0.0
        s.log.append(dict(n=int(w.sum()), recv_alone=int((r & ~q).sum()), send_alone=int((q & ~r).sum()), both=int((r & q).sum()), living=int(s.K.sum())))

def measures(s, held):
    W = s.A; aw = np.abs(W); big = aw.max(); live = s.K > 0; small = aw < .01 * big
    o = dict(end=s.score(held), largest=float(big), living=int(live.sum()), cells_with_none=int((live.sum(0) == 0).sum()), fewest_of_a_branch=int(live.sum(1).min()), most_of_a_branch=int(live.sum(1).max()),
             fewest_of_a_cell=int(live.sum(0).min()), most_of_a_cell=int(live.sum(0).max()), share_below_hundredth_of_all=float(small.mean()), share_below_hundredth_of_living=float(small[live].mean()))
    o["end_without_those"] = s.score(held, np.where(small, 0.0, W)); o["cos_R_w"] = float((s.R.T * W).sum() / (np.linalg.norm(s.R) * np.linalg.norm(W) + 1e-300))
    for k in ("recv_alone", "send_alone", "both"): o[k] = int(sum(e[k] for e in s.log))
    return o

def run(kw):
    t0 = time.process_time(); seed = int(kw.get("seed", 1)); rng = np.random.default_rng([seed, 99]); held, shown = {}, {}
    for op in "+-":
        o = rng.permutation(10000); P = [(int(q) // 100, int(q) % 100) for q in o]; held[op] = P[:500]; shown[op] = P[2000:]
    s = Sys2(seed, H=int(kw.get("H", 256))); who = kw.get("who", "N"); T = int(kw.get("T", 180000)); theta = float(kw.get("theta", .1))
    n = int(kw.get("lines", 400000)); every = int(kw.get("every", 50000)); fast = int(kw.get("fast", 0)); out = dict(label=kw.get("label"), args={k: v for k, v in kw.items() if k != "sched_list"}, rows=[])
    sched = kw.get("sched_list") or (json.load(open(os.path.join(HERE, "res", f"w_{kw['sched']}.json")))["withdrawn"] if who == "X" else None)
    TAB = np.array([text("+", a, b) for a, b in shown["+"]], np.int32); it = 0; nb = 0; total = n * TAB.shape[1]
    def go(xs, ys):
        if fast: import fastw; fastw.run(s, xs, ys)
        else: s.steps(xs, ys)
    while it < n:
        nxt = min((it // every + 1) * every, n) - 1; X = TAB[rng.integers(8000, size=nxt - it + 1)]
        xs = X.ravel(); ys = np.concatenate([X[:, 1:], np.full((len(X), 1), -1, np.int32)], 1).ravel(); it = nxt + 1; p = 0
        while who != "N" and p < len(xs):               # up to the next moment of withdrawal
            e = min(len(xs), p + T - nb % T); go(xs[p:e], ys[p:e]); nb += e - p; p = e
            if nb % T == 0 and nb < total: s.withdraw(who, theta, sched[len(s.log)] if who == "X" else None)
        if who == "N": go(xs, ys)                        # one call for each block, as in joint.py
        if (nxt + 1) % every == 0: out["rows"].append(dict(lines=nxt + 1, add=s.score(held["+"]), living=int(s.K.sum())))
    out["last3"] = float(np.mean([r["add"] for r in out["rows"][-3:]])) if out["rows"] else None
    out["withdrawn"] = [e["n"] for e in s.log]; out["living_after"] = [e["living"] for e in s.log]
    out["m"] = measures(s, held["+"]); out["cpu"] = round(time.process_time() - t0, 1); run.last = s
    if not kw.get("nowrite"):
        os.makedirs(os.path.join(HERE, "res"), exist_ok=True); json.dump(out, open(os.path.join(HERE, "res", f"w_{kw['label']}.json"), "w"))
    return out

if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[2:]); kw["label"] = sys.argv[1]; o = run(kw); m = o["m"]
    print(o["label"], "last3", round(o["last3"], 1), "| " + " ".join(f"{r['lines']//1000}k {r['add']:.0f}/{r['living']}" for r in o["rows"]), f"| end {m['end']:.1f} living {m['living']} cells with none {m['cells_with_none']}",
          f"| below a hundredth: {100*m['share_below_hundredth_of_all']:.1f}% of all, {100*m['share_below_hundredth_of_living']:.1f}% of living; without those {m['end_without_those']:.1f}",
          f"| by recv {m['recv_alone']} send {m['send_alone']} both {m['both']} | cos {m['cos_R_w']:.3f}", o["cpu"], "s")
