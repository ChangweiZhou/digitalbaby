#!/usr/bin/env python3
"""nano/keep, PLAN_2.md and later: the system of cells2.py in which the cells in between join over the life and their steps depend on their own
quantities. numpy is the reference; fast=1 runs the same arithmetic compiled (kage.c through fastage.py; check_kage.py compares the two).
    python3 kage.py <label> seed=1 plan=add:400000,sub:200000 [reh=32] [fast=1] H=384 [H0=1] [life=5400000] [law=0|exp|hyp] [tau=..] [young=..]
                    [use=sig|out] [join=sig theta=..] [every=..] and the options of cells2.py that concern the plain system (eta1, eta2, lam)
    H: all cells in between. H0: cells present from the first byte. The others join one by one, cell H0+i at learned byte (i+1)*life/(H-H0+1):
       a steady rate fixed before the life begins; nothing in the program refers to the change of task. Before it joins a cell is silent and unchanged.
    law, tau: a joined cell of age a (learned bytes since it joined) has rate 1 (law=0), exp(-a/tau) (exp) or 1/(1+a/tau) (hyp). The rate multiplies the
       cell's own steps (strengths from lines, threshold), the answering cell's step on the strength from that cell, the return strength's step, and the
       fading of those two strengths. The answering cell's constants change as in the plain system.
    young=y (PLAN_3.md): a strength from a line is closed at birth and opens when its line is active while the cell is younger than y learned bytes;
       later, a cell with a closed strength on an active line is silent at that byte (and so does not change).
    use=sig|out, tau=k (PLAN_4.md): the rate is 1/(1+U/k), U the cell's own accumulated use (sig: sizes of the signal it acted on, before the rate; out: its outputs).
    join=sig theta=T (PLAN_4.md): cell H0+i joins when the common signal C passes (i+1)*T; C is the running sum of E, and E the answering cell's running mean
       (rate rhoE, default 1e-3) of its squared mismatch, a signal that reaches every cell alike.
With H0=H, law=0 and nothing else the arithmetic is that of the plain system of cells2.py."""
import sys
import numpy as np
import cells2
from cells2 import OFF, D, S, NL, sg
OPT = {}
class AgeSys(cells2.Sys):
    def __init__(s, seed, **k):
        super().__init__(seed, **k); o = OPT; H = s.H; H0 = int(o.get("H0", H)); life = int(o.get("life", 5400000))
        s.law = {"0": 0, "exp": 1, "hyp": 2}[o.get("law", "0")]; s.tau = float(o.get("tau", 1e5)); s.young = int(float(o.get("young", 0)))
        s.usemode = {"0": 0, "sig": 1, "out": 2}[o.get("use", "0")]; s.sigjoin = int(o.get("join", "0") == "sig"); s.rhoE = float(o.get("rhoE", 1e-3))
        s.born = np.zeros(H, np.int64); i = np.arange(1, H - H0 + 1)
        s.thr = np.zeros(H)
        if s.sigjoin: s.born[H0:] = np.iinfo(np.int64).max; s.thr[H0:] = i * float(o["theta"])
        else: s.born[H0:] = (i * life) // (H - H0 + 1)
        s.n = 0; s.nl = 0; s.E = 0.0; s.C = 0.0; s.U = np.zeros(H); s.seen = np.zeros((H, NL), np.uint8)
    def join(s):
        if s.sigjoin:
            while s.nl < s.H and s.thr[s.nl] <= s.C: s.born[s.nl] = s.n; s.nl += 1
        else:
            while s.nl < s.H and s.born[s.nl] <= s.n: s.nl += 1
    def rate(s):
        live = (np.arange(s.H) < s.nl).astype(float); age = np.where(live > 0, s.n - s.born, 0).astype(float)
        if s.usemode: return live / (1 + s.U / s.tau)
        return live * (np.exp(-age / s.tau) if s.law == 1 else 1 / (1 + age / s.tau) if s.law == 2 else 1.0)
    def learn(s, y):
        s.join(); rate = s.rate(); live = (np.arange(s.H) < s.nl); act = OFF + np.asarray(s.hist); h = sg(s.W1[:, act].sum(1) + s.b1) * live
        if s.young:
            yg = np.nonzero(live & (s.n - s.born < s.young))[0]; s.seen[np.ix_(yg, act)] = 1; h = h * s.seen[:, act].all(1)
        m = -(s.W2 @ h + s.b2); m[y] += 1; g = s.eta2 / (h @ h + 1)
        raw = (s.R @ m) * h * (1 - h); sig = raw * rate; d2 = np.outer(g * m, h * rate)
        s.W1[:, act] += s.eta1 * sig[:, None]; s.b1 += s.eta1 * sig
        s.W2 += d2; s.W2 *= 1 - s.lam * rate; s.R += d2.T; s.R *= 1 - s.lam * rate[:, None]; s.b2 += g * m
        if s.usemode == 1: s.U += np.abs(raw)
        elif s.usemode == 2: s.U += h
        s.E += s.rhoE * (m @ m - s.E); s.C += s.E; s.n += 1
    def flush(s):
        if len(s.bx): import fastage; fastage.run(s)
    def out(s, u):
        """Scoring, many lines at once: here u is (drive, open) made by score(); cells not yet joined and cells with a closed strength on an active line are silent."""
        raise NotImplementedError
    def score(s, op, pairs, place=False):
        s.flush(); Ls = np.array([cells2.text(op, a, b) for a, b in pairs]); ok = np.ones(len(pairs), bool); W1T = s.W1.T; pl = np.zeros(3); live = np.arange(s.H) < s.nl
        for t in (5, 6, 7):
            hist = np.full((len(pairs), D), cells2.END); hist[:, :t + 1] = Ls[:, t::-1]; h = sg(W1T[OFF + hist].sum(1) + s.b1) * live
            if s.young: h = h * s.seen.T[OFF + hist].all(1)
            r = (h @ s.W2.T + s.b2).argmax(1) == Ls[:, t + 1]; ok &= r; pl[t - 5] = r.sum()
        return [int(round(100 * x / len(pairs))) for x in pl] if place else float(100 * ok.sum() / len(pairs))
    def census(s):
        return dict(joined=int(s.nl), n=int(s.n), E=float(s.E), C=float(s.C))

def run(kw):
    global OPT
    OPT = kw; cells2.Sys = AgeSys; o = cells2.run(kw); s = cells2.run.last; o["census"] = s.census(); return o

if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[2:]); kw["label"] = sys.argv[1]; o = run(kw); r0 = [r for r in o["rows"] if r["phase"] == 0]
    import json, os; json.dump(o, open(os.path.join(cells2.HERE, "res", f"c2_{kw['label']}.json"), "w"))
    print(o["label"], "| " + " ".join(f"{r['phase']}:{r['lines']//1000}k {r['add']:.0f}/{r['sub']:.0f}" for r in o["rows"]), "|", o["census"], o["cpu"], "s")
