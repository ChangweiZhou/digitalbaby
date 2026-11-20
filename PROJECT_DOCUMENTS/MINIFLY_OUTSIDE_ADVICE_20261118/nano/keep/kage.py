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
       later, a cell with a closed strength on an active line is silent at that byte and unchanged by it (its rate is nothing at that byte, so that
       the strengths to and from it do not fade either, as for a cell that has not joined).
    use=sig|out, tau=k (PLAN_4.md): the rate is 1/(1+U/k), U the cell's own accumulated use (sig: sizes of the signal it acted on, before the rate; out: its outputs).
    blaw=exp|hyp btau=t (PLAN_4.md): the answering cell's steps on its constants are multiplied by exp(-n/t) or 1/(1+n/t), n its age in learned bytes.
    join=sig theta=T [delta=.25 rhoE=1e-3 rhoS=1e-5] (PLAN_5.md): the answering cell keeps two running means of its squared mismatch, a fast one E (rate rhoE)
       and a slow one G (rate rhoS); the common signal is the excess X = max(0, E - (1+delta) G), which reaches every cell alike; cell H0+i joins when
       the running sum C of X passes (i+1)*T. No clock and no length of life is given.
    join=hear K=k theta=T [rhoA=1e-2] (PLAN_6.md): the answering cell keeps a running mean A (rate rhoA) of the number of cells it hears (joined and not
       silenced at the byte); the common signal is the shortfall X = max(0, 1 - A/k); cell H0+i joins when the running sum of X passes (i+1)*T.
    meet=c (PLAN_7.md; with young): a strength is open after youth if its line was active on at least c of the cell's young bytes (c=1: PLAN_3.md's rule;
       the count stops at 255). During its youth a cell is not silenced.
    openall=1 (PLAN_8.md): a young cell counts a line also at the one byte of each text line from which the system does not learn (the end of the line,
       after which nothing is named). Without it a line that is active only at that byte is never met.
    subtext=swap (PLAN_8.md; a text, not a part of the system): subtraction is written with the bytes of addition, `aa=bb+ccc` for aa-bb=ccc.
With H0=H, law=0 and nothing else the arithmetic is that of the plain system of cells2.py."""
import sys
import numpy as np
import cells2
from cells2 import OFF, D, S, NL, sg
OPT = {}; TEXT0 = cells2.text
class AgeSys(cells2.Sys):
    def __init__(s, seed, **k):
        super().__init__(seed, **k); o = OPT; H = s.H; H0 = int(o.get("H0", H)); life = int(o.get("life", 5400000))
        s.law = {"0": 0, "exp": 1, "hyp": 2}[o.get("law", "0")]; s.tau = float(o.get("tau", 1e5)); s.young = int(float(o.get("young", 0)))
        s.usemode = {"0": 0, "sig": 1, "out": 2}[o.get("use", "0")]; s.sigjoin = {"0": 0, "sig": 1, "hear": 2}[o.get("join", "0")]; s.K = float(o.get("K", 128)); s.rhoA = float(o.get("rhoA", 1e-2)); s.A = 0.0; s.meet = int(o.get("meet", 1))
        assert 1 <= s.meet <= 255; s.openall = int(o.get("openall", 0)); s.rhoE = float(o.get("rhoE", 1e-3)); s.rhoS = float(o.get("rhoS", 1e-5)); s.delta = float(o.get("delta", .25))
        s.blaw = {"0": 0, "exp": 1, "hyp": 2}[o.get("blaw", "0")]; s.btau = float(o.get("btau", 1e5)); s.G = 0.0
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
            ygm = live & (s.n - s.born < s.young); yg = np.nonzero(ygm)[0]; blk = s.seen[np.ix_(yg, act)]; s.seen[np.ix_(yg, act)] = np.where(blk < 255, blk + 1, 255)
            op = ygm | (s.seen[:, act] >= s.meet).all(1); h = h * op; rate = rate * op      # silent at this byte: unchanged, the fading included
            heard = int((live & op).sum())
        else: heard = int(live.sum())
        m = -(s.W2 @ h + s.b2); m[y] += 1; g = s.eta2 / (h @ h + 1)
        raw = (s.R @ m) * h * (1 - h); sig = raw * rate; d2 = np.outer(g * m, h * rate)
        s.W1[:, act] += s.eta1 * sig[:, None]; s.b1 += s.eta1 * sig
        s.W2 += d2; s.W2 *= 1 - s.lam * rate; s.R += d2.T; s.R *= 1 - s.lam * rate[:, None]
        s.b2 += g * m * (np.exp(-s.n / s.btau) if s.blaw == 1 else 1 / (1 + s.n / s.btau) if s.blaw == 2 else 1.0)
        if s.usemode == 1: s.U += np.abs(raw)
        elif s.usemode == 2: s.U += h
        s.E += s.rhoE * (m @ m - s.E); s.G += s.rhoS * (m @ m - s.G); s.A += s.rhoA * (heard - s.A)
        s.C += max(0.0, 1 - s.A / s.K) if s.sigjoin == 2 else max(0.0, s.E - (1 + s.delta) * s.G); s.n += 1
    def line(s, L):
        if s.fast: return super().line(L)
        for t in range(len(L) - 1):
            s.hist.insert(0, L[t]); s.hist.pop(); s.learn(L[t + 1])
        s.hist.insert(0, L[-1]); s.hist.pop()
        if s.openall and s.young:                     # the byte nothing is learned from: young cells count their active lines all the same
            act = OFF + np.asarray(s.hist); yg = np.nonzero((np.arange(s.H) < s.nl) & (s.n - s.born < s.young))[0]; blk = s.seen[np.ix_(yg, act)]; s.seen[np.ix_(yg, act)] = np.where(blk < 255, blk + 1, 255)
    def flush(s):
        if len(s.bx): import fastage; fastage.run(s)
    def out(s, u):
        """Scoring, many lines at once: here u is (drive, open) made by score(); cells not yet joined and cells with a closed strength on an active line are silent."""
        raise NotImplementedError
    def score(s, op, pairs, place=False):
        s.flush(); Ls = np.array([cells2.text(op, a, b) for a, b in pairs]); ok = np.ones(len(pairs), bool); W1T = s.W1.T; pl = np.zeros(3); live = np.arange(s.H) < s.nl
        for t in (5, 6, 7):
            hist = np.full((len(pairs), D), cells2.END); hist[:, :t + 1] = Ls[:, t::-1]; h = sg(W1T[OFF + hist].sum(1) + s.b1) * live
            if s.young: h = h * ((s.seen.T[OFF + hist] >= s.meet).all(1) | (live & (s.n - s.born < s.young)))
            r = (h @ s.W2.T + s.b2).argmax(1) == Ls[:, t + 1]; ok &= r; pl[t - 5] = r.sum()
        return [int(round(100 * x / len(pairs))) for x in pl] if place else float(100 * ok.sum() / len(pairs))
    def census(s, held=None):
        c = dict(joined=int(s.nl), n=int(s.n), E=float(s.E), G=float(s.G), C=float(s.C), A=float(s.A), born=[int(x) for x in s.born[:s.nl]])
        if s.young and held:                          # a measure, not used by the system: joined cells that are not silenced on never-shown lines at the answer positions
            live = np.arange(s.H) < s.nl; o = {}
            for op in "+-":
                Ls = np.array([cells2.text(op, a, b) for a, b in held[op]]); m = []
                for t in (5, 6, 7):
                    hist = np.full((len(Ls), D), cells2.END); hist[:, :t + 1] = Ls[:, t::-1]; m.append((s.seen.T[OFF + hist] >= s.meet).all(1) & live)
                o[op] = np.concatenate(m).mean(0)      # for each cell, the share of such lines on which it may speak
            c.update(cells_speaking_on_addition=float(o["+"].sum()), cells_speaking_on_subtraction=float(o["-"].sum()), cells_speaking_on_both=int(((o["+"] > .5) & (o["-"] > .5)).sum()),
                     cells_mostly_on_addition_only=int(((o["+"] > .5) & (o["-"] <= .5)).sum()), cells_mostly_on_subtraction_only=int(((o["+"] <= .5) & (o["-"] > .5)).sum()))
        return c

def run(kw):
    global OPT
    OPT = kw; cells2.Sys = AgeSys
    if kw.get("subtext") == "swap":
        def text(op, a, b):
            if op == "-" and a < b: a, b = b, a
            return [cells2.SI[c] for c in (f"{a:02d}+{b:02d}={a + b:03d}\n" if op == "+" else f"{a:02d}={b:02d}+{a - b:03d}\n").encode()]
        cells2.text = text
    else: cells2.text = TEXT0
    o = cells2.run(kw); s = cells2.run.last; rng = np.random.default_rng([int(kw.get("seed", 1)), 99]); held = {}
    for op in "+-": q = rng.permutation(10000); held[op] = [(int(x) // 100, int(x) % 100) for x in q[:500]]      # the never-shown pairs, drawn as cells2.run draws them
    o["census"] = s.census(held); return o

if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[2:]); kw["label"] = sys.argv[1]; o = run(kw); r0 = [r for r in o["rows"] if r["phase"] == 0]
    import json, os; json.dump(o, open(os.path.join(cells2.HERE, "res", f"c2_{kw['label']}.json"), "w"))
    print(o["label"], "| " + " ".join(f"{r['phase']}:{r['lines']//1000}k {r['add']:.0f}/{r['sub']:.0f}" for r in o["rows"]), "|", o["census"], o["cpu"], "s")
