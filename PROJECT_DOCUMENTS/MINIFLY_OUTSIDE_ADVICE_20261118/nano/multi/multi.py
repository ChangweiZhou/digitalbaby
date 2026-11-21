#!/usr/bin/env python3
"""nano/multi, PLAN_1.md: the system of loop_small_case/cells2.py (one step of relay) in which a cell in between may hold, beside connections
from single lines, pair-connections: each names two lines and is active when both are active. numpy is the reference; fast=1 runs kernm.c.
    python3 multi.py <label> seed=1 [nline=64|all] [npair=0] [rewire=4096] [H=256] [lines=400000] [every=50000] [fast=1] [eta1=1/4] [eta2=1/16] [lam=1e-5] [adj=0] [pdraw=uniform|active]
    nline=K: a cell reads K single lines, drawn at birth (cells2's fan=K); nline=all: every line.  npair=P: a cell holds P pair-connections.
    rewire=T: every T bytes each cell withdraws the weakest of all its connections that are at least four periods old (single lines and pairs
      compared together by the size of their strengths) and tries another of the same kind at strength 0.
    adj=1: the two lines of a pair are at neighbouring delays.  pdraw=active: a pair tried at a turnover is drawn among the pairs of lines active at that byte.
With npair=0 the arithmetic and every random draw are those of cells2.py with fan= and rewire= (commit 260c5d7)."""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
SYMS = list(b"0123456789+=\n-"); SI = {c: i for i, c in enumerate(SYMS)}; END = SI[10]

def text(op, a, b):
    if op == "-" and a < b: a, b = b, a
    return [SI[c] for c in f"{a:02d}{op}{b:02d}={(a + b if op == '+' else a - b):03d}\n".encode()]

def sg(u): return 1 / (1 + np.exp(-u))

class Sys:
    def __init__(s, seed, S=14, D=10, H=256, nline=None, npair=0, rewire=0, eta1=.25, eta2=1 / 16, lam=1e-5, adj=0, comp=0, pad=END, pdraw="uniform", pscale=.5):
        wr = np.random.default_rng([seed, 20261119]); s.g2 = g2 = np.random.default_rng([seed, 20261121]); s.g3 = np.random.default_rng([seed, 20261122])
        s.S, s.D, s.NL, s.H, s.eta1, s.eta2, s.lam, s.rewire, s.adj, s.comp, s.pdraw = S, D, D * S, H, eta1, eta2, lam, rewire, adj, comp, pdraw
        NL = s.NL; s.OFF = np.arange(D) * S; s.nline = nline
        s.W1 = wr.uniform(-.5, .5, (H, NL)); s.b1 = np.zeros(H); s.R = wr.uniform(-1, 1, (H, S)); s.W2 = np.zeros((S, H)); s.b2 = np.zeros(S)
        s.M = None; s.made = 0; s.madep = 0; s.nbytes = 0
        if nline is not None:
            s.M = np.zeros((H, NL), bool)
            for j in range(H): s.M[j, g2.choice(NL, nline, replace=False)] = True
            s.W1 *= s.M; s.age = np.zeros((H, NL), np.int32)
        g2.uniform(-1, 1, (H, S))                       # cells2.py draws a second answering cell's return strengths here; drawn and dropped, so that the stream of g2 is the same
        s.P = npair; s.PA = np.zeros((H, max(npair, 1)), np.int64); s.PB = np.zeros((H, max(npair, 1)), np.int64); s.PW = np.zeros((H, max(npair, 1))); s.Pag = np.zeros((H, max(npair, 1)), np.int32)
        if npair:
            for j in range(H):
                a, b = s.draw(npair)
                while True:
                    code = a * NL + b; _, first = np.unique(code, return_index=True); dup = np.ones(npair, bool); dup[first] = False
                    if not dup.any(): break
                    a[dup], b[dup] = s.draw(int(dup.sum()))
                s.PA[j], s.PB[j] = a, b
            s.PW = s.g3.uniform(-pscale, pscale, (H, npair))
        s.hist = [pad] * D; s.fast = 0; s.right = 0; s.nats = 0.0; s.seen = 0
    def draw(s, n):
        """n pairs of lines at two different delays (the nearer first); with adj, at neighbouring delays."""
        g = s.g3; D, S = s.D, s.S
        if s.adj: lo = g.integers(D - 1, size=n); hi = lo + 1
        else: d1 = g.integers(D, size=n); d2 = (d1 + 1 + g.integers(D - 1, size=n)) % D; lo = np.minimum(d1, d2); hi = np.maximum(d1, d2)
        return lo * S + g.integers(S, size=n), hi * S + g.integers(S, size=n)
    def fwd(s, hist):
        act = s.OFF + np.asarray(hist); u = s.W1[:, act].sum(1) + s.b1; pa = None
        if s.P:
            on = np.zeros(s.NL, bool); on[act] = True; pa = on[s.PA] & on[s.PB]; u = u + (s.PW * pa).sum(1)
        h = sg(u); return act, pa, h, s.W2 @ h + s.b2
    def step(s, x, y):
        """The byte x is read; if y >= 0 the system names the next byte, y arrives, and every cell changes its own strengths."""
        s.hist.insert(0, int(x)); s.hist.pop()
        if y < 0: return
        act, pa, h, a = s.fwd(s.hist); s.right += int(a.argmax() == y); s.seen += 1
        if s.comp:                                    # branches that compete: the mismatch is the byte that arrived less the share given to each branch
            p = np.exp(a - a.max()); p /= p.sum(); s.nats -= float(np.log(p[y])); m = -p
        else: m = -a
        m[y] += 1; g = s.eta2 / (h @ h + 1)
        sig = (s.R @ m) * h * (1 - h); d2 = np.outer(g * m, h); d = s.eta1 * sig
        s.R += d2.T; d1 = np.repeat(d[:, None], s.D, 1)
        if s.M is not None: d1 = d1 * s.M[:, act]
        s.W1[:, act] += d1; s.b1 += d; s.W2 += d2; s.b2 += g * m; s.W2 *= 1 - s.lam; s.R *= 1 - s.lam
        if s.P: s.PW += d[:, None] * pa               # a pair-connection changes as a line's does, when it is active
        if s.rewire:
            s.nbytes += 1
            if s.nbytes % s.rewire == 0: s.turnover()
    def turnover(s):
        """Each cell withdraws the weakest of its connections (single lines and pairs together) that are at least four periods old, and makes one of
        the same kind, at strength 0, that it does not hold. The cell compares only the sizes of its own strengths."""
        H, NL = s.H, s.NL; j = np.arange(H); wl = np.full(H, np.inf); wp = np.full(H, np.inf)
        if s.M is not None and s.nline:
            s.age[s.M] += 1; aw = np.where(s.M & (s.age >= 4), np.abs(s.W1), np.inf); weak = aw.argmin(1); wl = aw[j, weak]
            free = s.g2.random((H, NL)); free[s.M] = -1.0; new = free.argmax(1)
        if s.P:
            s.Pag += 1; ap = np.where(s.Pag >= 4, np.abs(s.PW), np.inf); pweak = ap.argmin(1); wp = ap[j, pweak]
            if s.pdraw == "active":                   # among the pairs of lines active at this byte
                act = s.OFF + np.asarray(s.hist)
                if s.adj: lo = s.g3.integers(s.D - 1, size=H); hi = lo + 1
                else: d1 = s.g3.integers(s.D, size=H); d2 = (d1 + 1 + s.g3.integers(s.D - 1, size=H)) % s.D; lo = np.minimum(d1, d2); hi = np.maximum(d1, d2)
                a, b = act[lo], act[hi]; dup = ((s.PA == a[:, None]) & (s.PB == b[:, None])).any(1)
            else:
                a, b = s.draw(H)
                for _ in range(8):
                    dup = ((s.PA == a[:, None]) & (s.PB == b[:, None])).any(1)
                    if not dup.any(): break
                    a[dup], b[dup] = s.draw(int(dup.sum()))
                dup = ((s.PA == a[:, None]) & (s.PB == b[:, None])).any(1)
        dol = np.isfinite(wl) & (wl <= wp); dop = np.isfinite(wp) & (wp < wl)
        if dol.any():
            jj = j[dol]; s.M[jj, weak[dol]] = False; s.W1[jj, weak[dol]] = 0.0; s.age[jj, weak[dol]] = 0; s.M[jj, new[dol]] = True; s.made += len(jj)
        if s.P:
            dop &= ~dup
            if dop.any():
                jj = j[dop]; k = pweak[dop]; s.PA[jj, k] = a[dop]; s.PB[jj, k] = b[dop]; s.PW[jj, k] = 0.0; s.Pag[jj, k] = 0; s.madep += len(jj)
    def feed(s, xs, ys):
        if s.fast: import fastm; fastm.run(s, xs, ys)
        else:
            for x, y in zip(np.asarray(xs).tolist(), np.asarray(ys).tolist()): s.step(x, y)
    def drive(s, hist):
        """The drives of the cells for many positions at once (hist: positions by delays, newest first); the same arithmetic as fwd."""
        if s.fast: import fastm; return fastm.drive(s, hist)
        act = s.OFF + hist; u = s.W1.T[act].sum(1) + s.b1
        if s.P:
            for i in range(0, len(hist), 256):
                X = np.zeros((len(act[i:i + 256]), s.NL), bool); X[np.arange(len(X))[:, None], act[i:i + 256]] = True; u[i:i + 256] += ((X[:, s.PA] & X[:, s.PB]) * s.PW).sum(2)
        return u
    def out(s, hist): return sg(s.drive(hist)) @ s.W2.T + s.b2
    def counts(s):
        """Numbers that learn: strengths a cell in between holds (single lines, pairs, threshold); the answering cell; the return strengths."""
        nl = s.NL if s.M is None else int(s.nline); return dict(lines_a_cell=nl, pairs_a_cell=s.P, strengths_a_cell=nl + s.P, cells=s.H,
            in_between=s.H * (nl + s.P + 1), answering=s.S * s.H + s.S, return_=s.H * s.S, total=s.H * (nl + s.P + 1) + 2 * s.S * s.H + s.S)

def score(s, pairs):
    """Whole answer right on never-shown sums, %, and each place of the answer."""
    D = s.D; Ls = np.array([text("+", a, b) for a, b in pairs]); ok = np.ones(len(pairs), bool); pl = []
    for t in (5, 6, 7):
        hist = np.full((len(pairs), D), END); hist[:, :t + 1] = Ls[:, t::-1]; r = s.out(hist).argmax(1) == Ls[:, t + 1]; ok &= r; pl.append(float(100 * r.mean()))
    return float(100 * ok.mean()), pl

def run(kw):
    t0 = time.process_time(); seed = int(kw.get("seed", 1)); rng = np.random.default_rng([seed, 99]); held, shown = {}, {}
    for op in "+-":                                     # the same draws as cells2.py: the first 2,000 of a permutation are never shown, 500 of them scored
        o = rng.permutation(10000); P = [(int(q) // 100, int(q) % 100) for q in o]; held[op] = P[:500]; shown[op] = P[2000:]
    nl = kw.get("nline", "all")
    s = Sys(seed, H=int(kw.get("H", 256)), nline=None if nl == "all" else int(nl), npair=int(kw.get("npair", 0)), rewire=int(kw.get("rewire", 0)), adj=int(kw.get("adj", 0)),
            eta1=float(eval(kw.get("eta1", "1/4"))), eta2=float(eval(kw.get("eta2", "1/16"))), lam=float(kw.get("lam", 1e-5)), pdraw=kw.get("pdraw", "uniform"), pscale=float(kw.get("pscale", .5)))
    s.fast = int(kw.get("fast", 0)); n = int(kw.get("lines", 400000)); every = int(kw.get("every", 50000)); TAB = np.array([text("+", a, b) for a, b in shown["+"]], np.int32)
    out = dict(label=kw.get("label"), args=kw, counts=s.counts(), rows=[]); it = 0
    while it < n:
        nxt = min(n, (it // every + 1) * every); X = TAB[rng.integers(8000, size=nxt - it)]
        s.feed(X.ravel(), np.concatenate([X[:, 1:], np.full((len(X), 1), -1, np.int32)], 1).ravel()); it = nxt
        if it % every == 0: a, pl = score(s, held["+"]); out["rows"].append(dict(lines=it, add=a, place=pl))
    out["made_lines"] = s.made; out["made_pairs"] = s.madep; out["cpu"] = round(time.process_time() - t0, 1)
    out["last3"] = float(np.mean([r["add"] for r in out["rows"][-3:]])) if out["rows"] else None; run.last = s
    if s.P:                                             # which delays the pairs held at the end name (a count for each pair of delays)
        dd = np.zeros((s.D, s.D), int); np.add.at(dd, (s.PA // s.S, s.PB // s.S), 1); out["pair_delays"] = dd.tolist(); out["pair_abs_mean"] = float(np.abs(s.PW).mean())
    if s.M is not None and s.nline: out["line_abs_mean"] = float(np.abs(s.W1[s.M]).mean())
    if kw.get("nowrite"): return out
    os.makedirs(os.path.join(HERE, "res"), exist_ok=True); json.dump(out, open(os.path.join(HERE, "res", f"m1_{kw['label']}.json"), "w")); return out

if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[2:]); kw["label"] = sys.argv[1]; o = run(kw)
    print(o["label"], "last3", round(o["last3"], 1), "| " + " ".join(f"{r['lines']//1000}k {r['add']:.0f}" for r in o["rows"]), "| place", [round(x) for x in o["rows"][-1]["place"]],
          "| made", o["made_lines"], o["made_pairs"], "| strengths a cell", o["counts"]["strengths_a_cell"], o["cpu"], "s")
