#!/usr/bin/env python3
"""nano/multi, PLAN_3.md: the system with two steps of relay of loop_small_case/cells2.py (`H2=`: lines, cells A, cells B, the answering cell) on
two-digit sums, in which a cell B holds a fixed number of connections: edges, each from one cell A, and pair-connections, each from two cells A and
active in proportion to the product of their two outputs. numpy is the reference; fast=1 runs kernq.c.
    python3 prod.py <label> seed=1 [H=128] [H2=128] [ke=all|K] [kp=0] [rewire=4096] [joint=0] [lines=400000] [every=50000] [fast=1] [rho=1/1024]
    ke=all, kp=0, rewire=0: cells2.py with H=128 H2=128, the same draws and arithmetic.
    rewire=T: every T bytes each cell B withdraws the weakest (size of strength) of all its connections at least four periods old and tries another of the same kind at strength 0.
    joint=1: every cell of a pair-connection can set it to zero. Each sending cell keeps, for each pair-connection it sends into, a running mean of the
      size of what it receives back over that connection; at a turnover each cell A withdraws from the one, at least four periods old, over which it
      has received least. The connection is then at zero whatever its strength was, and its cell B tries another pair."""
import json, os, sys, time
import numpy as np
from multi import SYMS, SI, END, text, sg
HERE = os.path.dirname(os.path.abspath(__file__)); S = 14; D = 10; NL = D * S; OFF = np.arange(D) * S

class Sys3:
    def __init__(s, seed, H=128, H2=128, ke=None, kp=0, rewire=0, joint=0, eta1=.25, eta2=1 / 16, lam=1e-5, rho=1 / 1024):
        wr = np.random.default_rng([seed, 20261119]); g2 = np.random.default_rng([seed, 20261121]); s.g3 = g3 = np.random.default_rng([seed, 20261123])
        s.H, s.H2, s.eta1, s.eta2, s.lam, s.rewire, s.joint, s.rho = H, H2, eta1, eta2, lam, rewire, joint, rho
        s.W1 = wr.uniform(-.5, .5, (H, NL)); s.b1 = np.zeros(H); s.R = wr.uniform(-1, 1, (H2, S)); s.W2 = np.zeros((S, H2)); s.b2 = np.zeros(S); s.bB = np.zeros(H2)
        if ke is None:                                  # every cell B reads every cell A: the births of cells2.py
            assert not kp and not rewire; s.KE = H; s.EA = np.tile(np.arange(H), (H2, 1)); s.EW = g2.uniform(-1, 1, (H2, H)) / np.sqrt(H / 3); s.ER = np.ascontiguousarray(g2.uniform(-1, 1, (H, H2)).T)
        else:
            K = ke + kp; s.KE = ke; s.EA = np.zeros((H2, ke), np.int64)
            for k in range(H2): s.EA[k] = g3.choice(H, ke, replace=False)
            s.EW = g3.uniform(-1, 1, (H2, ke)) / np.sqrt(K / 3); s.ER = g3.uniform(-1, 1, (H2, ke))
        K = s.KE + kp; s.KP = kp; s.PA = np.zeros((H2, kp), np.int64); s.PB = np.zeros((H2, kp), np.int64)
        for k in range(H2):
            a, b = s.draw(kp)
            while kp:
                code = a * H + b; _, first = np.unique(code, return_index=True); dup = np.ones(kp, bool); dup[first] = False
                if not dup.any(): break
                a[dup], b[dup] = s.draw(int(dup.sum()))
            s.PA[k], s.PB[k] = a, b
        s.PW = g3.uniform(-1, 1, (H2, kp)) / np.sqrt(K / 3); s.PR = g3.uniform(-1, 1, (H2, kp)); s.QA = np.zeros((H2, kp)); s.QB = np.zeros((H2, kp))
        s.Eage = np.zeros((H2, s.KE), np.int32); s.Page = np.zeros((H2, kp), np.int32); s.nbytes = 0; s.made = [0, 0, 0]      # connections made after a withdrawal: edges by B, pairs by B, pairs after a sending cell withdrew
        s.hist = [END] * D; s.fast = 0; s.right = 0
    def draw(s, n):
        """n pairs of two different cells A, the lower index first."""
        a = s.g3.integers(s.H, size=n); b = (a + 1 + s.g3.integers(s.H - 1, size=n)) % s.H; return np.minimum(a, b), np.maximum(a, b)
    def step(s, x, y):
        s.hist.insert(0, int(x)); s.hist.pop()
        if y < 0: return
        act = OFF + np.asarray(s.hist); hA = sg(s.W1[:, act].sum(1) + s.b1)
        aE = hA[s.EA]; aP = hA[s.PA] * hA[s.PB]                                   # a pair-connection is active in proportion to the product of its two sending cells' outputs
        hB = sg((s.EW * aE).sum(1) + (s.PW * aP).sum(1) + s.bB); a = s.W2 @ hB + s.b2; s.right += int(a.argmax() == y)
        m = -a; m[y] += 1; g = s.eta2 / (hB @ hB + 1); sig = (s.R @ m) * hB * (1 - hB)
        # what a cell B received, passed on over each connection's return strength; over a pair-connection, times the other sending cell's output
        raw = np.zeros(s.H); np.add.at(raw, s.EA, s.ER * sig[:, None]); cA = s.PR * sig[:, None] * hA[s.PB]; cB = s.PR * sig[:, None] * hA[s.PA]
        np.add.at(raw, s.PA, cA); np.add.at(raw, s.PB, cB); sA = raw * hA * (1 - hA)
        c = s.eta1 * D / ((aE * aE).sum(1) + (aP * aP).sum(1) + 1) * sig            # a cell B's step is divided by the summed squares of what its own connections carry
        dE = c[:, None] * aE; s.EW += dE; s.ER += dE; s.EW *= 1 - s.lam; s.ER *= 1 - s.lam
        dP = c[:, None] * aP; s.PW += dP; s.PR += dP; s.PW *= 1 - s.lam; s.PR *= 1 - s.lam; s.bB += s.eta1 * sig
        if s.joint: s.QA += s.rho * (np.abs(cA) - s.QA); s.QB += s.rho * (np.abs(cB) - s.QB)
        d = s.eta1 * sA; d2 = np.outer(g * m, hB); s.R += d2.T; s.W1[:, act] += d[:, None]; s.b1 += d; s.W2 += d2; s.b2 += g * m; s.W2 *= 1 - s.lam; s.R *= 1 - s.lam
        if s.rewire:
            s.nbytes += 1
            if s.nbytes % s.rewire == 0: s.turnover()
    def newpair(s, k):
        while True:
            a, b = s.draw(1)
            if not ((s.PA[k] == a[0]) & (s.PB[k] == b[0])).any(): return int(a[0]), int(b[0])
    def turnover(s):
        H, H2 = s.H, s.H2; j = np.arange(H2); wl = np.full(H2, np.inf); wp = np.full(H2, np.inf); s.Eage += 1; s.Page += 1
        if s.KE:
            aw = np.where(s.Eage >= 4, np.abs(s.EW), np.inf); weak = aw.argmin(1); wl = aw[j, weak]
            free = s.g3.random((H2, H)); free[j[:, None], s.EA] = -1.0; new = free.argmax(1)
        if s.KP: ap = np.where(s.Page >= 4, np.abs(s.PW), np.inf); pweak = ap.argmin(1); wp = ap[j, pweak]
        dol = np.isfinite(wl) & (wl <= wp); dop = np.isfinite(wp) & (wp < wl)
        if dol.any(): jj = j[dol]; q = weak[dol]; s.EA[jj, q] = new[dol]; s.EW[jj, q] = 0.0; s.ER[jj, q] = 0.0; s.Eage[jj, q] = 0; s.made[0] += len(jj)
        for k in j[dop]: s.put(k, pweak[k]); s.made[1] += 1
        if s.joint and s.KP:                           # each cell A withdraws from the pair-connection over which it has received least
            n = H2 * s.KP; snd = np.concatenate([s.PA.ravel(), s.PB.ravel()]); tr = np.concatenate([s.QA.ravel(), s.QB.ravel()]); ok = np.nonzero(np.concatenate([s.Page.ravel() >= 4] * 2))[0]
            if len(ok):
                o = ok[np.lexsort((ok, tr[ok], snd[ok]))]; _, first = np.unique(snd[o], return_index=True)
                for e in np.unique(o[first] % n): s.put(e // s.KP, e % s.KP); s.made[2] += 1
    def put(s, k, q):
        s.PA[k, q], s.PB[k, q] = s.newpair(k); s.PW[k, q] = 0.0; s.PR[k, q] = 0.0; s.QA[k, q] = 0.0; s.QB[k, q] = 0.0; s.Page[k, q] = 0
    def feed(s, xs, ys):
        if s.fast: import fastq; fastq.run(s, xs, ys)
        else:
            for x, y in zip(np.asarray(xs).tolist(), np.asarray(ys).tolist()): s.step(x, y)
    def out(s, hist):
        hA = sg(s.W1.T[OFF + hist].sum(1) + s.b1); u = (hA[:, s.EA] * s.EW).sum(2) + (hA[:, s.PA] * hA[:, s.PB] * s.PW).sum(2) + s.bB; return sg(u) @ s.W2.T + s.b2
    def counts(s):
        K = s.KE + s.KP; return dict(edges_a_cell_B=s.KE, pairs_a_cell_B=s.KP, strengths_a_cell_B=K, cells_A=s.H * (NL + 1), cells_B=s.H2 * (K + 1), return_to_A=s.H2 * K,
                                     answering=S * s.H2 + S, return_to_B=s.H2 * S, total=s.H * (NL + 1) + s.H2 * (2 * K + 1) + 2 * S * s.H2 + S)

def score(s, pairs):
    Ls = np.array([text("+", a, b) for a, b in pairs]); ok = np.ones(len(pairs), bool); pl = []
    for t in (5, 6, 7):
        hist = np.full((len(pairs), D), END); hist[:, :t + 1] = Ls[:, t::-1]; r = s.out(hist).argmax(1) == Ls[:, t + 1]; ok &= r; pl.append(float(100 * r.mean()))
    return float(100 * ok.mean()), pl

def run(kw):
    t0 = time.process_time(); seed = int(kw.get("seed", 1)); rng = np.random.default_rng([seed, 99]); held, shown = {}, {}
    for op in "+-": o = rng.permutation(10000); P = [(int(q) // 100, int(q) % 100) for q in o]; held[op] = P[:500]; shown[op] = P[2000:]
    ke = kw.get("ke", "all")
    s = Sys3(seed, H=int(kw.get("H", 128)), H2=int(kw.get("H2", 128)), ke=None if ke == "all" else int(ke), kp=int(kw.get("kp", 0)), rewire=int(kw.get("rewire", 0)), joint=int(kw.get("joint", 0)),
             eta1=float(eval(kw.get("eta1", "1/4"))), eta2=float(eval(kw.get("eta2", "1/16"))), lam=float(kw.get("lam", 1e-5)), rho=float(eval(kw.get("rho", "1/1024"))))
    s.fast = int(kw.get("fast", 0)); n = int(kw.get("lines", 400000)); every = int(kw.get("every", 50000)); TAB = np.array([text("+", a, b) for a, b in shown["+"]], np.int32)
    out = dict(label=kw.get("label"), args=kw, counts=s.counts(), rows=[]); it = 0
    while it < n:
        nxt = min(n, (it // every + 1) * every); X = TAB[rng.integers(8000, size=nxt - it)]
        s.feed(X.ravel(), np.concatenate([X[:, 1:], np.full((len(X), 1), -1, np.int32)], 1).ravel()); it = nxt
        if it % every == 0: a, pl = score(s, held["+"]); out["rows"].append(dict(lines=it, add=a, place=pl))
    out["made_edges_pairs_bysender"] = s.made; out["cpu"] = round(time.process_time() - t0, 1); out["last3"] = float(np.mean([r["add"] for r in out["rows"][-3:]])) if out["rows"] else None; run.last = s
    if s.KP: out["pair_abs_mean"] = float(np.abs(s.PW).mean()); out["ends_a_cell_A_min_max"] = [int(v) for v in (lambda c: (c.min(), c.max()))(np.bincount(np.concatenate([s.PA.ravel(), s.PB.ravel()]), minlength=s.H))]
    if s.KE: out["edge_abs_mean"] = float(np.abs(s.EW).mean())
    if kw.get("nowrite"): return out
    os.makedirs(os.path.join(HERE, "res"), exist_ok=True); json.dump(out, open(os.path.join(HERE, "res", f"p3_{kw['label']}.json"), "w")); return out

if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[2:]); kw["label"] = sys.argv[1]; o = run(kw)
    print(o["label"], "last3", round(o["last3"], 1), "| " + " ".join(f"{r['lines']//1000}k {r['add']:.0f}" for r in o["rows"]), "| place", [round(x) for x in o["rows"][-1]["place"]],
          "| made", o["made_edges_pairs_bysender"], "| strengths a cell B", o["counts"]["strengths_a_cell_B"], o["cpu"], "s")
