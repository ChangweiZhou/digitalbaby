#!/usr/bin/env python3
"""nano/keep2 (PLAN_1.md): my own program for the keeping line's final candidate, written from that line's prose and not from its code.
The plain system, the text, the split and the scoring are those of loop_small_case/cells2.py (imported, not changed).
    python3 rep.py <label> seed=1 [arm=cand|plain] [plan=add:400000,sub:200000] [reh=N] [harsh=1] [np=1] [every=50000]
                   [P=384] [H0=1] [K=128] [theta=20] [young=4500] [meet=30] [btau=1e4] [rate=1e-2] [fade_silent=1] [allbytes=1]
    arm=plain: 256 cells, all present, no rule. arm=cand: a pool of P cells, H0 present at birth, the three rules of PLAN_1.md section 2.
    harsh=1: subtraction is written aa=bb+ccc (larger number first). np=1: numpy (the reference); otherwise kern.c through ctypes.
    fade_silent=1: the answering and return strengths of a joined cell fade also while it is silent (the other reading of choice b).
    allbytes=1: a cell's youth and its meetings count every byte, the unpredicted one included (the other reading of choice a).
"""
import ctypes, json, os, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "..", "loop_small_case"))
import cells2
from cells2 import SI, S, D, NL, OFF, END, sg

def text(op, a, b, harsh=False):
    if op == "-" and harsh:
        if a < b: a, b = b, a
        return [SI[c] for c in f"{a:02d}={b:02d}+{a - b:03d}\n".encode()]
    return cells2.text(op, a, b)

class Par(ctypes.Structure):
    _fields_ = [(k, ctypes.c_double) for k in "eta1 eta2 lam K theta btau rate A sumX".split()] + [("nans", ctypes.c_long)] + [(k, ctypes.c_int) for k in "P H0 young meet njoined closed fade_silent allbytes join".split()]

_lib = None
def lib():
    global _lib
    if _lib is None:
        so = os.path.join(HERE, "kern.so"); src = os.path.join(HERE, "kern.c")
        if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src): subprocess.check_call(["gcc", "-O2", "-fPIC", "-shared", "-o", so, src, "-lm"])
        _lib = ctypes.CDLL(so)
    return _lib

class Cells:
    def __init__(s, seed, arm="cand", P=384, H0=1, K=128., theta=20., young=4500, meet=30, btau=1e4, rate=1e-2, fade_silent=0, allbytes=0, eta1=.25, eta2=1 / 16, lam=1e-5):
        plain = arm == "plain"; s.P = P = 256 if plain else P; base = cells2.Sys(seed, H=P)          # birth values as cells2.py draws them for P cells
        s.W1, s.b1, s.W2, s.b2, s.R = base.W1, base.b1, base.W2, base.b2, base.R
        s.eta1, s.eta2, s.lam, s.K, s.theta, s.young, s.meet, s.btau, s.rate = eta1, eta2, lam, K, theta, young, meet, (0.0 if plain else btau), rate
        s.closed = s.join = 0 if plain else 1; s.fade_silent, s.allbytes = fade_silent, allbytes; s.H0 = P if plain else H0; s.nj = s.H0
        s.cnt = np.zeros((P, NL), np.uint8); s.open = np.ones((P, NL), np.uint8); s.age = np.zeros(P, np.int32); s.A = 0.0; s.sumX = 0.0; s.nans = 0; s.hist = [END] * D
        if plain: s.age[:] = young
    def speak(s, act):
        """Rule 1: which of the cells speak at a byte whose active lines are act."""
        sp = np.zeros(s.P, bool); j = slice(0, s.nj)
        sp[j] = (s.age[j] < s.young) | s.open[j][:, act].all(1) if s.closed else True
        return sp
    def youth(s, act):
        y = np.flatnonzero((np.arange(s.P) < s.nj) & (s.age < s.young))
        if not len(y): return
        c = s.cnt[np.ix_(y, act)]; s.cnt[np.ix_(y, act)] = c + (c < 255); s.age[y] += 1
        for j in y[s.age[y] == s.young]: s.open[j] = s.cnt[j] >= s.meet
    def byte(s, x, y):
        s.hist.insert(0, x); s.hist.pop(); act = OFF + np.asarray(s.hist)
        if y < 0:
            if s.closed and s.allbytes: s.youth(act)
            return
        sp = s.speak(act); f = sp.astype(float); h = sg(s.W1[:, act].sum(1) + s.b1) * f; a = s.W2 @ h + s.b2
        m = -a; m[y] += 1; g = s.eta2 / (h @ h + 1); sig = (s.R @ m) * h * (1 - h); d2 = np.outer(g * m, h); d = s.eta1 * sig
        s.R += d2.T; s.W1[:, act] += d[:, None]; s.b1 += d; s.W2 += d2
        s.b2 += g * m * (np.exp(-s.nans / s.btau) if s.btau > 0 else 1.0); s.nans += 1
        fd = f.copy()
        if s.fade_silent: fd[:s.nj] = 1.0
        s.W2 *= 1 - s.lam * fd; s.R *= 1 - s.lam * fd[:, None]
        if s.closed: s.youth(act)
        if s.join:
            s.A += s.rate * (sp.sum() - s.A); s.sumX += max(0.0, 1 - s.A / s.K)
            while s.nj < s.P and s.sumX >= (s.nj - s.H0 + 1) * s.theta: s.age[s.nj] = 0; s.nj += 1
    def run(s, x, y, numpy_=False):
        if numpy_:
            for a, b in zip(x, y): s.byte(int(a), int(b))
            return
        p = Par(s.eta1, s.eta2, s.lam, s.K, s.theta, s.btau, s.rate, s.A, s.sumX, s.nans, s.P, s.H0, s.young, s.meet, s.nj, s.closed, s.fade_silent, s.allbytes, s.join)
        x = np.ascontiguousarray(x, np.int32); y = np.ascontiguousarray(y, np.int32); W2 = np.ascontiguousarray(s.W2.T); hist = np.array(s.hist, np.int32); h = np.zeros(s.P); sp = np.zeros(s.P, np.uint8)
        ptr = lambda a: a.ctypes.data_as(ctypes.c_void_p)
        lib().run(ctypes.byref(p), ctypes.c_long(len(x)), ptr(x), ptr(y), ptr(s.W1), ptr(s.b1), ptr(W2), ptr(s.b2), ptr(s.R), ptr(s.cnt), ptr(s.open), ptr(s.age), ptr(hist), ptr(h), ptr(sp))
        s.W2 = np.ascontiguousarray(W2.T); s.hist = hist.tolist(); s.A, s.sumX, s.nans, s.nj = p.A, p.sumX, p.nans, p.njoined
    def may(s, Ls):
        """For lines Ls (n by 10) and the three answer positions: which cells speak (3 by n by P), and the scores' inputs."""
        out = []
        for t in (5, 6, 7):
            hist = np.full((len(Ls), D), END); hist[:, :t + 1] = Ls[:, t::-1]; idx = OFF + hist
            sp = np.zeros((len(Ls), s.P), bool)
            sp[:, :s.nj] = ((s.age[:s.nj] < s.young)[None, :] | s.open.T[idx][:, :, :s.nj].all(1)) if s.closed else True
            out.append((idx, sp))
        return out
    def score(s, Ls):
        """cells2.py's score: each answer digit named from the true bytes before it; a line counts if all three are right. Returns right (n,)."""
        ok = np.ones(len(Ls), bool)
        for t, (idx, sp) in zip((5, 6, 7), s.may(Ls)):
            h = sg(s.W1.T[idx].sum(1) + s.b1) * sp; ok &= (h @ s.W2.T + s.b2).argmax(1) == Ls[:, t + 1]
        return ok

def run(kw):
    t0 = time.process_time(); seed = int(kw.get("seed", 1)); harsh = int(kw.get("harsh", 0)); reh = int(kw.get("reh", 0)); every = int(kw.get("every", 50000)); nump = int(kw.get("np", 0))
    rng = np.random.default_rng([seed, 99]); held, shown = {}, {}
    for op in "+-":                                        # the split of cells2.py
        o = rng.permutation(10000); Pp = [(int(q) // 100, int(q) % 100) for q in o]; held[op] = Pp[:500]; shown[op] = Pp[2000:]
    opt = {k: (float(kw[k]) if k in ("K", "theta", "btau", "rate") else int(float(kw[k]))) for k in ("P", "H0", "K", "theta", "young", "meet", "btau", "rate", "fade_silent", "allbytes") if k in kw}
    s = Cells(seed, arm=kw.get("arm", "cand"), **opt)
    TAB = {op: np.array([text(op, a, b, harsh) for a, b in shown[op]], np.int32) for op in "+-"}
    HL = {op: np.array([text(op, a, b, harsh) for a, b in held[op]], np.int32) for op in "+-"}
    written = {tuple(L) for L in TAB["-"].tolist()}; clean = np.array([tuple(L) not in written for L in HL["-"].tolist()])   # held-out subtraction lines that no shown pair writes
    assert not ({tuple(L) for L in TAB["+"].tolist()} & {tuple(L) for L in HL["+"].tolist()})
    out = dict(label=kw["label"], args=kw, rows=[], clean_sub_lines=int(clean.sum()))
    def read(pi, name, lines):
        a = s.score(HL["+"]); b = s.score(HL["-"])
        out["rows"].append(dict(phase=pi, name=name, lines=lines, add=float(100 * a.mean()), sub=float(100 * b.mean()), sub_clean=float(100 * b[clean].mean()), joined=int(s.nj)))
    for pi, ph in enumerate(kw.get("plan", "add:400000,sub:200000").split(",")):
        name, n = ph.split(":"); n = int(n); op0 = {"add": "+", "sub": "-"}[name]; it = 0
        while it < n:
            k = min(every - it % every, n - it); X = np.empty((k, 10), np.int32)
            for i in range(k):                                 # the draws of cells2.py's unbatched loop
                op = op0
                if reh and pi > 0 and rng.integers(reh) == 0: op = "+" if op0 == "-" else "-"
                X[i] = TAB[op][rng.integers(8000)]
            Y = np.concatenate([X[:, 1:], np.full((k, 1), -1, np.int32)], 1); s.run(X.ravel(), Y.ravel(), nump); it += k
            if it % every == 0: read(pi, name, it)
    # which joined cells may speak at the answer positions of held-out lines (on more than half of them)
    fa = np.mean([sp.mean(0) for _, sp in s.may(HL["+"])], 0); fs = np.mean([sp.mean(0) for _, sp in s.may(HL["-"])], 0); j = np.arange(s.P) < s.nj
    out["speak"] = dict(add_only=int((j & (fa > .5) & (fs <= .5)).sum()), sub_only=int((j & (fa <= .5) & (fs > .5)).sum()), both=int((j & (fa > .5) & (fs > .5)).sum()), neither=int((j & (fa <= .5) & (fs <= .5)).sum()))
    out["joined"] = int(s.nj); out["A"] = s.A; out["cpu"] = round(time.process_time() - t0, 1); run.last = s
    if not kw.get("nowrite"):
        os.makedirs(os.path.join(HERE, "res"), exist_ok=True); json.dump(out, open(os.path.join(HERE, "res", f"{kw['label']}.json"), "w"))
    return out

if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[2:]); kw["label"] = sys.argv[1]; o = run(kw)
    print(o["label"], " ".join(f"{r['phase']}:{r['lines'] // 1000}k {r['add']:.1f}/{r['sub']:.1f}({r['sub_clean']:.0f}) j{r['joined']}" for r in o["rows"]), o["speak"], o["cpu"], "s", flush=True)
