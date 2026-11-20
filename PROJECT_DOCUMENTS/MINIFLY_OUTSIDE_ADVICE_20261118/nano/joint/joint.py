#!/usr/bin/env python3
"""PLAN_1.md (line C, step C1): the plain system of loop_small_case/cells2.py (two-digit sums, 256 logistic cells in between, every line read, one
step of relay with learned return strengths) with the connection from a cell in between to a branch of the answering cell in one of three forms.
    python3 joint.py <label> seed=1 [form=1|2|3] [etaB=1] [b0=1] [lines=400000] [every=50000] [fast=1] [H=256]
    form=1: one-sided; the strength w = A is held and changed by the answering cell alone (cells2.py's plain system, which this must reproduce).
    form=2: w = A B; A held by the answering cell (born 0, fades), B by the sending cell (born 1, never below 0, does not fade);
            A += g m h B;  B += etaB g m h A   (both from the values before the change).
    form=3: w = sign(A) min(|A|, B); B born b0; only the end whose part is the smaller changes: A += g m h where |A| < B, else B += g m h sign(A).
    fast=1: the inner loop runs compiled (kernj.c through fastj.py); the numpy loop here is the reference (check_joint.py).
Copied from loop_small_case/cells2.py at commit fe397ca: the text, the split, the streams of random numbers, class Sys for the plain system, the
drawing of lines in blocks up to the next reading, and the scoring of all pairs at once."""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
SYMS = list(b"0123456789+=\n-"); S = len(SYMS); SI = {c: i for i, c in enumerate(SYMS)}; END = SI[10]; D = 10; NL = D * S; OFF = np.arange(D) * S

def text(op, a, b):
    if op == "-" and a < b: a, b = b, a
    return [SI[c] for c in f"{a:02d}{op}{b:02d}={(a + b if op == '+' else a - b):03d}\n".encode()]

def sg(u): return 1 / (1 + np.exp(-u))

class Sys:
    def __init__(s, seed, form=1, etaB=1.0, b0=1.0, H=256, eta1=.25, eta2=1 / 16, lam=1e-5):
        wr = np.random.default_rng([seed, 20261119])
        s.H, s.eta1, s.eta2, s.lam, s.form, s.etaB = H, eta1, eta2, lam, form, etaB
        s.W1 = wr.uniform(-.5, .5, (H, NL)); s.b1 = np.zeros(H); s.R = wr.uniform(-1, 1, (H, S)); s.A = np.zeros((S, H)); s.b2 = np.zeros(S)
        s.B = np.full((S, H), float(b0) if form == 3 else 1.0); s.hist = [END] * D
    def W(s):
        """The strengths of the connections, as both ends' parts give them."""
        if s.form == 1: return s.A
        if s.form == 2: return s.A * s.B
        return np.sign(s.A) * np.minimum(np.abs(s.A), s.B)
    def learn(s, y):
        act = OFF + np.asarray(s.hist); h = sg(s.W1[:, act].sum(1) + s.b1); a = s.W() @ h + s.b2; m = -a; m[y] += 1; g = s.eta2 / (h @ h + 1)
        sig = (s.R @ m) * h * (1 - h); d2 = np.outer(g * m, h); d = s.eta1 * sig
        s.R += d2.T; s.W1[:, act] += np.repeat(d[:, None], D, 1); s.b1 += d
        if s.form == 1: s.A += d2
        elif s.form == 2:
            dA = d2 * s.B; s.B = np.maximum(s.B + s.etaB * d2 * s.A, 0.0); s.A += dA
        else:
            low = np.abs(s.A) < s.B; s.B = np.maximum(s.B + np.where(low, 0.0, d2 * np.sign(s.A)), 0.0); s.A += np.where(low, d2, 0.0)
        s.b2 += g * m; s.A *= 1 - s.lam; s.R *= 1 - s.lam
    def steps(s, xs, ys):
        for x, y in zip(xs.tolist(), ys.tolist()):
            s.hist.insert(0, x); s.hist.pop()
            if y >= 0: s.learn(y)
    def score(s, pairs, W=None):
        W = s.W() if W is None else W; Ls = np.array([text("+", a, b) for a, b in pairs]); ok = np.ones(len(pairs), bool); W1T = s.W1.T
        for t in (5, 6, 7):
            hist = np.full((len(pairs), D), END); hist[:, :t + 1] = Ls[:, t::-1]
            ok &= (sg(W1T[OFF + hist].sum(1) + s.b1) @ W.T + s.b2).argmax(1) == Ls[:, t + 1]
        return float(100 * ok.sum() / len(pairs))

def measures(s, held):
    """PLAN_1.md, section 3: what is left of the connections at the end."""
    W = s.W(); aw = np.abs(W); big = aw.max(); small = aw < .01 * big; o = dict(end=s.score(held), largest=float(big), share_below_hundredth=float(small.mean()))
    o["end_without_those"] = s.score(held, np.where(small, 0.0, W)); order = np.argsort(aw, axis=None)
    for q in (25, 50, 75, 90):
        V = W.copy(); V.flat[order[:int(round(q / 100 * W.size))]] = 0.0; o[f"end_without_weakest_{q}"] = s.score(held, V)
    o["cos_R_w"] = float((s.R.T * W).sum() / (np.linalg.norm(s.R) * np.linalg.norm(W) + 1e-300))
    o["B"] = dict(min=float(s.B.min()), mean=float(s.B.mean()), max=float(s.B.max()), share_zero=float((s.B == 0).mean()))
    o["A"] = dict(mean_abs=float(np.abs(s.A).mean()), max_abs=float(np.abs(s.A).max()))
    if s.form == 3: o["share_sender_part_smaller"] = float((np.abs(s.A) >= s.B).mean())
    return o

def run(kw):
    t0 = time.process_time(); seed = int(kw.get("seed", 1)); rng = np.random.default_rng([seed, 99]); held, shown = {}, {}
    for op in "+-":                                   # both drawn, as in cells2.py, so that the stream is the same
        o = rng.permutation(10000); P = [(int(q) // 100, int(q) % 100) for q in o]; held[op] = P[:500]; shown[op] = P[2000:]
    s = Sys(seed, form=int(kw.get("form", 1)), etaB=float(eval(kw.get("etaB", "1"))), b0=float(eval(kw.get("b0", "1"))), H=int(kw.get("H", 256)))
    n = int(kw.get("lines", 400000)); every = int(kw.get("every", 50000)); fast = int(kw.get("fast", 0)); out = dict(label=kw.get("label"), args=kw, rows=[])
    TAB = np.array([text("+", a, b) for a, b in shown["+"]], np.int32); it = 0
    while it < n:
        nxt = min((it // every + 1) * every, n) - 1; X = TAB[rng.integers(8000, size=nxt - it + 1)]
        xs = X.ravel(); ys = np.concatenate([X[:, 1:], np.full((len(X), 1), -1, np.int32)], 1).ravel(); it = nxt + 1
        if fast: import fastj; fastj.run(s, xs, ys)
        else: s.steps(xs, ys)
        if (nxt + 1) % every == 0: out["rows"].append(dict(lines=nxt + 1, add=s.score(held["+"])))
    out["last3"] = float(np.mean([r["add"] for r in out["rows"][-3:]])) if out["rows"] else None
    out["finite"] = bool(np.isfinite(s.A).all() and np.isfinite(s.B).all() and np.isfinite(s.W1).all())
    out["m"] = measures(s, held["+"]); out["cpu"] = round(time.process_time() - t0, 1); run.last = s
    if not kw.get("nowrite"):
        os.makedirs(os.path.join(HERE, "res"), exist_ok=True); json.dump(out, open(os.path.join(HERE, "res", f"j_{kw['label']}.json"), "w"))
    return out

if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[2:]); kw["label"] = sys.argv[1]; o = run(kw); m = o["m"]
    print(o["label"], "last3", o["last3"], "| " + " ".join(f"{r['lines']//1000}k {r['add']:.0f}" for r in o["rows"]), f"| end {m['end']:.1f} small {100*m['share_below_hundredth']:.1f}% without {m['end_without_those']:.1f}",
          "| weakest 25/50/75/90 gone:", *(f"{m[f'end_without_weakest_{q}']:.1f}" for q in (25, 50, 75, 90)), f"| cos {m['cos_R_w']:.3f} B {m['B']}", m.get("share_sender_part_smaller", ""), o["cpu"], "s")
