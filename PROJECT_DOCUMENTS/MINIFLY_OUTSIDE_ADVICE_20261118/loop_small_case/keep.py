#!/usr/bin/env python3
"""PLAN_3.md: keeping. The system of cells_loop.py (text d2), with a second task and three arms. numpy only.
    python3 keep.py <arm> seed=.. [k=1] [l1=400000 l2=200000]      arms: plain stiffen rehearse mixed dreamN (in the second phase a line of its own before a real line with chance 1/N)
"""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
SYMS = list(b"0123456789+=\n-"); S = len(SYMS); SI = {c: i for i, c in enumerate(SYMS)}; END = SI[10]; D = 10; NL = D * S; OFF = np.arange(D) * S

def text(op, a, b):
    if op == "-" and a < b: a, b = b, a
    return [SI[c] for c in f"{a:02d}{op}{b:02d}={(a + b if op == '+' else a - b):03d}\n".encode()]

class Sys:
    def __init__(s, seed, H=256, eta1=.25, eta2=1 / 16, lam=1e-5, k=0.0):
        wr = np.random.default_rng([seed, 20261119]); s.eta1, s.eta2, s.lam, s.k = eta1, eta2, lam, k
        s.W1 = wr.uniform(-.5, .5, (H, NL)); s.b1 = np.zeros(H); s.R = wr.uniform(-1, 1, (H, S)); s.W2 = np.zeros((S, H)); s.b2 = np.zeros(S)
        s.sm = 0; s.I1 = np.zeros((H, NL)); s.I2 = np.zeros((S, H)); s.hist = [END] * D
    def fwd(s, hist):
        act = OFF + np.asarray(hist); h = 1 / (1 + np.exp(-(s.W1[:, act].sum(1) + s.b1))); return act, h, s.W2 @ h + s.b2
    def learn(s, y):
        act, h, a = s.fwd(s.hist); m = -a; m[y] += 1
        if s.sm:                                  # only the branch of the byte that arrived and the branch wrongly named change
            e = int(a.argmax()); keep = np.zeros(S, bool); keep[y] = keep[e] = True; m = m * keep
        g = s.eta2 / (h @ h + 1)
        d = s.eta1 * (s.R @ m) * h * (1 - h); d1 = np.repeat(d[:, None], D, 1); d2 = np.outer(g * m, h)
        if s.k:                                   # each strength divides its step by 1 + (total size of its own past changes) / k
            d1 = d1 / (1 + s.I1[:, act] / s.k); s.I1[:, act] += np.abs(d1)
            d2s = d2 / (1 + s.I2 / s.k); s.I2 += np.abs(d2s); s.R += d2s.T; d2 = d2s
        else: s.R += d2.T
        s.W1[:, act] += d1; s.b1 += d; s.W2 += d2; s.b2 += g * m
        s.W2 *= 1 - s.lam; s.R *= 1 - s.lam
    def line(s, L):
        for t in range(len(L) - 1):
            s.hist.insert(0, L[t]); s.hist.pop(); s.learn(L[t + 1])
        s.hist.insert(0, L[-1]); s.hist.pop()
    def dream(s, rng):
        """The system writes a line itself: at each byte it names the next (its largest branch when that is at least 1/2, otherwise a
        byte drawn in proportion to its positive branches), takes the byte it named as the byte that arrived, and learns from it."""
        for _ in range(12):
            a = s.fwd(s.hist)[2]; q = np.clip(a, 0, None)
            y = int(a.argmax()) if q.sum() <= 0 else int(rng.choice(S, p=q ** 4 / (q ** 4).sum()))
            s.learn(y); s.hist.insert(0, y); s.hist.pop()
            if y == END: break
    def score(s, op, pairs):
        n = 0
        for a, b in pairs:
            L = text(op, a, b); hist = [END] * D; ok = True
            for t in range(len(L) - 1):
                hist.insert(0, L[t]); hist.pop()
                if t >= 5 and L[t + 1] != END: ok = ok and int(s.fwd(hist)[2].argmax()) == L[t + 1]
            n += ok
        return 100 * n / len(pairs)

def run(arm, seed, k=1.0, l1=400000, l2=200000):
    t0 = time.process_time(); rng = np.random.default_rng([seed, 99]); held, shown = {}, {}
    for op in "+-":
        o = rng.permutation(10000); P = [(int(q) // 100, int(q) % 100) for q in o]; held[op] = P[:500]; shown[op] = P[2000:]
    s = Sys(seed, k=k if arm == "stiffen" else 0.0); dr = np.random.default_rng([seed, 7]); s.sm = int(arm.endswith('S')); arm = arm.rstrip('S'); out_arm = arm + ('S' if s.sm else ''); out = dict(arm=out_arm, seed=seed, k=k, rows=[])
    def phase(n, pick, name):
        for it in range(n):
            if name == "subtraction" and arm.startswith("dream") and dr.integers(int(arm[5:])) == 0: s.dream(dr)
            op = pick(); a, b = shown[op][rng.integers(8000)]; s.line(text(op, a, b))
            if (it + 1) % 100000 == 0: out["rows"].append(dict(phase=name, lines=it + 1, add=s.score("+", held["+"]), sub=s.score("-", held["-"])))
    if arm == "mixed": phase(l1 + l2, lambda: "+-"[rng.integers(2)], "mixed")
    else:
        phase(l1, lambda: "+", "addition")
        phase(l2, (lambda: "+" if rng.integers(8) == 0 else "-") if arm == "rehearse" else (lambda: "-"), "subtraction")
    out["cpu"] = round(time.process_time() - t0); os.makedirs(os.path.join(HERE, "res"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "res", f"keep_{out_arm}_k{k}_{seed}.json"), "w")); return out

if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[2:]); o = run(sys.argv[1], int(kw.get("seed", 1)), float(kw.get("k", 1)), int(kw.get("l1", 400000)), int(kw.get("l2", 200000)))
    print(o["arm"], o["seed"], o["k"], " | ".join(f"{r['phase'][:3]} {r['lines']//1000}k add {r['add']:.0f} sub {r['sub']:.0f}" for r in o["rows"]))
