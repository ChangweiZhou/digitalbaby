#!/usr/bin/env python3
"""PLAN_5.md: cells that hold bytes themselves; no delay chain; a trace at each connection, running forward. numpy only.
    python3 hold.py <label> text=d1|d2 seed=1 [H=256 K=16 eta1=1/4 lines=60000 every=10000]      eta1=0: the cells never change
"""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
SYMS = list(b"0123456789+=\n-"); S = len(SYMS); SI = {c: i for i, c in enumerate(SYMS)}; END = SI[10]; EQ = SI[ord("=")]

def text(tx, a, b): return [SI[c] for c in (f"{a}+{b}={a + b}\n" if tx == "d1" else f"{a:02d}+{b:02d}={a + b:03d}\n").encode()]
def sg(u): return 1 / (1 + np.exp(-u))

class Hold:
    def __init__(s, seed, H, K, eta1, eta2=1 / 16, lam=1e-5):
        g = np.random.default_rng([seed, 20261122]); s.H, s.eta1, s.eta2, s.lam = H, eta1, eta2, lam
        s.idx = np.stack([g.choice(np.delete(np.arange(H), j), K, replace=False) for j in range(H)])      # the K other cells that each cell reads
        s.Ac = g.uniform(-1, 1, (H, S)); s.Az = g.uniform(-1, 1, (H, S)); s.Bc = g.uniform(-1, 1, (H, K)) / np.sqrt(K); s.Bz = g.uniform(-1, 1, (H, K)) / np.sqrt(K)
        s.cc = np.zeros(H); s.cz = g.uniform(-1, 3, H)                                                    # gates' thresholds spread: cells hold for different times
        s.R = g.uniform(-1, 1, (H, S)); s.W2 = np.zeros((S, H)); s.b2 = np.zeros(S); s.h = np.zeros(H)
        s.Ecx = np.zeros((H, S)); s.Ezx = np.zeros((H, S)); s.Ecr = np.zeros((H, K)); s.Ezr = np.zeros((H, K)); s.Ecb = np.zeros(H); s.Ezb = np.zeros(H)
    def step(s, h, x):
        pre = h[s.idx]; z = sg(s.Az[:, x] + (s.Bz * pre).sum(1) + s.cz); c = sg(s.Ac[:, x] + (s.Bc * pre).sum(1) + s.cc)
        return pre, z, c, z * h + (1 - z) * c
    def read(s, x, y=None):
        """The byte x is read and every trace moves on; if a next byte y is given, the system names it first, y arrives, and the cells change."""
        hp = s.h; pre, z, c, h = s.step(hp, x); s.h = h
        kc = (1 - z) * c * (1 - c); kz = (hp - c) * z * (1 - z); zc = z[:, None]
        s.Ecx *= zc; s.Ecx[:, x] += kc; s.Ecr = zc * s.Ecr + kc[:, None] * pre; s.Ecb = z * s.Ecb + kc
        s.Ezx *= zc; s.Ezx[:, x] += kz; s.Ezr = zc * s.Ezr + kz[:, None] * pre; s.Ezb = z * s.Ezb + kz
        if y is None: return None
        a = s.W2 @ h + s.b2; ok = int(a.argmax()) == y; m = -a; m[y] += 1; g = s.eta2 / (h @ h + 1)
        if s.eta1:
            r = s.eta1 * (s.R @ m); rc = r[:, None]
            s.Ac += rc * s.Ecx; s.Bc += rc * s.Ecr; s.cc += r * s.Ecb; s.Az += rc * s.Ezx; s.Bz += rc * s.Ezr; s.cz += r * s.Ezb
        d2 = np.outer(g * m, h); s.W2 += d2; s.b2 += g * m; s.R += d2.T; s.W2 *= 1 - s.lam; s.R *= 1 - s.lam
        return ok
    def score(s, tx, pairs):
        n = 0; pl = np.zeros(3)
        for a, b in pairs:
            L = text(tx, a, b); eq = L.index(EQ); h = s.h; ok = True                                      # from the state left by the last line read
            for t in range(len(L) - 1):
                h = s.step(h, L[t])[3]
                if t >= eq and L[t + 1] != END:
                    r = int((s.W2 @ h + s.b2).argmax()) == L[t + 1]; ok = ok and r; pl[t - eq] += r
            n += ok
        return 100 * n / len(pairs), [round(100 * x / len(pairs)) for x in pl]

def run(kw):
    t0 = time.process_time(); tx = kw.get("text", "d1"); seed = int(kw.get("seed", 1)); lines = int(kw.get("lines", 60000)); every = int(kw.get("every", 10000))
    if tx == "d1":
        rng = np.random.default_rng(seed); P = [(a, b) for a in range(10) for b in range(10)]; rng.shuffle(P); held, shown = P[:20], P[20:]; shown_s = shown
    else:
        rng = np.random.default_rng([seed, 99]); o = rng.permutation(10000); P = [(int(q) // 100, int(q) % 100) for q in o]; held, shown = P[:500], P[2000:]; shown_s = shown[:500]
    s = Hold(seed, int(kw.get("H", 256)), int(kw.get("K", 16)), float(eval(kw.get("eta1", "1/4")))); out = dict(label=kw["label"], args=kw, rows=[]); right = n = 0
    for it in range(lines):
        a, b = shown[rng.integers(len(shown))]; L = text(tx, a, b)
        for t in range(len(L) - 1): right += s.read(L[t], L[t + 1]); n += 1
        s.read(L[-1])
        if (it + 1) % every == 0:
            sh, _ = s.score(tx, shown_s); nv, pl = s.score(tx, held)
            out["rows"].append(dict(lines=it + 1, shown=sh, never=nv, place=pl, bytes_right=100 * right / n, gate_mean=float(sg(s.cz).mean()))); right = n = 0
    out["cpu"] = round(time.process_time() - t0); out["last3_shown"] = float(np.mean([r["shown"] for r in out["rows"][-3:]])); out["last3_never"] = float(np.mean([r["never"] for r in out["rows"][-3:]]))
    os.makedirs(os.path.join(HERE, "res"), exist_ok=True); json.dump(out, open(os.path.join(HERE, "res", f"hold_{kw['label']}.json"), "w")); return out

if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[2:]); kw["label"] = sys.argv[1]; o = run(kw)
    print(o["label"], f"last3 shown {o['last3_shown']:.1f} never {o['last3_never']:.1f} |", " ".join(f"{r['lines']//1000}k {r['shown']:.0f}/{r['never']:.0f}" for r in o["rows"]), "| place", o["rows"][-1]["place"], "| bytes", round(o["rows"][-1]["bytes_right"]), o["cpu"], "s")
