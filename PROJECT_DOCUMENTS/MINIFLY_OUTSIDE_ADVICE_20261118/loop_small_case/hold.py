#!/usr/bin/env python3
"""PLAN_5.md: cells that hold bytes themselves; no delay chain; a trace at each connection, running forward. numpy only.
    python3 hold.py <label> text=d1|d2 seed=1 [H=256 K=16 eta1=1/4 lines=60000 every=10000]      eta1=0: the cells never change; save=1: the strengths are saved
PLAN_6.md: [cross=1] a cell that reads another passes on to it what it received, times its own sensitivity, over a return strength; the cell read
    changes by what its readers send times each trace as it stood one byte earlier. [ret=exact|learned] the return strength is the connection's
    forward strength, or a separate strength that receives the same changes and fades with it ([lamB=1e-5], the fading of both, a byte).
    ret=grown: a separate return strength that starts at nothing and receives the forward strength's changes; no fading unless lamB is given.
    ret=sign: the return strength has the sign of the forward strength and one fixed size (the sign-symmetric form of the record's sweep).
    ret=local: the return strength is kept by the cell that is read. Every reader sends the same two numbers to all the cells it reads (what it received
      times each of its two sensitivities); the cell read weighs them by its return strengths, which start at nothing and change by those numbers
      times its own output one byte earlier. Nothing of the reader's strengths or traces is used.
PLAN_7.md: text=g1|m1 [gap=4]: single-digit lines a+<0 to gap letters>b=c; g1: c = a + b; m1: c = (a + b) mod 10. Scored on the sums shown at each length of gap.
"""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
SYMS = list(b"0123456789+=\n-"); S = len(SYMS); SI = {c: i for i, c in enumerate(SYMS)}; END = SI[10]; EQ = SI[ord("=")]

LET = b"wxyz"
def setup(tx):
    """The texts with a gap have four more byte values, the letters."""
    global SYMS, S, SI
    if tx in ("g1", "m1") and len(SYMS) == 14: SYMS = SYMS + list(LET); S = len(SYMS); SI = {c: i for i, c in enumerate(SYMS)}
def fill(a, b, n): return bytes(LET[(7 * a + 3 * b + 5 * i * i + i) % 4] for i in range(n)).decode()          # the letters used when scoring: fixed by the pair
def text(tx, a, b, f=""):
    if tx == "d1": t = f"{a}+{b}={a + b}\n"
    elif tx == "d2": t = f"{a:02d}+{b:02d}={a + b:03d}\n"
    else: t = f"{a}+{f}{b}={(a + b) % 10 if tx == 'm1' else a + b}\n"
    return [SI[c] for c in t.encode()]
def sg(u): return 1 / (1 + np.exp(-u))

class Hold:
    def __init__(s, seed, H, K, eta1, eta2=1 / 16, lam=1e-5, cross=0, ret="exact", lamB=None):
        g = np.random.default_rng([seed, 20261122]); s.H, s.eta1, s.eta2, s.lam = H, eta1, eta2, lam
        s.idx = np.stack([g.choice(np.delete(np.arange(H), j), K, replace=False) for j in range(H)])      # the K other cells that each cell reads
        s.Ac = g.uniform(-1, 1, (H, S)); s.Az = g.uniform(-1, 1, (H, S)); s.Bc = g.uniform(-1, 1, (H, K)) / np.sqrt(K); s.Bz = g.uniform(-1, 1, (H, K)) / np.sqrt(K)
        s.cc = np.zeros(H); s.cz = g.uniform(-1, 3, H)                                                    # gates' thresholds spread: cells hold for different times
        s.R = g.uniform(-1, 1, (H, S)); s.W2 = np.zeros((S, H)); s.b2 = np.zeros(S); s.h = np.zeros(H)
        s.Ecx = np.zeros((H, S)); s.Ezx = np.zeros((H, S)); s.Ecr = np.zeros((H, K)); s.Ezr = np.zeros((H, K)); s.Ecb = np.zeros(H); s.Ezb = np.zeros(H)
        s.ret = ret; s.qs = 1 / np.sqrt(3 * K)                                                            # ret=sign: the size of every return strength (the birth strengths' root mean square)
        s.cross, s.learned = cross, bool(cross) and ret in ("learned", "grown", "local"); s.lamB = (1e-5 if ret == "learned" and cross else 0.0) if lamB is None else lamB
        if s.learned: g3 = np.random.default_rng([seed, 20261123]); s.Qc = g3.uniform(-1, 1, (H, K)) / np.sqrt(K); s.Qz = g3.uniform(-1, 1, (H, K)) / np.sqrt(K)
        if s.learned and ret in ("grown", "local"): s.Qc = np.zeros((H, K)); s.Qz = np.zeros((H, K))      # a return strength starts at nothing and receives the forward strength's changes
    def step(s, h, x):
        pre = h[s.idx]; z = sg(s.Az[:, x] + (s.Bz * pre).sum(1) + s.cz); c = sg(s.Ac[:, x] + (s.Bc * pre).sum(1) + s.cc)
        return pre, z, c, z * h + (1 - z) * c
    def read(s, x, y=None):
        """The byte x is read and every trace moves on; if a next byte y is given, the system names it first, y arrives, and the cells change."""
        hp = s.h; pre, z, c, h = s.step(hp, x); s.h = h
        kc = (1 - z) * c * (1 - c); kz = (hp - c) * z * (1 - z); zc = z[:, None]; X = None
        if y is not None and s.eta1 and s.cross:          # the crossing: every reader sends back what it received times its sensitivity to that connection
            r = s.eta1 * (s.R @ (np.eye(S)[y] - s.W2 @ h - s.b2)); Qc, Qz = (s.Qc, s.Qz) if s.learned else (np.sign(s.Bc) * s.qs, np.sign(s.Bz) * s.qs) if s.ret == "sign" else (s.Bc, s.Bz)
            rho = np.bincount(s.idx.ravel(), weights=(r[:, None] * (kc[:, None] * Qc + kz[:, None] * Qz)).ravel(), minlength=s.H); pc = rho[:, None]
            X = (pc * s.Ecx, pc * s.Ecr, rho * s.Ecb, pc * s.Ezx, pc * s.Ezr, rho * s.Ezb)                 # with the traces as they stood one byte earlier
        s.Ecx *= zc; s.Ecx[:, x] += kc; s.Ecr = zc * s.Ecr + kc[:, None] * pre; s.Ecb = z * s.Ecb + kc
        s.Ezx *= zc; s.Ezx[:, x] += kz; s.Ezr = zc * s.Ezr + kz[:, None] * pre; s.Ezb = z * s.Ezb + kz
        if y is None: return None
        a = s.W2 @ h + s.b2; ok = int(a.argmax()) == y; m = -a; m[y] += 1; g = s.eta2 / (h @ h + 1)
        if s.eta1:
            r = s.eta1 * (s.R @ m); rc = r[:, None]; Dl = [rc * s.Ecx, rc * s.Ecr, r * s.Ecb, rc * s.Ezx, rc * s.Ezr, r * s.Ezb]
            if X is not None: Dl = [d + e for d, e in zip(Dl, X)]
            s.Ac += Dl[0]; s.Bc += Dl[1]; s.cc += Dl[2]; s.Az += Dl[3]; s.Bz += Dl[4]; s.cz += Dl[5]
            if s.ret == "local": s.Qc += (r * kc)[:, None] * pre; s.Qz += (r * kz)[:, None] * pre         # at the cell read: what the reader sent out times that cell's own earlier output
            elif s.learned: s.Qc += Dl[1]; s.Qz += Dl[4]
            if s.learned: s.Qc *= 1 - s.lamB; s.Qz *= 1 - s.lamB
            if s.lamB: s.Bc *= 1 - s.lamB; s.Bz *= 1 - s.lamB
        d2 = np.outer(g * m, h); s.W2 += d2; s.b2 += g * m; s.R += d2.T; s.W2 *= 1 - s.lam; s.R *= 1 - s.lam
        return ok
    def line(s, L):
        right = sum(s.read(L[t], L[t + 1]) for t in range(len(L) - 1)); s.read(L[-1]); return right
    def score(s, tx, pairs, gap=0):
        n = 0; pl = np.zeros(3)
        for a, b in pairs:
            L = text(tx, a, b, fill(a, b, gap)); eq = L.index(EQ); h = s.h; ok = True                                      # from the state left by the last line read
            for t in range(len(L) - 1):
                h = s.step(h, L[t])[3]
                if t >= eq and L[t + 1] != END:
                    r = int((s.W2 @ h + s.b2).argmax()) == L[t + 1]; ok = ok and r; pl[t - eq] += r
            n += ok
        return 100 * n / len(pairs), [round(100 * x / len(pairs)) for x in pl]

def run(kw, cls=None):
    t0 = time.process_time(); tx = kw.get("text", "d1"); seed = int(kw.get("seed", 1)); lines = int(kw.get("lines", 60000)); every = int(kw.get("every", 10000))
    setup(tx); G = int(kw.get("gap", 4)) if tx in ("g1", "m1") else 0; gaps = sorted({0, G // 2, G})
    if tx in ("d1", "g1", "m1"):
        rng = np.random.default_rng(seed); P = [(a, b) for a in range(10) for b in range(10)]; rng.shuffle(P); held, shown = P[:20], P[20:]; shown_s = shown
    else:
        rng = np.random.default_rng([seed, 99]); o = rng.permutation(10000); P = [(int(q) // 100, int(q) % 100) for q in o]; held, shown = P[:500], P[2000:]; shown_s = shown[:500]
    s = (cls or Hold)(seed, int(kw.get("H", 256)), int(kw.get("K", 16)), float(eval(kw.get("eta1", "1/4"))), cross=int(kw.get("cross", 0)), ret=kw.get("ret", "exact"),
                      lamB=float(kw["lamB"]) if "lamB" in kw else None)
    s.X = kw.get("X"); out = dict(label=kw["label"], args=kw, rows=[]); right = n = 0
    for it in range(lines):
        a, b = shown[rng.integers(len(shown))]
        L = text(tx, a, b, bytes(LET[i] for i in rng.integers(4, size=rng.integers(G + 1))).decode()) if G else text(tx, a, b)
        right += s.line(L); n += len(L) - 1
        if (it + 1) % every == 0:
            sh, _ = s.score(tx, shown_s); nv, pl = s.score(tx, held)
            out["rows"].append(dict(lines=it + 1, shown=sh, never=nv, place=pl, bytes_right=100 * right / n, gate_mean=float(sg(s.cz).mean()))); right = n = 0
            if G: out["rows"][-1]["shown_by_gap"] = {g: s.score(tx, shown_s, g)[0] for g in gaps}; out["rows"][-1]["shown"] = float(np.mean(list(out["rows"][-1]["shown_by_gap"].values())))
    out["cpu"] = round(time.process_time() - t0); out["last3_shown"] = float(np.mean([r["shown"] for r in out["rows"][-3:]])); out["last3_never"] = float(np.mean([r["never"] for r in out["rows"][-3:]]))
    os.makedirs(os.path.join(HERE, "res"), exist_ok=True); json.dump(out, open(os.path.join(HERE, "res", f"hold_{kw['label']}.json"), "w"))
    if kw.get("save"): np.savez_compressed(os.path.join(HERE, "res", f"hold_{kw['label']}.npz"), **{k: getattr(s, k) for k in ("idx", "Ac", "Az", "Bc", "Bz", "cc", "cz", "W2", "b2", "R", "h")})
    return out

if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[2:]); kw["label"] = sys.argv[1]; o = run(kw)
    print(o["label"], f"last3 shown {o['last3_shown']:.1f} never {o['last3_never']:.1f} |", " ".join(f"{r['lines']//1000}k {r['shown']:.0f}/{r['never']:.0f}" for r in o["rows"]), "| place", o["rows"][-1]["place"], "| bytes", round(o["rows"][-1]["bytes_right"]),
          "| by gap " + " ".join(f"{g}:{v:.0f}" for g, v in o["rows"][-1]["shown_by_gap"].items()) if "shown_by_gap" in o["rows"][-1] else "", o["cpu"], "s")
