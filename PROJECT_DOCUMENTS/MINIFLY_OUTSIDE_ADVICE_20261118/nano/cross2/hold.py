#!/usr/bin/env python3
"""PLAN_5.md: cells that hold bytes themselves; no delay chain; a trace at each connection, running forward. numpy only.
    python3 hold.py <label> text=d1|d2 seed=1 [H=256 K=16 eta1=1/4 lines=60000 every=10000]      eta1=0: the cells never change; save=1: the strengths are saved
    chain=C (PLAN_9.md): the C bytes before the present one are on lines as well (a chain of C delay cells; the earlier plans' full chain had nine).
    fast=1: the inner loop runs compiled (kern.c through fastk.py; the same arithmetic, to rounding).
PLAN_6.md: [cross=1] a cell that reads another passes on to it what it received, times its own sensitivity, over a return strength; the cell read
    changes by what its readers send times each trace as it stood one byte earlier. [ret=exact|learned] the return strength is the connection's
    forward strength, or a separate strength that receives the same changes and fades with it ([lamB=1e-5], the fading of both, a byte).
    ret=grown: a separate return strength that starts at nothing and receives the forward strength's changes; no fading unless lamB is given.
    ret=sign: the return strength has the sign of the forward strength and one fixed size (the sign-symmetric form of the record's sweep).
    ret=local: the return strength is kept by the cell that is read. Every reader sends the same two numbers to all the cells it reads (what it received
      times each of its two sensitivities); the cell read weighs them by its return strengths, which start at nothing and change by those numbers
      times its own output one byte earlier. Nothing of the reader's strengths or traces is used.
    ret=local2: as local, and every reader also sends its gate and its two sensitivities at every byte, the same to all the cells it reads; the cell
      read keeps from them, and from its own earlier outputs, a trace for each reader (it equals the reader's own trace of that connection), and its
      return strengths change by what the reader received times that trace. No value differs from one cell read to another.
nano/cross2 (this copy): [cross=2] (ret=exact only) the second crossing, still forward in time. A cell j that was read passes on, to each cell i it
    reads, what reached it over the first crossing, times its own sensitivity to that connection as it was one byte earlier, times its own strength
    from i; cell i changes by what reaches it this way times each of its traces as it stood two bytes earlier. For this a cell keeps, beside each
    trace, the trace's value of one byte before (so that at the moment of a change it has the trace as it now is, as it stood one byte earlier and
    as it stood two bytes earlier), and its own two sensitivities (to its candidate and to its gate) of one byte earlier. No strength of another
    cell is used. fast=1 runs kern5.c through fastk5.py for every setting.
PLAN_11.md: text=g2 [gap=8]: two-digit lines aa+<0 to gap letters>bb=ccc; never-shown pairs scored at each length of gap.
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
    if tx in ("g1", "m1", "g2") and len(SYMS) == 14: SYMS = SYMS + list(LET); S = len(SYMS); SI = {c: i for i, c in enumerate(SYMS)}
def fill(a, b, n): return bytes(LET[(7 * a + 3 * b + 5 * i * i + i) % 4] for i in range(n)).decode()          # the letters used when scoring: fixed by the pair
def text(tx, a, b, f=""):
    if tx == "d1": t = f"{a}+{b}={a + b}\n"
    elif tx == "d2": t = f"{a:02d}+{b:02d}={a + b:03d}\n"
    elif tx == "g2": t = f"{a:02d}+{f}{b:02d}={a + b:03d}\n"
    else: t = f"{a}+{f}{b}={(a + b) % 10 if tx == 'm1' else a + b}\n"
    return [SI[c] for c in t.encode()]
def sg(u): return 1 / (1 + np.exp(-u))

class Hold:
    def __init__(s, seed, H, K, eta1, eta2=1 / 16, lam=1e-5, cross=0, ret="exact", lamB=None, chain=0):
        g = np.random.default_rng([seed, 20261122]); s.H, s.eta1, s.eta2, s.lam = H, eta1, eta2, lam
        s.idx = np.stack([g.choice(np.delete(np.arange(H), j), K, replace=False) for j in range(H)])      # the K other cells that each cell reads
        s.C = chain; NI = S * (1 + chain); s.prev = [END] * chain                                         # chain: the bytes just before the present one are on lines too
        s.Ac = g.uniform(-1, 1, (H, NI)); s.Az = g.uniform(-1, 1, (H, NI)); s.Bc = g.uniform(-1, 1, (H, K)) / np.sqrt(K); s.Bz = g.uniform(-1, 1, (H, K)) / np.sqrt(K)
        s.cc = np.zeros(H); s.cz = g.uniform(-1, 3, H)                                                    # gates' thresholds spread: cells hold for different times
        s.R = g.uniform(-1, 1, (H, S)); s.W2 = np.zeros((S, H)); s.b2 = np.zeros(S); s.h = np.zeros(H)
        s.Ecx = np.zeros((H, NI)); s.Ezx = np.zeros((H, NI)); s.Ecr = np.zeros((H, K)); s.Ezr = np.zeros((H, K)); s.Ecb = np.zeros(H); s.Ezb = np.zeros(H)
        s.fast = 0; s.bx = []; s.by = []; s.right_acc = 0
        s.ret = ret; s.qs = 1 / np.sqrt(3 * K)                                                            # ret=sign: the size of every return strength (the birth strengths' root mean square)
        s.cross, s.learned = cross, bool(cross) and ret in ("learned", "grown", "local", "local2"); s.lamB = (1e-5 if ret == "learned" and cross else 0.0) if lamB is None else lamB
        if s.learned: g3 = np.random.default_rng([seed, 20261123]); s.Qc = g3.uniform(-1, 1, (H, K)) / np.sqrt(K); s.Qz = g3.uniform(-1, 1, (H, K)) / np.sqrt(K)
        if s.learned and ret in ("grown", "local", "local2"): s.Qc = np.zeros((H, K)); s.Qz = np.zeros((H, K))      # a return strength starts at nothing and receives the forward strength's changes
        assert cross in (0, 1, 2) and (cross < 2 or ret == "exact"), "two crossings are built for the reader's own strengths only"
        s.Pcx = np.zeros((H, NI)); s.Pzx = np.zeros((H, NI)); s.Pcr = np.zeros((H, K)); s.Pzr = np.zeros((H, K)); s.Pcb = np.zeros(H); s.Pzb = np.zeros(H)      # cross=2: every trace as it stood one byte before the present value
        s.kc1 = np.zeros(H); s.kz1 = np.zeros(H)                                                          # cross=2: the cell's own two sensitivities of one byte earlier
    def cols(s, x, prev): return [x] + [S * (d + 1) + prev[d] for d in range(s.C)] if s.C else x
    def step(s, h, x):
        """x: the line of the present byte, or with a chain the list of the active lines."""
        pre = h[s.idx]
        if s.C: z = sg(s.Az[:, x].sum(1) + (s.Bz * pre).sum(1) + s.cz); c = sg(s.Ac[:, x].sum(1) + (s.Bc * pre).sum(1) + s.cc)
        else: z = sg(s.Az[:, x] + (s.Bz * pre).sum(1) + s.cz); c = sg(s.Ac[:, x] + (s.Bc * pre).sum(1) + s.cc)
        return pre, z, c, z * h + (1 - z) * c
    def read(s, x, y=None):
        """The byte x is read and every trace moves on; if a next byte y is given, the system names it first, y arrives, and the cells change."""
        if s.C: x0 = x; x = s.cols(x, s.prev); s.prev = [x0] + s.prev[:-1]
        hp = s.h; pre, z, c, h = s.step(hp, x); s.h = h
        kc = (1 - z) * c * (1 - c); kz = (hp - c) * z * (1 - z); zc = z[:, None]; X = None
        ka, kb = (kc[:, None], kz[:, None]) if s.C else (kc, kz)
        if y is not None and s.eta1 and s.cross:          # the crossing: every reader sends back what it received times its sensitivity to that connection
            r = s.eta1 * (s.R @ (np.eye(S)[y] - s.W2 @ h - s.b2)); Qc, Qz = (s.Qc, s.Qz) if s.learned else (np.sign(s.Bc) * s.qs, np.sign(s.Bz) * s.qs) if s.ret == "sign" else (s.Bc, s.Bz)
            rho = np.bincount(s.idx.ravel(), weights=(r[:, None] * (kc[:, None] * Qc + kz[:, None] * Qz)).ravel(), minlength=s.H); pc = rho[:, None]
            X = (pc * s.Ecx, pc * s.Ecr, rho * s.Ecb, pc * s.Ezx, pc * s.Ezr, rho * s.Ezb)                 # with the traces as they stood one byte earlier
            if s.cross == 2:                              # the second crossing: a cell that was read passes on what reached it, times its sensitivity of one byte earlier, times its own strength
                rho2 = np.bincount(s.idx.ravel(), weights=(rho[:, None] * (s.kc1[:, None] * s.Bc + s.kz1[:, None] * s.Bz)).ravel(), minlength=s.H); p2 = rho2[:, None]
                X = tuple(a + b for a, b in zip(X, (p2 * s.Pcx, p2 * s.Pcr, rho2 * s.Pcb, p2 * s.Pzx, p2 * s.Pzr, rho2 * s.Pzb)))      # with the traces as they stood two bytes earlier
        if s.cross == 2:
            s.Pcx[...] = s.Ecx; s.Pcr[...] = s.Ecr; s.Pcb[...] = s.Ecb; s.Pzx[...] = s.Ezx; s.Pzr[...] = s.Ezr; s.Pzb[...] = s.Ezb; s.kc1 = kc; s.kz1 = kz
        s.Ecx *= zc; s.Ecx[:, x] += ka; s.Ecr = zc * s.Ecr + kc[:, None] * pre; s.Ecb = z * s.Ecb + kc
        s.Ezx *= zc; s.Ezx[:, x] += kb; s.Ezr = zc * s.Ezr + kz[:, None] * pre; s.Ezb = z * s.Ezb + kz
        if y is None: return None
        a = s.W2 @ h + s.b2; ok = int(a.argmax()) == y; m = -a; m[y] += 1; g = s.eta2 / (h @ h + 1)
        if s.eta1:
            r = s.eta1 * (s.R @ m); rc = r[:, None]; Dl = [rc * s.Ecx, rc * s.Ecr, r * s.Ecb, rc * s.Ezx, rc * s.Ezr, r * s.Ezb]
            D1, D4 = Dl[1], Dl[4]
            if X is not None: Dl = [d + e for d, e in zip(Dl, X)]
            s.Ac += Dl[0]; s.Bc += Dl[1]; s.cc += Dl[2]; s.Az += Dl[3]; s.Bz += Dl[4]; s.cz += Dl[5]
            if s.ret == "local": s.Qc += (r * kc)[:, None] * pre; s.Qz += (r * kz)[:, None] * pre         # at the cell read: what the reader sent out times that cell's own earlier output
            elif s.ret == "local2": s.Qc += D1; s.Qz += D4                                              # at the cell read: what the reader received times a trace kept there (see the heading)
            elif s.learned: s.Qc += Dl[1]; s.Qz += Dl[4]
            if s.learned: s.Qc *= 1 - s.lamB; s.Qz *= 1 - s.lamB
            if s.lamB: s.Bc *= 1 - s.lamB; s.Bz *= 1 - s.lamB
        d2 = np.outer(g * m, h); s.W2 += d2; s.b2 += g * m; s.R += d2.T; s.W2 *= 1 - s.lam; s.R *= 1 - s.lam
        return ok
    def flush(s):
        if not len(s.bx): return
        import fastk5; fastk5.hold_run(s)
    def line(s, L):
        if s.fast:
            s.bx += L; s.by += L[1:]; s.by.append(-1)
            if len(s.bx) >= 500000: s.flush()
            return 0
        right = sum(s.read(L[t], L[t + 1]) for t in range(len(L) - 1)); s.read(L[-1]); return right
    def score(s, tx, pairs, gap=0):
        n = 0; pl = np.zeros(3); s.flush()
        for a, b in pairs:
            L = text(tx, a, b, fill(a, b, gap)); eq = L.index(EQ); h = s.h; ok = True; prev = list(s.prev)                 # from the state left by the last line read
            for t in range(len(L) - 1):
                h = s.step(h, s.cols(L[t], prev))[3]
                if s.C: prev = [L[t]] + prev[:-1]
                if t >= eq and L[t + 1] != END:
                    r = int((s.W2 @ h + s.b2).argmax()) == L[t + 1]; ok = ok and r; pl[t - eq] += r
            n += ok
        return 100 * n / len(pairs), [round(100 * x / len(pairs)) for x in pl]

def run(kw, cls=None):
    t0 = time.process_time(); tx = kw.get("text", "d1"); seed = int(kw.get("seed", 1)); lines = int(kw.get("lines", 60000)); every = int(kw.get("every", 10000))
    setup(tx); G = int(kw.get("gap", 4)) if tx in ("g1", "m1", "g2") else 0; gaps = sorted({0, G // 2, G})
    if tx in ("d1", "g1", "m1"):
        rng = np.random.default_rng(seed); P = [(a, b) for a in range(10) for b in range(10)]; rng.shuffle(P); held, shown = P[:20], P[20:]; shown_s = shown
    else:
        rng = np.random.default_rng([seed, 99]); o = rng.permutation(10000); P = [(int(q) // 100, int(q) % 100) for q in o]; held, shown = P[:500], P[2000:]; shown_s = shown[:500]
    s = (cls or Hold)(seed, int(kw.get("H", 256)), int(kw.get("K", 16)), float(eval(kw.get("eta1", "1/4"))), cross=int(kw.get("cross", 0)), ret=kw.get("ret", "exact"), chain=int(kw.get("chain", 0)),
                      lamB=float(kw["lamB"]) if "lamB" in kw else None)
    s.X = kw.get("X"); s.fast = int(kw.get("fast", 0)); out = dict(label=kw["label"], args=kw, rows=[]); right = n = 0
    for it in range(lines):
        a, b = shown[rng.integers(len(shown))]
        L = text(tx, a, b, bytes(LET[i] for i in rng.integers(4, size=rng.integers(G + 1))).decode()) if G else text(tx, a, b)
        right += s.line(L); n += len(L) - 1
        if (it + 1) % every == 0:
            s.flush(); right += s.right_acc; s.right_acc = 0; sh, _ = s.score(tx, shown_s); nv, pl = s.score(tx, held)
            out["rows"].append(dict(lines=it + 1, shown=sh, never=nv, place=pl, bytes_right=100 * right / n, gate_mean=float(sg(s.cz).mean()))); right = n = 0
            if G: out["rows"][-1]["shown_by_gap"] = {g: s.score(tx, shown_s, g)[0] for g in gaps}; out["rows"][-1]["shown"] = float(np.mean(list(out["rows"][-1]["shown_by_gap"].values())))
            if G and tx == "g2": out["rows"][-1]["never_by_gap"] = {g: s.score(tx, held, g)[0] for g in gaps}; out["rows"][-1]["never"] = float(np.mean(list(out["rows"][-1]["never_by_gap"].values())))
    out["cpu"] = round(time.process_time() - t0); out["last3_shown"] = float(np.mean([r["shown"] for r in out["rows"][-3:]])); out["last3_never"] = float(np.mean([r["never"] for r in out["rows"][-3:]]))
    s.flush(); run.last = s
    if kw.get("nowrite"): return out
    os.makedirs(os.path.join(HERE, "res"), exist_ok=True); json.dump(out, open(os.path.join(HERE, "res", f"hold_{kw['label']}.json"), "w"))
    if kw.get("save"): np.savez_compressed(os.path.join(HERE, "res", f"hold_{kw['label']}.npz"), **{k: getattr(s, k).astype(np.float32) if getattr(s, k).dtype == np.float64 else getattr(s, k) for k in ("idx", "Ac", "Az", "Bc", "Bz", "cc", "cz", "W2", "b2", "R", "h")})
    return out

if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[2:]); kw["label"] = sys.argv[1]; o = run(kw)
    print(o["label"], f"last3 shown {o['last3_shown']:.1f} never {o['last3_never']:.1f} |", " ".join(f"{r['lines']//1000}k {r['shown']:.0f}/{r['never']:.0f}" for r in o["rows"]), "| place", o["rows"][-1]["place"], "| bytes", round(o["rows"][-1]["bytes_right"]),
          "| by gap " + " ".join(f"{g}:{v:.0f}" for g, v in o["rows"][-1]["shown_by_gap"].items()) if "shown_by_gap" in o["rows"][-1] else "",
          "| never shown by gap " + " ".join(f"{g}:{v:.0f}" for g, v in o["rows"][-1]["never_by_gap"].items()) if "never_by_gap" in o["rows"][-1] else "", o["cpu"], "s")
