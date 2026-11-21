#!/usr/bin/env python3
"""Track M, line `nano/layered`: holding cells that read only lines, ordinary cells that read them at the same byte, the mismatch relayed over
two steps of return strengths. No crossing between holding cells. numpy only; this program is the reference for kernl.c (through fastl.py).
    python3 layered.py <label> text=d1|d2|g1|m1|g2 seed=1 [gap=8] [chain=C] [NA=128 NB=104] [etaA=1 etaB=1/4 nB=8 eta2=1/16 lam=1e-5]
                       [og=1 [ob=1]] [lines=60000 every=10000] [fast=1] [save=1]

Cells A (holding). Each reads the lines of the present byte, and with chain=C the lines of the C bytes before it. Gate z and candidate c are
    logistic functions of its strengths from the active lines plus a threshold; state = z * (state before) + (1 - z) * c. No cell A reads any
    other cell, so the trace that each strength keeps (how that strength has shaped its own cell's present state) is exact without a correction.
    og=1 (the crosswalk's item 5): a second gate o, logistic, from a third set of strengths from the same lines; the cell gives out o * state.
Cells B (ordinary, logistic). Each reads what every cell A gives out at the same byte, and the lines of the present byte.
One answering cell with a branch for each byte value reads the cells B; least squares, as in hold.py.
When the next byte arrives the mismatch goes to the cells B over return strengths R that they hold (R changes by the same product as the
    answering cell's strengths and fades with them). Each cell B changes by that sum times its own sensitivity b (1 - b). The cells B pass on
    what they received, times their sensitivity, to the cells A over a second set of return strengths RB, held by the cells A, which change by
    the same product as the forward strengths WB between the same two ends and fade with them (cells2.py does this for H2). Each cell A changes
    each strength by what reaches it times that strength's trace. What a cell B sends is one number, the same to every cell A; what reaches a
    cell A is weighed by strengths that cell A holds. Nothing passes backward over the text.

Copied from loop_small_case/hold.py at commit 0ba9f241 (branch claude/friendly-darwin-8mmhfm), unchanged in what they compute: the byte values
    and `setup`, `fill`, `text`, `sg`; `score` (the only change: the state stepped is the cells A's, and the byte named is read through the
    cells B by `out`); `run` (the only changes: the system constructed and its options, the name of the result file, and the count of the
    numbers that learn). The lines drawn for a seed are therefore the lines hold.py draws for that seed, and the scoring is hold.py's.
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

class Lay:
    def __init__(s, seed, NA=128, NB=104, etaA=1.0, etaB=.25, nB=8.0, eta2=1 / 16, lam=1e-5, chain=0, og=0, ob=1.0):
        g = np.random.default_rng([seed, 20261124]); s.NA, s.NB, s.etaA, s.etaB, s.nB, s.eta2, s.lam, s.og = NA, NB, etaA, etaB, nB, eta2, lam, og
        s.C = chain; NI = S * (1 + chain); s.prev = [END] * chain                                         # chain: the bytes just before the present one are on lines too
        s.Ac = g.uniform(-1, 1, (NA, NI)); s.Az = g.uniform(-1, 1, (NA, NI)); s.cc = np.zeros(NA); s.cz = g.uniform(-1, 3, NA)      # as hold.py: gates' thresholds spread at birth
        s.WB = g.uniform(-1, 1, (NB, NA)) / np.sqrt(NA / 3); s.V = g.uniform(-.5, .5, (NB, S)); s.bB = np.zeros(NB)                  # cells B: from the cells A, from the lines, threshold
        s.RB = g.uniform(-1, 1, (NA, NB)); s.R = g.uniform(-1, 1, (NB, S)); s.W2 = np.zeros((S, NB)); s.b2 = np.zeros(S)            # return strengths; the answering cell
        s.Ao = g.uniform(-1, 1, (NA, NI)) if og else np.zeros((0, NI)); s.co = np.full(NA if og else 0, float(ob))                 # og: the gate on what a cell A gives out
        s.h = np.zeros(NA); s.Ecx = np.zeros((NA, NI)); s.Ezx = np.zeros((NA, NI)); s.Ecb = np.zeros(NA); s.Ezb = np.zeros(NA)
        s.fast = 0; s.bx = []; s.by = []; s.right_acc = 0
    def count(s):
        """The numbers that learn: forward strengths and thresholds; the return strengths are counted apart."""
        fwd = dict(A_lines=s.Ac.size + s.Az.size + s.Ao.size, A_thresholds=s.cc.size + s.cz.size + s.co.size, B_from_A=s.WB.size, B_lines=s.V.size, B_thresholds=s.bB.size, answering=s.W2.size + s.b2.size)
        return dict(fwd, forward_total=sum(fwd.values()), return_strengths=s.R.size + s.RB.size)
    def cols(s, x, prev): return [x] + [S * (d + 1) + prev[d] for d in range(s.C)]
    def step(s, h, x):
        """x: the list of the active lines (the present byte's first). Returns gate, candidate, new state."""
        z = sg(s.Az[:, x].sum(1) + s.cz); c = sg(s.Ac[:, x].sum(1) + s.cc); return z, c, z * h + (1 - z) * c
    def gives(s, h, x):
        """What the cells A give out: their state, or with og the state times a second gate."""
        if not s.og: return None, h
        o = sg(s.Ao[:, x].sum(1) + s.co); return o, o * h
    def out(s, h, x):
        """From the cells A's states and the active lines: the cells B and the answering cell's branches."""
        o, u = s.gives(h, x); b = sg(s.WB @ u + s.V[:, x[0]] + s.bB); return o, u, b, s.W2 @ b + s.b2
    def read(s, x, y=None):
        """The byte x is read and every trace moves on; if a next byte y is given, the system names it first, y arrives, and the cells change."""
        x0 = x; x = s.cols(x, s.prev)
        if s.C: s.prev = [x0] + s.prev[:-1]
        hp = s.h; z, c, h = s.step(hp, x); s.h = h
        kc = (1 - z) * c * (1 - c); kz = (hp - c) * z * (1 - z); zc = z[:, None]
        s.Ecx *= zc; s.Ecx[:, x] += kc[:, None]; s.Ecb = z * s.Ecb + kc
        s.Ezx *= zc; s.Ezx[:, x] += kz[:, None]; s.Ezb = z * s.Ezb + kz
        if y is None: return None
        o, u, b, a = s.out(h, x); ok = int(a.argmax()) == y; m = -a; m[y] += 1; g = s.eta2 / (b @ b + 1)
        sig = (s.R @ m) * b * (1 - b)                              # what a cell B receives over its return strengths, times its own sensitivity
        if s.etaA:
            rA = s.etaA * (s.RB @ sig)                             # what reaches each cell A: the cells B's signals over the return strengths the cell A holds
            ra = rA * o if s.og else rA; rc = ra[:, None]
            if s.og:                                               # the output gate's strengths: no trace is needed, the gate acts at this byte
                q = rA * h * o * (1 - o); s.Ao[:, x] += q[:, None]; s.co += q
            s.Ac += rc * s.Ecx; s.cc += ra * s.Ecb; s.Az += rc * s.Ezx; s.cz += ra * s.Ezb
        if s.etaB:
            dB = np.outer(s.etaB * s.nB / (u @ u + 1) * sig, u); s.WB += dB; s.RB += dB.T; s.WB *= 1 - s.lam; s.RB *= 1 - s.lam
            s.V[:, x[0]] += s.etaB * sig; s.bB += s.etaB * sig
        d2 = np.outer(g * m, b); s.W2 += d2; s.b2 += g * m; s.R += d2.T; s.W2 *= 1 - s.lam; s.R *= 1 - s.lam
        return ok
    def flush(s):
        if not len(s.bx): return
        import fastl; fastl.run(s)
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
                x = s.cols(L[t], prev); h = s.step(h, x)[2]
                if s.C: prev = [L[t]] + prev[:-1]
                if t >= eq and L[t + 1] != END:
                    r = int(s.out(h, x)[3].argmax()) == L[t + 1]; ok = ok and r; pl[t - eq] += r
            n += ok
        return 100 * n / len(pairs), [round(100 * x / len(pairs)) for x in pl]

def run(kw):
    t0 = time.process_time(); tx = kw.get("text", "d1"); seed = int(kw.get("seed", 1)); lines = int(kw.get("lines", 60000)); every = int(kw.get("every", 10000))
    setup(tx); G = int(kw.get("gap", 4)) if tx in ("g1", "m1", "g2") else 0; gaps = sorted({0, G // 2, G})
    if tx in ("d1", "g1", "m1"):
        rng = np.random.default_rng(seed); P = [(a, b) for a in range(10) for b in range(10)]; rng.shuffle(P); held, shown = P[:20], P[20:]; shown_s = shown
    else:
        rng = np.random.default_rng([seed, 99]); o = rng.permutation(10000); P = [(int(q) // 100, int(q) % 100) for q in o]; held, shown = P[:500], P[2000:]; shown_s = shown[:500]
    F = lambda k, d: float(eval(kw.get(k, d)))
    s = Lay(seed, NA=int(kw.get("NA", 128)), NB=int(kw.get("NB", 104)), etaA=F("etaA", "1"), etaB=F("etaB", "1/4"), nB=F("nB", "8"), eta2=F("eta2", "1/16"), lam=F("lam", "1e-5"),
            chain=int(kw.get("chain", 0)), og=int(kw.get("og", 0)), ob=F("ob", "1"))
    s.fast = int(kw.get("fast", 0)); out = dict(label=kw["label"], args=kw, numbers=s.count(), rows=[]); right = n = 0
    for it in range(lines):
        a, b = shown[rng.integers(len(shown))]
        L = text(tx, a, b, bytes(LET[i] for i in rng.integers(4, size=rng.integers(G + 1))).decode()) if G else text(tx, a, b)
        right += s.line(L); n += len(L) - 1
        if (it + 1) % every == 0:
            s.flush(); right += s.right_acc; s.right_acc = 0; sh, _ = s.score(tx, shown_s); nv, pl = s.score(tx, held)
            out["rows"].append(dict(lines=it + 1, shown=sh, never=nv, place=pl, bytes_right=100 * right / n, gate_mean=float(sg(s.cz).mean()))); right = n = 0
            if G: out["rows"][-1]["shown_by_gap"] = {g: s.score(tx, shown_s, g)[0] for g in gaps}; out["rows"][-1]["shown"] = float(np.mean(list(out["rows"][-1]["shown_by_gap"].values())))
            if G and tx == "g2": out["rows"][-1]["never_by_gap"] = {g: s.score(tx, held, g)[0] for g in gaps}; out["rows"][-1]["never"] = float(np.mean(list(out["rows"][-1]["never_by_gap"].values())))
            if G and tx == "g2": out["rows"][-1]["place_never_at_gap"] = s.score(tx, held, G)[1]
    out["cpu"] = round(time.process_time() - t0); out["last3_shown"] = float(np.mean([r["shown"] for r in out["rows"][-3:]])); out["last3_never"] = float(np.mean([r["never"] for r in out["rows"][-3:]]))
    if G: out["last3_shown_by_gap"] = {g: float(np.mean([r["shown_by_gap"][g] for r in out["rows"][-3:]])) for g in gaps}
    if G and tx == "g2": out["last3_never_by_gap"] = {g: float(np.mean([r["never_by_gap"][g] for r in out["rows"][-3:]])) for g in gaps}
    s.flush(); run.last = s
    if kw.get("nowrite"): return out
    os.makedirs(os.path.join(HERE, "res"), exist_ok=True); json.dump(out, open(os.path.join(HERE, "res", f"lay_{kw['label']}.json"), "w"))
    if kw.get("save"): np.savez_compressed(os.path.join(HERE, "res", f"lay_{kw['label']}.npz"), **{k: getattr(s, k) for k in ("Ac", "Az", "Ao", "cc", "cz", "co", "WB", "V", "bB", "RB", "W2", "b2", "R", "h")})
    return out

if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[2:]); kw["label"] = sys.argv[1]; o = run(kw)
    print(o["label"], f"last3 shown {o['last3_shown']:.1f} never {o['last3_never']:.1f} |", " ".join(f"{r['lines']//1000}k {r['shown']:.0f}/{r['never']:.0f}" for r in o["rows"]), "| place", o["rows"][-1]["place"], "| bytes", round(o["rows"][-1]["bytes_right"]),
          "| by gap " + " ".join(f"{g}:{v:.0f}" for g, v in o["rows"][-1]["shown_by_gap"].items()) if "shown_by_gap" in o["rows"][-1] else "",
          "| never shown by gap " + " ".join(f"{g}:{v:.0f}" for g, v in o["rows"][-1]["never_by_gap"].items()) if "never_by_gap" in o["rows"][-1] else "",
          "| numbers", o["numbers"]["forward_total"], "+", o["numbers"]["return_strengths"], "return |", o["cpu"], "s")
