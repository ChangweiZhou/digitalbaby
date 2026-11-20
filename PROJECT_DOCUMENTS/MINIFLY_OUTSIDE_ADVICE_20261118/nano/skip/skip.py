#!/usr/bin/env python3
"""nano/skip: the delay-chain system of loop_small_case/cells2.py (the plain system: logistic cells in between, one answering cell, the mismatch
relayed back over return strengths that receive the forward strengths' changes), with a clock for the chain. numpy is the reference; fast=1 runs
the same arithmetic compiled (kern_s.c; check_skip.py compares the two).

Copied from loop_small_case and said so: the system and its constants from cells2.py (Sys.fwd, Sys.learn, the plain options only); the texts g2,
g1, m1, the letters used when scoring (fill) and the split into pairs shown and never shown from hold.py; the lazy fading from kern3.c.

    python3 skip.py <label> text=g2|m1|g1|d2 seed=1 lines=400000 every=50000 [gap=8] [H=256] [fast=1] [clock=...] [lam1=0] [theta=0.5] [ref=birth|mean]

The lines. Line 0 of each byte value carries the present byte. The chain is lines 1 to 9: the last nine bytes that ENTERED the chain. When a new byte
arrives, the byte that was present either enters the chain (which moves on by one) or is passed over and is on no line any more.
clock=none    every byte enters the chain (the system of cells2.py on this text).
clock=oracle  the letters are passed over. The program is told which bytes are letters: a check of the program only.
clock=values:<n>   n byte values drawn by chance (from the seed) are passed over (a control).
clock=bytes:<q>    each byte is passed over with chance q, whatever it is (a control).
clock=rule    the rule, which names no byte: a byte value v enters the chain only while Q(v) >= theta * Q0(v), where Q(v) is the sum, over the nine
              chain lines of v and over the cells in between, of the sizes of the strengths that the cells hold from those lines, and Q0(v) is the same
              sum at birth (ref=birth) or the mean of Q over the byte values at this moment (ref=mean).
clock=trial   (PLAN_2.md) each byte value has an entry to the chain that holds one number u, its tendency to be closed. Every P bytes of the stream
              ([P=256]) each entry is closed with chance c = logistic(u), kept between eps and 1 - eps ([eps=0.01]), and open otherwise; while it is
              closed its byte value is passed over. The answering cell sends out one number at every byte, the same to all entries: the sum of its
              squared mismatches. At the end of a period, with r that number's mean over the period, b the running mean of r and w the running mean of
              |r - b| ([beta=0.02]): every u moves by alpha * z * ((1 if it was closed, else 0) - c), with z = (b - r) / w kept within -3 and 3
              ([alpha=0.05]); u starts at logit(p0) ([p0=0.2]). alpha=0: the tendencies never change (the closing by chance, a control).
              form=evid (addition 1 to PLAN_2.md): an entry keeps instead four numbers, the sum and the count of z over the periods in which it was closed
              and over those in which it was open, all four fading by 1/win a period ([win=3000]); u = logit(p0) + alpha * t, with t the difference of
              the two means of z (closed less open) over the root of (1/(count closed + 1) + 1/(count open + 1)). alpha=0: closing by chance at p0.
              The scores are taken with every entry in its likelier state (closed where u > 0) and no drawing.
lam1          a cell's strengths from the lines that are active fade by this fraction at every byte at which they are used (0: as in cells2.py).
"""
import ctypes, json, os, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
SYMS = list(b"0123456789+=\n-wxyz"); S = len(SYMS); SI = {c: i for i, c in enumerate(SYMS)}; END = SI[10]; EQ = SI[ord("=")]; D = 10; NL = D * S; OFF = np.arange(D) * S
LET = b"wxyz"; LETS = [SI[c] for c in LET]

def fill(a, b, n): return bytes(LET[(7 * a + 3 * b + 5 * i * i + i) % 4] for i in range(n)).decode()          # the letters used when scoring: fixed by the pair (hold.py)
def text(tx, a, b, f=""):
    if tx in ("d2", "g2"): t = f"{a:02d}+{f}{b:02d}={a + b:03d}\n"
    else: t = f"{a}+{f}{b}={(a + b) % 10 if tx == 'm1' else a + b}\n"
    return [SI[c] for c in t.encode()]
def sg(u): return 1 / (1 + np.exp(-u))

# ---------------------------------------------------------------- the compiled kernel
I64, F64, PD, PI, P32, PU = ctypes.c_int64, ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_uint8)
class KNet(ctypes.Structure):
    _fields_ = [(k, I64) for k in "H NL S D mode pend".split()] + [(k, F64) for k in "eta1 eta2 lam lam1 theta".split()] + [(k, PD) for k in "W1 b1 W2 b2 R L0".split()] + [("hist", PI), ("nskip", PI), ("esum", F64), ("ecnt", I64)]
_lib = None
def lib():
    global _lib
    if _lib is None:
        so, src = os.path.join(HERE, "kern_s.so"), os.path.join(HERE, "kern_s.c")
        if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
            tmp = f"{so}.{os.getpid()}.tmp"; subprocess.check_call(["gcc", "-O3", "-march=native", "-ffast-math", "-shared", "-fPIC", "-o", tmp, src, "-lm"]); os.replace(tmp, so)
        _lib = ctypes.CDLL(so); _lib.skip_steps.restype = I64; _lib.skip_steps.argtypes = [ctypes.POINTER(KNet), I64, P32, P32, PU]
    return _lib
def pd(a):
    assert a.dtype == np.float64 and a.flags.c_contiguous and a.flags.writeable; return a.ctypes.data_as(PD)

# ---------------------------------------------------------------- the system
class Sk:
    def __init__(s, seed, H=256, eta1=.25, eta2=1 / 16, lam=1e-5, lam1=0.0):
        wr = np.random.default_rng([seed, 20261119]); s.H, s.eta1, s.eta2, s.lam, s.lam1 = H, eta1, eta2, lam, lam1
        s.W1 = wr.uniform(-.5, .5, (H, NL)); s.b1 = np.zeros(H); s.R = wr.uniform(-1, 1, (H, S)); s.W2 = np.zeros((S, H)); s.b2 = np.zeros(S)
        s.hist = [END] * D; s.pend = 1; s.rule = False; s.theta = 0.5; s.ref = "birth"; s.Q0 = s.Q(); s.nskip = np.zeros(S, np.int64); s.fast = 0; s.esum = 0.0; s.ecnt = 0
    def Qmat(s): return np.abs(s.W1).sum(0).reshape(D, S)                 # for each line: the sum of the sizes of the strengths that the cells hold from it
    def Q(s): return s.Qmat()[1:].sum(0)                                  # for each byte value: that sum over its nine chain lines
    def refQ(s): return s.Q0 if s.ref == "birth" else np.full(S, s.Q().mean())
    def enters(s, v):                                                     # the rule, for the byte value v
        q = np.abs(s.W1[:, OFF[1:] + v]).sum(); return bool(q >= s.theta * (s.Q0[v] if s.ref == "birth" else s.Q().mean()))
    def fwd(s, hist):
        h = sg(s.W1[:, OFF + np.asarray(hist)].sum(1) + s.b1); return h, s.W2 @ h + s.b2
    def read(s, x, y, fl=None):
        """The byte x arrives; the byte that was present enters the chain or is passed over. If y >= 0 the system names the next byte, y arrives, the cells change."""
        p = s.hist[0]; adm = bool(s.pend) if fl is not None else (s.enters(p) if s.rule else True)
        if adm: s.hist = [x, p] + s.hist[1:-1]
        else: s.hist[0] = x; s.nskip[p] += 1
        if fl is not None: s.pend = int(fl)
        if y < 0: return
        act = OFF + np.asarray(s.hist); h, a = s.fwd(s.hist); m = -a; m[y] += 1; s.esum += float(m @ m); s.ecnt += 1; g = s.eta2 / (h @ h + 1)
        d = s.eta1 * (s.R @ m) * h * (1 - h); d2 = np.outer(g * m, h)
        s.R += d2.T; s.W1[:, act] = (s.W1[:, act] + d[:, None]) * (1 - s.lam1); s.b1 += d; s.W2 += d2; s.b2 += g * m; s.W2 *= 1 - s.lam; s.R *= 1 - s.lam
    def run(s, xs, ys, fl=None):
        if not s.fast:
            for i in range(len(xs)): s.read(int(xs[i]), int(ys[i]), None if fl is None else fl[i])
            return
        n = KNet(); hist = np.asarray(s.hist, np.int64); L0 = np.ascontiguousarray(s.refQ(), np.float64)
        assert s.ref == "birth" or not s.rule, "the compiled kernel has the rule with ref=birth only"
        n.H, n.NL, n.S, n.D, n.mode, n.pend = s.H, NL, S, D, 2 if s.rule else 0, s.pend; n.esum = 0.0; n.ecnt = 0; n.eta1, n.eta2, n.lam, n.lam1, n.theta = s.eta1, s.eta2, s.lam, s.lam1, s.theta
        n.W1, n.b1, n.W2, n.b2, n.R, n.L0 = pd(s.W1), pd(s.b1), pd(s.W2), pd(s.b2), pd(s.R), pd(L0); n.hist = hist.ctypes.data_as(PI); n.nskip = s.nskip.ctypes.data_as(PI)
        xs = np.ascontiguousarray(xs, np.int32); ys = np.ascontiguousarray(ys, np.int32); f = None if fl is None else np.ascontiguousarray(fl, np.uint8)
        lib().skip_steps(ctypes.byref(n), len(xs), xs.ctypes.data_as(P32), ys.ctypes.data_as(P32), None if f is None else f.ctypes.data_as(PU))
        s.hist = [int(v) for v in hist]; s.pend = int(n.pend); s.esum += float(n.esum); s.ecnt += int(n.ecnt)
    def score(s, tx, pairs, gap, admit):
        """Whole answer right, %, and by place. Every line is read from the state left by the last line of the stream; nothing changes.
        admit(byte value) says whether a byte enters the chain (the clock as it stands now)."""
        hists = []; want = []; own = []
        for k, (a, b) in enumerate(pairs):
            L = text(tx, a, b, fill(a, b, gap)); eq = L.index(EQ); hist = list(s.hist); pend = bool(s.pend)
            for t in range(len(L) - 1):
                p = hist[0]
                if pend: hist = [L[t], p] + hist[1:-1]
                else: hist[0] = L[t]
                pend = admit(L[t])
                if t >= eq and L[t + 1] != END: hists.append(list(hist)); want.append(L[t + 1]); own.append((k, t - eq))
        hists = np.array(hists); out = sg(s.W1.T[OFF + hists].sum(1) + s.b1) @ s.W2.T + s.b2; r = out.argmax(1) == np.array(want)
        ok = np.ones(len(pairs), bool); npl = max(o[1] for o in own) + 1; pl = np.zeros(npl)
        for (k, j), rr in zip(own, r): ok[k] &= bool(rr); pl[j] += rr
        return float(100 * ok.mean()), [int(round(100 * x / len(pairs))) for x in pl]

def stream(tx, shown, G, n, rng):
    """n lines drawn from the pairs shown, with 0 to G letters drawn afresh for every line; the bytes, and for each byte the next one (-1 at a line's end)."""
    P = np.asarray(shown)[rng.integers(len(shown), size=n)]; a, b = P[:, 0], P[:, 1]; k = rng.integers(G + 1, size=n) if G else np.zeros(n, np.int64)
    let = np.asarray(LETS)[rng.integers(4, size=(n, max(G, 1)))]; let = np.where(np.arange(max(G, 1))[None, :] < k[:, None], let, -1)
    col = lambda v: np.asarray(v)[:, None]; const = lambda c: np.full((n, 1), SI[ord(c)] if isinstance(c, str) else c)
    if tx in ("d2", "g2"): c = a + b; M = [col(a // 10), col(a % 10), const("+"), let, col(b // 10), col(b % 10), const("="), col(c // 100), col(c // 10 % 10), col(c % 10), const(END)]
    elif tx == "m1": M = [col(a), const("+"), let, col(b), const("="), col((a + b) % 10), const(END)]
    else:
        c = a + b; M = [col(a), const("+"), let, col(b), const("="), np.where(col(c) >= 10, 1, -1), col(c % 10), const(END)]
    M = np.concatenate(M, 1); xs = M[M >= 0].astype(np.int32); ys = np.empty_like(xs); ys[:-1] = xs[1:]; ys[-1] = -1; ys[xs == END] = -1
    return xs, ys

def run(kw):
    t0 = time.process_time(); tx = kw.get("text", "g2"); seed = int(kw.get("seed", 1)); lines = int(kw.get("lines", 400000)); every = int(kw.get("every", 50000))
    G = int(kw.get("gap", 8)) if tx in ("g1", "m1", "g2") else 0; gaps = sorted({0, G // 2, G})
    if tx in ("g1", "m1"):
        rng = np.random.default_rng(seed); P = [(a, b) for a in range(10) for b in range(10)]; rng.shuffle(P); held, shown = P[:20], P[20:]; shown_s = shown
    else:
        rng = np.random.default_rng([seed, 99]); o = rng.permutation(10000); P = [(int(q) // 100, int(q) % 100) for q in o]; held, shown = P[:500], P[2000:]; shown_s = shown[:500]
    s = Sk(seed, H=int(kw.get("H", 256)), lam1=float(kw.get("lam1", 0))); s.fast = int(kw.get("fast", 0)); s.theta = float(kw.get("theta", 0.5)); s.ref = kw.get("ref", "birth")
    ck = kw.get("clock", "none"); crng = np.random.default_rng([seed, 20261124]); passed = np.zeros(S, bool); q = 0.0
    if ck == "oracle": passed[LETS] = True
    elif ck.startswith("values:"): used = [v for v in range(S) if v != SI[ord("-")]]; passed[crng.choice(used, int(ck[7:]), replace=False)] = True      # among the 17 byte values that occur
    elif ck.startswith("bytes:"): q = float(ck[6:])
    elif ck == "rule": s.rule = True
    elif ck == "trial":
        P = int(kw.get("P", 256)); alpha = float(kw.get("alpha", 0.05)); eps = float(kw.get("eps", 0.01)); beta = float(kw.get("beta", 0.02)); p0 = float(kw.get("p0", 0.2))
        u = np.full(S, np.log(p0 / (1 - p0))); tr = dict(b=None, w=None, n=0); form = kw.get("form", "grad"); gam = 1 - 1 / float(kw.get("win", 3000)); ev = np.zeros((4, S))
    else: assert ck == "none", ck
    trial = ck == "trial"; given = ck != "none" and not s.rule and not trial
    def trial_block(xs, ys):
        """The stream in periods of P bytes; in each, the entries draw their states, the bytes are read, and the tendencies move."""
        nonlocal u
        for i in range(0, len(xs), P):
            c = np.clip(sg(u), eps, 1 - eps); closed = crng.random(S) < c; x = xs[i:i + P]; s.esum = 0.0; s.ecnt = 0; s.run(x, ys[i:i + P], ~closed[x])
            if s.ecnt < P // 4: continue                                   # a short remainder at the end of a block of lines: no change
            r = s.esum / s.ecnt; tr["n"] += 1
            if tr["b"] is None: tr["b"] = r; tr["w"] = 0.05 * r; continue
            z = float(np.clip((tr["b"] - r) / max(tr["w"], 1e-12), -3, 3))
            if form == "grad": u = np.clip(u + alpha * z * (closed - c), -9, 9)
            else:                                                          # form=evid (PLAN_2.md, addition 1): ev = sums and counts of z over the closed and the open periods, fading
                ev *= gam; ev[0] += z * closed; ev[1] += closed; ev[2] += z * ~closed; ev[3] += ~closed
                u = np.log(p0 / (1 - p0)) + alpha * (ev[0] / (ev[1] + 1) - ev[2] / (ev[3] + 1)) / np.sqrt(1 / (ev[1] + 1) + 1 / (ev[3] + 1))
            tr["w"] += beta * (abs(r - tr["b"]) - tr["w"]); tr["b"] += beta * (r - tr["b"])
    def admit_now():
        if s.rule: e = [s.enters(v) for v in range(S)]; return lambda v: e[v]
        if trial: e = u <= 0; return lambda v: bool(e[v])
        if q: return lambda v: bool(crng.random() >= q)
        return lambda v: not passed[v]
    out = dict(label=kw["label"], args=kw, rows=[], passed_by_program=[chr(SYMS[v]) for v in np.flatnonzero(passed)]); done = 0; nb = 0
    while done < lines:
        n = min(every, lines - done); xs, ys = stream(tx, shown, G, n, rng)
        fl = None if not given else (crng.random(len(xs)) >= q) if q else ~passed[xs]
        if trial: trial_block(xs, ys)
        else: s.run(xs, ys, fl)
        done += n; nb += len(xs); adm = admit_now()
        row = dict(lines=done, bytes=nb, never_by_gap={}, shown_by_gap={}, place={})
        for g in gaps:
            row["shown_by_gap"][g], pl = s.score(tx, shown_s, g, adm)
            if tx not in ("g1", "m1"): row["never_by_gap"][g], pl = s.score(tx, held, g, adm)
            row["place"][g] = pl
        row["Q"] = [round(float(x), 2) for x in s.Q()]; row["Qmat"] = [[round(float(x), 1) for x in r] for r in s.Qmat()]; row["nskip"] = [int(x) for x in s.nskip]
        if s.rule: row["passed_over_now"] = [chr(SYMS[v]) for v in range(S) if not s.enters(v)]
        if trial: row["passed_over_now"] = [chr(SYMS[v]) for v in range(S) if u[v] > 0]; row["u"] = [round(float(x), 3) for x in u]; row["mismatch_mean"] = tr["b"]; row["periods"] = tr["n"]
        out["rows"].append(row)
    out["Q0"] = [round(float(x), 2) for x in s.Q0]; out["syms"] = [chr(c) for c in SYMS]; out["cpu"] = round(time.process_time() - t0, 1)
    key = "shown_by_gap" if tx in ("g1", "m1") else "never_by_gap"; out["scored"] = key
    out["last3_by_gap"] = {g: float(np.mean([r[key][g] for r in out["rows"][-3:]])) for g in gaps}; out["last3_min"] = min(out["last3_by_gap"].values())
    run.last = s
    if not kw.get("nowrite"): os.makedirs(os.path.join(HERE, "res"), exist_ok=True); json.dump(out, open(os.path.join(HERE, "res", f"{kw['label']}.json"), "w"))
    return out

if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[2:]); kw["label"] = sys.argv[1]; o = run(kw); r = o["rows"][-1]
    print(o["label"], "| last3 by gap", " ".join(f"{g}:{v:.1f}" for g, v in o["last3_by_gap"].items()), "|", " ".join(f"{r_['lines']//1000}k " + "/".join(f"{v:.0f}" for v in r_[o['scored']].values()) for r_ in o["rows"]),
          "| place", r["place"], "| passed over", [(r_["lines"] // 1000, "".join(r_["passed_over_now"]).replace("\n", "N")) for r_ in o["rows"]] if "u" in r else r.get("passed_over_now", o["passed_by_program"]),
          ("| u " + " ".join(f"{chr(c) if c != 10 else 'nl'}:{x:.1f}" for c, x in zip(SYMS, r["u"]))) if "u" in r else "", "| Q/Q0", " ".join(f"{chr(c) if c != 10 else 'nl'}:{a / b:.2f}" for c, a, b in zip(SYMS, r["Q"], o["Q0"])), "|", o["cpu"], "s")
