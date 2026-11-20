#!/usr/bin/env python3
"""PLAN_4.md: the system of keep.py (two-digit sums, cells in between that learn from a relayed mismatch) with options. numpy only.
    python3 cells2.py <label> seed=1 [plan=add:400000[,sub:200000[,add:40000]]] [every=50000] [fine=1] [reh=N] [swap=1]
                               [fan=K] [rewire=T] [dale=1] [H=256] [H2=0] [ans2=1] [freeze=1] [stiff1=k] [reserve=N] [lam2=x] [fast=1|2|3] [clip=p] [kap=1e-3] [cap=1] [eta1=1/4] [eta2=1/16] [lam=1e-5] [rscale=1]
    ans2=1: two answering cells; which one answers and learns is given by the line's operator (a control, laid down by the program).
    reserve=N: N further cells in between are silent and unchanged in the first phase; from the second phase on they alone change, with the answering
      cell's strengths from them, and everything the first phase wrote is left as it is (a control: the program decides where the new task is written).
    lam2=x: the fading of the answering cell's strengths and of the return connections is x from the second phase on (a control).
    fast=1: the inner loop runs compiled (kern.c through fastk.py; the same arithmetic, to rounding; not with ans2 or stiff1).
    clip=p (PLAN_8.md): a cell in between is silent below its threshold: output = drive cut off at 0 and 1; after every byte its threshold moves by
      kap times (its output less p). cap=0: no cut-off at 1. fast=2: the compiled kernel that works only on the active cells (clip only; none of the other options).
    fast=3: the dense kernel with the fading done as one scale factor (the plain system only: no fan, rewire, dale, H2, reserve, clip).
    nano/kinds (a copy of loop_small_case/cells2.py at 260c5d73, with two options; with both off the arithmetic is the original's; plain system only):
    clock=B: each cell in between changes its own strengths at its own rate, eta1 times B to the power u, u uniform from -1 to 1 at birth (its own
      stream), shifted so that the mean of u is nought (the geometric mean of the rates is eta1). clockam=1 (a control): every cell has the arithmetic mean of those rates.
    sig=beta [sigp=1 sigk=1]: the answering cell keeps a running mean a of the size of its mismatch (rate beta a byte) and the mean A of it over the run
      so far and sends a/A to every cell in between, which multiplies its step by sigk times (a/A) to the power sigp (the value sent before the byte).
    Not in this copy: fast=1, fast=2 for cells2 (kern2.c was not copied; kern.c's loop_steps is unchanged and does not have the two options).
    freeze=1: after the first phase the cells in between do not change. stiff1=k: each strength of a cell in between divides its step by 1 + (total size of its past changes)/k.
"""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
SYMS = list(b"0123456789+=\n-"); S = len(SYMS); SI = {c: i for i, c in enumerate(SYMS)}; END = SI[10]; D = 10; NL = D * S; OFF = np.arange(D) * S

def text(op, a, b):
    if op == "-" and a < b: a, b = b, a
    return [SI[c] for c in f"{a:02d}{op}{b:02d}={(a + b if op == '+' else a - b):03d}\n".encode()]

def sg(u): return 1 / (1 + np.exp(-u))

class Sys:
    def __init__(s, seed, H=256, eta1=.25, eta2=1 / 16, lam=1e-5, fan=0, rewire=0, dale=0, H2=0):
        wr = np.random.default_rng([seed, 20261119]); s.g2 = g2 = np.random.default_rng([seed, 20261121])
        s.H, s.eta1, s.eta2, s.lam, s.rewire, s.dale, s.H2 = H, eta1, eta2, lam, rewire, dale, H2; top = H2 or H
        s.W1 = wr.uniform(-.5, .5, (H, NL)); s.b1 = np.zeros(H); s.R = wr.uniform(-1, 1, (top, S)); s.W2 = np.zeros((S, top)); s.b2 = np.zeros(S)
        if H2: s.WB = g2.uniform(-1, 1, (H2, H)) / np.sqrt(H / 3); s.bB = np.zeros(H2); s.RB = g2.uniform(-1, 1, (H, H2))
        s.M = None; s.made = 0; s.two = 0; s.live = None; s.late = False
        if fan:
            s.M = np.zeros((H, NL), bool)
            for j in range(H): s.M[j, g2.choice(NL, fan, replace=False)] = True
            s.W1 *= s.M; s.age = np.zeros((H, NL), np.int32); s.nbytes = 0
        if dale:
            assert not H2
            s.W1 = np.abs(s.W1); s.b1 = np.full(H, -D * .25 * (fan / NL if fan else 1.0))      # lines excite; the threshold starts at the mean drive
            s.sign = np.where(g2.random(H) < .25, -1.0, 1.0)
        s.A = [(s.W2, s.b2, s.R), (np.zeros((S, top)), np.zeros(S), g2.uniform(-1, 1, (top, S)))]; s.stiff1 = 0.0; s.I1 = np.zeros((H, NL))
        s.hist = [END] * D; s.fast = 0; s.bx = []; s.by = []; s.clip = 0.0; s.kap = 0.0; s.cap = 1.0; s.stat = [0, 0, 0]
        s.rate = None; s.sig = 0.0; s.sigp = 1.0; s.sigk = 1.0; s.sa = 0.0; s.sS = 0.0; s.sn = 0; s.sM = 0.0
    def use(s, k): s.W2, s.b2, s.R = s.A[k]
    def fwd(s, hist):
        act = OFF + np.asarray(hist); u = s.W1[:, act].sum(1) + s.b1
        if s.clip: hA = np.clip(u, 0.0, s.cap); s.sens = ((u > 0) & (u < s.cap)).astype(float)
        else: hA = sg(u)
        if s.live is not None: hA = hA * s.live
        if s.H2:
            hB = sg(s.WB @ hA + s.bB); return act, hA, hB, s.W2 @ hB + s.b2
        return act, hA, hA, s.W2 @ hA + s.b2
    def learn(s, y):
        act, hA, h, a = s.fwd(s.hist); m = -a; m[y] += 1; g = s.eta2 / (h @ h + 1)
        sig = (s.R @ m) * (s.sens if s.clip else h * (1 - h))      # what a cell read by the answering cell receives, times its own sensitivity
        d2 = np.outer(g * m, h)
        if s.live is not None: sig = sig * s.free; d2 = d2 * s.free
        if s.H2:                                      # cells B change, and pass on what they received to cells A over their own return connections
            sA = (s.RB @ sig) * hA * (1 - hA); dB = np.outer(s.eta1 * D / (hA @ hA + 1) * sig, hA)
            s.WB += dB; s.bB += s.eta1 * sig; s.RB += dB.T; s.WB *= 1 - s.lam; s.RB *= 1 - s.lam; d = s.eta1 * sA
        else: d = s.eta1 * sig
        if s.rate is not None: d = d * s.rate                                                           # each cell's own clock
        if s.sig:                                                                                       # the common signal, as sent before this byte
            M = s.sigk * (s.sa / (s.sS / s.sn)) ** s.sigp if s.sn else s.sigk; d = d * M
            sz = float(np.sqrt(m @ m)); s.sa = sz if not s.sn else s.sa + s.sig * (sz - s.sa); s.sS += sz; s.sn += 1; s.sM += M
        s.R += d2.T; d1 = np.repeat(d[:, None], D, 1)
        if s.M is not None: d1 = d1 * s.M[:, act]
        if s.stiff1: d1 = d1 / (1 + s.I1[:, act] / s.stiff1); s.I1[:, act] += np.abs(d1)
        s.W1[:, act] += d1; s.b1 += d; s.W2 += d2
        if s.live is None: s.b2 += g * m; s.W2 *= 1 - s.lam; s.R *= 1 - s.lam
        else:                                         # reserve: only the cells that may change fade; the constants are the first phase's
            if not s.late: s.b2 += g * m
            s.W2 *= 1 - s.lam * s.free; s.R *= 1 - s.lam * s.free[:, None]
        if s.clip: s.b1 -= s.kap * (hA - s.clip); s.stat[0] += 1; s.stat[1] += int((hA > 0).sum()); s.stat[2] += int(s.sens.sum())
        if s.dale:
            s.W1[:, act] = np.maximum(s.W1[:, act], 0); s.W2 = np.where(s.W2 * s.sign > 0, s.W2, 0.0); s.R = np.where(s.R * s.sign[:, None] > 0, s.R, 0.0)
        if s.rewire:
            s.nbytes += 1
            if s.nbytes % s.rewire == 0: s.turnover()
    def turnover(s):
        """Each cell withdraws the weakest of its connections that are at least four periods old and makes one to a line it does not read."""
        s.age[s.M] += 1; aw = np.where(s.M & (s.age >= 4), np.abs(s.W1), np.inf); j = np.arange(s.H); weak = aw.argmin(1); ok = np.isfinite(aw[j, weak])
        free = s.g2.random((s.H, NL)); free[s.M] = -1.0; new = free.argmax(1); jj = j[ok]
        s.M[jj, weak[ok]] = False; s.W1[jj, weak[ok]] = 0.0; s.age[jj, weak[ok]] = 0; s.M[jj, new[ok]] = True; s.made += len(jj)
    def flush(s):
        if not len(s.bx): return
        assert s.fast == 3, "this copy has the dense kernel only (fast=3)"
        import fastk3; fastk3.run(s)
    def line(s, L):
        if s.fast:
            s.bx += L; s.by += L[1:]; s.by.append(-1)
            if len(s.bx) >= 500000: s.flush()
            return
        for t in range(len(L) - 1):
            s.hist.insert(0, L[t]); s.hist.pop(); s.learn(L[t + 1])
        s.hist.insert(0, L[-1]); s.hist.pop()
    def out(s, u):
        """The answering cell's branches for many lines at once (u: lines by cells); the same arithmetic as fwd."""
        hA = np.clip(u, 0.0, s.cap) if s.clip else sg(u)
        if s.live is not None: hA = hA * s.live
        return (sg(hA @ s.WB.T + s.bB) if s.H2 else hA) @ s.W2.T + s.b2
    def score(s, op, pairs, place=False):
        n = 0; pl = np.zeros(3); s.flush()
        if s.two: s.use(0 if op == '+' else 1)
        if s.fast:                                    # all pairs at once
            Ls = np.array([text(op, a, b) for a, b in pairs]); ok = np.ones(len(pairs), bool); W1T = s.W1.T
            for t in (5, 6, 7):
                hist = np.full((len(pairs), D), END); hist[:, :t + 1] = Ls[:, t::-1]; r = s.out(W1T[OFF + hist].sum(1) + s.b1).argmax(1) == Ls[:, t + 1]; ok &= r; pl[t - 5] = r.sum()
            return [int(round(100 * x / len(pairs))) for x in pl] if place else float(100 * ok.sum() / len(pairs))
        for a, b in pairs:
            L = text(op, a, b); hist = [END] * D; ok = True
            for t in range(len(L) - 1):
                hist.insert(0, L[t]); hist.pop()
                if t >= 5 and L[t + 1] != END:
                    r = int(s.fwd(hist)[3].argmax()) == L[t + 1]; ok = ok and r; pl[t - 5] += r
            n += ok
        return [round(100 * x / len(pairs)) for x in pl] if place else 100 * n / len(pairs)

def run(kw):
    t0 = time.process_time(); seed = int(kw.get("seed", 1)); rng = np.random.default_rng([seed, 99]); held, shown = {}, {}
    for op in "+-":
        o = rng.permutation(10000); P = [(int(q) // 100, int(q) % 100) for q in o]; held[op] = P[:500]; shown[op] = P[2000:]
    s = Sys(seed, H=int(kw.get("H", 256)) + int(kw.get("reserve", 0)), fan=int(kw.get("fan", 0)), rewire=int(kw.get("rewire", 0)), dale=int(kw.get("dale", 0)), H2=int(kw.get("H2", 0)),
            eta1=float(eval(kw.get("eta1", "1/4"))), eta2=float(eval(kw.get("eta2", "1/16"))), lam=float(kw.get("lam", 1e-5)))
    if "rscale" in kw: s.R *= float(kw["rscale"])              # the size of the return strengths at birth (0: they start at nothing and equal the forward strengths from then on)
    nres = int(kw.get("reserve", 0))
    if float(kw.get("clock", 0)): u = np.random.default_rng([seed, 20261124]).uniform(-1, 1, s.H); s.rate = float(kw["clock"]) ** (u - u.mean())
    if s.rate is not None and int(kw.get("clockam", 0)): s.rate = np.full(s.H, s.rate.mean())      # a control: all cells alike at the arithmetic mean of those rates
    s.sig = float(eval(kw.get("sig", "0"))); s.sigp = float(kw.get("sigp", 1)); s.sigk = float(kw.get("sigk", 1))
    assert not ((s.rate is not None or s.sig) and (s.H2 or nres or "fan" in kw or "dale" in kw or "clip" in kw)), "clock and sig are written for the plain system"
    if nres: s.live = np.arange(s.H) < s.H - nres; s.free = s.live.astype(float)
    s.clip = float(eval(kw.get("clip", "0"))); s.kap = float(kw.get("kap", 1e-3)); s.cap = 1.0 if int(kw.get("cap", 1)) else np.inf
    if s.clip: assert not s.H2; s.b1 -= 1.0                     # thresholds start where about one cell in seven is active
    s.two = int(kw.get("ans2", 0)); s.stiff1 = float(kw.get("stiff1", 0)); freeze = int(kw.get("freeze", 0)); s.fast = int(kw.get("fast", 0))
    reh = int(kw.get("reh", 0)); every = int(kw.get("every", 50000)); fine = int(kw.get("fine", 0)); out = dict(label=kw.get("label"), args=kw, rows=[]); snaps = {}
    TAB = {op: np.array([text(op, a, b) for a, b in shown[op]], np.int32) for op in "+-"}; TL = {op: TAB[op].tolist() for op in "+-"}
    def reading(it): return (it + 1) % every == 0 or (fine and it + 1 <= 20000 and (it + 1) % 2000 == 0) or (fine and it + 1 <= 100000 and (it + 1) % 10000 == 0)
    for pi, ph in enumerate(kw.get("plan", "add:400000").split(",")):
        name, n = ph.split(":"); op0 = {"add": "+", "sub": "-", "mix": None}[name]; s.flush()
        if freeze and pi > 0: s.eta1 = 0.0
        if nres and pi > 0: s.free = (~(np.arange(s.H) < s.H - nres)).astype(float); s.live = np.ones(s.H, bool); s.late = True
        if "lam2" in kw and pi > 0: s.lam = float(kw["lam2"])
        block = s.fast and op0 is not None and not (reh and pi > 0) and not s.two; it = 0      # one task, no rehearsal: the lines up to the next reading are drawn at once (the same stream)
        while block and it < int(n):
            nxt = next(j for j in range(it, int(n)) if reading(j) or j == int(n) - 1); X = TAB[op0][rng.integers(8000, size=nxt - it + 1)]
            s.bx = X.ravel(); s.by = np.concatenate([X[:, 1:], np.full((len(X), 1), -1, np.int32)], 1).ravel(); s.flush(); it = nxt + 1
            if reading(nxt): out["rows"].append(dict(phase=pi, name=name, lines=nxt + 1, add=s.score("+", held["+"]), sub=s.score("-", held["-"])))
        for it in range(0 if not block else int(n), int(n)):
            if op0 is None: op = "+-"[rng.integers(2)]
            elif reh and pi > 0 and rng.integers(reh) == 0: op = "+" if op0 == "-" else "-"
            else: op = op0
            L = TL[op][rng.integers(8000)]
            if s.two: s.use(0 if op == '+' else 1)
            s.line(L)
            if reading(it):
                out["rows"].append(dict(phase=pi, name=name, lines=it + 1, add=s.score("+", held["+"]), sub=s.score("-", held["-"])))
        s.flush(); snaps[pi] = {k: getattr(s, k).copy() for k in ("W1", "b1", "W2", "b2")}
        if pi == 1 and kw.get("swap"):
            cur = snaps[1]; old = snaps[0]; put = lambda d: [setattr(s, k, v.copy()) for k, v in d.items()]
            put(dict(cur, W2=old["W2"], b2=old["b2"])); a1 = s.score("+", held["+"])
            put(dict(cur, W1=old["W1"], b1=old["b1"])); a2 = s.score("+", held["+"]); put(cur)
            rel = lambda k: float(np.linalg.norm(cur[k] - old[k]) / np.linalg.norm(old[k]))
            out["where"] = dict(addition_with_the_answering_cell_put_back=a1, addition_with_the_cells_in_between_put_back=a2, change_of_W1_over_its_size=rel("W1"), change_of_W2_over_its_size=rel("W2"))
    out["by_place_add"] = s.score("+", held["+"], place=True); out["made"] = s.made; out["cpu"] = round(time.process_time() - t0, 1)
    if s.clip and s.stat[0]: out["active_cells_a_byte"] = s.stat[1] / s.stat[0]; out["changing_cells_a_byte"] = s.stat[2] / s.stat[0]
    r0 = [r for r in out["rows"] if r["phase"] == 0 and r["lines"] % every == 0]; out["last3_add_phase0"] = float(np.mean([r["add"] for r in r0[-3:]])) if r0 else None
    run.last = s
    if s.sig: out["sig_mean"] = s.sM / max(s.sn, 1); out["sig_ratio_end"] = s.sa / (s.sS / max(s.sn, 1)); out["sig_size_mean"] = s.sS / max(s.sn, 1)
    if kw.get("nowrite"): return out
    os.makedirs(os.path.join(HERE, "res"), exist_ok=True); json.dump(out, open(os.path.join(HERE, "res", f"c2_{kw['label']}.json"), "w")); return out

if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[2:]); kw["label"] = sys.argv[1]; o = run(kw)
    print(o["label"], "last3", o["last3_add_phase0"], "| " + " ".join(f"{r['phase']}:{r['lines']//1000}k {r['add']:.0f}/{r['sub']:.0f}" for r in o["rows"]), "| place", o["by_place_add"], "| made", o["made"], o.get("where", ""), (f"| active {o['active_cells_a_byte']:.1f} changing {o['changing_cells_a_byte']:.1f}" if "active_cells_a_byte" in o else ""), o["cpu"], "s")
